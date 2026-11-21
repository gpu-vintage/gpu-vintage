"""Decoding with CUDA graphs, and prefill in micro-batches.

A decode step of a small model launches several hundred short kernels; launched
one by one from Python, their overhead depends on the host CPU and can exceed
the GPU time itself, which would make GPUs of different speed look alike. The
step is therefore captured once per (batch size, attention span) in a CUDA
graph and replayed: the GPU then runs the step back to back, as optimized
serving engines do. Spans come in buckets of `bucket` positions, so a step at
position p reads the smallest bucket >= p + 1 of the cache, not all of it.
"""

from __future__ import annotations

import torch

from .model import Model


def spans_for(start: int, max_len: int, bucket: int) -> list[int]:
    """Bucket ends covering positions start .. max_len-1."""
    spans, s = [], ((start // bucket) + 1) * bucket
    while s < max_len:
        spans.append(s)
        s += bucket
    spans.append(max_len)
    return spans


def span_for(pos: int, spans: list[int]) -> int:
    for s in spans:
        if s >= pos + 1:
            return s
    raise ValueError(f"position {pos} beyond the cache")


@torch.inference_mode()
def prefill_all(model: Model, prompts: torch.Tensor, cache: torch.Tensor,
                max_tokens: int = 8192) -> torch.Tensor:
    """Prefill every row, at most max_tokens prompt tokens per forward pass.
    Returns the first generated token of each row (greedy)."""
    B, t = prompts.shape
    rows = max(1, max_tokens // t)
    out = []
    for r0 in range(0, B, rows):
        logits = model.prefill(prompts[r0:r0 + rows], cache, row0=r0)
        out.append(torch.argmax(logits, dim=-1))
    return torch.cat(out)


class GraphDecoder:
    """Greedy decoding of every row of `cache`, one CUDA graph per span bucket.

    The graphs read the current tokens from `self.tok` and the position from
    `self.pos`, write the next tokens back into `self.tok` and advance
    `self.pos`, so the host only chooses which graph to replay.
    """

    def __init__(self, model: Model, cache: torch.Tensor, start: int, max_len: int,
                 bucket: int = 64, use_graphs: bool = True) -> None:
        self.model = model
        self.cache = cache
        self.start = start
        self.max_len = max_len
        B = cache.shape[2]
        dev = cache.device
        self.tok = torch.zeros(B, dtype=torch.long, device=dev)
        self.pos = torch.full((1,), start, dtype=torch.long, device=dev)
        self.finite = torch.ones((), dtype=torch.bool, device=dev)
        self.spans = spans_for(start, max_len, bucket)
        self.graphs: dict[int, torch.cuda.CUDAGraph] = {}
        self.use_graphs = use_graphs and dev.type == "cuda"
        if self.use_graphs:
            self._capture()

    @torch.inference_mode()
    def _eager(self, span: int) -> None:
        logits = self.model.decode_step(self.tok, self.pos, self.cache, span)
        self.finite.copy_(torch.isfinite(logits).all())
        self.tok.copy_(torch.argmax(logits, dim=-1))
        self.pos.add_(1)

    def _capture(self) -> None:
        saved_tok, saved_pos = self.tok.clone(), self.pos.clone()
        side = torch.cuda.Stream()
        side.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(side):
            for span in self.spans:  # warm-up, as torch.cuda.graph requires
                self.pos.fill_(span - 1)
                for _ in range(2):
                    self.pos.fill_(span - 1)
                    self._eager(span)
        torch.cuda.current_stream().wait_stream(side)
        torch.cuda.synchronize()
        pool = torch.cuda.graph_pool_handle()
        for span in self.spans:
            self.pos.fill_(span - 1)
            g = torch.cuda.CUDAGraph()
            with torch.cuda.graph(g, pool=pool):
                self._eager(span)
            self.graphs[span] = g
        torch.cuda.synchronize()
        self.tok.copy_(saved_tok)
        self.pos.copy_(saved_pos)

    def reset(self, first_tokens: torch.Tensor) -> None:
        """Start a new pass at position `start` from the prefill's first tokens.
        The keys and values of positions >= start are overwritten as decoding goes."""
        self.tok.copy_(first_tokens)
        self.pos.fill_(self.start)

    def step(self, p: int) -> None:
        """Enqueue the step that writes position p (self.pos must equal p on the GPU)."""
        span = span_for(p, self.spans)
        if self.use_graphs:
            self.graphs[span].replay()
        else:
            self._eager(span)

    def run(self, steps: int) -> None:
        """Enqueue `steps` decoding steps from position `start` (no synchronization)."""
        if self.start + steps > self.max_len:
            raise ValueError("more steps than the cache holds")
        for s in range(steps):
            self.step(self.start + s)

    def close(self) -> None:
        self.graphs.clear()
