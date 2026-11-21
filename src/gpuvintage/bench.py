"""The measurement protocol: one model on one GPU, a list of batch sizes, one
request shape (prompt and output lengths). Writes three JSON sidecars into the
run directory: config.json (what was asked), env.json (where it ran) and
metrics.json (what was measured), the last after every batch size so that a
resubmitted job skips the batch sizes already done.

For each batch size B:

- prefill window: whole prefills of B prompts of `prompt_len` tokens, repeated
  until the window lasts at least `min_seconds`, after `warmup_seconds` of the
  same work;
- decode window: passes of `gen_len` greedy steps over positions prompt_len ..
  prompt_len+gen_len-1 (each pass restarts at prompt_len, from the prefill's
  first tokens), repeated likewise.

Energy is the GPU board energy over each window (power.PowerMeter). An
out-of-memory error at some B is recorded as such, and larger B are not tried.
"""

from __future__ import annotations

import gc
import json
import os
import sys
import time
from pathlib import Path

import torch

from .engine import GraphDecoder, prefill_all
from .model import Model
from .power import PowerMeter

SCHEMA = 1


def _write(path: Path, obj: dict) -> None:
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(obj, indent=1, sort_keys=True) + "\n")
    os.replace(tmp, path)


def resolve_model(name: str) -> tuple[Path, str]:
    """A local checkpoint directory, or a Hugging Face repo id read from the
    local cache only (jobs run offline). Returns (path, revision)."""
    p = Path(name)
    if p.is_dir():
        return p, "local"
    from huggingface_hub import snapshot_download

    path = Path(snapshot_download(name, local_files_only=True))
    return path, path.name


def environment(meter: PowerMeter | None) -> dict:
    """Library versions, thread counts and the GPU model; nothing that names the machine."""
    env = {
        "cpus_allowed": len(os.sched_getaffinity(0)) if hasattr(os, "sched_getaffinity") else None,
        "python": sys.version.split()[0],
        "torch": torch.__version__,
        "torch_cuda": torch.version.cuda,
        "cudnn": torch.backends.cudnn.version() if torch.cuda.is_available() else None,
        "torch_threads": torch.get_num_threads(),
    }
    if torch.cuda.is_available():
        props = torch.cuda.get_device_properties(0)
        env["gpu"] = {
            "torch_name": props.name,
            "capability": f"{props.major}.{props.minor}",
            "multiprocessors": props.multi_processor_count,
            "total_memory_bytes": props.total_memory,
        }
    if meter is not None:
        env["nvml"] = meter.static_info()
    return env


def prompts_for(batch: int, length: int, vocab: int, seed: int = 0) -> torch.Tensor:
    """Synthetic prompts: token ids drawn uniformly from [1000, min(vocab, 100000)).
    Dense-model compute does not depend on which tokens are fed."""
    g = torch.Generator().manual_seed(seed)
    return torch.randint(1000, min(vocab, 100_000), (batch, length), generator=g)


def _window(work, meter: PowerMeter, min_seconds: float, warmup_seconds: float,
            min_iters: int = 1) -> dict:
    """Run `work()` (enqueues one unit, returns tokens in it) for warmup_seconds,
    then measure a window of at least min_seconds and min_iters units."""
    t0 = time.perf_counter()
    while True:
        work()
        torch.cuda.synchronize()
        if time.perf_counter() - t0 >= warmup_seconds:
            break
    mark = meter.begin()
    t1 = time.perf_counter()
    iters, tokens = 0, 0
    while True:
        tokens += work()
        iters += 1
        torch.cuda.synchronize()
        if iters >= min_iters and time.perf_counter() - t1 >= min_seconds:
            break
    out = meter.end(mark)
    out.update(iterations=iters, tokens=tokens)
    return out


def check_graphs(model: Model, prompt_len: int, steps: int = 8, batch: int = 2) -> dict:
    """Decode a few steps with graphs and eagerly from the same prefill, and
    compare the tokens (a correctness check of the capture)."""
    prompts = prompts_for(batch, prompt_len, model.cfg.vocab_size, seed=1).to(model.device)
    max_len = prompt_len + steps
    out = {}
    for mode in ("eager", "graph"):
        cache = model.new_cache(batch, max_len)
        first = prefill_all(model, prompts, cache)
        dec = GraphDecoder(model, cache, prompt_len, max_len, use_graphs=(mode == "graph"))
        dec.reset(first)
        toks = []
        for s in range(steps):
            dec.step(prompt_len + s)
            toks.append(dec.tok.clone())
        out[mode] = torch.stack(toks, 1).cpu()
        dec.close()
        del cache, dec
    agree = (out["eager"] == out["graph"]).float().mean().item()
    return {"graph_eager_token_agreement": agree, "steps": steps, "batch": batch}


def run(args) -> int:
    run_dir = Path(args.run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    dtype = {"fp16": torch.float16, "fp32": torch.float32, "bf16": torch.bfloat16}[args.dtype]
    model_path, revision = resolve_model(args.model)
    config = {
        "schema": SCHEMA,
        "model": args.model,
        "model_revision": revision,
        "dtype": args.dtype,
        "prompt_len": args.prompt_len,
        "gen_len": args.gen_len,
        "batches": args.batches,
        "min_seconds": args.min_seconds,
        "warmup_seconds": args.warmup_seconds,
        "idle_seconds": args.idle_seconds,
        "prefill_max_tokens": args.prefill_max_tokens,
        "bucket": args.bucket,
        "use_graphs": not args.no_graphs,
        "gpu_tag": args.gpu_tag,
        "command": " ".join(sys.argv),
    }
    cfg_path, met_path = run_dir / "config.json", run_dir / "metrics.json"
    if cfg_path.is_file():
        old = json.loads(cfg_path.read_text())
        keys = [k for k in config if k not in ("command", "batches")]
        if any(old.get(k) != config[k] for k in keys):
            sys.exit(f"error: {cfg_path} exists with another configuration; use a new --run-dir")
    _write(cfg_path, config)
    metrics = json.loads(met_path.read_text()) if met_path.is_file() else {"schema": SCHEMA, "batches": {}}

    torch.manual_seed(0)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    meter = PowerMeter(0)
    try:
        _write(run_dir / "env.json", environment(meter))
        t = time.perf_counter()
        model = Model.from_pretrained(model_path, device="cuda", dtype=dtype)
        torch.cuda.synchronize()
        metrics["load_seconds"] = time.perf_counter() - t
        metrics["parameters"] = model.num_parameters()
        metrics["weight_bytes"] = model.weight_bytes()
        metrics["model_config"] = model.cfg.to_dict()
        metrics["other_processes_at_start"] = meter.other_processes()

        if "idle" not in metrics:
            torch.cuda.synchronize()
            time.sleep(2.0)
            mark = meter.begin()
            time.sleep(args.idle_seconds)
            metrics["idle"] = meter.end(mark)
            _write(met_path, metrics)
        if "graph_check" not in metrics and not args.no_graphs:
            metrics["graph_check"] = check_graphs(model, min(args.prompt_len, 128))
            _write(met_path, metrics)
            torch.cuda.empty_cache()

        max_len = args.prompt_len + args.gen_len
        for B in args.batches:
            key = str(B)
            if key in metrics["batches"]:
                print(f"B={B}: done already ({metrics['batches'][key].get('status')})", flush=True)
                if metrics["batches"][key].get("status") == "oom":
                    break
                continue
            rec: dict = {"batch": B, "cache_bytes": model.cache_bytes(B, max_len)}
            cache = dec = None
            try:
                torch.cuda.reset_peak_memory_stats()
                cache = model.new_cache(B, max_len)
                prompts = prompts_for(B, args.prompt_len, model.cfg.vocab_size).to(model.device)
                rows = max(1, args.prefill_max_tokens // args.prompt_len)
                logits = model.prefill(prompts[:rows], cache)
                rec["prefill_logits_finite"] = bool(torch.isfinite(logits).all().item())
                del logits
                first = prefill_all(model, prompts, cache, args.prefill_max_tokens)
                torch.cuda.synchronize()
                t = time.perf_counter()
                dec = GraphDecoder(model, cache, args.prompt_len, max_len, bucket=args.bucket,
                                   use_graphs=not args.no_graphs)
                torch.cuda.synchronize()
                rec["capture_seconds"] = time.perf_counter() - t

                def prefill_work() -> int:
                    prefill_all(model, prompts, cache, args.prefill_max_tokens)
                    return B * args.prompt_len

                def decode_work() -> int:
                    dec.reset(first)
                    dec.run(args.gen_len)
                    return B * args.gen_len

                rec["prefill"] = _window(prefill_work, meter, args.min_seconds, args.warmup_seconds)
                rec["decode"] = _window(decode_work, meter, args.min_seconds, args.warmup_seconds)
                rec["decode_logits_finite"] = bool(dec.finite.item())
                rec["peak_memory_bytes"] = torch.cuda.max_memory_allocated()
                rec["other_processes"] = meter.other_processes()
                rec["status"] = "ok"
                p, d = rec["prefill"], rec["decode"]
                print(f"B={B}: prefill {p['tokens'] / p['seconds']:.0f} tok/s, "
                      f"decode {d['tokens'] / d['seconds']:.1f} tok/s, "
                      f"decode {(d['energy_counter_j'] or d['energy_sampled_j']) / d['tokens']:.4f} J/tok",
                      flush=True)
            except torch.cuda.OutOfMemoryError as e:
                rec["status"] = "oom"
                rec["error"] = str(e).split("\n")[0][:300]
                print(f"B={B}: out of memory", flush=True)
            finally:
                if dec is not None:
                    dec.close()
                del cache, dec
                gc.collect()
                torch.cuda.empty_cache()
            metrics["batches"][key] = rec
            _write(met_path, metrics)
            if rec["status"] == "oom":
                break
        metrics["finished"] = True
        _write(met_path, metrics)
    finally:
        meter.close()
    return 0
