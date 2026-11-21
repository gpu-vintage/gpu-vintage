#!/usr/bin/env python3
"""E3: the same requests through Hugging Face `generate` (eager PyTorch, dynamic
cache, SDPA attention), end to end, with the same energy meter as bench.py.
Compared with the engine's prefill + decode time per request, it shows how much
of the measured gap between GPUs the software stack moves.

    python scripts/bench_hf.py --model Qwen/Qwen3-1.7B --batches 1 16 --run-dir out/runs/<name>
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import torch  # noqa: E402

from gpuvintage.bench import _window, _write, environment, prompts_for, resolve_model  # noqa: E402
from gpuvintage.power import PowerMeter  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", required=True)
    ap.add_argument("--dtype", default="fp16", choices=["fp16", "fp32"])
    ap.add_argument("--prompt-len", type=int, default=512)
    ap.add_argument("--gen-len", type=int, default=256)
    ap.add_argument("--batches", type=int, nargs="+", default=[1, 16])
    ap.add_argument("--min-seconds", type=float, default=10.0)
    ap.add_argument("--warmup-seconds", type=float, default=2.0)
    ap.add_argument("--gpu-tag", default="")
    ap.add_argument("--run-dir", required=True)
    a = ap.parse_args()

    from transformers import AutoModelForCausalLM

    run_dir = Path(a.run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    path, rev = resolve_model(a.model)
    dtype = {"fp16": torch.float16, "fp32": torch.float32}[a.dtype]
    _write(run_dir / "config.json", {"schema": 1, "kind": "hf_generate", "model": a.model,
                                     "model_revision": rev, "dtype": a.dtype,
                                     "prompt_len": a.prompt_len, "gen_len": a.gen_len,
                                     "batches": a.batches, "min_seconds": a.min_seconds,
                                     "warmup_seconds": a.warmup_seconds, "gpu_tag": a.gpu_tag,
                                     "attn_implementation": "sdpa", "command": " ".join(sys.argv)})
    met_path = run_dir / "metrics.json"
    metrics = json.loads(met_path.read_text()) if met_path.is_file() else {"schema": 1, "batches": {}}
    meter = PowerMeter(0)
    try:
        import transformers
        env = environment(meter)
        env["transformers"] = transformers.__version__
        _write(run_dir / "env.json", env)
        model = AutoModelForCausalLM.from_pretrained(path, torch_dtype=dtype,
                                                     attn_implementation="sdpa").cuda().eval()
        for B in a.batches:
            if str(B) in metrics["batches"]:
                continue
            prompts = prompts_for(B, a.prompt_len, model.config.vocab_size).cuda()
            mask = torch.ones_like(prompts)

            def work() -> int:
                with torch.inference_mode():
                    out = model.generate(prompts, attention_mask=mask, max_new_tokens=a.gen_len,
                                         min_new_tokens=a.gen_len, do_sample=False,
                                         pad_token_id=model.config.eos_token_id)
                assert out.shape[1] == a.prompt_len + a.gen_len
                return B * a.gen_len

            try:
                rec = {"batch": B, "request": _window(work, meter, a.min_seconds, a.warmup_seconds),
                       "status": "ok"}
            except torch.cuda.OutOfMemoryError as e:
                rec = {"batch": B, "status": "oom", "error": str(e).split("\n")[0][:300]}
            metrics["batches"][str(B)] = rec
            _write(met_path, metrics)
            torch.cuda.empty_cache()
            if rec["status"] == "ok":
                r = rec["request"]
                print(f"B={B}: {r['tokens'] / r['seconds']:.1f} output tok/s end to end", flush=True)
        metrics["finished"] = True
        _write(met_path, metrics)
    finally:
        meter.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
