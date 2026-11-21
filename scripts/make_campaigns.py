#!/usr/bin/env python3
"""Write the campaign lists (campaigns/*.tsv), one row per run: the run name,
the GPU it runs on, a time limit and the benchmark arguments.

    python3 scripts/make_campaigns.py

E1: five Qwen3 models on every GPU that holds them, batches of 1 to 256.
E2: single precision on the two Pascal cards.
E3: Hugging Face `generate` against the engine, on four architectures.
E4: two other request shapes (long outputs, long prompts).
Every run is repeated three times (-r1, -r2, -r3).
"""

from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

# GPU tag -> memory in GB
GPUS = {
    "1080ti": 11,    # GeForce GTX 1080 Ti
    "p100": 12,      # Tesla P100 PCIe 12 GB
    "v100": 16,      # Tesla V100 PCIe 16 GB
    "t4": 16,        # Tesla T4
    "a100": 40,      # A100 SXM4 40 GB
    "a40": 48,       # A40
    "ada6000": 48,   # RTX 6000 Ada Generation
}

# Qwen3 dense models and their fp16 weight size in GB (parameters x 2 bytes)
MODELS = {
    "Qwen/Qwen3-0.6B": 1.5,
    "Qwen/Qwen3-1.7B": 4.1,
    "Qwen/Qwen3-4B": 8.0,
    "Qwen/Qwen3-8B": 16.4,
    "Qwen/Qwen3-14B": 29.5,
}

REPS = (1, 2, 3)

# The GTX 1080 Ti (GP102) runs fp16 GEMMs at its native fp16 rate, 1/64 of fp32,
# so fp32 is its working precision: E2 measures it, and E3/E4 use it. E1 keeps
# only its 0.6B fp16 runs, to document the penalty; larger fp16 runs would
# exceed any time limit.
DTYPE = {"1080ti": "fp32"}
FP16_ONLY_SMALL = {"1080ti": ("Qwen/Qwen3-0.6B",)}
HEADER = "run\tgpu\ttime\targs"


def short(model: str) -> str:
    return model.split("/")[-1].lower().replace("qwen3-", "q3-")


def fits(model: str, mem_gb: int, dtype: str = "fp16") -> bool:
    w = MODELS[model] * (2 if dtype == "fp32" else 1)
    return w + 1.5 <= mem_gb  # weights plus room for activations and a small cache


def row(name, gpu, time, args):
    return f"{name}\t{gpu}\t{time}\t{args} --gpu-tag {gpu}"


def e1() -> list[str]:
    rows = []
    for gpu, mem in GPUS.items():
        for model in MODELS:
            if not fits(model, mem):
                continue
            if gpu in FP16_ONLY_SMALL and model not in FP16_ONLY_SMALL[gpu]:
                continue
            for r in REPS:
                name = f"e1-{gpu}-{short(model)}-fp16-r{r}"
                rows.append(row(name, gpu, "01:30:00",
                                f"--model {model} --dtype fp16 --batches 1 4 16 64 256"))
    return rows


def e2() -> list[str]:
    """fp32 on the Pascal cards (fp16 comes from E1)."""
    rows = []
    for gpu in ("1080ti", "p100"):
        for model in ("Qwen/Qwen3-0.6B", "Qwen/Qwen3-1.7B"):
            for r in REPS:
                name = f"e2-{gpu}-{short(model)}-fp32-r{r}"
                rows.append(row(name, gpu, "01:30:00",
                                f"--model {model} --dtype fp32 --batches 1 4 16 64 256"))
    return rows


def e4() -> list[str]:
    """Two other request shapes, Qwen3-1.7B, B = 1 and 16, on every GPU."""
    rows = []
    for gpu in GPUS:
        for tag, (p, g) in {"dec": (128, 1024), "pre": (2048, 64)}.items():
            for r in REPS:
                name = f"e4-{gpu}-q3-1.7b-{tag}-r{r}"
                rows.append(row(name, gpu, "01:00:00",
                                f"--model Qwen/Qwen3-1.7B --dtype {DTYPE.get(gpu, 'fp16')} "
                                f"--prompt-len {p} --gen-len {g} --batches 1 16"))
    return rows


def e3() -> list[str]:
    """Hugging Face generate against the engine, on four architectures."""
    rows = []
    for gpu in ("1080ti", "t4", "a100", "ada6000"):
        for r in REPS:
            name = f"e3-{gpu}-q3-1.7b-hf-r{r}"
            rows.append(row(name, gpu, "01:00:00",
                            f"scripts/bench_hf.py --model Qwen/Qwen3-1.7B --dtype {DTYPE.get(gpu, 'fp16')} "
                            "--batches 1 16"))
    return rows


def main() -> None:
    out = ROOT / "campaigns"
    out.mkdir(exist_ok=True)
    for name, fn in (("e1", e1), ("e2", e2), ("e3", e3), ("e4", e4)):
        rows = fn()
        (out / f"{name}.tsv").write_text(HEADER + "\n" + "\n".join(rows) + "\n")
        print(f"{name}: {len(rows)} runs")


if __name__ == "__main__":
    main()
