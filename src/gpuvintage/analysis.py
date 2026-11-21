"""From run sidecars (results/runs/<name>/{config,env,metrics}.json) to tidy rows.

One row per run and batch size. Energy per window is the NVML energy counter
where the GPU has one (Volta and newer) and the integral of sampled power
otherwise (Pascal); both are kept where both exist.
"""

from __future__ import annotations

import json
import math
import re
import statistics
from pathlib import Path

RUN_RE = re.compile(r"^(?P<campaign>e\d)-(?P<gpu>[a-z0-9]+)-(?P<rest>.+)-r(?P<rep>\d+)$")


def _energy(w: dict) -> tuple[float, str]:
    if w.get("energy_counter_j") is not None:
        return float(w["energy_counter_j"]), "counter"
    return float(w["energy_sampled_j"]), "sampled"


def _window(prefix: str, w: dict) -> dict:
    e, src = _energy(w)
    out = {
        f"{prefix}_tok_s": w["tokens"] / w["seconds"],
        f"{prefix}_j_tok": e / w["tokens"],
        f"{prefix}_w": e / w["seconds"],
        f"{prefix}_seconds": w["seconds"],
        f"{prefix}_energy_source": src,
        f"{prefix}_sm_mhz": w.get("sm_clock_mhz_mean"),
        f"{prefix}_temp_max": w.get("temp_c_max"),
    }
    if w.get("energy_counter_j") is not None and w.get("energy_sampled_j"):
        out[f"{prefix}_sampled_over_counter"] = w["energy_sampled_j"] / w["energy_counter_j"]
    return out


def load_runs(root: str | Path = "results/runs") -> list[dict]:
    rows = []
    for d in sorted(Path(root).iterdir()):
        m = RUN_RE.match(d.name)
        if not m or not (d / "metrics.json").is_file():
            continue
        cfg = json.loads((d / "config.json").read_text())
        met = json.loads((d / "metrics.json").read_text())
        env = json.loads((d / "env.json").read_text()) if (d / "env.json").is_file() else {}
        kind = cfg.get("kind", "engine")
        base = {
            "run": d.name, "campaign": m["campaign"], "gpu": m["gpu"], "rep": int(m["rep"]),
            "kind": kind, "model": cfg["model"], "dtype": cfg["dtype"],
            "prompt_len": cfg["prompt_len"], "gen_len": cfg["gen_len"],
            "finished": bool(met.get("finished")),
            "gpu_name": env.get("nvml", {}).get("name"),
            "power_limit_w": env.get("nvml", {}).get("power_limit_w"),
            "parameters": met.get("parameters"),
            "graph_agreement": (met.get("graph_check") or {}).get("graph_eager_token_agreement"),
        }
        idle = met.get("idle")
        if idle:
            e, _ = _energy(idle)
            base["idle_w"] = e / idle["seconds"]
        for b, rec in met.get("batches", {}).items():
            row = dict(base, batch=int(b), status=rec.get("status"))
            row["other_processes"] = len(rec.get("other_processes") or [])
            if rec.get("status") == "ok":
                if kind == "engine":
                    row.update(_window("prefill", rec["prefill"]))
                    row.update(_window("decode", rec["decode"]))
                    row["finite"] = bool(rec.get("prefill_logits_finite", True) and rec.get("decode_logits_finite", True))
                    row["peak_memory_gb"] = rec.get("peak_memory_bytes", 0) / 1e9
                else:
                    row.update(_window("request", rec["request"]))
            rows.append(row)
    return rows


def request_metrics(prefill_tok_s: float, prefill_j_tok: float, decode_tok_s: float,
                    decode_j_tok: float, batch: int, prompt_len: int, gen_len: int) -> dict:
    """A static batch of `batch` requests, each prompt_len tokens in and gen_len
    out: output throughput (tok/s), energy per output token (J, prompt
    included) and mean power while busy (W)."""
    t_pre = batch * prompt_len / prefill_tok_s
    t_dec = batch * gen_len / decode_tok_s
    e = batch * prompt_len * prefill_j_tok + batch * gen_len * decode_j_tok
    out_tokens = batch * gen_len
    return {"out_tok_s": out_tokens / (t_pre + t_dec), "j_per_out_tok": e / out_tokens,
            "busy_w": e / (t_pre + t_dec), "prefill_time_share": t_pre / (t_pre + t_dec)}


def aggregate(rows: list[dict], keys=("campaign", "kind", "gpu", "model", "dtype", "prompt_len", "gen_len", "batch"),
              values=("prefill_tok_s", "prefill_j_tok", "prefill_w", "decode_tok_s", "decode_j_tok",
                      "decode_w", "request_tok_s", "request_j_tok", "request_w", "idle_w",
                      "decode_sampled_over_counter", "prefill_sampled_over_counter")) -> list[dict]:
    """Mean, standard deviation and count over repetitions, ok rows only;
    status 'oom' when every repetition ran out of memory."""
    groups: dict[tuple, list[dict]] = {}
    for r in rows:
        groups.setdefault(tuple(r[k] for k in keys), []).append(r)
    out = []
    for k, rs in sorted(groups.items(), key=lambda kv: tuple(str(x) for x in kv[0])):
        ok = [r for r in rs if r["status"] == "ok"]
        a = dict(zip(keys, k))
        a["n"] = len(ok)
        a["status"] = "ok" if ok else (rs[0]["status"] or "missing")
        for v in values:
            xs = [r[v] for r in ok if r.get(v) is not None and not math.isnan(r[v])]
            if xs:
                a[v] = statistics.fmean(xs)
                a[v + "_sd"] = statistics.stdev(xs) if len(xs) > 1 else 0.0
        if ok and a.get("prefill_tok_s") and a.get("decode_tok_s"):
            a.update(request_metrics(a["prefill_tok_s"], a["prefill_j_tok"], a["decode_tok_s"],
                                     a["decode_j_tok"], a["batch"], a["prompt_len"], a["gen_len"]))
        out.append(a)
    return out
