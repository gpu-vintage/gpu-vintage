"""GPU board energy through NVML.

Two sources, recorded side by side:

- the cumulative energy counter (nvmlDeviceGetTotalEnergyConsumption, in mJ),
  which NVIDIA supports from Volta on;
- the integral of the board power, sampled by a background thread every
  `period` seconds (nvmlDeviceGetPowerUsage, in mW), on every GPU; on Pascal it
  is the only source.

The device is the one the process computes on: NVML is matched to the torch
device by UUID, so neither CUDA_VISIBLE_DEVICES nor a node that exposes all of
its GPUs to every job (as some visualization partitions do) can point the meter
at another card.
"""

from __future__ import annotations

import os
import threading
import time
from bisect import bisect_left, bisect_right

import pynvml


def _s(x) -> str:
    return x.decode() if isinstance(x, bytes) else str(x)


class PowerMeter:
    def __init__(self, torch_index: int = 0, period: float = 0.02, aux_every: int = 25) -> None:
        import torch

        pynvml.nvmlInit()
        self.handle = self._handle_for(torch, torch_index)
        h = self.handle
        self.name = _s(pynvml.nvmlDeviceGetName(h))
        self.uuid = _s(pynvml.nvmlDeviceGetUUID(h))
        self.driver = _s(pynvml.nvmlSystemGetDriverVersion())
        try:
            pynvml.nvmlDeviceGetTotalEnergyConsumption(h)
            self.has_counter = True
        except pynvml.NVMLError:
            self.has_counter = False
        self.period = period
        self.aux_every = aux_every
        self._t: list[float] = []
        self._w: list[float] = []
        self.aux: list[tuple[float, int, int, int, int]] = []  # t, sm MHz, mem MHz, temp C, event reasons
        self._stop = threading.Event()
        self._lock = threading.Lock()
        self._thread = threading.Thread(target=self._run, name="nvml-power", daemon=True)
        self._thread.start()

    # ------------------------------------------------------------------ device
    @staticmethod
    def _handle_for(torch, idx: int):
        props = torch.cuda.get_device_properties(idx)
        uuid = getattr(props, "uuid", None)
        if uuid is not None:
            s = str(uuid)
            if not s.startswith("GPU-"):
                s = "GPU-" + s
            try:
                return pynvml.nvmlDeviceGetHandleByUUID(s)
            except pynvml.NVMLError:
                pass
        bus = getattr(props, "pci_bus_id", None)
        if bus is not None:
            dom = getattr(props, "pci_domain_id", 0)
            dev = getattr(props, "pci_device_id", 0)
            busid = f"{dom:08x}:{bus:02x}:{dev:02x}.0"
            try:
                return pynvml.nvmlDeviceGetHandleByPciBusId(busid)
            except pynvml.NVMLError:
                pass
        if pynvml.nvmlDeviceGetCount() == 1:
            return pynvml.nvmlDeviceGetHandleByIndex(0)
        raise RuntimeError("cannot match the torch device to an NVML device")

    def static_info(self) -> dict:
        h = self.handle
        info = {"name": self.name, "driver": self.driver,
                "energy_counter": self.has_counter}
        for key, fn in (
            ("power_limit_w", lambda: pynvml.nvmlDeviceGetEnforcedPowerLimit(h) / 1000.0),
            ("power_default_limit_w", lambda: pynvml.nvmlDeviceGetPowerManagementDefaultLimit(h) / 1000.0),
            ("memory_total_bytes", lambda: pynvml.nvmlDeviceGetMemoryInfo(h).total),
            ("persistence_mode", lambda: pynvml.nvmlDeviceGetPersistenceMode(h)),
            ("max_sm_clock_mhz", lambda: pynvml.nvmlDeviceGetMaxClockInfo(h, pynvml.NVML_CLOCK_SM)),
            ("max_mem_clock_mhz", lambda: pynvml.nvmlDeviceGetMaxClockInfo(h, pynvml.NVML_CLOCK_MEM)),
            ("cuda_capability", lambda: pynvml.nvmlDeviceGetCudaComputeCapability(h)),
        ):
            try:
                info[key] = fn()
            except pynvml.NVMLError as e:
                info[key] = f"unavailable: {e}"
        return info

    def other_processes(self) -> list[dict]:
        """Compute processes on this GPU other than ours (a shared card would
        contaminate the energy reading)."""
        me = os.getpid()
        try:
            procs = pynvml.nvmlDeviceGetComputeRunningProcesses(self.handle)
        except pynvml.NVMLError:
            return []
        return [{"pid": p.pid, "used_bytes": getattr(p, "usedGpuMemory", None)}
                for p in procs if p.pid != me]

    # ------------------------------------------------------------------ sampling
    def _events(self) -> int:
        fn = getattr(pynvml, "nvmlDeviceGetCurrentClocksEventReasons", None) or \
            getattr(pynvml, "nvmlDeviceGetCurrentClocksThrottleReasons")
        try:
            return int(fn(self.handle))
        except pynvml.NVMLError:
            return -1

    def _run(self) -> None:
        h = self.handle
        n = 0
        nxt = time.perf_counter()
        while not self._stop.is_set():
            t = time.perf_counter()
            try:
                w = pynvml.nvmlDeviceGetPowerUsage(h) / 1000.0
            except pynvml.NVMLError:
                w = float("nan")
            with self._lock:
                self._t.append(t)
                self._w.append(w)
            if n % self.aux_every == 0:
                try:
                    self.aux.append((t,
                                     pynvml.nvmlDeviceGetClockInfo(h, pynvml.NVML_CLOCK_SM),
                                     pynvml.nvmlDeviceGetClockInfo(h, pynvml.NVML_CLOCK_MEM),
                                     pynvml.nvmlDeviceGetTemperature(h, pynvml.NVML_TEMPERATURE_GPU),
                                     self._events()))
                except pynvml.NVMLError:
                    pass
            n += 1
            nxt += self.period
            delay = nxt - time.perf_counter()
            if delay > 0:
                time.sleep(delay)
            else:
                nxt = time.perf_counter()

    def counter_mj(self) -> int | None:
        if not self.has_counter:
            return None
        return int(pynvml.nvmlDeviceGetTotalEnergyConsumption(self.handle))

    def begin(self) -> dict:
        return {"t": time.perf_counter(), "mj": self.counter_mj()}

    def end(self, mark: dict) -> dict:
        t1 = time.perf_counter()
        mj1 = self.counter_mj()
        t0 = mark["t"]
        # let the sampler record one more point past the end of the window
        time.sleep(min(2 * self.period, 0.1))
        with self._lock:
            ts = list(self._t)
            ws = list(self._w)
        sampled_j, n = integrate(ts, ws, t0, t1)
        aux = [a for a in self.aux if t0 <= a[0] <= t1]
        out = {
            "seconds": t1 - t0,
            "energy_counter_j": None if mj1 is None or mark["mj"] is None else (mj1 - mark["mj"]) / 1000.0,
            "energy_sampled_j": sampled_j,
            "power_samples": n,
        }
        if aux:
            out["sm_clock_mhz_mean"] = sum(a[1] for a in aux) / len(aux)
            out["mem_clock_mhz_mean"] = sum(a[2] for a in aux) / len(aux)
            out["temp_c_max"] = max(a[3] for a in aux)
            out["clock_event_reasons"] = sorted({a[4] for a in aux})
        return out

    def close(self) -> None:
        self._stop.set()
        self._thread.join(timeout=2)
        try:
            pynvml.nvmlShutdown()
        except pynvml.NVMLError:
            pass


def integrate(ts: list[float], ws: list[float], t0: float, t1: float) -> tuple[float, int]:
    """Energy in J over [t0, t1] from power samples (W) at times ts, by the
    trapezoid rule, with the power at t0 and t1 interpolated linearly."""
    if t1 <= t0 or len(ts) < 2:
        return float("nan"), 0
    i0 = bisect_left(ts, t0)
    i1 = bisect_right(ts, t1)

    def at(t: float) -> float:
        j = bisect_left(ts, t)
        if j <= 0:
            return ws[0]
        if j >= len(ts):
            return ws[-1]
        ta, tb = ts[j - 1], ts[j]
        wa, wb = ws[j - 1], ws[j]
        return wa + (wb - wa) * (t - ta) / (tb - ta) if tb > ta else wb

    pts = [(t0, at(t0))] + list(zip(ts[i0:i1], ws[i0:i1])) + [(t1, at(t1))]
    e = 0.0
    for (ta, wa), (tb, wb) in zip(pts, pts[1:]):
        e += 0.5 * (wa + wb) * (tb - ta)
    return e, i1 - i0
