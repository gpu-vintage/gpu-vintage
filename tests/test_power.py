from __future__ import annotations

import math

from gpuvintage.power import integrate


def test_integrate_constant_and_ramp():
    ts = [0.0, 0.5, 1.0, 1.5, 2.0]
    assert math.isclose(integrate(ts, [100.0] * 5, 0.25, 1.75)[0], 150.0)
    ws = [0.0, 50.0, 100.0, 150.0, 200.0]  # 100 W/s ramp
    e, n = integrate(ts, ws, 0.0, 2.0)
    assert math.isclose(e, 200.0) and n == 5
    e, _ = integrate(ts, ws, 0.25, 0.75)  # interpolated ends: mean 50 W over 0.5 s
    assert math.isclose(e, 25.0)


def test_integrate_degenerate():
    assert math.isnan(integrate([0.0], [1.0], 0.0, 1.0)[0])
    assert math.isnan(integrate([0.0, 1.0], [1.0, 1.0], 1.0, 1.0)[0])
