from __future__ import annotations

import math

import pytest

from gpuvintage import econ


def test_crf():
    assert econ.crf(0.0, 4) == pytest.approx(0.25)
    assert econ.crf(0.05, 10) == pytest.approx(0.129505, rel=1e-5)


def test_operating_cost_units():
    op = econ.Operation(utilization=1.0, pue=1.0, host_w=0.0, electricity=0.10,
                        carbon_intensity=500, carbon_price=100)
    # 1 kW for a year: 8760 kWh, 876 USD of electricity, 4.38 t, 438 USD of carbon
    assert econ.operating_cost_per_year(1000, 0, op) == pytest.approx(876 + 438)
    # 1000 tok/s for a year is 31,536 Mtok
    assert econ.mtok_per_year(1000, 1.0) == pytest.approx(31536)


@pytest.mark.parametrize("theta,r", [(0.2, 0.05), (0.4, 0.08), (0.1, 0.0)])
def test_vintage_zero_profit_and_scrapping(theta, r):
    K, Y, o = 10_000.0, 5_000.0, 0.05
    v = econ.solve_vintage(K, Y, o, theta, r)
    # the card earns exactly its price, and its rent is zero when scrapped
    assert econ.value_at_age(0.0, v, Y, o, theta, r) == pytest.approx(K, rel=1e-9)
    assert v.c0 * math.exp(-theta * v.life) == pytest.approx(o, rel=1e-9)
    assert econ.value_at_age(v.life, v, Y, o, theta, r) == 0.0
    ages = [0.25 * i for i in range(int(4 * v.life))]
    vals = [econ.value_at_age(a, v, Y, o, theta, r) for a in ages]
    assert all(a > b for a, b in zip(vals, vals[1:]))


def test_comparative_statics():
    K, Y = 10_000.0, 5_000.0
    base = econ.solve_vintage(K, Y, 0.05, 0.3, 0.05)
    # dearer energy, or faster progress, shortens the life
    assert econ.solve_vintage(K, Y, 0.10, 0.3, 0.05).life < base.life
    assert econ.solve_vintage(K, Y, 0.05, 0.6, 0.05).life < base.life
    # a dearer card (relative to its running cost) lives longer
    assert econ.solve_vintage(2 * K, Y, 0.05, 0.3, 0.05).life > base.life


def test_carbon():
    assert econ.payback_years(30.0, 0.5, 0.2) == pytest.approx(0.1)
    assert econ.payback_years(30.0, 0.2, 0.5) == math.inf
    assert econ.breakeven_carbon_price(0.02, 2e-4, 0.10, 0.5e-4) == pytest.approx(0.08 / 1.5e-4)
    assert econ.breakeven_carbon_price(0.2, 2e-4, 0.10, 0.5e-4) == 0.0
    assert econ.breakeven_carbon_price(0.02, 1e-4, 0.10, 2e-4) == math.inf


def test_capacity_cost():
    op = econ.Operation(utilization=1.0, pue=1.2, host_w=100.0, electricity=0.0,
                        carbon_intensity=0.0, capacity_w_year=1.5)
    # (300 W card + 100 W host) * 1.2 * 1.5 USD/W/year = 720 USD/year
    assert econ.operating_cost_per_year(250, 50, op, rated_w=300) == pytest.approx(720.0)
    assert econ.operating_cost_per_year(250, 50, op) == pytest.approx(180.0)  # host share only
