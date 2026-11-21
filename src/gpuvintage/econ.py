"""The economics of GPU vintages for LLM inference.

Units: money in USD, time in years, output in millions of tokens (Mtok).

Cost of a token on one GPU at one operating point
-------------------------------------------------
A GPU serving requests at utilization u (share of hours busy) produces
Y = 3600 * 8760 * u * y / 1e6 Mtok per year, y being its output throughput
(tokens/s) while busy. Its electricity draw is u * P_busy + (1 - u) * P_idle + h
watts (h: host share per GPU), times the data centre's PUE. The operating cost
per Mtok is that energy priced at p_e (USD/kWh) plus its emissions priced at the
carbon price tau (USD/tCO2), plus other running costs if any:

    o = 8.76 * PUE * P_avg[W] * (p_e + tau * CI[g/kWh] / 1e6) / Y.

Where data-centre capacity is scarce, a card also ties up the power capacity it
is provisioned for: its rated power plus its host share, times PUE, priced at
phi USD per watt of facility power per year (the annualized cost of building
that capacity). That cost is part of the operating cost of keeping a card, since
retiring it frees the capacity for a new one; with spare capacity, phi = 0.

The capital cost per Mtok of a new card bought at K (its price times one plus
the share of server and infrastructure) is K * CRF(r, L) / Y over a life of L
years at discount rate r.

Vintages (Salter 1960; Malcomson 1975)
--------------------------------------
New capacity sets the price of a token at its full cost, which falls at the rate
theta of embodied technical progress: p(t) = c0 * exp(-theta t). A card bought
at t = 0 earns the quasi-rent p(t) - o per Mtok, o its own operating cost, and is
scrapped when that rent reaches zero, at

    T* = ln(c0 / o) / theta.

Zero profit on new capacity pins c0: the discounted quasi-rents over the
card's life repay its price,

    K / Y = c0 (1 - e^{-(theta + r) T*}) / (theta + r) - o (1 - e^{-r T*}) / r.

Its value at age a is V(a) = Y * integral_a^T* (p(s) - o) e^{-r (s - a)} ds,
V(0) = K: the path of economic depreciation, which falls to zero at T*.
Physical wear is assumed away (computers depreciate by obsolescence: Geske,
Ramey and Shapiro 2007; Doms et al. 2004).
"""

from __future__ import annotations

import math
from dataclasses import dataclass

HOURS_PER_YEAR = 8760.0


def crf(r: float, n: float) -> float:
    """Capital recovery factor: the annual payment that repays 1 over n years at rate r."""
    if n <= 0:
        raise ValueError("n must be positive")
    if r == 0:
        return 1.0 / n
    return r / (1.0 - (1.0 + r) ** (-n))


@dataclass(frozen=True)
class Operation:
    """How a card is run. Power in W, prices in USD."""

    utilization: float = 0.5
    pue: float = 1.3
    host_w: float = 100.0          # host (CPU, memory, fans, network) per GPU
    electricity: float = 0.12      # USD/kWh
    carbon_intensity: float = 400  # gCO2/kWh
    carbon_price: float = 0.0      # USD/tCO2
    capacity_w_year: float = 0.0   # USD per watt of facility power per year (phi)


def mtok_per_year(y_tok_s: float, utilization: float) -> float:
    return 3600.0 * HOURS_PER_YEAR * utilization * y_tok_s / 1e6


def average_power(p_busy: float, p_idle: float, op: Operation) -> float:
    return op.utilization * p_busy + (1.0 - op.utilization) * p_idle + op.host_w


def energy_kwh_per_year(p_busy: float, p_idle: float, op: Operation) -> float:
    return op.pue * average_power(p_busy, p_idle, op) * HOURS_PER_YEAR / 1000.0


def emissions_t_per_year(p_busy: float, p_idle: float, op: Operation) -> float:
    return energy_kwh_per_year(p_busy, p_idle, op) * op.carbon_intensity / 1e6


def capacity_cost_per_year(rated_w: float, op: Operation) -> float:
    """Facility power capacity a card ties up: (rated power + host share) * PUE * phi."""
    return op.capacity_w_year * op.pue * (rated_w + op.host_w)


def operating_cost_per_year(p_busy: float, p_idle: float, op: Operation, rated_w: float = 0.0) -> float:
    return (energy_kwh_per_year(p_busy, p_idle, op) * op.electricity
            + emissions_t_per_year(p_busy, p_idle, op) * op.carbon_price
            + capacity_cost_per_year(rated_w, op))


def operating_cost_per_mtok(y_tok_s: float, p_busy: float, p_idle: float, op: Operation,
                            rated_w: float = 0.0) -> float:
    return operating_cost_per_year(p_busy, p_idle, op, rated_w) / mtok_per_year(y_tok_s, op.utilization)


def capital_cost_per_mtok(K: float, y_tok_s: float, op: Operation, r: float, life: float) -> float:
    return K * crf(r, life) / mtok_per_year(y_tok_s, op.utilization)


# ---------------------------------------------------------------- vintages
def _pv_rents(c0: float, o: float, theta: float, r: float, T: float) -> float:
    """Present value at t = 0 of (c0 e^{-theta t} - o) over [0, T], per unit output per year."""
    a = theta + r
    first = c0 * (1.0 - math.exp(-a * T)) / a
    second = o * ((1.0 - math.exp(-r * T)) / r if r > 0 else T)
    return first - second


def scrap_age(c0: float, o: float, theta: float) -> float:
    if c0 <= o:
        return 0.0
    return math.log(c0 / o) / theta


@dataclass(frozen=True)
class Vintage:
    c0: float          # full cost per Mtok of new capacity at purchase (price level)
    life: float        # economic service life T*, years
    k_per_mtok: float  # implied capital charge c0 - o


def solve_vintage(K: float, Y: float, o: float, theta: float, r: float) -> Vintage:
    """The price level c0 at which a card of price K, output Y Mtok/year and
    operating cost o per Mtok earns exactly K over its economic life, under
    embodied progress theta and discount rate r; and that life."""
    if K <= 0 or Y <= 0 or o <= 0 or theta <= 0:
        raise ValueError("K, Y, o and theta must be positive")
    target = K / Y

    def excess(c0: float) -> float:
        return _pv_rents(c0, o, theta, r, scrap_age(c0, o, theta)) - target

    lo, hi = o, 2.0 * o
    while excess(hi) < 0:
        hi *= 2.0
        if hi > 1e12 * o:
            raise RuntimeError("no price level repays the card")
    for _ in range(200):  # the present value increases with c0
        mid = 0.5 * (lo + hi)
        if excess(mid) < 0:
            lo = mid
        else:
            hi = mid
    c0 = 0.5 * (lo + hi)
    return Vintage(c0=c0, life=scrap_age(c0, o, theta), k_per_mtok=c0 - o)


def value_at_age(a: float, v: Vintage, Y: float, o: float, theta: float, r: float) -> float:
    """Value of the card at age a (USD): its remaining discounted quasi-rents."""
    if a >= v.life:
        return 0.0
    c_a = v.c0 * math.exp(-theta * a)  # the token price at age a
    return Y * _pv_rents(c_a, o, theta, r, v.life - a)


def straight_line(a: float, K: float, life: float) -> float:
    return max(0.0, K * (1.0 - a / life))


# ---------------------------------------------------------------- carbon
def payback_years(embodied_kg: float, old_t_per_year: float, new_t_per_year: float) -> float:
    """Years for a replacement's operating savings to repay its embodied carbon,
    both cards producing the same output (the new one, faster, runs less)."""
    saving = old_t_per_year - new_t_per_year
    if saving <= 0:
        return math.inf
    return embodied_kg / 1000.0 / saving


def breakeven_carbon_price(o_old: float, e_old: float, c_new: float, e_new: float) -> float:
    """Carbon price (USD/t) above which an old card should give way to new
    capacity: its operating cost per Mtok, o_old + tau * e_old, reaches the full
    cost per Mtok of new capacity, c_new + tau * e_new (o_old and c_new without
    carbon; e_old and e_new the emissions in t/Mtok). 0 if the old card is
    already dearer; inf if it emits no more than new capacity."""
    if o_old >= c_new:
        return 0.0
    if e_old <= e_new:
        return math.inf
    return (c_new - o_old) / (e_old - e_new)
