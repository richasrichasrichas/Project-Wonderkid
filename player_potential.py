"""
player_potential.py

Expands the potential-estimation piece of the overall model using a
DESCENDING LOGARITHMIC CURVE as the regression model: the older a player
gets, the smaller the remaining "room to explode" — with the steepest
drop-off happening at younger ages (18-23) and the curve flattening out
as it approaches the peak age, matching how breakout development
realistically tapers off in football.

This module imports from player_overall.py but never modifies it.
"""

from __future__ import annotations
import math
from dataclasses import dataclass
from typing import Dict, List, Optional

from player_overall import base_overall, POSITION_WEIGHTS  # noqa: F401  (re-exported for convenience)


# ---------------------------------------------------------------------------
# 1. DESCENDING LOGARITHMIC EXPLOSION FACTOR (parametric, no data required)
# ---------------------------------------------------------------------------

def explosion_factor(age: float, min_age: float = 15, peak_age: float = 27) -> float:
    """
    Returns a value in [0, 1] representing how much "explosion room" a
    player still has at a given age, using a descending logarithmic
    curve:

        factor(age) = 1 - ln(1 + (age - min_age)) / ln(1 + (peak_age - min_age))

    Shape (why log, not linear):
      - The factor drops FAST between min_age and the early twenties —
        this is where most real breakouts get "used up".
      - The drop then FLATTENS as age approaches peak_age — a 24 and a
        26 year-old with the same profile aren't that different anymore,
        both are already mostly "who they are".
      - factor(min_age) = 1.0 (maximum remaining explosion room)
      - factor(peak_age) = 0.0 (no more explosion room, by definition)
      - Ages beyond peak_age are clamped to 0.0.
    """
    if age <= min_age:
        return 1.0
    if age >= peak_age:
        return 0.0

    numerator = math.log(1 + (age - min_age))
    denominator = math.log(1 + (peak_age - min_age))
    return float(min(1.0, max(0.0, 1 - numerator / denominator)))


def estimate_potential_heuristic(base_ov: float, age: float,
                                  min_age: float = 15, peak_age: float = 27,
                                  k: float = 0.6) -> Dict[str, float]:
    """
    Estimates potential without needing historical data: takes the
    player's headroom to 100 and scales it by the descending log
    explosion_factor for their age.

        Potential = BaseOverall + k * (100 - BaseOverall) * explosion_factor(age)

    k controls how much of the theoretical headroom is realistically
    reachable even for the youngest players (0.6 is a reasonable
    starting point — full headroom is rarely achieved by anyone).
    """
    factor = explosion_factor(age, min_age=min_age, peak_age=peak_age)
    headroom = 100 - base_ov
    potential = min(100, base_ov + k * headroom * factor)

    return {
        "potential": float(potential),
        "explosion_factor": factor,
        "headroom": float(headroom),
        "method": "heuristic_log",
    }


# ---------------------------------------------------------------------------
# 2. LOG-LINEAR REGRESSION FITTED ON REAL DATA (growth_rate ~ b0 + b1*ln(age))
# ---------------------------------------------------------------------------

@dataclass
class GrowthObservation:
    """One (age, year-over-year overall change) data point, typically
    aggregated across many players of the same age to fit the population
    growth curve rather than a single player's noisy trajectory."""
    age: float
    overall_delta: float  # e.g. this_season_overall - last_season_overall


def fit_log_growth_curve(observations: List[GrowthObservation]) -> Dict[str, float]:
    """
    Fits growth_rate ~= b0 + b1 * ln(age) via ordinary least squares on a
    population of (age, yearly overall delta) observations.

    Because players develop fastest early in their careers and plateau
    or decline as they age, b1 comes out NEGATIVE in practice — which is
    exactly the descending logarithmic curve the model calls for: growth
    potential shrinks as age increases, fastest at first, then flattening.

    Requires at least 3 distinct-age observations to fit reliably.
    """
    n = len(observations)
    if n < 3:
        raise ValueError("Need at least 3 observations to fit the log-growth curve")

    x = [math.log(obs.age) for obs in observations]
    y = [obs.overall_delta for obs in observations]

    mean_x = sum(x) / n
    mean_y = sum(y) / n

    cov_xy = sum((xi - mean_x) * (yi - mean_y) for xi, yi in zip(x, y))
    var_x = sum((xi - mean_x) ** 2 for xi, yi in zip(x, y))

    if var_x == 0:
        raise ValueError("All observations have the same age — cannot fit a slope")

    b1 = cov_xy / var_x
    b0 = mean_y - b1 * mean_x

    # R^2 for diagnostic purposes
    y_pred = [b0 + b1 * xi for xi in x]
    ss_res = sum((yi - ypi) ** 2 for yi, ypi in zip(y, y_pred))
    ss_tot = sum((yi - mean_y) ** 2 for yi in y)
    r_squared = 1 - (ss_res / ss_tot) if ss_tot != 0 else 0.0

    return {"b0": b0, "b1": b1, "r_squared": float(r_squared), "n_observations": n}


def projected_growth_rate(age: float, b0: float, b1: float) -> float:
    """
    Growth rate implied by the fitted log-linear curve at a given age.
    Clamped at 0 — a fitted curve can technically go negative past some
    age (implying decline), which is a real signal but shouldn't be
    added as "growth" toward potential.
    """
    if age <= 0:
        raise ValueError("age must be positive")
    rate = b0 + b1 * math.log(age)
    return max(0.0, rate)


def estimate_potential_regression(base_ov: float, age: float, b0: float, b1: float,
                                   peak_age: float = 32) -> Dict[str, float]:
    """
    Projects potential by summing the fitted growth curve year-by-year
    from the player's current age up to peak_age. This is the
    data-driven counterpart to estimate_potential_heuristic: instead of
    a fixed-shape curve, the decay rate comes from fitting real
    (age, overall_delta) observations.
    """
    if age >= peak_age:
        return {
            "potential": float(base_ov),
            "projected_growth": 0.0,
            "years_projected": 0,
            "method": "regression_log",
        }

    total_growth = 0.0
    current_age = math.floor(age) + 1
    years_projected = 0
    while current_age <= peak_age:
        total_growth += projected_growth_rate(current_age, b0, b1)
        current_age += 1
        years_projected += 1

    potential = min(100.0, base_ov + total_growth)
    return {
        "potential": float(potential),
        "projected_growth": float(total_growth),
        "years_projected": years_projected,
        "method": "regression_log",
    }


# ---------------------------------------------------------------------------
# 3. ORCHESTRATOR — picks regression when data is available, else heuristic
# ---------------------------------------------------------------------------

def estimate_potential(base_ov: float, age: float,
                        growth_observations: Optional[List[GrowthObservation]] = None,
                        min_age: float = 15, peak_age: float = 27,
                        regression_peak_age: float = 32, k: float = 0.6) -> Dict[str, float]:
    """
    Single entry point for potential estimation.

    If `growth_observations` is provided (a population dataset of
    (age, overall_delta) pairs), fits the log-linear regression and uses
    it. Otherwise falls back to the parametric descending-log heuristic,
    which needs no historical data at all.
    """
    result: Dict[str, float]
    if growth_observations and len(growth_observations) >= 3:
        try:
            fit = fit_log_growth_curve(growth_observations)
            result = estimate_potential_regression(
                base_ov, age, fit["b0"], fit["b1"], peak_age=regression_peak_age
            )
            result["fit_r_squared"] = fit["r_squared"]
            return result
        except ValueError:
            pass  # fall through to heuristic if the fit fails (e.g. degenerate data)

    result = estimate_potential_heuristic(base_ov, age, min_age=min_age, peak_age=peak_age, k=k)
    return result


# ---------------------------------------------------------------------------
# 4. USAGE EXAMPLE
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    # Reuse the forward example from player_overall.py
    attributes = {
        "finishing": 82, "passing_vision": 65, "dribbling": 78,
        "tackling": 30, "physical": 70, "pace": 85, "positioning": 75,
    }
    base_ov = base_overall(attributes, position="FWD")
    print(f"Base Overall: {base_ov:.1f}\n")

    print("--- Descending log explosion factor by age ---")
    for age in [17, 19, 21, 23, 25, 27, 30]:
        f = explosion_factor(age)
        print(f"  age {age:>2}: explosion_factor = {f:.3f}")

    print("\n--- Heuristic potential (no historical data needed) ---")
    for age in [18, 20, 24, 28]:
        r = estimate_potential_heuristic(base_ov, age)
        print(f"  age {age}: potential = {r['potential']:.1f} "
              f"(explosion_factor = {r['explosion_factor']:.3f})")

    print("\n--- Regression-based potential (fitted on population data) ---")
    # Synthetic population sample: younger ages show bigger average yearly
    # overall gains, tapering off logarithmically with age.
    sample_observations = [
        GrowthObservation(age=17, overall_delta=6.5),
        GrowthObservation(age=18, overall_delta=5.8),
        GrowthObservation(age=19, overall_delta=4.9),
        GrowthObservation(age=20, overall_delta=4.1),
        GrowthObservation(age=22, overall_delta=2.6),
        GrowthObservation(age=24, overall_delta=1.5),
        GrowthObservation(age=27, overall_delta=0.4),
        GrowthObservation(age=30, overall_delta=-0.8),
        GrowthObservation(age=33, overall_delta=-2.1),
    ]
    fit = fit_log_growth_curve(sample_observations)
    print(f"  Fitted curve: growth_rate = {fit['b0']:.2f} + {fit['b1']:.2f} * ln(age) "
          f"(R^2 = {fit['r_squared']:.3f})")

    for age in [18, 20, 24]:
        r = estimate_potential_regression(base_ov, age, fit["b0"], fit["b1"], peak_age=32)
        print(f"  age {age}: potential = {r['potential']:.1f} "
              f"(+{r['projected_growth']:.1f} over {r['years_projected']} years)")

    print("\n--- Orchestrator: estimate_potential() picks the right method ---")
    without_data = estimate_potential(base_ov, age=20)
    print(f"  Without data -> method={without_data['method']}, potential={without_data['potential']:.1f}")

    with_data = estimate_potential(base_ov, age=20, growth_observations=sample_observations)
    print(f"  With data    -> method={with_data['method']}, potential={with_data['potential']:.1f}")