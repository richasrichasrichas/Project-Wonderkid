"""
player_overall.py

Mathematical model to calculate:
  1. Base Overall       -> the player's "pure" underlying ability
  2. Current-Form Overall -> adjusted by recent performance
  3. Estimated Potential  -> projection based on age / historical trajectory
"""

from __future__ import annotations
import math
from dataclasses import dataclass
from typing import Dict, List, Optional


# ---------------------------------------------------------------------------
# 1. ATTRIBUTE NORMALIZATION
# ---------------------------------------------------------------------------

def normalize_attribute(value: float, mean: float, std: float,
                         center: float = 50, scale: float = 10) -> float:
    """
    Converts a raw attribute (e.g. goals/90min, pass completion %, distance
    covered) into a 0-100 scale using a z-score.

    The mean of the comparison group (same position/league/season) maps to
    `center` (50 by default), and each standard deviation is worth `scale`
    points (10 by default).
    """
    if std == 0:
        return center
    z = (value - mean) / std
    return float(min(100, max(0, center + scale * z)))


def normalize_dataset(values: List[float]) -> List[float]:
    """
    Normalizes a list of raw values using the list itself as the reference
    group (sample mean and standard deviation). Useful when you already
    have the full dataset for a given position on hand.
    """
    n = len(values)
    if n == 0:
        return []
    mean = sum(values) / n
    var = sum((v - mean) ** 2 for v in values) / max(1, n - 1)
    std = math.sqrt(var)
    return [normalize_attribute(v, mean, std) for v in values]


# ---------------------------------------------------------------------------
# 2. POSITION WEIGHTS (adjust freely to match your position model)
# ---------------------------------------------------------------------------

POSITION_WEIGHTS: Dict[str, Dict[str, float]] = {
    "FWD": {  # Forward
        "finishing": 0.25, "passing_vision": 0.10, "dribbling": 0.15,
        "tackling": 0.05, "physical": 0.15, "pace": 0.15, "positioning": 0.15,
    },
    "MID": {  # Midfielder
        "finishing": 0.10, "passing_vision": 0.25, "dribbling": 0.15,
        "tackling": 0.10, "physical": 0.10, "pace": 0.10, "positioning": 0.20,
    },
    "CB": {  # Center-back
        "finishing": 0.02, "passing_vision": 0.10, "dribbling": 0.02,
        "tackling": 0.30, "physical": 0.25, "pace": 0.08, "positioning": 0.23,
    },
    "FB": {  # Full-back
        "finishing": 0.02, "passing_vision": 0.15, "dribbling": 0.10,
        "tackling": 0.20, "physical": 0.15, "pace": 0.20, "positioning": 0.18,
    },
    "GK": {  # Goalkeeper (different attribute set)
        "reflexes": 0.40, "gk_positioning": 0.25, "distribution": 0.15,
        "sweeping": 0.10, "physical": 0.10,
    },
}


def validate_weights(weights: Dict[str, float], tol: float = 1e-6) -> None:
    total = sum(weights.values())
    if abs(total - 1.0) > tol:
        raise ValueError(f"Weights must sum to 1.0 (current sum: {total:.4f})")


# ---------------------------------------------------------------------------
# 3. BASE OVERALL
# ---------------------------------------------------------------------------

def base_overall(normalized_attributes: Dict[str, float], position: str) -> float:
    """
    normalized_attributes: dict {attribute_name: value 0-100}
    position: one of the keys in POSITION_WEIGHTS ("FWD", "MID", "CB", "FB", "GK")
    """
    if position not in POSITION_WEIGHTS:
        raise ValueError(f"Position '{position}' not mapped in POSITION_WEIGHTS")

    weights = POSITION_WEIGHTS[position]
    validate_weights(weights)

    missing = set(weights) - set(normalized_attributes)
    if missing:
        raise ValueError(f"Missing attributes for position {position}: {missing}")

    return sum(weights[attr] * normalized_attributes[attr] for attr in weights)


# ---------------------------------------------------------------------------
# 4. CURRENT-FORM OVERALL (recent form with exponential decay)
# ---------------------------------------------------------------------------

@dataclass
class Match:
    days_ago: float         # how many days ago this match was played
    performance: float      # match score, already normalized 0-100


def recent_form_index(matches: List[Match], lam: float = 0.05) -> float:
    """
    Computes the recent performance index using exponential decay weighting.
    lam (lambda) controls the "memory window":
      - small lam (~0.02)  -> long memory, last 2-3 months carry weight
      - large lam (~0.15+) -> short memory, last 3-4 matches dominate
    """
    if not matches:
        return 50.0  # neutral when no data is available

    weights = [math.exp(-lam * m.days_ago) for m in matches]
    total_weight = sum(weights)
    if total_weight == 0:
        return 50.0

    return sum(w * m.performance for w, m in zip(weights, matches)) / total_weight


def current_form_overall(base_ov: float, matches: List[Match],
                          alpha: float = 0.7, lam: float = 0.05) -> float:
    """
    CurrentFormOverall = alpha * BaseOverall + (1 - alpha) * RecentFormIndex
    alpha between 0.6 and 0.8 is a reasonable starting point (anchors on
    the base rating, while recent form nudges the number up or down).
    """
    recent_index = recent_form_index(matches, lam=lam)
    return alpha * base_ov + (1 - alpha) * recent_index


# ---------------------------------------------------------------------------
# 5. ESTIMATED POTENTIAL
# ---------------------------------------------------------------------------

def heuristic_potential(base_ov: float, age: float,
                         peak_age: float = 27, min_age: float = 16,
                         k: float = 0.6) -> float:
    """
    Heuristic approach based on an age curve.
    k controls how much of the "headroom to 100" is realistically reachable.
    Younger players (and those with strong physical attributes -> higher k)
    get more projected growth room.
    """
    if age >= peak_age:
        age_factor = 0.0
    else:
        age_factor = max(0.0, (peak_age - age) / (peak_age - min_age))

    headroom = 100 - base_ov
    return float(min(100, base_ov + k * headroom * age_factor))


def regression_potential(overall_history: List[float],
                          age_history: List[float],
                          fallback_peak_age: Optional[float] = None) -> Dict[str, float]:
    """
    Quadratic regression approach: Overall(age) ~= b0 + b1*age + b2*age^2

    Fits a parabola to the player's (or a comparable group's) historical
    (age, overall) points and returns:
      - the estimated potential (the parabola's maximum, or the value at
        `fallback_peak_age` if b2 >= 0 and the curve has no real maximum)
      - the age at which that peak occurs
      - the fitted coefficients

    Requires at least 3 (age, overall) pairs to fit the quadratic.
    Implemented via manual least squares (no numpy) to avoid an external
    dependency — swap in np.polyfit if you prefer.
    """
    n = len(age_history)
    if n < 3 or len(overall_history) != n:
        raise ValueError("Need at least 3 (age, overall) pairs to fit a quadratic regression")

    x = age_history
    y = overall_history

    # Build the normal equations for degree-2 polynomial regression: X'X b = X'y
    S0, S1, S2, S3, S4 = n, sum(x), sum(v**2 for v in x), sum(v**3 for v in x), sum(v**4 for v in x)
    Sy0 = sum(y)
    Sy1 = sum(xi * yi for xi, yi in zip(x, y))
    Sy2 = sum((xi**2) * yi for xi, yi in zip(x, y))

    # 3x3 system: [[S0,S1,S2],[S1,S2,S3],[S2,S3,S4]] * [b0,b1,b2]^T = [Sy0,Sy1,Sy2]
    A = [[S0, S1, S2], [S1, S2, S3], [S2, S3, S4]]
    B = [Sy0, Sy1, Sy2]

    b0, b1, b2 = _solve_3x3_system(A, B)

    if b2 < 0:
        peak_age = -b1 / (2 * b2)
        max_overall = b0 + b1 * peak_age + b2 * peak_age**2
    else:
        # No real maximum (curve is flat/convex upward) -> use fallback peak age
        peak_age = fallback_peak_age or 27
        max_overall = b0 + b1 * peak_age + b2 * peak_age**2

    return {
        "potential": float(min(100, max(0, max_overall))),
        "peak_age": float(peak_age),
        "b0": b0, "b1": b1, "b2": b2,
    }


def _solve_3x3_system(A: List[List[float]], B: List[float]) -> List[float]:
    """Solves Ax = B for a 3x3 system via simple Gaussian elimination."""
    M = [row[:] + [B[i]] for i, row in enumerate(A)]
    n = 3
    for i in range(n):
        pivot = M[i][i]
        if abs(pivot) < 1e-12:
            raise ValueError("Singular system — insufficient or collinear data for quadratic fit")
        for j in range(i, n + 1):
            M[i][j] /= pivot
        for k in range(n):
            if k != i:
                factor = M[k][i]
                for j in range(i, n + 1):
                    M[k][j] -= factor * M[i][j]
    return [M[i][n] for i in range(n)]


# ---------------------------------------------------------------------------
# 6. USAGE EXAMPLE
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    # --- Hypothetical forward, attributes already normalized 0-100 ---
    attributes = {
        "finishing": 82, "passing_vision": 65, "dribbling": 78,
        "tackling": 30, "physical": 70, "pace": 85, "positioning": 75,
    }
    base_ov = base_overall(attributes, position="FWD")
    print(f"Base Overall: {base_ov:.1f}")

    # --- Last 5 matches (days ago, performance 0-100) ---
    matches = [
        Match(days_ago=3, performance=88),
        Match(days_ago=10, performance=70),
        Match(days_ago=17, performance=92),
        Match(days_ago=24, performance=60),
        Match(days_ago=31, performance=75),
    ]
    form_ov = current_form_overall(base_ov, matches, alpha=0.7, lam=0.05)
    print(f"Current-Form Overall: {form_ov:.1f}")

    # --- Potential (age-based heuristic) ---
    heur_pot = heuristic_potential(base_ov, age=20, k=0.65)
    print(f"Potential (heuristic, age 20): {heur_pot:.1f}")

    # --- Potential (quadratic regression on player history) ---
    age_hist = [17, 18, 19, 20]
    overall_hist = [58, 63, 68, 74]
    regression_result = regression_potential(overall_hist, age_hist)
    print(f"Potential (regression): {regression_result['potential']:.1f} "
          f"at age {regression_result['peak_age']:.1f}")