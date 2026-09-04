from __future__ import annotations

import math
from typing import Iterable, Sequence

CODEBOOK_SIZE = 64
OUTAGE_THRESHOLDS_DB = (1.0, 3.0, 6.0)
REFERENCE_SNR_DB = (0.0, 10.0, 20.0)

class EvaluationError(ValueError):
    pass

def _validate_power_vector(powers: Sequence[float]) -> list[float]:
    p = [float(x) for x in powers]
    if len(p) != CODEBOOK_SIZE:
        raise EvaluationError(f"expected {CODEBOOK_SIZE} powers, got {len(p)}")
    if not all(math.isfinite(x) and x > 0.0 for x in p):
        raise EvaluationError("all measured powers must be finite and strictly positive")
    return p

def _validate_selected(selected: Iterable[int]) -> list[int]:
    s = sorted(set(int(x) for x in selected))
    if not s:
        raise EvaluationError("selected beam set is empty")
    if s[0] < 0 or s[-1] >= CODEBOOK_SIZE:
        raise EvaluationError("selected beam index outside 0..63")
    return s

def label_consistent_with_power(label: int, powers: Sequence[float], *, rtol=1e-10, atol=1e-12) -> bool:
    p = _validate_power_vector(powers)
    label = int(label)
    if not (0 <= label < CODEBOOK_SIZE):
        return False
    m = max(p)
    return math.isclose(p[label], m, rel_tol=rtol, abs_tol=atol)

def evaluate_selected_set(
    selected: Iterable[int],
    powers: Sequence[float],
    ground_truth_label: int,
) -> dict:
    p = _validate_power_vector(powers)
    s = _validate_selected(selected)
    y = int(ground_truth_label)
    if not (0 <= y < CODEBOOK_SIZE):
        raise EvaluationError("ground-truth beam label outside 0..63")
    if not label_consistent_with_power(y, p):
        raise EvaluationError("ground-truth label is inconsistent with measured-power maximum")

    p_opt = max(p)
    p_sel = max(p[i] for i in s)
    ratio = p_sel / p_opt
    gap_db = 10.0 * math.log10(ratio)
    loss_db = -gap_db
    k = len(s)

    outages = {f"{int(t)}dB": bool(gap_db < -float(t)) for t in OUTAGE_THRESHOLDS_DB}

    se = {}
    for snr_db in REFERENCE_SNR_DB:
        gamma = 10.0 ** (snr_db / 10.0)
        se_opt = math.log2(1.0 + gamma)
        se_sel = math.log2(1.0 + gamma * ratio)
        se[f"{int(snr_db)}dB"] = {
            "optimum_bps_per_hz": se_opt,
            "selected_bps_per_hz": se_sel,
            "gap_bps_per_hz": se_opt - se_sel,
        }

    return {
        "coverage": int(y in s),
        "K": k,
        "probing_overhead": k / CODEBOOK_SIZE,
        "overhead_reduction_vs_exhaustive": 1.0 - k / CODEBOOK_SIZE,
        "selected_best_measured_power": p_sel,
        "optimum_measured_power": p_opt,
        "power_ratio": ratio,
        "measured_power_gap_db": gap_db,
        "measured_power_loss_db": loss_db,
        "outage": outages,
        "normalized_spectral_efficiency": se,
    }

def adaptive_mass_covering_set(probabilities: Sequence[float], q: float) -> list[int]:
    probs = [float(x) for x in probabilities]
    if len(probs) != CODEBOOK_SIZE:
        raise EvaluationError("expected 64 probabilities")
    if not all(math.isfinite(x) and x >= 0.0 for x in probs):
        raise EvaluationError("probabilities must be finite and nonnegative")
    total = sum(probs)
    if total <= 0.0:
        raise EvaluationError("probability mass must be positive")
    probs = [x / total for x in probs]
    if not (0.0 < float(q) <= 1.0):
        raise EvaluationError("q must lie in (0,1]")
    order = sorted(range(CODEBOOK_SIZE), key=lambda i: (-probs[i], i))
    acc = 0.0
    out = []
    for i in order:
        out.append(i)
        acc += probs[i]
        if acc + 1e-15 >= q:
            return out
    return order
