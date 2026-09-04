from __future__ import annotations

import math
from typing import Sequence

import numpy as np

CODEBOOK_SIZE = 64

def exhaustive_fallback(reason: str) -> dict:
    return {
        "policy_name": "exhaustive_64",
        "requested_policy_name": None,
        "selected_beams": list(range(CODEBOOK_SIZE)),
        "K": CODEBOOK_SIZE,
        "requested_mass": None,
        "achieved_probability_mass": None,
        "fallback_triggered": True,
        "fallback_reason": str(reason),
        "measured_power_accessed": False,
    }

def calibrated_probabilities_from_logits(logits: Sequence[float], temperature: float) -> list[float]:
    z = np.asarray(logits, dtype=np.float64)
    if z.shape != (CODEBOOK_SIZE,):
        raise ValueError(f"logits must have shape (64,), got {z.shape}")
    if not np.isfinite(z).all():
        raise ValueError("logits contain non-finite values")
    t = float(temperature)
    if not math.isfinite(t) or t <= 0.0:
        raise ValueError("temperature must be finite and positive")
    z = z / t
    z = z - np.max(z)
    e = np.exp(z)
    denom = float(np.sum(e))
    if not math.isfinite(denom) or denom <= 0.0:
        raise ValueError("invalid softmax denominator")
    p = e / denom
    if p.shape != (CODEBOOK_SIZE,) or not np.isfinite(p).all() or np.any(p < 0.0):
        raise ValueError("invalid calibrated probabilities")
    return [float(x) for x in p]

def safe_select(policy_module, policy_name: str, logits: Sequence[float], temperature: float) -> dict:
    try:
        probs = calibrated_probabilities_from_logits(logits, temperature)
        out = dict(policy_module.policy_dispatch(policy_name, probs))
        out["fallback_triggered"] = False
        out["fallback_reason"] = None
        out["requested_policy_name"] = policy_name
        out["measured_power_accessed"] = False
        return out
    except Exception as exc:
        out = exhaustive_fallback(f"{type(exc).__name__}: {exc}")
        out["requested_policy_name"] = policy_name
        return out
