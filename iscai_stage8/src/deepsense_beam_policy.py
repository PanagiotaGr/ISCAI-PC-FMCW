from __future__ import annotations

import hashlib
import math
from typing import Iterable, Sequence

CODEBOOK_SIZE = 64

class PolicyError(ValueError):
    pass

def normalize_probabilities(probabilities: Sequence[float]) -> list[float]:
    p = [float(x) for x in probabilities]
    if len(p) != CODEBOOK_SIZE:
        raise PolicyError(f"expected {CODEBOOK_SIZE} probabilities, got {len(p)}")
    if not all(math.isfinite(x) and x >= 0.0 for x in p):
        raise PolicyError("probabilities must be finite and nonnegative")
    total = sum(p)
    if total <= 0.0:
        raise PolicyError("probability mass must be positive")
    return [x / total for x in p]

def deterministic_ranking(probabilities: Sequence[float]) -> tuple[list[int], list[float]]:
    p = normalize_probabilities(probabilities)
    order = sorted(range(CODEBOOK_SIZE), key=lambda i: (-p[i], i))
    return order, p

def fixed_topk(probabilities: Sequence[float], k: int) -> dict:
    k = int(k)
    if not 1 <= k <= CODEBOOK_SIZE:
        raise PolicyError("K must be in 1..64")
    order, p = deterministic_ranking(probabilities)
    selected = order[:k]
    mass = sum(p[i] for i in selected)
    return {
        "selected_beams": selected,
        "K": k,
        "requested_mass": None,
        "achieved_probability_mass": mass,
        "measured_power_accessed": False,
    }

def adaptive_topk(probabilities: Sequence[float], q: float) -> dict:
    q = float(q)
    if not 0.0 < q <= 1.0:
        raise PolicyError("q must lie in (0,1]")
    order, p = deterministic_ranking(probabilities)
    selected = []
    mass = 0.0
    for i in order:
        selected.append(i)
        mass += p[i]
        if mass + 1e-15 >= q:
            break
    return {
        "selected_beams": selected,
        "K": len(selected),
        "requested_mass": q,
        "achieved_probability_mass": mass,
        "measured_power_accessed": False,
    }

def exhaustive(probabilities: Sequence[float]) -> dict:
    # Still validate the controller input but do not use measured powers.
    _, p = deterministic_ranking(probabilities)
    return {
        "selected_beams": list(range(CODEBOOK_SIZE)),
        "K": CODEBOOK_SIZE,
        "requested_mass": None,
        "achieved_probability_mass": 1.0,
        "measured_power_accessed": False,
    }

def policy_dispatch(policy_name: str, probabilities: Sequence[float]) -> dict:
    if policy_name == "adaptive_topk_q090":
        out = adaptive_topk(probabilities, 0.90)
    elif policy_name == "adaptive_topk_q095":
        out = adaptive_topk(probabilities, 0.95)
    elif policy_name == "adaptive_topk_q0975":
        out = adaptive_topk(probabilities, 0.975)
    elif policy_name == "adaptive_topk_q099":
        out = adaptive_topk(probabilities, 0.99)
    elif policy_name == "fixed_top1":
        out = fixed_topk(probabilities, 1)
    elif policy_name == "fixed_top3":
        out = fixed_topk(probabilities, 3)
    elif policy_name == "fixed_top5":
        out = fixed_topk(probabilities, 5)
    elif policy_name == "exhaustive_64":
        out = exhaustive(probabilities)
    else:
        raise PolicyError(f"unsupported controller policy: {policy_name}")
    out = dict(out)
    out["policy_name"] = policy_name
    return out

def stable_sample_id(
    scenario: int,
    role: str,
    csv_member: str,
    row_index: int,
    lidar_ref: str,
    power_ref: str,
) -> str:
    payload = f"{int(scenario)}|{role}|{csv_member}|{int(row_index)}|{lidar_ref}|{power_ref}"
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]
