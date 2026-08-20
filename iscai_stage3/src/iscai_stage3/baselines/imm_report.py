from __future__ import annotations

from dataclasses import dataclass, asdict
import hashlib
import json


@dataclass(frozen=True)
class IMMReportRecord:

    track_index: int
    time_index: int

    cv_probability: float
    ca_probability: float
    ctrv_probability: float

    cv_log_likelihood: float
    ca_log_likelihood: float
    ctrv_log_likelihood: float

    selected_mode: str



def select_mode(
    *,
    cv: float,
    ca: float,
    ctrv: float,
) -> str:

    modes = {
        "CV": cv,
        "CA": ca,
        "CTRV": ctrv,
    }

    return max(
        modes,
        key=modes.get,
    )



def report_sha256(
    records: tuple[IMMReportRecord, ...],
) -> str:

    payload = json.dumps(
        [
            asdict(record)
            for record in records
        ],
        sort_keys=True,
        separators=(
            ",",
            ":",
        ),
    )

    return hashlib.sha256(
        payload.encode("utf-8")
    ).hexdigest()
