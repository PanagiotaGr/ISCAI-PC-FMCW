from __future__ import annotations

import hashlib
import json
import math
import os
import shutil
import subprocess
import sys
import textwrap
from pathlib import Path


ROOT = Path("/home/agni/waymo")
S4 = ROOT / "iscai_stage4"
S5 = ROOT / "iscai_stage5"

AUDIT = ROOT / "audits/stage5_fullpdf_v2"
AUDIT.mkdir(parents=True, exist_ok=True)

SRC = (
    S5
    / "src/iscai_stage5/fullpdf_v2.py"
)

TEST = (
    S5
    / "tests/test_stage5_fullpdf_v2.py"
)

CONFIG = (
    S5
    / "configs/stage5_fullpdf_v2_protocol.json"
)

REPORT = (
    S5
    / "reports/stage5_fullpdf_v2_repair.json"
)

STATUS = (
    S5
    / "reports/stage5_fullpdf_v2_status.json"
)

S4_HANDOFF = (
    S4
    / "artifacts/fullpdf_v2/"
    "stage4_to_stage5_handoff_fullpdf_v2.json"
)

S4_CLOSURE = (
    S4
    / "reports/stage4_fullpdf_v2_final_closure.json"
)

OLD_S5_CLOSURE = (
    S5
    / "reports/stage5_final_closure.json"
)

OLD_S5_HANDOFF = (
    S5
    / "artifacts/block510/stage5_to_stage6_handoff.json"
)


def sha256(path: Path) -> str:
    h = hashlib.sha256()

    with path.open("rb") as f:
        for block in iter(
            lambda: f.read(1024 * 1024),
            b"",
        ):
            h.update(block)

    return h.hexdigest()


def atomic_json(path, value):
    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    tmp = path.with_name(
        path.name + ".tmp"
    )

    tmp.write_text(
        json.dumps(
            value,
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )

    os.replace(tmp, path)


def frozen_json(path, value):
    if path.exists():
        old = json.loads(
            path.read_text(
                encoding="utf-8"
            )
        )

        if old != value:
            raise RuntimeError(
                f"Frozen protocol differs: {path}"
            )

    else:
        atomic_json(
            path,
            value,
        )


# ============================================================
# A. Upstream Stage4 Full-PDF binding
# ============================================================

if not S4_HANDOFF.is_file():
    raise RuntimeError(
        "Missing Stage4 Full-PDF V2 handoff."
    )

if not S4_CLOSURE.is_file():
    raise RuntimeError(
        "Missing Stage4 Full-PDF V2 closure."
    )

handoff = json.loads(
    S4_HANDOFF.read_text(
        encoding="utf-8"
    )
)

closure = json.loads(
    S4_CLOSURE.read_text(
        encoding="utf-8"
    )
)

if (
    closure.get("status")
    != "COMPLETE_FROZEN_FULLPDF_V2"
):
    raise RuntimeError(
        "Stage4 is not frozen Full-PDF V2."
    )

if (
    closure.get(
        "Stage4_PDF_compliance"
    )
    != "PASS"
):
    raise RuntimeError(
        "Stage4 PDF compliance not PASS."
    )

if not handoff.get(
    "Stage5_allowed",
    False,
):
    raise RuntimeError(
        "Stage4 did not unlock Stage5."
    )

cal = handoff.get(
    "calibrator",
    {},
)

if not cal.get(
    "already_applied_to_predictive_covariance",
    False,
):
    raise RuntimeError(
        "Stage4 covariance is not marked calibrated."
    )

if not cal.get(
    "Stage5_must_not_reapply",
    False,
):
    raise RuntimeError(
        "Stage4 handoff lacks no-recalibration guard."
    )


# ============================================================
# B. Freeze new protocol BEFORE new Stage5 formal outcomes
# ============================================================

PROTOCOL = {
    "stage": 5,
    "version": "fullpdf_v2",

    "status":
        "FROZEN_BEFORE_STAGE5_V2_FORMAL",

    "upstream": {
        "Stage4_status":
            "COMPLETE_FROZEN_FULLPDF_V2",

        "Stage4_handoff_sha256":
            sha256(S4_HANDOFF),

        "Stage4_closure_sha256":
            sha256(S4_CLOSURE),

        "posterior":
            "calibrated_full_3D_Gaussian_GRU",

        "Stage4_calibration_reapplied":
            False,
    },

    "receiver_geometry": {
        "modes": [
            "centroid",
            "known",
            "uncertain",
        ],

        # PDF-listed physical placement:
        # choose rear-center as a declared,
        # constructed receiver assumption.
        #
        # body x is longitudinal.
        "known":
            "rear_face_center",

        "known_offset_body":
            "[-length/2, 0, 0]",

        # No arbitrary tuned sigma:
        # uncertain receiver is uniform over
        # the complete rear face.
        "uncertain_distribution":
            "x=-length/2; "
            "y~U[-width/2,width/2]; "
            "z~U[-height/2,height/2]",

        "uncertain_mean_body":
            "[-length/2,0,0]",

        "uncertain_covariance_body":
            "diag(0,width^2/12,height^2/12)",

        "measured_receiver_placement":
            False,

        "declared_assumption":
            True,
    },

    "angular_posterior": {
        "method":
            "deterministic_seeded_Monte_Carlo",

        "sample_count":
            2048,

        "trajectory_covariance_input":
            "already_calibrated_Stage4_predictive_covariance",

        "receiver_uncertainty_integrated":
            True,

        "future_GT_heading":
            False,
    },

    "codebooks": {
        "sizes": [
            16,
            32,
            64,
        ],

        "azimuth_support_deg": [
            -12.0,
            12.0,
        ],

        "support_semantics":
            "constructed_PartA_visualization_FOV_assumption",

        "outside_support_mass":
            "explicit",

        "azimuth_clipping":
            False,
    },

    "adaptive_TopK": {
        "coverage_targets": [
            0.90,
            0.95,
            0.975,
            0.99,
        ],

        "nominal":
            0.95,

        "rule":
            "minimum_beam_set_reaching_requested_mass",

        "hysteresis":
            "retain_previous_primary_if_still_in_"
            "current_mass_covering_set",

        "persistence":
            True,

        "local_neighbor_sweep":
            "immediate_adjacent_cells_on_recovery",

        "widened_fallback":
            "actual_next_coarser_beam_geometry; "
            "coarsest_uses_union_of_local_cells",

        "loss_of_lock":
            "full_exhaustive_base_codebook",

        "all_evaluated_beams_charged":
            True,

        "free_hidden_probes":
            False,
    },

    "optical": {
        "existing_Stage5_chain_reused":
            True,

        "required_chain": [
            "pointing_error",
            "optical_gain",
            "received_power",
            "SNR",
            "DPSK_BER",
            "effective_rate",
        ],

        "claim":
            "normalized_constructed_optical_surrogate",

        "measured_optical_claim":
            False,

        "probe_count_source":
            "exact_actual_probe_plan_only",
    },

    "latency": {
        "latency_aware_horizon":
            True,

        "existing_frozen_Stage5_timing_contract_reused":
            True,
    },

    "formal": {
        "run_in_repair":
            False,

        "parameter_tuning_on_formal":
            False,

        "completion": [
            "requested_empirical_coverage",
            "reduced_overhead_vs_fixed_or_exhaustive",
        ],
    },
}

frozen_json(
    CONFIG,
    PROTOCOL,
)


# ============================================================
# C. Preserve historical Stage5 evidence
# ============================================================

history = (
    AUDIT
    / "historical_pre_fullpdf_v2"
)

history.mkdir(
    parents=True,
    exist_ok=True,
)

for source in (
    OLD_S5_CLOSURE,
    OLD_S5_HANDOFF,
):
    if source.is_file():
        destination = (
            history
            / source.name
        )

        if not destination.exists():
            shutil.copy2(
                source,
                destination,
            )


# ============================================================
# D. New corrected runtime
# ============================================================

SOURCE = r'''
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import math
from typing import Callable, Iterable

import numpy as np


MC_SAMPLES = 2048
FOV_HALF_RAD = math.radians(12.0)

CODEBOOK_SIZES = (
    16,
    32,
    64,
)

COVERAGE_TARGETS = (
    0.90,
    0.95,
    0.975,
    0.99,
)


def _vector3(x):
    value = np.asarray(
        x,
        dtype=np.float64,
    )

    if value.shape != (3,):
        raise ValueError(
            "Expected 3-vector."
        )

    if not np.all(
        np.isfinite(value)
    ):
        raise ValueError(
            "Non-finite 3-vector."
        )

    return value


def _cov3(x):
    value = np.asarray(
        x,
        dtype=np.float64,
    )

    if value.shape != (3, 3):
        raise ValueError(
            "Expected 3x3 covariance."
        )

    if not np.all(
        np.isfinite(value)
    ):
        raise ValueError(
            "Non-finite covariance."
        )

    value = (
        value + value.T
    ) / 2.0

    eig = np.linalg.eigvalsh(
        value
    )

    if np.min(eig) < -1.0e-9:
        raise ValueError(
            "Covariance is not PSD."
        )

    return value


def rotation_z(heading_rad):
    c = math.cos(
        float(heading_rad)
    )

    s = math.sin(
        float(heading_rad)
    )

    return np.asarray(
        [
            [c, -s, 0.0],
            [s,  c, 0.0],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float64,
    )


@dataclass(frozen=True)
class ReceiverGeometry:
    mode: str
    mean_offset_body_m: tuple[
        float,
        float,
        float,
    ]
    covariance_body_m2: tuple[
        tuple[float, float, float],
        tuple[float, float, float],
        tuple[float, float, float],
    ]
    distribution: str


def receiver_geometry(
    mode,
    *,
    length_m,
    width_m,
    height_m,
):
    L = float(length_m)
    W = float(width_m)
    H = float(height_m)

    if not (
        L > 0
        and W > 0
        and H > 0
    ):
        raise ValueError(
            "Vehicle dimensions must be positive."
        )

    if mode == "centroid":
        mean = (
            0.0,
            0.0,
            0.0,
        )

        covariance = np.zeros(
            (3, 3),
            dtype=np.float64,
        )

        distribution = (
            "point_mass_at_actor_centroid"
        )

    elif mode == "known":
        mean = (
            -0.5 * L,
            0.0,
            0.0,
        )

        covariance = np.zeros(
            (3, 3),
            dtype=np.float64,
        )

        distribution = (
            "point_mass_at_rear_face_center"
        )

    elif mode == "uncertain":
        mean = (
            -0.5 * L,
            0.0,
            0.0,
        )

        # Exact covariance of uniform placement
        # over the complete rear face.
        covariance = np.diag(
            [
                0.0,
                W * W / 12.0,
                H * H / 12.0,
            ]
        )

        distribution = (
            "uniform_over_complete_rear_face"
        )

    else:
        raise ValueError(
            f"Unknown receiver mode: {mode}"
        )

    return ReceiverGeometry(
        mode=mode,

        mean_offset_body_m=tuple(
            float(x)
            for x in mean
        ),

        covariance_body_m2=tuple(
            tuple(
                float(x)
                for x in row
            )
            for row in covariance
        ),

        distribution=distribution,
    )


def receiver_mean_and_covariance_h0(
    actor_mean_h0_m,
    already_calibrated_predictive_covariance_h0_m2,
    *,
    heading_h0_rad,
    geometry,
):
    """
    IMPORTANT:
      input covariance is ALREADY CALIBRATED by Stage4.

    There is intentionally no variance-scale or calibrator
    parameter in this API.
    """
    actor_mean = _vector3(
        actor_mean_h0_m
    )

    actor_cov = _cov3(
        already_calibrated_predictive_covariance_h0_m2
    )

    offset_mean_body = _vector3(
        geometry.mean_offset_body_m
    )

    offset_cov_body = _cov3(
        geometry.covariance_body_m2
    )

    R = rotation_z(
        heading_h0_rad
    )

    receiver_mean = (
        actor_mean
        +
        R @ offset_mean_body
    )

    receiver_cov = (
        actor_cov
        +
        R
        @ offset_cov_body
        @ R.T
    )

    return (
        receiver_mean,
        receiver_cov,
    )


def deterministic_seed(key):
    digest = hashlib.sha256(
        str(key).encode(
            "utf-8"
        )
    ).digest()

    return int.from_bytes(
        digest[:8],
        "little",
        signed=False,
    )


def receiver_samples_h0(
    actor_mean_h0_m,
    already_calibrated_predictive_covariance_h0_m2,
    *,
    heading_h0_rad,
    geometry,
    length_m,
    width_m,
    height_m,
    sample_key,
    sample_count=MC_SAMPLES,
):
    """
    Jointly samples trajectory uncertainty and receiver
    placement uncertainty.

    No Stage4 covariance scaling occurs here.
    """
    n = int(sample_count)

    if n <= 0:
        raise ValueError(
            "sample_count must be positive."
        )

    mean = _vector3(
        actor_mean_h0_m
    )

    covariance = _cov3(
        already_calibrated_predictive_covariance_h0_m2
    )

    rng = np.random.default_rng(
        deterministic_seed(
            sample_key
        )
    )

    trajectory = (
        rng.multivariate_normal(
            mean,
            covariance,
            size=n,
            check_valid="raise",
        )
    )

    L = float(length_m)
    W = float(width_m)
    H = float(height_m)

    if geometry.mode == "centroid":
        offsets = np.zeros(
            (n, 3),
            dtype=np.float64,
        )

    elif geometry.mode == "known":
        offsets = np.repeat(
            np.asarray(
                [
                    [
                        -0.5 * L,
                        0.0,
                        0.0,
                    ]
                ],
                dtype=np.float64,
            ),
            n,
            axis=0,
        )

    elif geometry.mode == "uncertain":
        offsets = np.empty(
            (n, 3),
            dtype=np.float64,
        )

        offsets[:, 0] = (
            -0.5 * L
        )

        offsets[:, 1] = (
            rng.uniform(
                -0.5 * W,
                0.5 * W,
                size=n,
            )
        )

        offsets[:, 2] = (
            rng.uniform(
                -0.5 * H,
                0.5 * H,
                size=n,
            )
        )

    else:
        raise ValueError(
            "Unknown geometry mode."
        )

    R = rotation_z(
        heading_h0_rad
    )

    offsets_h0 = (
        offsets
        @ R.T
    )

    return (
        trajectory
        +
        offsets_h0
    )


def receiver_angles(samples_h0):
    points = np.asarray(
        samples_h0,
        dtype=np.float64,
    )

    if (
        points.ndim != 2
        or points.shape[1] != 3
    ):
        raise ValueError(
            "Expected Nx3 samples."
        )

    x = points[:, 0]
    y = points[:, 1]
    z = points[:, 2]

    horizontal = np.sqrt(
        x * x
        + y * y
    )

    radius = np.sqrt(
        horizontal * horizontal
        + z * z
    )

    valid = (
        horizontal
        > 1.0e-9
    )

    if not np.all(valid):
        raise ValueError(
            "Receiver at undefined azimuth."
        )

    azimuth = np.arctan2(
        y,
        x,
    )

    elevation = np.arctan2(
        z,
        horizontal,
    )

    return (
        radius,
        azimuth,
        elevation,
    )


@dataclass(frozen=True)
class Beam:
    codebook_size: int
    index: int
    lower_azimuth_rad: float
    upper_azimuth_rad: float
    center_azimuth_rad: float
    width_rad: float
    provenance: str = "base_codebook"

    def contains(self, azimuth):
        value = float(azimuth)

        if self.index == (
            self.codebook_size
            - 1
        ):
            return (
                self.lower_azimuth_rad
                <= value
                <= self.upper_azimuth_rad
            )

        return (
            self.lower_azimuth_rad
            <= value
            < self.upper_azimuth_rad
        )


def build_codebook(
    size,
    *,
    half_fov_rad=FOV_HALF_RAD,
):
    size = int(size)

    if size not in CODEBOOK_SIZES:
        raise ValueError(
            "Codebook must be 16, 32 or 64."
        )

    half = float(
        half_fov_rad
    )

    if half <= 0:
        raise ValueError(
            "half_fov_rad must be positive."
        )

    edges = np.linspace(
        -half,
        half,
        size + 1,
    )

    beams = []

    for index in range(size):
        lower = float(
            edges[index]
        )

        upper = float(
            edges[index + 1]
        )

        beams.append(
            Beam(
                codebook_size=size,
                index=index,
                lower_azimuth_rad=lower,
                upper_azimuth_rad=upper,
                center_azimuth_rad=(
                    0.5
                    * (
                        lower
                        + upper
                    )
                ),
                width_rad=(
                    upper
                    - lower
                ),
            )
        )

    return tuple(beams)


@dataclass(frozen=True)
class BeamPosterior:
    probabilities: tuple[float, ...]
    outside_support_probability: float
    sample_count: int

    @property
    def in_support_probability(self):
        return float(
            sum(
                self.probabilities
            )
        )

    @property
    def total_probability(self):
        return (
            self.in_support_probability
            +
            float(
                self.outside_support_probability
            )
        )


def beam_probability_mass(
    azimuth_samples_rad,
    codebook,
):
    azimuth = np.asarray(
        azimuth_samples_rad,
        dtype=np.float64,
    )

    if azimuth.ndim != 1:
        raise ValueError(
            "Expected 1-D azimuth samples."
        )

    n = azimuth.size

    if n <= 0:
        raise ValueError(
            "No azimuth samples."
        )

    counts = np.zeros(
        len(codebook),
        dtype=np.int64,
    )

    outside = 0

    lower = min(
        beam.lower_azimuth_rad
        for beam in codebook
    )

    upper = max(
        beam.upper_azimuth_rad
        for beam in codebook
    )

    for value in azimuth:
        # CRITICAL:
        # no clipping to edge beams.
        if (
            value < lower
            or value > upper
        ):
            outside += 1
            continue

        assigned = False

        for beam in codebook:
            if beam.contains(value):
                counts[
                    beam.index
                ] += 1

                assigned = True
                break

        if not assigned:
            outside += 1

    result = BeamPosterior(
        probabilities=tuple(
            float(x / n)
            for x in counts
        ),

        outside_support_probability=float(
            outside / n
        ),

        sample_count=int(n),
    )

    if not math.isclose(
        result.total_probability,
        1.0,
        rel_tol=0.0,
        abs_tol=1.0e-12,
    ):
        raise RuntimeError(
            "Beam posterior is not a complete partition."
        )

    return result


@dataclass(frozen=True)
class AdaptiveSelection:
    requested_mass: float
    selected_indices: tuple[int, ...]
    selected_mass: float
    outside_support_probability: float
    attainable_inside_support: bool
    primary_index: int | None
    loss_of_lock: bool


def adaptive_topk(
    posterior,
    requested_mass,
    *,
    previous_primary_index=None,
):
    q = float(
        requested_mass
    )

    if not (
        0.0 < q <= 1.0
    ):
        raise ValueError(
            "requested mass must be in (0,1]."
        )

    probabilities = np.asarray(
        posterior.probabilities,
        dtype=np.float64,
    )

    ranking = sorted(
        range(
            probabilities.size
        ),
        key=lambda index:
            (
                -probabilities[index],
                index,
            ),
    )

    attainable = (
        posterior.in_support_probability
        + 1.0e-15
        >= q
    )

    selected = []
    mass = 0.0

    for index in ranking:
        selected.append(
            index
        )

        mass += float(
            probabilities[index]
        )

        if mass + 1.0e-15 >= q:
            break

    if not attainable:
        # Honest complete in-support ranking,
        # followed by explicit recovery.
        selected = ranking
        mass = float(
            posterior.in_support_probability
        )

    primary = (
        selected[0]
        if selected
        else None
    )

    # Parameter-free hysteresis:
    # keep the previous primary ONLY if it remains
    # inside the current minimal mass-covering set.
    if (
        previous_primary_index
        is not None
        and previous_primary_index
        in selected
    ):
        selected = [
            previous_primary_index
        ] + [
            index
            for index in selected
            if index
            != previous_primary_index
        ]

        primary = int(
            previous_primary_index
        )

    return AdaptiveSelection(
        requested_mass=q,

        selected_indices=tuple(
            int(x)
            for x in selected
        ),

        selected_mass=float(
            mass
        ),

        outside_support_probability=float(
            posterior
            .outside_support_probability
        ),

        attainable_inside_support=bool(
            attainable
        ),

        primary_index=primary,

        loss_of_lock=(
            not attainable
        ),
    )


@dataclass(frozen=True)
class Probe:
    beam: Beam
    phase: str


@dataclass(frozen=True)
class ProbePlan:
    probes: tuple[Probe, ...]
    loss_of_lock: bool
    used_local_neighbor_sweep: bool
    used_widened_fallback: bool
    used_exhaustive_sweep: bool

    @property
    def probing_beam_count(self):
        return len(
            self.probes
        )


def _beam_key(beam):
    return (
        beam.codebook_size,
        beam.index,
        round(
            beam.lower_azimuth_rad,
            15,
        ),
        round(
            beam.upper_azimuth_rad,
            15,
        ),
        beam.provenance,
    )


def recovery_probe_plan(
    selection,
    codebook,
):
    probes = []
    seen = set()

    def add(beam, phase):
        key = _beam_key(
            beam
        )

        if key in seen:
            return

        seen.add(key)

        probes.append(
            Probe(
                beam=beam,
                phase=phase,
            )
        )

    # Nominal adaptive Top-K.
    for index in (
        selection.selected_indices
    ):
        add(
            codebook[index],
            "adaptive_topk",
        )

    local_used = False
    widened_used = False
    exhaustive_used = False

    if selection.loss_of_lock:
        primary = (
            selection.primary_index
        )

        if primary is not None:
            for index in (
                primary - 1,
                primary,
                primary + 1,
            ):
                if (
                    0
                    <= index
                    < len(codebook)
                ):
                    add(
                        codebook[index],
                        "local_neighbor_recovery",
                    )

                    local_used = True

            size = len(
                codebook
            )

            if size in (32, 64):
                coarse = build_codebook(
                    size // 2
                )

                center = (
                    codebook[
                        primary
                    ]
                    .center_azimuth_rad
                )

                fallback = min(
                    coarse,
                    key=lambda beam:
                        abs(
                            beam.center_azimuth_rad
                            - center
                        ),
                )

                fallback = Beam(
                    codebook_size=(
                        fallback
                        .codebook_size
                    ),

                    index=(
                        fallback.index
                    ),

                    lower_azimuth_rad=(
                        fallback
                        .lower_azimuth_rad
                    ),

                    upper_azimuth_rad=(
                        fallback
                        .upper_azimuth_rad
                    ),

                    center_azimuth_rad=(
                        fallback
                        .center_azimuth_rad
                    ),

                    width_rad=(
                        fallback
                        .width_rad
                    ),

                    provenance=(
                        "next_coarser_"
                        "widened_fallback"
                    ),
                )

            else:
                left = max(
                    0,
                    primary - 1,
                )

                right = min(
                    len(codebook) - 1,
                    primary + 1,
                )

                lower = (
                    codebook[left]
                    .lower_azimuth_rad
                )

                upper = (
                    codebook[right]
                    .upper_azimuth_rad
                )

                fallback = Beam(
                    codebook_size=0,
                    index=-1,
                    lower_azimuth_rad=lower,
                    upper_azimuth_rad=upper,
                    center_azimuth_rad=(
                        0.5
                        * (
                            lower
                            + upper
                        )
                    ),
                    width_rad=(
                        upper - lower
                    ),
                    provenance=(
                        "coarsest_local_union_"
                        "widened_fallback"
                    ),
                )

            add(
                fallback,
                "widened_fallback",
            )

            widened_used = True

        # Full base-codebook reacquisition.
        for beam in codebook:
            add(
                beam,
                "exhaustive_loss_of_lock",
            )

        exhaustive_used = True

    return ProbePlan(
        probes=tuple(
            probes
        ),

        loss_of_lock=(
            selection.loss_of_lock
        ),

        used_local_neighbor_sweep=(
            local_used
        ),

        used_widened_fallback=(
            widened_used
        ),

        used_exhaustive_sweep=(
            exhaustive_used
        ),
    )


def best_link_over_probed_beams(
    plan,
    evaluator: Callable[[Beam], float],
):
    """
    Evaluate exactly and only beams that were actually
    probed and charged in ProbePlan.

    No hidden neighbour/oracle/fallback beam is evaluated.
    """
    if not plan.probes:
        raise ValueError(
            "Cannot evaluate empty probe plan."
        )

    results = []

    for probe in plan.probes:
        score = float(
            evaluator(
                probe.beam
            )
        )

        if not math.isfinite(
            score
        ):
            raise ValueError(
                "Non-finite link score."
            )

        results.append(
            (
                score,
                probe,
            )
        )

    best = max(
        results,
        key=lambda item:
            (
                item[0],
                -item[1].beam.index,
            ),
    )

    return {
        "score":
            best[0],

        "probe":
            best[1],

        "evaluated_beam_count":
            len(
                results
            ),

        "charged_beam_count":
            plan
            .probing_beam_count,

        "free_probe_count":
            0,
    }
'''

SRC.parent.mkdir(
    parents=True,
    exist_ok=True,
)

SRC.write_text(
    textwrap.dedent(
        SOURCE
    ).lstrip(),
    encoding="utf-8",
)


# ============================================================
# E. Focused tests
# ============================================================

TEST_SOURCE = r'''
from __future__ import annotations

import inspect
import math
import unittest

import numpy as np

from iscai_stage5.fullpdf_v2 import (
    COVERAGE_TARGETS,
    adaptive_topk,
    beam_probability_mass,
    best_link_over_probed_beams,
    build_codebook,
    receiver_angles,
    receiver_geometry,
    receiver_mean_and_covariance_h0,
    receiver_samples_h0,
    recovery_probe_plan,
)


class TestStage5FullPdfV2(
    unittest.TestCase
):

    def test_01_receiver_modes_are_physically_distinct(self):
        centroid = receiver_geometry(
            "centroid",
            length_m=4.0,
            width_m=2.0,
            height_m=1.6,
        )

        known = receiver_geometry(
            "known",
            length_m=4.0,
            width_m=2.0,
            height_m=1.6,
        )

        uncertain = receiver_geometry(
            "uncertain",
            length_m=4.0,
            width_m=2.0,
            height_m=1.6,
        )

        self.assertNotEqual(
            centroid.mean_offset_body_m,
            known.mean_offset_body_m,
        )

        self.assertEqual(
            known.mean_offset_body_m,
            uncertain.mean_offset_body_m,
        )

        self.assertGreater(
            np.trace(
                np.asarray(
                    uncertain
                    .covariance_body_m2
                )
            ),
            0.0,
        )

    def test_02_known_offset_rotates_with_heading(self):
        geometry = receiver_geometry(
            "known",
            length_m=4.0,
            width_m=2.0,
            height_m=1.6,
        )

        mean, covariance = (
            receiver_mean_and_covariance_h0(
                [10.0, 0.0, 0.0],
                np.eye(3),

                heading_h0_rad=(
                    math.pi / 2.0
                ),

                geometry=geometry,
            )
        )

        self.assertAlmostEqual(
            mean[0],
            10.0,
            places=12,
        )

        self.assertAlmostEqual(
            mean[1],
            -2.0,
            places=12,
        )

        self.assertTrue(
            np.allclose(
                covariance,
                np.eye(3),
            )
        )

    def test_03_uncertain_covariance_propagates(self):
        geometry = receiver_geometry(
            "uncertain",
            length_m=4.0,
            width_m=2.0,
            height_m=1.6,
        )

        _, covariance = (
            receiver_mean_and_covariance_h0(
                [10.0, 0.0, 0.0],
                np.eye(3),

                heading_h0_rad=0.0,

                geometry=geometry,
            )
        )

        self.assertGreater(
            covariance[1, 1],
            1.0,
        )

        self.assertGreater(
            covariance[2, 2],
            1.0,
        )

    def test_04_no_calibration_scale_argument_exists(self):
        parameters = (
            inspect.signature(
                receiver_mean_and_covariance_h0
            )
            .parameters
        )

        forbidden = {
            "variance_scale",
            "calibrator",
            "calibration_scale",
        }

        self.assertTrue(
            forbidden.isdisjoint(
                parameters
            )
        )

    def test_05_receiver_mc_is_exactly_reproducible(self):
        geometry = receiver_geometry(
            "uncertain",
            length_m=4.0,
            width_m=2.0,
            height_m=1.6,
        )

        kwargs = dict(
            actor_mean_h0_m=[
                20.0,
                1.0,
                0.5,
            ],

            already_calibrated_predictive_covariance_h0_m2=(
                np.diag(
                    [
                        1.0,
                        0.5,
                        0.2,
                    ]
                )
            ),

            heading_h0_rad=0.2,

            geometry=geometry,

            length_m=4.0,
            width_m=2.0,
            height_m=1.6,

            sample_key="fixed-event",
        )

        a = receiver_samples_h0(
            **kwargs
        )

        b = receiver_samples_h0(
            **kwargs
        )

        self.assertTrue(
            np.array_equal(
                a,
                b,
            )
        )

    def test_06_outside_fov_is_not_clipped(self):
        codebook = build_codebook(
            16
        )

        _, azimuth, _ = (
            receiver_angles(
                [
                    [10.0, 0.0, 0.0],
                ]
            )
        )

        values = np.asarray(
            [
                azimuth[0],
                math.radians(30.0),
            ]
        )

        posterior = (
            beam_probability_mass(
                values,
                codebook,
            )
        )

        self.assertAlmostEqual(
            posterior
            .outside_support_probability,
            0.5,
        )

        self.assertAlmostEqual(
            posterior
            .total_probability,
            1.0,
        )

    def test_07_all_pdf_probability_targets_supported(self):
        self.assertEqual(
            COVERAGE_TARGETS,
            (
                0.90,
                0.95,
                0.975,
                0.99,
            ),
        )

    def test_08_adaptive_topk_is_minimum_mass_set(self):
        codebook = build_codebook(
            16
        )

        # 100 samples:
        # beam 4 = .60
        # beam 5 = .35
        # beam 6 = .05
        samples = []

        for index, count in (
            (4, 60),
            (5, 35),
            (6, 5),
        ):
            samples.extend(
                [
                    codebook[index]
                    .center_azimuth_rad
                ]
                * count
            )

        posterior = (
            beam_probability_mass(
                samples,
                codebook,
            )
        )

        selection = adaptive_topk(
            posterior,
            0.95,
        )

        self.assertEqual(
            selection
            .selected_indices,
            (4, 5),
        )

    def test_09_hysteresis_only_keeps_previous_inside_set(self):
        codebook = build_codebook(
            16
        )

        samples = (
            [
                codebook[4]
                .center_azimuth_rad
            ]
            * 60
            +
            [
                codebook[5]
                .center_azimuth_rad
            ]
            * 40
        )

        posterior = (
            beam_probability_mass(
                samples,
                codebook,
            )
        )

        selection = adaptive_topk(
            posterior,
            0.95,
            previous_primary_index=5,
        )

        self.assertEqual(
            selection.primary_index,
            5,
        )

    def test_10_unattainable_mass_is_explicit_loss_of_lock(self):
        codebook = build_codebook(
            16
        )

        samples = (
            [
                codebook[4]
                .center_azimuth_rad
            ]
            * 70
            +
            [
                math.radians(30.0)
            ]
            * 30
        )

        posterior = (
            beam_probability_mass(
                samples,
                codebook,
            )
        )

        selection = adaptive_topk(
            posterior,
            0.95,
        )

        self.assertFalse(
            selection
            .attainable_inside_support
        )

        self.assertTrue(
            selection
            .loss_of_lock
        )

    def test_11_widened_fallback_is_real_beam_geometry(self):
        codebook = build_codebook(
            64
        )

        samples = (
            [
                codebook[30]
                .center_azimuth_rad
            ]
            * 50
            +
            [
                math.radians(30.0)
            ]
            * 50
        )

        posterior = (
            beam_probability_mass(
                samples,
                codebook,
            )
        )

        selection = adaptive_topk(
            posterior,
            0.95,
        )

        plan = recovery_probe_plan(
            selection,
            codebook,
        )

        fallback = [
            probe.beam
            for probe in plan.probes
            if probe.phase
            == "widened_fallback"
        ]

        self.assertEqual(
            len(fallback),
            1,
        )

        self.assertGreater(
            fallback[0].width_rad,
            codebook[0].width_rad,
        )

    def test_12_no_free_best_link_probes(self):
        codebook = build_codebook(
            16
        )

        samples = [
            codebook[7]
            .center_azimuth_rad
        ] * 100

        posterior = (
            beam_probability_mass(
                samples,
                codebook,
            )
        )

        selection = adaptive_topk(
            posterior,
            0.95,
        )

        plan = recovery_probe_plan(
            selection,
            codebook,
        )

        calls = []

        def evaluator(beam):
            calls.append(
                (
                    beam.codebook_size,
                    beam.index,
                )
            )

            return -abs(
                beam
                .center_azimuth_rad
            )

        result = (
            best_link_over_probed_beams(
                plan,
                evaluator,
            )
        )

        self.assertEqual(
            len(calls),
            plan.probing_beam_count,
        )

        self.assertEqual(
            result[
                "evaluated_beam_count"
            ],
            result[
                "charged_beam_count"
            ],
        )

        self.assertEqual(
            result[
                "free_probe_count"
            ],
            0,
        )

    def test_13_existing_optical_chain_remains_available(self):
        import iscai_stage5.optical_link as optical

        self.assertTrue(
            callable(
                optical.pointing_error
            )
        )

        self.assertTrue(
            callable(
                optical.optical_gain_for_cell
            )
        )

        self.assertTrue(
            callable(
                optical.normalized_received_power
            )
        )

        self.assertTrue(
            callable(
                optical.effective_rate
            )
        )

        self.assertGreater(
            float(
                optical
                .PARTA_REFERENCE_SNR_LINEAR
            ),
            0.0,
        )

    def test_14_existing_latency_contract_uses_probe_count(self):
        import iscai_stage5.beam_latency as latency

        signature = inspect.signature(
            latency.beam_probing_time_s
        )

        self.assertIn(
            "probing_beam_count",
            signature.parameters,
        )


if __name__ == "__main__":
    unittest.main()
'''

TEST.parent.mkdir(
    parents=True,
    exist_ok=True,
)

TEST.write_text(
    textwrap.dedent(
        TEST_SOURCE
    ).lstrip(),
    encoding="utf-8",
)


# ============================================================
# F. Compile + regression
# ============================================================

env = os.environ.copy()

env["PYTHONPATH"] = os.pathsep.join(
    [
        str(
            ROOT
            / "iscai_stage0/src"
        ),
        str(
            ROOT
            / "iscai_stage1/src"
        ),
        str(
            ROOT
            / "iscai_stage2/src"
        ),
        str(
            ROOT
            / "iscai_stage3/src"
        ),
        str(
            ROOT
            / "iscai_stage4/src"
        ),
        str(
            ROOT
            / "iscai_stage5/src"
        ),
        str(
            ROOT
            / "iscai_stage5/tests"
        ),
        env.get(
            "PYTHONPATH",
            "",
        ),
    ]
)

compile_result = subprocess.run(
    [
        sys.executable,
        "-m",
        "py_compile",
        str(SRC),
        str(TEST),
    ],
    env=env,
)

if compile_result.returncode != 0:
    raise RuntimeError(
        "Stage5 V2 compile failed."
    )


targeted = subprocess.run(
    [
        sys.executable,
        "-B",
        "-m",
        "unittest",
        "-v",
        "test_stage5_fullpdf_v2",
    ],
    cwd=S5 / "tests",
    env=env,
    stdout=subprocess.PIPE,
    stderr=subprocess.STDOUT,
    text=True,
)

print(targeted.stdout)

if targeted.returncode != 0:
    raise RuntimeError(
        "Stage5 V2 focused tests failed."
    )


full = subprocess.run(
    [
        sys.executable,
        "-B",
        "-m",
        "unittest",
        "discover",
        "-s",
        "tests",
        "-p",
        "test_*.py",
    ],
    cwd=S5,
    env=env,
    stdout=subprocess.PIPE,
    stderr=subprocess.STDOUT,
    text=True,
)

print(full.stdout)

if full.returncode != 0:
    raise RuntimeError(
        "Full Stage5 regression failed."
    )


# ============================================================
# G. Pre-formal repair report
# ============================================================

report = {
    "stage": 5,

    "repair":
        "fullpdf_v2_1_of_2",

    "status":
        "REPAIRED_AND_FROZEN_PRE_FORMAL",

    "formal_evaluation_run":
        False,

    "Stage4_retraining":
        False,

    "Stage4_recalibration":
        False,

    "Stage4_calibration_reapplied":
        False,

    "repairs": {
        "receiver_geometry_distinct":
            True,

        "centroid_known_uncertain":
            True,

        "known_receiver":
            "rear_face_center",

        "uncertain_receiver":
            "uniform_rear_face",

        "receiver_uncertainty_integrated":
            True,

        "outside_support_mass_explicit":
            True,

        "azimuth_edge_clipping":
            False,

        "adaptive_targets":
            [
                0.90,
                0.95,
                0.975,
                0.99,
            ],

        "previous_beam_persistence":
            True,

        "hysteresis":
            True,

        "local_neighbor_recovery":
            True,

        "widened_fallback_real_geometry":
            True,

        "exhaustive_loss_of_lock":
            True,

        "all_evaluated_beams_charged":
            True,

        "free_link_probes":
            False,

        "optical_chain_retained":
            True,

        "latency_contract_retained":
            True,
    },

    "files": {
        "runtime": {
            "path":
                str(SRC),

            "sha256":
                sha256(SRC),
        },

        "tests": {
            "path":
                str(TEST),

            "sha256":
                sha256(TEST),
        },

        "protocol": {
            "path":
                str(CONFIG),

            "sha256":
                sha256(CONFIG),
        },

        "Stage4_handoff": {
            "path":
                str(S4_HANDOFF),

            "sha256":
                sha256(S4_HANDOFF),
        },
    },

    "targeted_tests":
        "PASS",

    "full_regression":
        "PASS",

    "Stage6_allowed":
        False,

    "next":
        "independent Stage5 Full-PDF V2 "
        "formal checker 2/2",
}

atomic_json(
    REPORT,
    report,
)

atomic_json(
    STATUS,
    {
        "stage":
            5,

        "status":
            "FULLPDF_V2_PRE_FORMAL_FROZEN",

        "PDF_full_compliance":
            False,

        "formal_evaluation_run":
            False,

        "Stage6_allowed":
            False,

        "repair_report":
            str(REPORT),
    },
)


print()
print("=" * 72)
print(
    "STAGE5 FULL-PDF V2 REPAIR 1/2 = PASS PRE-FORMAL"
)
print("=" * 72)
print(
    "Stage4 recalibration       = NO"
)
print(
    "receiver geometries       = DISTINCT"
)
print(
    "outside-support clipping  = REMOVED"
)
print(
    "all evaluated probes      = CHARGED"
)
print(
    "formal evaluation         = NO"
)
print(
    "Stage6                    = BLOCKED"
)
print(
    "report =",
    REPORT,
)
