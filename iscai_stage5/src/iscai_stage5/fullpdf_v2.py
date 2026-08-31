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
