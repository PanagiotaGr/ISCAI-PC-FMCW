from __future__ import annotations

from iscai_stage3.geometry.spherical import (
    spherical_to_cartesian,
    spherical_position_jacobian,
)

from iscai_stage3.uncertainty.propagation import (
    propagate_position_covariance,
)

from iscai_stage3.beam.probability import (
    beam_cross_track_error,
    gaussian_beam_confidence,
)

from iscai_stage3.adb.controller import (
    ADBTargetScore,
    compute_adb_utility,
    choose_best_target,
)



def evaluate_target(
    *,
    target_id: int,
    range_m: float,
    azimuth_rad: float,
    elevation_rad: float,
    measurement_covariance,
    priority: float,
):

    position = spherical_to_cartesian(
        range_m,
        azimuth_rad,
        elevation_rad,
    )


    J = spherical_position_jacobian(
        range_m,
        azimuth_rad,
        elevation_rad,
    )


    covariance = (
        propagate_position_covariance(
            J,
            measurement_covariance,
        )
    )


    # beam axis = vehicle forward axis
    error = beam_cross_track_error(
        position,
        (0.0,0.0,0.0),
        (1.0,0.0,0.0),
    )


    sigma = (
        covariance[1][1]
    ) ** 0.5


    confidence = gaussian_beam_confidence(
        error,
        sigma,
    )


    utility = compute_adb_utility(
        beam_confidence=confidence,
        priority=priority,
        uncertainty_penalty=sigma,
    )


    return ADBTargetScore(
        target_id=target_id,
        beam_confidence=confidence,
        priority=priority,
        uncertainty_penalty=sigma,
        utility=utility,
    )



def select_target(targets):

    return choose_best_target(
        tuple(targets)
    )
from iscai_stage2.observations.ideal import (
    IdealCausalObservable,
)


def evaluate_causal_observable(
    *,
    observation: IdealCausalObservable,
    target_id: int,
    measurement_covariance,
    priority: float,
):
    """
    Stage3 causal entry point.

    Consumes Stage2 IdealCausalObservable.
    No manual range/angle injection.
    """

    if not observation.geometry_valid:
        raise ValueError(
            "Cannot evaluate invalid geometry observation."
        )

    if (
        observation.range_m is None
        or observation.azimuth_rad is None
        or observation.elevation_rad is None
    ):
        raise ValueError(
            "Incomplete causal observation."
        )

    return evaluate_target(
        target_id=target_id,
        range_m=observation.range_m,
        azimuth_rad=observation.azimuth_rad,
        elevation_rad=observation.elevation_rad,
        measurement_covariance=measurement_covariance,
        priority=priority,
    )
from iscai_stage2.observations.womd_ideal_adapter import (
    ActorIdealObservationSeries,
)


def evaluate_observation_series(
    *,
    series: ActorIdealObservationSeries,
    measurement_covariance,
):
    """
    Evaluate a real Stage2 causal observation series.

    No manual range/angle injection.
    """

    results = []

    for observation in series.observations:

        if not observation.geometry_valid:
            continue

        if (
            observation.range_m is None
            or observation.azimuth_rad is None
            or observation.elevation_rad is None
        ):
            continue

        result = evaluate_target(
            target_id=series.track_index,

            range_m=observation.range_m,
            azimuth_rad=observation.azimuth_rad,
            elevation_rad=observation.elevation_rad,

            measurement_covariance=measurement_covariance,

            # metadata only, not predictor feature
            priority=1.0,
        )

        results.append(result)

    if not results:
        raise ValueError(
            "Observation series contains no valid geometry."
        )

    return tuple(results)
