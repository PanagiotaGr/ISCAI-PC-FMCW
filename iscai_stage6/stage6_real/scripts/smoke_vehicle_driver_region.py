from pathlib import Path
import importlib.util
import numpy as np


EVAL = Path(
    "stage6_real/scripts/"
    "evaluate_stage6_streaming.py"
)

PRED = Path(
    "stage6_real/data/"
    "real_stage6_predictions_balanced.npz"
)


spec = importlib.util.spec_from_file_location(
    "stage6_eval",
    EVAL,
)

m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)


def vehicle_relevant_face(
    center_xy,
    length,
    width,
    heading,
):
    """
    Geometry-only Stage-6 vehicle glare surrogate.

    No true windshield / mirror location exists in WOMD.

    If the vehicle forward axis points toward the ego/headlamp
    origin, use the FRONT face as a windshield surrogate.

    Otherwise use the REAR face as a rear/mirror surrogate.
    """

    center_xy = np.asarray(
        center_xy,
        dtype=np.float64,
    )

    forward = np.asarray(
        [
            np.cos(heading),
            np.sin(heading),
        ],
        dtype=np.float64,
    )

    lateral = np.asarray(
        [
            -np.sin(heading),
            np.cos(heading),
        ],
        dtype=np.float64,
    )

    to_ego = -center_xy

    approaching = (
        np.dot(
            forward,
            to_ego,
        )
        >
        0.0
    )

    face_sign = (
        +1.0
        if approaching
        else
        -1.0
    )

    face_center = (
        center_xy
        +
        face_sign
        *
        0.5
        *
        length
        *
        forward
    )

    p_left = (
        face_center
        +
        0.5
        *
        width
        *
        lateral
    )

    p_right = (
        face_center
        -
        0.5
        *
        width
        *
        lateral
    )

    return {
        "mode":
            (
                "front_windshield_surrogate"
                if approaching
                else
                "rear_mirror_surrogate"
            ),

        "points":
            np.stack(
                [
                    p_left,
                    p_right,
                ],
                axis=0,
            ),
    }


def angular_span_deg(points):

    angles = np.unwrap(
        np.arctan2(
            points[:, 1],
            points[:, 0],
        )
    )

    return float(
        np.degrees(
            angles.max()
            -
            angles.min()
        )
    )


def main():

    data = np.load(
        PRED
    )

    classes = data[
        "actor_class"
    ]

    mu = data[
        "future_mean_H_m"
    ]

    length = data[
        "length_m"
    ]

    width = data[
        "width_m"
    ]

    heading = data[
        "current_heading_H_rad"
    ]


    vehicle_idx = np.flatnonzero(
        classes
        ==
        "VEHICLE"
    )


    front_count = 0
    rear_count = 0

    full_spans = []
    surrogate_spans = []


    for actor in vehicle_idx:

        actor = int(actor)

        for t in range(
            mu.shape[1]
        ):

            center = mu[
                actor,
                t,
                :2
            ]

            full = m.box_corners(
                center,
                float(
                    length[
                        actor
                    ]
                ),
                float(
                    width[
                        actor
                    ]
                ),
                float(
                    heading[
                        actor
                    ]
                ),
            )

            surrogate = (
                vehicle_relevant_face(
                    center,
                    float(
                        length[
                            actor
                        ]
                    ),
                    float(
                        width[
                            actor
                        ]
                    ),
                    float(
                        heading[
                            actor
                        ]
                    ),
                )
            )

            if (
                surrogate["mode"]
                ==
                "front_windshield_surrogate"
            ):
                front_count += 1
            else:
                rear_count += 1

            full_spans.append(
                angular_span_deg(
                    full
                )
            )

            surrogate_spans.append(
                angular_span_deg(
                    surrogate[
                        "points"
                    ]
                )
            )


    full_spans = np.asarray(
        full_spans
    )

    surrogate_spans = np.asarray(
        surrogate_spans
    )


    print("=" * 78)
    print("STAGE 6 VEHICLE DRIVER-REGION SURROGATE SMOKE TEST")
    print("=" * 78)

    print(
        "vehicle actors =",
        len(vehicle_idx)
    )

    print(
        "vehicle future states =",
        len(full_spans)
    )

    print()
    print(
        "front / windshield states =",
        front_count
    )

    print(
        "rear / mirror states       =",
        rear_count
    )

    print()
    print(
        "mean full-box angular span =",
        float(
            full_spans.mean()
        ),
        "deg"
    )

    print(
        "mean surrogate angular span =",
        float(
            surrogate_spans.mean()
        ),
        "deg"
    )

    print(
        "median full-box span =",
        float(
            np.median(
                full_spans
            )
        ),
        "deg"
    )

    print(
        "median surrogate span =",
        float(
            np.median(
                surrogate_spans
            )
        ),
        "deg"
    )

    print()
    print(
        "surrogate <= full box fraction =",
        float(
            np.mean(
                surrogate_spans
                <=
                full_spans
                +
                1e-9
            )
        )
    )

    assert np.all(
        np.isfinite(
            surrogate_spans
        )
    )

    assert np.all(
        surrogate_spans
        <=
        full_spans
        +
        1e-9
    )

    assert (
        front_count
        +
        rear_count
        ==
        len(full_spans)
    )

    print()
    print(
        "SURROGATE GEOMETRY SMOKE TEST = PASS"
    )

    print()
    print(
        "Scientific semantics:"
    )

    print(
        "This is a geometry-based surrogate, "
        "NOT measured windshield/mirror geometry."
    )


if __name__ == "__main__":
    main()
