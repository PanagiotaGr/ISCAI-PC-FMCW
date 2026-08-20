from pathlib import Path
import re


ROOT = Path("stage6_real")

FILES = list(
    ROOT.rglob("*.py")
)

text = "\n".join(
    p.read_text(errors="ignore")
    for p in FILES
)


def check(name, patterns):

    found = []

    for pattern in patterns:
        if re.search(
            pattern,
            text,
            re.I,
        ):
            found.append(pattern)

    status = (
        "FOUND"
        if found
        else
        "MISSING"
    )

    print(
        f"{status:8s} {name}"
    )

    return bool(found)


print("=" * 78)
print("STAGE 6 PDF REQUIREMENTS AUDIT")
print("=" * 78)

checks = {}


# ------------------------------------------------------------
# Stage-6 core
# ------------------------------------------------------------

checks["future_boxes"] = check(
    "future box projection",
    [
        r"future.*box",
        r"box_corners",
    ],
)

checks["probabilistic_occupancy"] = check(
    "probabilistic occupancy / masks",
    [
        r"probabilistic_occupancy",
        r"occupancy.*gamma",
    ],
)

checks["covariance"] = check(
    "predictive covariance",
    [
        r"future_std",
        r"rho",
        r"covariance",
    ],
)

checks["class_margin"] = check(
    "class-dependent margins",
    [
        r"lateral_margin_scale",
        r"class.*margin",
    ],
)


# ------------------------------------------------------------
# Part-A illumination function
# ------------------------------------------------------------

checks["radial_threshold"] = check(
    "radial thresholds",
    [
        r"RADIAL_MARGIN",
        r"r_min",
        r"r_max",
    ],
)

checks["angular_shadow"] = check(
    "angular shadow zones",
    [
        r"theta_min",
        r"theta_max",
    ],
)

checks["raised_cosine"] = check(
    "raised-cosine transition",
    [
        r"raised.cosine",
        r"np\.cos",
    ],
)

checks["minimum_floor"] = check(
    "minimum illumination floor",
    [
        r"floor",
        r"intensity_floor",
    ],
)

checks["temporal_smoothing"] = check(
    "temporal smoothing CONTROLLER",
    [
        r"temporal_smooth",
        r"smooth.*alpha",
        r"exponential.*smooth",
    ],
)

checks["actuation_limit"] = check(
    "actuation-rate limiting",
    [
        r"actuation.*limit",
        r"rate.*limit",
        r"max.*change",
    ],
)


# ------------------------------------------------------------
# Class-aware details
# ------------------------------------------------------------

checks["vehicle_driver_region"] = check(
    "vehicle windshield/mirror/driver surrogate",
    [
        r"windshield",
        r"mirror",
        r"driver.*region",
    ],
)

checks["closing_speed"] = check(
    "vehicle closing-speed dependent margin",
    [
        r"closing.*speed",
        r"relative.*speed",
    ],
)

checks["pedestrian_visibility_policy"] = check(
    "pedestrian visibility / no-blackout policy",
    [
        r"PEDESTRIAN",
        r"pedestrian.*floor",
    ],
)

checks["cyclist_crossing"] = check(
    "cyclist crossing / lateral maneuver policy",
    [
        r"crossing",
        r"lateral.*maneuver",
    ],
)


# ------------------------------------------------------------
# ADB baselines
# ------------------------------------------------------------

checks["static"] = check(
    "static ADB baseline",
    [
        r"static.*adb",
        r"static.*mask",
    ],
)

checks["reactive"] = check(
    "Part-A reactive baseline",
    [
        r"reactive",
    ],
)

checks["deterministic"] = check(
    "deterministic predictive baseline",
    [
        r"deterministic",
    ],
)

checks["uncertainty"] = check(
    "uncertainty-aware predictive baseline",
    [
        r"uncertainty_gamma",
    ],
)

checks["class_agnostic"] = check(
    "class-agnostic predictive baseline",
    [
        r"class.agnostic",
        r"uncertainty_class_agnostic",
    ],
)

checks["class_aware"] = check(
    "class-aware predictive ADB",
    [
        r"class_aware",
    ],
)

checks["oracle"] = check(
    "oracle future ADB",
    [
        r"oracle_D",
        r"oracle.*mask",
    ],
)


# ------------------------------------------------------------
# PDF ADB metrics
# ------------------------------------------------------------

metric_patterns = {
    "mask IoU":
        [r"def iou"],

    "vehicle shadow-zone violation":
        [r"shadow_violation"],

    "glare-risk exposure":
        [r"glare.*exposure"],

    "over-masking":
        [r"overmask"],

    "road illumination retention":
        [r"road_retention"],

    "pedestrian/cyclist visibility":
        [r"oracle_region_visibility"],

    "false dimming":
        [r"false_dimming"],

    "temporal smoothness metric":
        [r"temporal_smoothness"],

    "flicker/change rate":
        [r"flicker_rate"],

    "energy consumption":
        [r"energy.*consumption", r"energy_metric"],

    "actuation latency":
        [r"actuation.*latency"],
}

for name, patterns in metric_patterns.items():

    checks[
        "metric_" + name
    ] = check(
        "metric: " + name,
        patterns,
    )


# ------------------------------------------------------------
# Summary
# ------------------------------------------------------------

passed = sum(
    bool(v)
    for v in checks.values()
)

total = len(
    checks
)

print()
print("=" * 78)
print("SUMMARY")
print("=" * 78)

print(
    "requirements found =",
    passed,
    "/",
    total,
)

missing = [
    name
    for name, ok
    in checks.items()
    if not ok
]

print()
print("MISSING / NEEDS MANUAL VERIFICATION:")

for name in missing:
    print(
        " -",
        name
    )

print()
print(
    "IMPORTANT: FOUND means only that matching implementation "
    "exists somewhere in stage6_real."
)

print(
    "It does NOT yet prove scientific correctness."
)
