from dataclasses import dataclass


@dataclass
class ADBDecision:
    actor_type: str
    priority: float
    illuminate: bool



CLASS_PRIORITY = {

    "PEDESTRIAN": 1.0,

    "CYCLIST": 0.9,

    "VEHICLE": 0.7,

}



def compute_adb_priority(
    actor_type,
    illumination_probability,
):
    base = CLASS_PRIORITY.get(
        actor_type,
        0.5
    )

    return base * illumination_probability



def decide_illumination(
    actor_type,
    illumination_probability,
    threshold=0.3,
):

    priority = compute_adb_priority(
        actor_type,
        illumination_probability
    )


    return ADBDecision(

        actor_type=actor_type,

        priority=priority,

        illuminate=priority >= threshold

    )
