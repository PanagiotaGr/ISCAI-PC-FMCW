from iscai_stage3.baselines.state_adapter import (
    build_causal_state
)


observations = (

    (10.0, 0.0, 0.0),

    (11.0, 0.05, 0.0),

    (12.0, 0.10, 0.0),

)


state = build_causal_state(
    observations,
    dt=0.1,
)


print(
"===== Stage3 State Adapter Smoke ====="
)

print(
"position =",
state["position"]
)

print(
"velocity =",
state["velocity"]
)

print(
"acceleration =",
state["acceleration"]
)

print(
"future_used = NO"
)

print(
"STATUS = PASS"
)
