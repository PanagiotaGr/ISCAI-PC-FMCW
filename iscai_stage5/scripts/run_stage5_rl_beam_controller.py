from pathlib import Path
import json
import numpy as np

from iscai_stage5.rl.beam_environment import (
    BeamEnvironment
)

from iscai_stage5.rl.beam_agent import (
    BeamAgent
)


OUTPUT = Path(
    "reports/stage5_rl/rl_beam_controller_test.json"
)



def create_test_posterior(
    num_beams=64
):

    posterior = np.zeros(
        num_beams,
        dtype=np.float32
    )


    # simulated uncertainty distribution
    posterior[20] = 0.45
    posterior[21] = 0.30
    posterior[22] = 0.15
    posterior[23] = 0.10


    posterior /= posterior.sum()


    return posterior



def main():


    print("==============================")
    print("STAGE 5 RL BEAM CONTROLLER")
    print("==============================")


    posterior = create_test_posterior()


    env = BeamEnvironment(
        num_beams=64,
        max_active_beams=4,
    )


    state = env.reset(
        angular_posterior=posterior,
        uncertainty=0.25,
    )


    agent = BeamAgent(
        num_beams=64,
        max_active_beams=4,
    )


    selected = agent.select_action(
        state
    )


    next_state, reward, done, info = env.step(
        selected
    )


    result = {

        "status":
            "PASS",

        "selected_beams":
            [
                int(b)
                for b in selected
            ],

        "reward":
            float(reward),

        "metrics":
            {
                k: float(v)
                for k, v in info.items()
            },

        "uncertainty":
            float(
                state["uncertainty"]
            ),

    }


    OUTPUT.parent.mkdir(
        parents=True,
        exist_ok=True
    )


    OUTPUT.write_text(
        json.dumps(
            result,
            indent=2
        )
    )


    print()
    print("Selected beams:")
    print(selected)

    print()
    print("Reward:")
    print(reward)

    print()
    print("Saved:")
    print(OUTPUT)



if __name__ == "__main__":
    main()
