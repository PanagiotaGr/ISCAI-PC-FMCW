import json
from pathlib import Path

import numpy as np


from iscai_stage5.adapters.stage4_prediction_loader import (
    Stage4PredictionLoader
)

from iscai_stage5.adapters.angular_posterior import (
    AngularPosterior
)

from iscai_stage5.rl.beam_agent import (
    BeamAgent
)

from iscai_stage5.rl.beam_environment import (
    BeamEnvironment
)



STAGE4_ARTIFACT = (
    "/home/agni/waymo/iscai_stage4/"
    "stage4_final_package/artifacts/"
    "stage4_prediction_artifact.json"
)


OUTPUT = Path(
    "reports/stage5_rl/"
    "predictive_adb_real_test.json"
)



def main():

    print("==============================")
    print("STAGE 5 PREDICTIVE ADB")
    print("==============================")


    # Load Stage 4 artifact

    loader = Stage4PredictionLoader(
        STAGE4_ARTIFACT
    )


    gaussian_records = (
        loader.get_gaussian_predictions()
    )


    if len(gaussian_records) == 0:

        raise RuntimeError(
            "No GAUSSIAN_GRU records found"
        )


    # Take first actor for smoke integration

    record = gaussian_records[0]


    trajectory = np.array(
        record["trajectory"],
        dtype=np.float32
    )


    uncertainty = float(
        record.get(
            "uncertainty",
            0.0
        )
    )


    print()
    print(
        "Actor:",
        record["actor_class"]
    )

    print(
        "Scenario:",
        record["scenario_id"]
    )


    # trajectory -> beam posterior

    posterior_engine = AngularPosterior(
        num_beams=64
    )


    posterior = (
        posterior_engine.compute(
            trajectory,
            uncertainty
        )
    )


    top_beams = np.argsort(
        posterior
    )[-5:]


    print()
    print(
        "Posterior top beams:",
        top_beams.tolist()
    )


    # RL controller

    env = BeamEnvironment(
        num_beams=64,
        max_active_beams=4
    )


    state = env.reset(
        angular_posterior=posterior,
        uncertainty=uncertainty
    )


    agent = BeamAgent(
        num_beams=64,
        max_active_beams=4
    )


    selected = agent.select_action(
        state
    )


    _, reward, _, info = env.step(
        selected
    )


    result = {

        "status":
            "PASS",

        "model":
            record["model_name"],

        "actor_class":
            record["actor_class"],

        "selected_beams":
            [
                int(x)
                for x in selected
            ],

        "posterior_top5":
            [
                int(x)
                for x in top_beams
            ],

        "reward":
            float(reward),

        "uncertainty":
            uncertainty,

        "metrics":
            {
                k: float(v)
                for k,v in info.items()
            }

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
    print(
        "Selected beams:",
        selected
    )

    print(
        "Reward:",
        reward
    )

    print()
    print(
        "Saved:",
        OUTPUT
    )



if __name__ == "__main__":
    main()
