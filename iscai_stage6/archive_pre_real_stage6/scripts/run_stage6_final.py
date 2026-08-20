from pathlib import Path
import json
import time


STAGE4_REPORT = Path(
    "/home/agni/waymo/iscai_stage4/reports/stage4_final/stage4_summary.json"
)

STAGE5_REPORT = Path(
    "/home/agni/waymo/iscai_stage5/reports/stage5_scientific_evaluation_summary.json"
)

OUTPUT = Path(
    "reports/stage6_final/stage6_final_report.json"
)


def load_if_exists(path):

    if path.exists():
        return json.loads(
            path.read_text()
        )

    return {
        "status": "NOT_FOUND",
        "path": str(path)
    }



def main():

    print("==============================")
    print("STAGE 6 FINAL INTEGRATION")
    print("==============================")


    print()
    print("Loading Stage 4...")
    stage4 = load_if_exists(
        STAGE4_REPORT
    )


    print(
        "Stage 4:",
        stage4.get(
            "experiments",
            stage4.get("status")
        )
    )


    print()
    print("Loading Stage 5...")
    stage5 = load_if_exists(
        STAGE5_REPORT
    )


    print(
        "Stage 5:",
        stage5.get(
            "status"
        )
    )


    final = {

        "status": "READY",

        "timestamp": time.time(),

        "components": {

            "stage4_prediction":
                stage4,

            "stage5_adaptive_sensing":
                stage5,

        },

        "pipeline":

            [
                "trajectory_prediction",
                "uncertainty_estimation",
                "adaptive_beam_selection",
                "closed_loop_evaluation"
            ]

    }


    OUTPUT.parent.mkdir(
        parents=True,
        exist_ok=True
    )


    OUTPUT.write_text(
        json.dumps(
            final,
            indent=2
        )
    )


    print()
    print("SAVED:")
    print(OUTPUT)



if __name__ == "__main__":
    main()
