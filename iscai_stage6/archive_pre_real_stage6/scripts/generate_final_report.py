from pathlib import Path
import json


STAGE4 = Path(
    "/home/agni/waymo/iscai_stage4/reports/stage4_final/stage4_summary.json"
)

STAGE5 = Path(
    "/home/agni/waymo/iscai_stage5/reports/stage5_scientific_evaluation_summary.json"
)

OUTPUT = Path(
    "reports/stage6_final/final_system_report.json"
)


def read_json(path):

    if path.exists():
        return json.loads(
            path.read_text()
        )

    return {
        "status": "MISSING",
        "file": str(path)
    }



def main():

    print("==============================")
    print("FINAL ISCAI SYSTEM REPORT")
    print("==============================")


    stage4 = read_json(STAGE4)
    stage5 = read_json(STAGE5)


    report = {

        "project":
            "ISCAI-PC-FMCW",

        "pipeline":

            {
                "stage4":
                    {
                        "trajectory_prediction":
                            stage4
                    },

                "stage5":
                    {
                        "adaptive_sensing":
                            stage5
                    },

                "stage6":
                    {
                        "status":
                            "INTEGRATED"
                    }
            },


        "evaluation":

            [
                "trajectory prediction",
                "uncertainty estimation",
                "adaptive beam selection",
                "closed loop evaluation"
            ],


        "reproducibility":

            {
                "stage4_report":
                    str(STAGE4),

                "stage5_report":
                    str(STAGE5)
            }
    }


    OUTPUT.parent.mkdir(
        parents=True,
        exist_ok=True
    )


    OUTPUT.write_text(
        json.dumps(
            report,
            indent=2
        )
    )


    print()
    print("CREATED:")
    print(OUTPUT)



if __name__ == "__main__":
    main()
