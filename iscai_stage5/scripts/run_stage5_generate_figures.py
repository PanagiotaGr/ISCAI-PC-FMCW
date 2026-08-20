
from pathlib import Path
import json

import matplotlib.pyplot as plt


OUT = Path(
    "reports/figures"
)

OUT.mkdir(
    parents=True,
    exist_ok=True
)


# ---------- FIGURE 1 IMM ----------

imm = json.loads(
    Path(
        "reports/block5_measurement_driven_imm_v2.json"
    ).read_text()
)


plt.figure(
    figsize=(6,4)
)

plt.bar(
    [
        "normalized\nentropy",
        "nonuniform\nfraction"
    ],
    [
        imm["normalized_entropy"],
        imm["nonuniform_fraction"]
    ]
)

plt.ylabel(
    "value"
)

plt.title(
    "IMM Model Discrimination"
)

plt.tight_layout()

plt.savefig(
    OUT / "fig1_imm_entropy.png",
    dpi=300
)

plt.close()



# ---------- FIGURE 2 MHT ----------

mht = json.loads(
    Path(
        "reports/block5_mht_lambda_sweep.json"
    ).read_text()
)


lam = []
change = []
ctrv = []


for k,v in mht["lambda_results"].items():

    lam.append(
        float(k)
    )

    change.append(
        v["change_rate"]
    )

    ctrv.append(
        v["ctrv_selection_rate"]
    )


plt.figure(
    figsize=(6,4)
)

plt.plot(
    lam,
    change,
    marker="o",
    label="ranking change"
)

plt.plot(
    lam,
    ctrv,
    marker="o",
    label="CTRV selection"
)

plt.xlabel(
    "lambda"
)

plt.ylabel(
    "rate"
)

plt.title(
    "IMM-MHT Lambda Calibration"
)

plt.legend()

plt.tight_layout()

plt.savefig(
    OUT / "fig2_mht_lambda_sweep.png",
    dpi=300
)

plt.close()



# ---------- FIGURE 3 Scheduler ----------

sched = json.loads(
    Path(
        "reports/block5_scheduler_comparison.json"
    ).read_text()
)


classes = [
    "vehicle",
    "pedestrian",
    "cyclist"
]


v1 = [
    sched["v1"][c+"_coverage"]
    for c in classes
]


v2 = [
    sched["v2"][c+"_coverage"]
    for c in classes
]


x = range(
    len(classes)
)


plt.figure(
    figsize=(6,4)
)

plt.bar(
    [i-0.2 for i in x],
    v1,
    width=0.4,
    label="V1"
)

plt.bar(
    [i+0.2 for i in x],
    v2,
    width=0.4,
    label="V2"
)

plt.xticks(
    list(x),
    classes
)

plt.ylabel(
    "coverage"
)

plt.title(
    "ADB Scheduler Comparison"
)

plt.legend()

plt.tight_layout()

plt.savefig(
    OUT / "fig3_scheduler_balance.png",
    dpi=300
)

plt.close()



# ---------- FIGURE 4 Temporal ----------

temp = json.loads(
    Path(
        "reports/block5_temporal_closed_loop_evaluation.json"
    ).read_text()
)


plt.figure(
    figsize=(5,4)
)

plt.bar(
    [
        "retention",
        "switch rate"
    ],
    [
        temp["retention_rate"],
        temp["switch_rate"]
    ]
)

plt.ylabel(
    "rate"
)

plt.title(
    "Temporal Beam Persistence"
)

plt.tight_layout()

plt.savefig(
    OUT / "fig4_temporal_persistence.png",
    dpi=300
)

plt.close()



print(
    "===== Stage5 Figures Generation ====="
)

print(
    "figures = 4"
)

print(
    "output =",
    OUT
)

print(
    "STATUS = PASS"
)
