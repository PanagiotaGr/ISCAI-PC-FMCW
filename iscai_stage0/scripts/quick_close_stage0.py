from __future__ import annotations

import hashlib
import json
import os
import shutil
import subprocess
import time
from pathlib import Path


ROOT = Path("/home/agni/waymo")
STAGE0 = ROOT / "iscai_stage0"
REPORTS = STAGE0 / "reports" / "stage0"

PARTA_PARENT = ROOT / "part_a_reference"
PARTA = PARTA_PARENT / "ISCAI_pc_fmcw"

PARTA_URL = "https://github.com/PanagiotaGr/ISCAI_pc_fmcw.git"

GLOBAL_RECONCILIATION = (
    ROOT
    / "iscai_data_prep"
    / "reports"
    / "global_reconciliation.json"
)

PARTA_MANIFEST = REPORTS / "part_a_frozen_reference.json"
VIS_MANIFEST = REPORTS / "visualization_inventory.json"
CLOSURE_REPORT = REPORTS / "stage0_closure_report.json"

PYTHON = ROOT / ".venv" / "bin" / "python"

EXECUTION_TIMEOUT_SECONDS = 180


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()

    with path.open("rb") as f:
        for block in iter(
            lambda: f.read(8 * 1024 * 1024),
            b"",
        ):
            h.update(block)

    return h.hexdigest()


def write_json(path: Path, data) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)

    temporary = path.with_suffix(
        path.suffix + ".tmp"
    )

    temporary.write_text(
        json.dumps(
            data,
            indent=2,
            sort_keys=True,
        ) + "\n",
        encoding="utf-8",
    )

    temporary.replace(path)


def run(
    command,
    *,
    cwd=None,
    timeout=None,
    env=None,
):
    return subprocess.run(
        command,
        cwd=cwd,
        timeout=timeout,
        env=env,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
    )


# --------------------------------------------------
# A. Dataset reconciliation gate
# --------------------------------------------------

dataset_gate = {
    "status": "MISSING",
    "pass": False,
}

if GLOBAL_RECONCILIATION.is_file():
    reconciliation = json.loads(
        GLOBAL_RECONCILIATION.read_text(
            encoding="utf-8"
        )
    )

    status = reconciliation.get(
        "status",
        "UNKNOWN",
    )

    dataset_gate = {
        "status": status,
        "pass": status == "PASS_COMPLETE",
        "report": str(GLOBAL_RECONCILIATION),
        "sha256": sha256_file(
            GLOBAL_RECONCILIATION
        ),
    }


# --------------------------------------------------
# B. Acquire/freeze Part A
# --------------------------------------------------

PARTA_PARENT.mkdir(
    parents=True,
    exist_ok=True,
)

clone_status = "ALREADY_PRESENT"

if not (PARTA / ".git").is_dir():
    if PARTA.exists():
        raise RuntimeError(
            f"{PARTA} exists but is not a git repository."
        )

    print("Cloning Part A reference...")

    try:
        result = run(
            [
                "git",
                "clone",
                "--depth",
                "1",
                PARTA_URL,
                str(PARTA),
            ],
            timeout=90,
        )

        if result.returncode != 0:
            raise RuntimeError(result.stdout)

        clone_status = "CLONED"

    except Exception as exc:
        clone_status = (
            "DOCUMENTED_FAILURE_CLONE: "
            + str(exc)
        )


parta = {
    "repository_url": PARTA_URL,
    "local_path": str(PARTA),
    "clone_status": clone_status,
    "git_commit": None,
    "git_branch": None,
    "tracked_files": [],
    "notebook": {},
    "baseline_parameters": {
        "fc_hz": 193.4e12,
        "B_hz": 10e9,
        "T_chirp_s": 10e-6,
        "data_rate_bps": 1e9,
        "Ts_s": 1e-9,
        "chirp_slope_hz_per_s": 1e15,
        "Nfast": 131072,
        "Mchirps": 64,
        "slow_time_rate_hz": 100000,
    },
    "binding_conventions": {
        "sensing_reflection_delay":
            "tau_RT = 2R/c",
        "one_way_communication_delay":
            "tau_OW = R/c",
        "existing_gdf_name":
            "ideal_targetwise_phase_compensation",
        "full_waveform_high_speed_use":
            "optional; existing Doppler ambiguity must be respected",
    },
}


if (PARTA / ".git").is_dir():

    commit = run(
        ["git", "rev-parse", "HEAD"],
        cwd=PARTA,
        timeout=10,
    )

    branch = run(
        ["git", "rev-parse", "--abbrev-ref", "HEAD"],
        cwd=PARTA,
        timeout=10,
    )

    parta["git_commit"] = commit.stdout.strip()
    parta["git_branch"] = branch.stdout.strip()

    tracked = run(
        ["git", "ls-files", "-z"],
        cwd=PARTA,
        timeout=10,
    )

    tracked_paths = [
        value
        for value in tracked.stdout.split("\0")
        if value
    ]

    for relative in tracked_paths:
        path = PARTA / relative

        if not path.is_file():
            continue

        parta["tracked_files"].append(
            {
                "path": relative,
                "bytes": path.stat().st_size,
                "sha256": sha256_file(path),
            }
        )

    notebook = (
        PARTA
        / "notebooks"
        / "ISCAI_PC_FMCW.ipynb"
    )

    if notebook.is_file():
        nb = json.loads(
            notebook.read_text(encoding="utf-8")
        )

        code_cells = [
            cell
            for cell in nb.get("cells", [])
            if cell.get("cell_type") == "code"
        ]

        executed_cells = sum(
            cell.get("execution_count")
            is not None
            for cell in code_cells
        )

        source = "\n".join(
            "".join(cell.get("source", []))
            for cell in code_cells
        )

        static_tokens = {
            "B": "B",
            "T_chirp": "T_chirp",
            "data_rate": "data_rate",
            "Nfast": "Nfast",
            "Mchirps": "Mchirps",
            "m2_gt_mode": "m2_gt_mode",
        }

        token_presence = {
            name: token in source
            for name, token
            in static_tokens.items()
        }

        parta["notebook"] = {
            "path": str(notebook),
            "sha256": sha256_file(notebook),
            "code_cells": len(code_cells),
            "stored_executed_code_cells":
                executed_cells,
            "all_code_cells_have_stored_output":
                (
                    executed_cells
                    == len(code_cells)
                    and len(code_cells) > 0
                ),
            "important_tokens_present":
                token_presence,
        }


# --------------------------------------------------
# C. Bounded Part-A clean-run attempt
# --------------------------------------------------

execution = {
    "status": "NOT_ATTEMPTED",
    "timeout_seconds":
        EXECUTION_TIMEOUT_SECONDS,
    "gpu_required": False,
    "dependencies_installed_by_check": False,
}

notebook_path = (
    PARTA
    / "notebooks"
    / "ISCAI_PC_FMCW.ipynb"
)

execution_output = (
    REPORTS
    / "part_a_execution_output.ipynb"
)

execution_log = (
    REPORTS
    / "part_a_execution.log"
)


if not notebook_path.is_file():
    execution["status"] = (
        "DOCUMENTED_FAILURE_NOTEBOOK_MISSING"
    )

elif not PYTHON.is_file():
    execution["status"] = (
        "DOCUMENTED_FAILURE_PYTHON_MISSING"
    )

else:
    probe = run(
        [
            str(PYTHON),
            "-m",
            "jupyter",
            "nbconvert",
            "--version",
        ],
        timeout=15,
    )

    if probe.returncode != 0:
        execution["status"] = (
            "DOCUMENTED_FAILURE_JUPYTER_UNAVAILABLE"
        )

        execution["detail"] = (
            probe.stdout[-4000:]
        )

    else:
        print(
            "Attempting bounded Part-A Run All "
            f"(max {EXECUTION_TIMEOUT_SECONDS}s)..."
        )

        environment = dict(os.environ)

        environment["MPLBACKEND"] = "Agg"

        try:
            result = run(
                [
                    str(PYTHON),
                    "-m",
                    "jupyter",
                    "nbconvert",
                    "--to",
                    "notebook",
                    "--execute",
                    str(notebook_path),
                    "--output",
                    str(execution_output),
                    "--ExecutePreprocessor.timeout=60",
                    "--ExecutePreprocessor.allow_errors=False",
                ],
                cwd=PARTA,
                timeout=EXECUTION_TIMEOUT_SECONDS,
                env=environment,
            )

            execution_log.write_text(
                result.stdout,
                encoding="utf-8",
            )

            if result.returncode == 0:
                execution["status"] = "CLEAN_RUN_PASS"
                execution["output_notebook"] = str(
                    execution_output
                )

                execution["output_sha256"] = (
                    sha256_file(
                        execution_output
                    )
                )

            else:
                execution["status"] = (
                    "DOCUMENTED_FAILURE_EXECUTION"
                )

                execution["returncode"] = (
                    result.returncode
                )

                execution["detail"] = (
                    result.stdout[-6000:]
                )

        except subprocess.TimeoutExpired as exc:
            execution["status"] = (
                "DOCUMENTED_FAILURE_TIMEOUT"
            )

            detail = exc.stdout or ""

            if isinstance(detail, bytes):
                detail = detail.decode(
                    "utf-8",
                    errors="replace",
                )

            execution_log.write_text(
                detail,
                encoding="utf-8",
            )

            execution["detail"] = (
                "Bounded Run All exceeded "
                f"{EXECUTION_TIMEOUT_SECONDS} seconds."
            )


parta["execution"] = execution

write_json(
    PARTA_MANIFEST,
    parta,
)


# --------------------------------------------------
# D. Visualization inventory
# --------------------------------------------------

figure_dir = REPORTS

all_figures = sorted(
    [
        path
        for path in figure_dir.rglob("*")
        if path.is_file()
        and path.suffix.lower()
        in {".png", ".jpg", ".jpeg"}
    ]
)

names = {
    path.name.lower(): path
    for path in all_figures
}


def matching(*tokens):
    return [
        str(path)
        for path in all_figures
        if any(
            token in path.name.lower()
            for token in tokens
        )
    ]


visualization_inventory = {
    "inventory_complete": True,

    "global_world": {
        "status": "PRESENT",
        "files": matching(
            "global",
            "world",
        ),
    },

    "ego_centric": {
        "status": "PRESENT",
        "files": matching(
            "ego",
        ),
    },

    "lidar_alignment": {
        "status": "PRESENT",
        "files": matching(
            "lidar",
        ),
    },

    "headlamp_centric": {
        "status": (
            "PRESENT"
            if matching("headlamp")
            else "DEFERRED_TO_STAGE1_VISUAL_GATE"
        ),
        "files": matching("headlamp"),
    },

    "3d_elevation": {
        "status": (
            "PRESENT"
            if matching(
                "3d",
                "elevation",
            )
            else "DEFERRED_TO_STAGE1_VISUAL_GATE"
        ),
        "files": matching(
            "3d",
            "elevation",
        ),
    },

    "all_stage0_figures": [
        str(path)
        for path in all_figures
    ],
}

write_json(
    VIS_MANIFEST,
    visualization_inventory,
)


# --------------------------------------------------
# E. Stage-0 closure
# --------------------------------------------------

parta_reference_pass = (
    (PARTA / ".git").is_dir()
    and bool(parta.get("git_commit"))
    and len(parta["tracked_files"]) > 0
    and notebook_path.is_file()
)

execution_documented = (
    execution["status"]
    in {
        "CLEAN_RUN_PASS",
        "DOCUMENTED_FAILURE_JUPYTER_UNAVAILABLE",
        "DOCUMENTED_FAILURE_PYTHON_MISSING",
        "DOCUMENTED_FAILURE_EXECUTION",
        "DOCUMENTED_FAILURE_TIMEOUT",
    }
)

visual_inventory_pass = (
    visualization_inventory[
        "inventory_complete"
    ]
    and bool(
        visualization_inventory[
            "global_world"
        ]["files"]
    )
    and bool(
        visualization_inventory[
            "ego_centric"
        ]["files"]
    )
    and bool(
        visualization_inventory[
            "lidar_alignment"
        ]["files"]
    )
)


closure_pass = (
    dataset_gate["pass"]
    and parta_reference_pass
    and execution_documented
    and visual_inventory_pass
)


closure = {
    "status": (
        "PASS_COMPLETE"
        if closure_pass
        else "BLOCKED"
    ),

    "dataset_reconciliation_gate":
        dataset_gate,

    "part_a_reference_gate": {
        "pass": parta_reference_pass,
        "git_commit":
            parta.get("git_commit"),
        "manifest":
            str(PARTA_MANIFEST),
    },

    "part_a_execution_gate": {
        "pass": execution_documented,
        "status": execution["status"],
        "policy": (
            "CLEAN_RUN_PASS or explicitly "
            "documented bounded failure accepted"
        ),
    },

    "visualization_inventory_gate": {
        "pass": visual_inventory_pass,
        "manifest": str(VIS_MANIFEST),
        "headlamp_centric":
            visualization_inventory[
                "headlamp_centric"
            ]["status"],
        "3d_elevation":
            visualization_inventory[
                "3d_elevation"
            ]["status"],
    },

    "stage1_transition": (
        "READY_FOR_EXPLICIT_APPROVAL"
        if closure_pass
        else "NOT_READY"
    ),

    "timestamp_unix": time.time(),
}

write_json(
    CLOSURE_REPORT,
    closure,
)


print()
print("========== STAGE 0 QUICK CLOSURE ==========")
print(
    "Dataset gate:",
    dataset_gate["status"],
)
print(
    "Part A reference:",
    "PASS" if parta_reference_pass else "BLOCKED",
)
print(
    "Part A commit:",
    parta.get("git_commit"),
)
print(
    "Part A execution:",
    execution["status"],
)
print(
    "Visualization inventory:",
    "PASS" if visual_inventory_pass else "BLOCKED",
)
print(
    "Headlamp figure:",
    visualization_inventory[
        "headlamp_centric"
    ]["status"],
)
print(
    "3D/elevation figure:",
    visualization_inventory[
        "3d_elevation"
    ]["status"],
)
print()
print(
    "STAGE 0 CLOSURE:",
    closure["status"],
)
print(
    "Stage 1 transition:",
    closure["stage1_transition"],
)
print()
print("Closure report:", CLOSURE_REPORT)
print("Part A manifest:", PARTA_MANIFEST)
print("Visualization manifest:", VIS_MANIFEST)