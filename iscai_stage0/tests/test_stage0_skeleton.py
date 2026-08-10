from __future__ import annotations

import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

from audit_dataset_layout import build_report  # noqa: E402
from iscai_stage0.common import load_config, write_json  # noqa: E402


def test_config_loads() -> None:
    config = load_config(PROJECT_ROOT / "configs" / "stage0.json")
    assert config["dataset_root"] == "/waymo"


def test_json_writer_round_trip(tmp_path: Path) -> None:
    target = tmp_path / "out.json"
    write_json(target, {"ok": True})
    assert json.loads(target.read_text(encoding="utf-8")) == {"ok": True}


def test_layout_audit_does_not_infer_schema(tmp_path: Path) -> None:
    (tmp_path / "training").mkdir()
    (tmp_path / "training" / "sample.tfrecord").write_bytes(b"abc")
    report = build_report(tmp_path, max_depth=3, sample_n=4)
    assert report["file_groups"]["train:tfrecord"] == 1
    assert "no dataset field is considered confirmed" in report["note"]
