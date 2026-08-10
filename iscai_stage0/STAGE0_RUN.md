# Stage 0 — first run only

Run these commands **inside the Docker container that exposes `/waymo`**.

```bash
cd /path/to/iscai_part_b_stage0
python3 scripts/audit_environment.py
python3 scripts/audit_dataset_layout.py
python3 -m pytest -q
```

Expected outputs:

```text
reports/stage0/environment_manifest.json
reports/stage0/dataset_layout.json
```

Stop after these two reports. Do not install Waymo/TensorFlow packages and do not start schema parsing yet; the next step is chosen from the actual environment and file layout reported here.
