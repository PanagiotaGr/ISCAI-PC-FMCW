from hashlib import sha256
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np
import torch

from iscai_stage4.ml.reproducibility_runtime import (
    FORMAL_ARRAY_KEYS,
    canonical_json_sha256,
    file_sha256,
    formal_cache_content_sha256,
    state_dict_sha256,
)


class TestBlock49Reproducibility(
    unittest.TestCase
):

    def test_file_sha256(self):
        with tempfile.TemporaryDirectory() as root:
            path = (
                Path(root)
                /
                "value.bin"
            )

            path.write_bytes(
                b"Agni"
            )

            expected = sha256(
                b"Agni"
            ).hexdigest()

            self.assertEqual(
                file_sha256(
                    path
                ),
                expected,
            )

    def test_canonical_json_ignores_key_order(self):
        first = {
            "b": 2,
            "a": 1,
        }

        second = {
            "a": 1,
            "b": 2,
        }

        self.assertEqual(
            canonical_json_sha256(
                first
            ),
            canonical_json_sha256(
                second
            ),
        )

    def test_state_dict_hash_exact_repeat(self):
        state = {
            "a":
                torch.tensor(
                    [
                        1.0,
                        2.0,
                    ]
                ),

            "b":
                torch.tensor(
                    [
                        3.0
                    ]
                ),
        }

        first = state_dict_sha256(
            state
        )

        second = state_dict_sha256(
            {
                "b":
                    state[
                        "b"
                    ].clone(),

                "a":
                    state[
                        "a"
                    ].clone(),
            }
        )

        self.assertEqual(
            first,
            second,
        )

    def test_state_dict_hash_detects_change(self):
        first = {
            "weight":
                torch.zeros(
                    2
                )
        }

        second = {
            "weight":
                torch.tensor(
                    [
                        0.0,
                        1.0,
                    ]
                )
        }

        self.assertNotEqual(
            state_dict_sha256(
                first
            ),
            state_dict_sha256(
                second
            ),
        )

    def _write_shard(
        self,
        path,
        *,
        scenario_id,
        prediction_sha,
        value,
    ):
        arrays = {}

        for key in (
            FORMAL_ARRAY_KEYS
        ):
            if key in (
                "selected_class_id",
                "eligible_class_id",
            ):
                arrays[
                    key
                ] = np.asarray(
                    [
                        1
                    ],
                    dtype=np.int64,
                )

            elif key in (
                "validity",
                "eligible_validity",
            ):
                arrays[
                    key
                ] = np.ones(
                    (
                        1,
                        4,
                    ),
                    dtype=np.bool_,
                )

            elif key == "gaussian_scale":
                arrays[
                    key
                ] = (
                    np.eye(
                        3,
                        dtype=np.float32,
                    )[
                        None,
                        None,
                        :,
                        :
                    ]
                    .repeat(
                        4,
                        axis=1,
                    )
                )

            elif key == "gmm_logits":
                arrays[
                    key
                ] = np.zeros(
                    (
                        1,
                        3,
                    ),
                    dtype=np.float32,
                )

            elif key == "gmm_means":
                arrays[
                    key
                ] = np.full(
                    (
                        1,
                        3,
                        4,
                        3,
                    ),
                    value,
                    dtype=np.float32,
                )

            elif key == "gmm_scale":
                arrays[
                    key
                ] = (
                    np.eye(
                        3,
                        dtype=np.float32,
                    )[
                        None,
                        None,
                        None,
                        :,
                        :
                    ]
                    .repeat(
                        3,
                        axis=1,
                    )
                    .repeat(
                        4,
                        axis=2,
                    )
                )

            else:
                arrays[
                    key
                ] = np.full(
                    (
                        1,
                        4,
                        3,
                    ),
                    value,
                    dtype=np.float32,
                )

        np.savez(
            path,
            scenario_id=np.asarray(
                scenario_id
            ),
            cache_contract_sha256=np.asarray(
                "c" * 64
            ),
            all_prediction_sha256=np.asarray(
                prediction_sha
            ),
            matched_formal_target_count=np.asarray(
                1,
                dtype=np.int64,
            ),
            eligible_formal_target_count=np.asarray(
                1,
                dtype=np.int64,
            ),
            tracks_to_predict_accessed_after_prediction=np.asarray(
                True,
                dtype=np.bool_,
            ),
            **arrays,
        )

    def test_formal_cache_digest_repeat(self):
        with tempfile.TemporaryDirectory() as root:
            root = Path(
                root
            )

            rows = [
                {
                    "scenario_id":
                        "scene_a"
                },
                {
                    "scenario_id":
                        "scene_b"
                },
            ]

            self._write_shard(
                root
                /
                "001_scene_a.npz",
                scenario_id="scene_a",
                prediction_sha=(
                    "a" * 64
                ),
                value=1.0,
            )

            self._write_shard(
                root
                /
                "002_scene_b.npz",
                scenario_id="scene_b",
                prediction_sha=(
                    "b" * 64
                ),
                value=2.0,
            )

            first = (
                formal_cache_content_sha256(
                    rows,
                    root,
                )
            )

            second = (
                formal_cache_content_sha256(
                    rows,
                    root,
                )
            )

            self.assertEqual(
                first[
                    "sha256"
                ],
                second[
                    "sha256"
                ],
            )

            self.assertEqual(
                first[
                    "shard_count"
                ],
                2,
            )

    def test_formal_cache_rejects_wrong_scene(self):
        with tempfile.TemporaryDirectory() as root:
            root = Path(
                root
            )

            rows = [
                {
                    "scenario_id":
                        "expected"
                }
            ]

            self._write_shard(
                root
                /
                "001_expected.npz",
                scenario_id="wrong",
                prediction_sha=(
                    "a" * 64
                ),
                value=0.0,
            )

            with self.assertRaises(
                RuntimeError
            ):
                formal_cache_content_sha256(
                    rows,
                    root,
                )


if __name__ == "__main__":
    unittest.main()
