from pathlib import Path
import tempfile
import unittest

from iscai_stage4.ml.closure_runtime import (
    acceptance_matrix,
    file_sha256,
    recursive_hex64_values,
    report_contains_sha,
    require_false,
    require_true,
)


class TestBlock410Closure(
    unittest.TestCase
):

    def test_recursive_hash_discovery(self):
        value = {
            "a": [
                {
                    "hash":
                        "a" * 64
                }
            ]
        }

        self.assertEqual(
            recursive_hex64_values(
                value
            ),
            (
                "a" * 64,
            ),
        )

    def test_report_contains_expected_hash(self):
        payload = {
            "implementation": {
                "sha256":
                    "b" * 64
            }
        }

        self.assertTrue(
            report_contains_sha(
                payload,
                "b" * 64,
            )
        )

    def test_require_true(self):
        self.assertTrue(
            require_true(
                True,
                name="criterion",
            )
        )

    def test_require_false(self):
        self.assertTrue(
            require_false(
                False,
                name="forbidden",
            )
        )

    def test_acceptance_blocks_mandatory_failure(self):
        result = acceptance_matrix(
            [
                {
                    "id":
                        "required",

                    "mandatory":
                        True,

                    "pass":
                        False,
                },
                {
                    "id":
                        "diagnostic",

                    "mandatory":
                        False,

                    "pass":
                        False,
                },
            ]
        )

        self.assertFalse(
            result[
                "closure_ready"
            ]
        )

        self.assertEqual(
            result[
                "mandatory_failures"
            ],
            [
                "required"
            ],
        )

    def test_acceptance_ignores_nonmandatory_failure(self):
        result = acceptance_matrix(
            [
                {
                    "id":
                        "required",

                    "mandatory":
                        True,

                    "pass":
                        True,
                },
                {
                    "id":
                        "diagnostic",

                    "mandatory":
                        False,

                    "pass":
                        False,
                },
            ]
        )

        self.assertTrue(
            result[
                "closure_ready"
            ]
        )


if __name__ == "__main__":
    unittest.main()
