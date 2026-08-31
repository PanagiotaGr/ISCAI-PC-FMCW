import unittest

from pathlib import Path


ROOT = Path(
    "/home/agni/waymo/iscai_stage3"
)

SRC = ROOT / "src/iscai_stage3"


class TestStageBoundaries(unittest.TestCase):

    def _python_texts(self, directory):
        if not directory.exists():
            return ()

        return tuple(
            (
                path,
                path.read_text(
                    encoding="utf-8"
                ),
            )
            for path in directory.rglob("*.py")
        )

    def test_no_legacy_panagiota_imports(
        self,
    ):
        offenders = []

        for path, text in self._python_texts(
            SRC
        ):
            if "iscai_stage3_panagiota" in text:
                offenders.append(str(path))

        self.assertEqual(
            offenders,
            [],
        )

    def test_no_downstream_stage_imports(
        self,
    ):
        forbidden = (
            "iscai_stage4",
            "iscai_stage5",
            "iscai_stage6",
            "iscai_stage7",
        )

        offenders = []

        for path, text in self._python_texts(
            SRC
        ):
            if any(
                token in text
                for token in forbidden
            ):
                offenders.append(str(path))

        self.assertEqual(
            offenders,
            [],
        )

    def test_algorithm_packages_do_not_import_truth(
        self,
    ):
        algorithm_roots = (
            SRC / "observations",
            SRC / "association",
            SRC / "baselines",
        )

        forbidden = (
            "DetectionTruthEntry",
            "DetectionTruthSidecar",
            "EvaluatorTruthSequence",
            "evaluation.truth",
        )

        offenders = []

        for directory in algorithm_roots:
            for path, text in self._python_texts(
                directory
            ):
                if any(
                    token in text
                    for token in forbidden
                ):
                    offenders.append(
                        str(path)
                    )

        self.assertEqual(
            offenders,
            [],
        )


if __name__ == "__main__":
    unittest.main()
