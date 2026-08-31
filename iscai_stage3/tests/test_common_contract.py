import unittest

from iscai_stage3.contracts import (
    AlgorithmObservationSequence,
    AssociationMode,
)

from iscai_stage3.observations import (
    algorithm_sequence_from_degraded_scene,
)


class _TruthBomb:
    def __get__(self, instance, owner):
        raise AssertionError(
            "Algorithm bridge accessed truth."
        )


class _FakeScene:
    frames = ()
    measured_fmcw = False

    truth_sidecars = _TruthBomb()


class _FakeMeasuredScene:
    frames = ()
    measured_fmcw = True


class TestCommonContract(unittest.TestCase):

    def test_empty_structural_sequence_allowed(
        self,
    ):
        result = AlgorithmObservationSequence(
            scenario_id="scenario-test",
            frames=(),
        )

        self.assertEqual(
            result.frames,
            (),
        )

        self.assertFalse(
            result.measured_fmcw
        )

    def test_bridge_never_reads_truth(
        self,
    ):
        result = (
            algorithm_sequence_from_degraded_scene(
                scenario_id="scenario-test",
                scene=_FakeScene(),
            )
        )

        self.assertEqual(
            result.frames,
            (),
        )

    def test_measured_fmcw_rejected(
        self,
    ):
        with self.assertRaises(
            ValueError
        ):
            algorithm_sequence_from_degraded_scene(
                scenario_id="scenario-test",
                scene=_FakeMeasuredScene(),
            )

    def test_association_modes_are_explicit(
        self,
    ):
        self.assertEqual(
            set(AssociationMode),
            {
                AssociationMode.ORACLE_IDENTITY_DIAGNOSTIC,
                AssociationMode.ORACLE_ASSOCIATION_DIAGNOSTIC,
                AssociationMode.ESTIMATED_ASSOCIATION,
                AssociationMode.TRACK_BEFORE_DETECT,
            },
        )


if __name__ == "__main__":
    unittest.main()
