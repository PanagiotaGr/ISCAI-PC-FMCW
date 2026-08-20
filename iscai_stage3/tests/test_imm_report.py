import unittest

from iscai_stage3.baselines.imm_report import (
    IMMReportRecord,
    select_mode,
    report_sha256,
)


class TestIMMReport(unittest.TestCase):


    def test_mode_selection(self):

        mode = select_mode(
            cv=0.7,
            ca=0.2,
            ctrv=0.1,
        )

        self.assertEqual(
            mode,
            "CV",
        )


    def test_hash_repeatable(self):

        records = (
            IMMReportRecord(
                track_index=1,
                time_index=2,

                cv_probability=0.5,
                ca_probability=0.3,
                ctrv_probability=0.2,

                cv_log_likelihood=1.0,
                ca_log_likelihood=0.5,
                ctrv_log_likelihood=0.1,

                selected_mode="CV",
            ),
        )


        h1 = report_sha256(records)
        h2 = report_sha256(records)


        self.assertEqual(
            h1,
            h2,
        )


if __name__ == "__main__":
    unittest.main()
