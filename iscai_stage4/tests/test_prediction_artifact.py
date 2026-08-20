import unittest
import tempfile

from pathlib import Path


from iscai_stage4.predictor.prediction_artifact import (
    PredictionArtifact,
    save_prediction_artifacts,
)



class TestPredictionArtifact(
    unittest.TestCase
):


    def test_save(self):

        artifact = PredictionArtifact(

            actor_id=1,

            model_name="GMM",

            trajectory=(

                (
                    1.0,
                    0.0,
                    0.0,
                ),

                (
                    2.0,
                    0.0,
                    0.0,
                ),
            ),

            uncertainty=0.2,
        )


        artifact.validate()


        path = (
            Path(
                tempfile.mkdtemp()
            )
            /
            "prediction.json"
        )


        save_prediction_artifacts(
            [
                artifact
            ],
            str(path),
        )


        self.assertTrue(
            path.exists()
        )



if __name__ == "__main__":

    unittest.main()
