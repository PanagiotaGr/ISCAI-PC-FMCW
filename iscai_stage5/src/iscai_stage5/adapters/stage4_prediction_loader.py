import json
from pathlib import Path


class Stage4PredictionLoader:


    def __init__(self, artifact_path):

        self.path = Path(artifact_path)



    def load(self):

        with open(self.path, "r") as f:
            return json.load(f)



    def get_gaussian_predictions(self):

        data = self.load()

        gaussian = [
            r
            for r in data
            if r.get("model_name")
            ==
            "GAUSSIAN_GRU"
        ]

        return gaussian
