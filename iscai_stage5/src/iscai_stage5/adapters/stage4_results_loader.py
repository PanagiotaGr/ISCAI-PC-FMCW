import json
from pathlib import Path


class Stage4ResultsLoader:


    def __init__(self, results_path):

        self.path = Path(results_path)



    def load_results(self):

        with open(self.path, "r") as f:
            return json.load(f)



    def get_gaussian_results(self):

        results = self.load_results()

        return [
            r
            for r in results
            if "gaussian" in r["name"].lower()
        ]



    def get_best_gaussian(self):

        gaussian = self.get_gaussian_results()

        if not gaussian:
            raise RuntimeError(
                "No Gaussian experiments finished yet"
            )


        best = min(
            gaussian,
            key=lambda x:
                x["metrics"]["FDE"]
        )


        return best
