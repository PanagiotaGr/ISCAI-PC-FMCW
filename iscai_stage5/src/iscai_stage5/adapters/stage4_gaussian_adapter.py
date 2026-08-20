import numpy as np


class Stage4GaussianAdapter:
    """
    Adapter between Stage 4 probabilistic
    trajectory prediction and Stage 5
    beam selection.

    Input:
        trajectory mean
        trajectory covariance

    Output:
        angular posterior over beams
    """


    def __init__(
        self,
        num_beams=64
    ):

        self.num_beams = num_beams



    def trajectory_to_angles(
        self,
        trajectory_mean
    ):
        """
        Convert future trajectory points
        into angular directions.

        trajectory_mean:
            [T,2]
            x,y positions

        return:
            angles in radians
        """

        trajectory_mean = np.asarray(
            trajectory_mean
        )


        x = trajectory_mean[:,0]
        y = trajectory_mean[:,1]


        angles = np.arctan2(
            y,
            x
        )


        return angles



    def angles_to_posterior(
        self,
        angles,
        covariance=None
    ):
        """
        Convert angular uncertainty
        into beam probability distribution.
        """


        posterior = np.zeros(
            self.num_beams,
            dtype=np.float32
        )


        for angle in angles:

            normalized = (
                angle + np.pi
            ) / (
                2*np.pi
            )


            beam = int(
                normalized *
                self.num_beams
            )


            beam = max(
                0,
                min(
                    self.num_beams-1,
                    beam
                )
            )


            posterior[beam] += 1.0



        if posterior.sum() > 0:

            posterior /= posterior.sum()


        return posterior



    def predict_posterior(
        self,
        trajectory_mean,
        covariance=None
    ):
        """
        Full Stage4 -> Stage5 conversion.
        """

        angles = self.trajectory_to_angles(
            trajectory_mean
        )


        posterior = self.angles_to_posterior(
            angles,
            covariance
        )


        return posterior
