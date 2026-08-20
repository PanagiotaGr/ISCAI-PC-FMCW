import numpy as np


class AngularPosterior:


    def __init__(
        self,
        num_beams=64
    ):

        self.num_beams = num_beams



    def trajectory_to_angles(
        self,
        trajectory
    ):
        """
        Convert trajectory points
        (x,y,z) to azimuth angles.
        """

        trajectory = np.asarray(
            trajectory
        )


        x = trajectory[:,0]
        y = trajectory[:,1]


        angles = np.arctan2(
            y,
            x
        )


        return angles



    def angles_to_beam_distribution(
        self,
        angles,
        uncertainty=0.0
    ):
        """
        Create P(beam | trajectory)

        Higher uncertainty spreads
        probability over neighboring beams.
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


            center = int(
                normalized *
                self.num_beams
            )


            center = max(
                0,
                min(
                    self.num_beams-1,
                    center
                )
            )


            posterior[center] += 1.0


            # uncertainty spreading
            if uncertainty > 0:

                spread = int(
                    uncertainty *
                    5
                )

                for offset in range(
                    -spread,
                    spread+1
                ):

                    beam = center + offset

                    if (
                        0 <= beam <
                        self.num_beams
                    ):
                        posterior[beam] += (
                            1.0 /
                            (abs(offset)+1)
                        )



        if posterior.sum() > 0:

            posterior /= posterior.sum()


        return posterior



    def compute(
        self,
        trajectory,
        uncertainty=0.0
    ):

        angles = self.trajectory_to_angles(
            trajectory
        )


        posterior = self.angles_to_beam_distribution(
            angles,
            uncertainty
        )


        return posterior
