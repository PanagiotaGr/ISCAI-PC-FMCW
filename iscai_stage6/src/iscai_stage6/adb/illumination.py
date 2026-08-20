import math


def gaussian_probability(distance, sigma):
    """
    Simple Gaussian occupancy probability.
    """

    if sigma <= 0:
        return 0.0

    return math.exp(
        -0.5 * (distance / sigma) ** 2
    )



def trajectory_to_illumination_mask(
    angular_path,
    sigma=0.05,
    azimuth_bins=16,
    elevation_bins=8,
):
    """
    Convert angular trajectory uncertainty
    into probabilistic illumination mask.
    """

    mask = []

    for point in angular_path:

        az = point["azimuth_rad"]
        el = point["elevation_rad"]

        az_bin = int(
            ((az + math.pi) /
            (2 * math.pi))
            * azimuth_bins
        )

        el_bin = int(
            ((el + math.pi/2) /
            math.pi)
            * elevation_bins
        )

        probability = gaussian_probability(
            0.0,
            sigma
        )

        mask.append(
            {
                "azimuth_bin": az_bin,
                "elevation_bin": el_bin,
                "probability": probability,
            }
        )

    return mask
