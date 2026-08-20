import math


def cartesian_to_angles(x, y, z=0.0):
    """
    Convert headlamp-frame coordinates
    to azimuth/elevation.
    """

    r_xy = math.sqrt(
        x*x + y*y
    )

    azimuth = math.atan2(
        y,
        x
    )

    elevation = math.atan2(
        z,
        r_xy
    )

    return azimuth, elevation



def trajectory_to_angular_path(
    trajectory
):
    """
    Convert future trajectory points
    into angular occupancy path.
    """

    output = []

    for point in trajectory:

        x, y, z = point

        az, el = cartesian_to_angles(
            x,
            y,
            z
        )

        output.append(
            {
                "azimuth_rad": az,
                "elevation_rad": el,
            }
        )

    return output
