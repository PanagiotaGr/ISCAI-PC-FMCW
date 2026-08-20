def compute_visibility_score(
    illumination_probability
):
    """
    Visibility contribution from illumination.
    """

    return max(
        0.0,
        min(
            1.0,
            illumination_probability
        )
    )



def compute_glare_penalty(
    illumination_probability,
    actor_type
):

    glare_factor = {

        "VEHICLE": 0.2,

        "PEDESTRIAN": 0.05,

        "CYCLIST": 0.1,

    }.get(
        actor_type,
        0.2
    )


    return (
        illumination_probability
        *
        glare_factor
    )



def evaluate_visibility(
    actors
):

    visibility = []

    glare = []

    classes = {}


    for actor in actors:

        score = compute_visibility_score(
            actor["probability"]
        )

        penalty = compute_glare_penalty(
            actor["probability"],
            actor["actor_type"]
        )

        visibility.append(score)

        glare.append(penalty)


        classes.setdefault(
            actor["actor_type"],
            []
        ).append(score)


    class_coverage = {

        k:
        sum(v)/len(v)

        for k,v in classes.items()

    }


    return {

        "mean_visibility":
            sum(visibility)/len(visibility),

        "mean_glare_penalty":
            sum(glare)/len(glare),

        "class_coverage":
            class_coverage,

    }
