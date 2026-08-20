import torch


def gaussian_nll(
    pred,
    target,
    eps=1e-6
):

    mu = pred["mu"]

    std = pred["std"] + eps

    rho = pred["rho"].squeeze(-1)


    dx = target[...,0] - mu[...,0]
    dy = target[...,1] - mu[...,1]


    sx = std[...,0]
    sy = std[...,1]


    z = (
        (dx / sx)**2
        +
        (dy / sy)**2
        -
        2*rho*(dx/sx)*(dy/sy)
    )


    denom = 2*(1-rho**2)


    nll = (
        0.5*z/denom
        +
        torch.log(
            sx*sy*torch.sqrt(
                1-rho**2
            )
        )
    )


    return nll.mean()
