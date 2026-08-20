import numpy as np

from rl_real.src.horizon_env import HorizonBeamEnvironment


DATA = "rl_real/data/real_rl_states.npz"


def main():

    print("=" * 72)
    print("REAL RL HORIZON ENVIRONMENT SMOKE TEST")
    print("=" * 72)

    data = np.load(DATA)

    posterior = data["posterior"]
    trajectory_id = data["trajectory_id"]
    future_step = data["future_step"]

    print("posterior shape =", posterior.shape)
    print("trajectory shape =", trajectory_id.shape)
    print("future_step shape =", future_step.shape)

    # --------------------------------------------------------
    # Basic dataset checks
    # --------------------------------------------------------

    assert posterior.ndim == 2
    assert posterior.shape[1] == 64
    assert len(posterior) == len(trajectory_id)
    assert len(posterior) == len(future_step)

    assert np.isfinite(posterior).all()
    assert np.all(posterior >= 0.0)

    sums = posterior.sum(axis=1)

    assert np.allclose(
        sums,
        1.0,
        atol=1e-5,
    )

    # --------------------------------------------------------
    # Environment
    # --------------------------------------------------------

    env = HorizonBeamEnvironment(
        posteriors=posterior,
        trajectory_ids=trajectory_id,
        future_steps=future_step,
        target_mass=0.95,
        beam_cost_weight=0.15,
        switching_cost_weight=0.10,
        outage_penalty=1.0,
    )

    print()
    print("state_dim =", env.state_dim)
    print("num_actions =", env.num_actions)
    print("actions K =", env.ACTION_K.tolist())

    assert env.state_dim == 129
    assert env.num_actions == 6

    assert np.array_equal(
        env.ACTION_K,
        np.asarray(
            [1, 2, 3, 4, 5, 64]
        ),
    )

    # --------------------------------------------------------
    # Pick one real trajectory
    # --------------------------------------------------------

    tid = int(
        np.unique(
            trajectory_id
        )[0]
    )

    indices = env.trajectory_to_indices[
        tid
    ]

    horizons = future_step[
        indices
    ]

    print()
    print("trajectory_id =", tid)
    print("indices =", indices.tolist())
    print("horizons =", horizons.tolist())

    assert len(indices) == 10

    assert np.array_equal(
        horizons,
        np.arange(10),
    )

    # --------------------------------------------------------
    # Reset
    # --------------------------------------------------------

    state = env.reset(
        tid
    )

    print()
    print("RESET")
    print("state shape =", state.shape)
    print(
        "posterior sum =",
        state[:64].sum()
    )
    print(
        "previous mask sum =",
        state[64:128].sum()
    )
    print(
        "horizon normalized =",
        state[128]
    )

    assert state.shape == (129,)

    assert np.isclose(
        state[:64].sum(),
        1.0,
        atol=1e-5,
    )

    assert np.isclose(
        state[64:128].sum(),
        0.0,
    )

    assert np.isclose(
        state[128],
        0.0,
    )

    # --------------------------------------------------------
    # Test K=1 at horizon 0
    # --------------------------------------------------------

    next_state, reward, done, info = env.step(
        0
    )

    print()
    print("ACTION 0 -> K=1")
    print("K =", info["k"])
    print(
        "selected =",
        info["selected_beams"]
    )
    print(
        "covered mass =",
        info["covered_mass"]
    )
    print(
        "switching cost =",
        info["switching_cost"]
    )
    print(
        "outage =",
        info["outage"]
    )
    print(
        "reward =",
        reward
    )
    print(
        "next horizon =",
        next_state[128]
    )

    assert info["k"] == 1
    assert len(
        info["selected_beams"]
    ) == 1

    # First decision has no previous beam set.
    assert np.isclose(
        info["switching_cost"],
        0.0,
    )

    assert done is False

    assert np.isclose(
        next_state[128],
        1.0 / 9.0,
        atol=1e-6,
    )

    # Previous mask in next state must contain K=1 beam.
    assert np.isclose(
        next_state[64:128].sum(),
        1.0,
    )

    # --------------------------------------------------------
    # Test K=5 at horizon 1
    # --------------------------------------------------------

    next_state, reward, done, info = env.step(
        4
    )

    print()
    print("ACTION 4 -> K=5")
    print("K =", info["k"])
    print(
        "selected =",
        info["selected_beams"]
    )
    print(
        "covered mass =",
        info["covered_mass"]
    )
    print(
        "switching cost =",
        info["switching_cost"]
    )
    print(
        "outage =",
        info["outage"]
    )
    print(
        "next horizon =",
        next_state[128]
    )

    assert info["k"] == 5

    assert len(
        info["selected_beams"]
    ) == 5

    assert done is False

    assert np.isclose(
        next_state[128],
        2.0 / 9.0,
        atol=1e-6,
    )

    assert np.isclose(
        next_state[64:128].sum(),
        5.0,
    )

    # --------------------------------------------------------
    # Separate reset: fallback K=64
    # --------------------------------------------------------

    state = env.reset(
        tid
    )

    next_state, reward, done, info = env.step(
        5
    )

    print()
    print("ACTION 5 -> FALLBACK")
    print("K =", info["k"])
    print(
        "selected count =",
        len(
            info["selected_beams"]
        )
    )
    print(
        "covered mass =",
        info["covered_mass"]
    )
    print(
        "outage =",
        info["outage"]
    )
    print(
        "beam cost =",
        info["beam_cost"]
    )

    assert info["k"] == 64

    assert len(
        info["selected_beams"]
    ) == 64

    assert np.isclose(
        info["covered_mass"],
        1.0,
        atol=1e-5,
    )

    assert info["outage"] is False

    assert np.isclose(
        info["beam_cost"],
        1.0,
    )

    assert np.isclose(
        next_state[64:128].sum(),
        64.0,
    )

    # --------------------------------------------------------
    # Full episode and terminal transition
    # --------------------------------------------------------

    state = env.reset(
        tid
    )

    visited_horizons = []

    for step in range(10):

        visited_horizons.append(
            float(
                state[128]
            )
        )

        # K=1 for this deterministic test.
        state, reward, done, info = env.step(
            0
        )

        if step < 9:
            assert done is False

        else:
            assert done is True

            assert np.allclose(
                state,
                0.0,
            )

    expected = [
        i / 9.0
        for i in range(10)
    ]

    assert np.allclose(
        visited_horizons,
        expected,
        atol=1e-6,
    )

    print()
    print("visited normalized horizons =")
    print(
        [
            round(x, 6)
            for x in visited_horizons
        ]
    )

    print()
    print("=" * 72)
    print("ALL SMOKE TESTS PASSED")
    print("=" * 72)


if __name__ == "__main__":
    main()
