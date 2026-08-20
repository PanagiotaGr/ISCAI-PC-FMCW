from pathlib import Path

import numpy as np
import torch
import torch.nn as nn


class QNetwork(nn.Module):

    def __init__(
        self,
        state_dim,
        num_actions,
    ):
        super().__init__()

        self.net = nn.Sequential(
            nn.Linear(
                state_dim,
                128,
            ),
            nn.ReLU(),

            nn.Linear(
                128,
                128,
            ),
            nn.ReLU(),

            nn.Linear(
                128,
                num_actions,
            ),
        )

    def forward(
        self,
        x,
    ):
        return self.net(x)


class ReplayBuffer:

    def __init__(
        self,
        capacity,
        state_dim,
        seed=42,
    ):
        self.capacity = int(
            capacity
        )

        self.state_dim = int(
            state_dim
        )

        self.states = np.zeros(
            (
                self.capacity,
                self.state_dim,
            ),
            dtype=np.float32,
        )

        self.next_states = np.zeros_like(
            self.states
        )

        self.actions = np.zeros(
            self.capacity,
            dtype=np.int64,
        )

        self.rewards = np.zeros(
            self.capacity,
            dtype=np.float32,
        )

        self.dones = np.zeros(
            self.capacity,
            dtype=np.float32,
        )

        self.position = 0
        self.size = 0

        self.rng = np.random.default_rng(
            seed
        )

    def add(
        self,
        state,
        action,
        reward,
        next_state,
        done,
    ):
        i = self.position

        self.states[i] = state
        self.actions[i] = action
        self.rewards[i] = reward
        self.next_states[i] = (
            next_state
        )
        self.dones[i] = float(
            done
        )

        self.position = (
            self.position + 1
        ) % self.capacity

        self.size = min(
            self.size + 1,
            self.capacity,
        )

    def sample(
        self,
        batch_size,
    ):
        idx = self.rng.integers(
            0,
            self.size,
            size=batch_size,
        )

        return (
            self.states[idx],
            self.actions[idx],
            self.rewards[idx],
            self.next_states[idx],
            self.dones[idx],
        )


class DQNBeamAgent:

    def __init__(
        self,
        state_dim,
        num_actions,
        device,
        lr=1e-3,
        gamma=0.95,
        seed=42,
    ):
        torch.manual_seed(
            seed
        )

        self.device = torch.device(
            device
        )

        self.gamma = float(
            gamma
        )

        self.num_actions = int(
            num_actions
        )

        self.online = QNetwork(
            state_dim,
            num_actions,
        ).to(
            self.device
        )

        self.target = QNetwork(
            state_dim,
            num_actions,
        ).to(
            self.device
        )

        self.target.load_state_dict(
            self.online.state_dict()
        )

        self.target.eval()

        self.optimizer = (
            torch.optim.Adam(
                self.online.parameters(),
                lr=lr,
            )
        )

        self.loss_fn = (
            nn.SmoothL1Loss()
        )

        self.rng = np.random.default_rng(
            seed
        )

    def select_action(
        self,
        state,
        epsilon=0.0,
    ):
        if (
            self.rng.random()
            <
            epsilon
        ):
            return int(
                self.rng.integers(
                    0,
                    self.num_actions,
                )
            )

        state_t = torch.as_tensor(
            state,
            dtype=torch.float32,
            device=self.device,
        ).unsqueeze(0)

        with torch.no_grad():
            q = self.online(
                state_t
            )

        return int(
            torch.argmax(
                q,
                dim=1,
            ).item()
        )

    def train_step(
        self,
        replay,
        batch_size=256,
    ):
        (
            states,
            actions,
            rewards,
            next_states,
            dones,
        ) = replay.sample(
            batch_size
        )

        states = torch.as_tensor(
            states,
            device=self.device,
        )

        next_states = torch.as_tensor(
            next_states,
            device=self.device,
        )

        actions = torch.as_tensor(
            actions,
            dtype=torch.long,
            device=self.device,
        )

        rewards = torch.as_tensor(
            rewards,
            device=self.device,
        )

        dones = torch.as_tensor(
            dones,
            device=self.device,
        )

        q = self.online(
            states
        ).gather(
            1,
            actions.unsqueeze(1),
        ).squeeze(1)

        with torch.no_grad():
            next_q = self.target(
                next_states
            ).max(
                dim=1
            ).values

            target = (
                rewards
                +
                self.gamma
                *
                (1.0 - dones)
                *
                next_q
            )

        loss = self.loss_fn(
            q,
            target,
        )

        self.optimizer.zero_grad()

        loss.backward()

        torch.nn.utils.clip_grad_norm_(
            self.online.parameters(),
            5.0,
        )

        self.optimizer.step()

        return float(
            loss.item()
        )

    def update_target(self):
        self.target.load_state_dict(
            self.online.state_dict()
        )

    def save(
        self,
        path,
    ):
        path = Path(path)

        path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        torch.save(
            {
                "online":
                    self.online.state_dict(),

                "target":
                    self.target.state_dict(),

                "optimizer":
                    self.optimizer.state_dict(),

                "gamma":
                    self.gamma,

                "num_actions":
                    self.num_actions,
            },
            path,
        )
