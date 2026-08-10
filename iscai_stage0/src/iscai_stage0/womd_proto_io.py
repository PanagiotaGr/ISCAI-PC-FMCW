from __future__ import annotations

import struct
from dataclasses import dataclass
from pathlib import Path

from waymo_open_dataset.protos import scenario_pb2
from collections.abc import Iterator


def iter_tfrecord_payloads(path: Path) -> Iterator[bytes]:
    """Yield raw payloads from a TFRecord file without TensorFlow."""
    if not path.is_file():
        raise FileNotFoundError(f"TFRecord not found: {path}")

    with path.open("rb") as f:
        while True:
            length_bytes = f.read(8)

            if not length_bytes:
                return

            if len(length_bytes) != 8:
                raise RuntimeError("Invalid TFRecord length header.")

            record_length = struct.unpack("<Q", length_bytes)[0]

            if len(f.read(4)) != 4:
                raise RuntimeError("Missing TFRecord length CRC.")

            payload = f.read(record_length)
            if len(payload) != record_length:
                raise RuntimeError("Incomplete TFRecord payload.")

            if len(f.read(4)) != 4:
                raise RuntimeError("Missing TFRecord data CRC.")

            yield payload


def iter_scenarios(
    path: Path,
    limit: int | None = None,
) -> Iterator[scenario_pb2.Scenario]:
    """Yield parsed WOMD Scenario messages."""
    for index, payload in enumerate(iter_tfrecord_payloads(path)):
        if limit is not None and index >= limit:
            return

        scenario = scenario_pb2.Scenario()
        scenario.ParseFromString(payload)
        yield scenario

def read_first_scenario(path: Path) -> scenario_pb2.Scenario:
    if not path.is_file():
        raise FileNotFoundError(f"TFRecord not found: {path}")

    with path.open("rb") as f:
        length_bytes = f.read(8)
        if len(length_bytes) != 8:
            raise RuntimeError("Invalid TFRecord length header.")

        record_length = struct.unpack("<Q", length_bytes)[0]

        if len(f.read(4)) != 4:
            raise RuntimeError("Missing TFRecord length CRC.")

        payload = f.read(record_length)
        if len(payload) != record_length:
            raise RuntimeError("Incomplete TFRecord payload.")

        if len(f.read(4)) != 4:
            raise RuntimeError("Missing TFRecord data CRC.")

    scenario = scenario_pb2.Scenario()
    scenario.ParseFromString(payload)
    return scenario


@dataclass(frozen=True)
class CausalScenarioView:
    scenario: scenario_pb2.Scenario

    @property
    def anchor_index(self) -> int:
        return self.scenario.current_time_index

    @property
    def timestamps(self) -> tuple[float, ...]:
        return tuple(
            self.scenario.timestamps_seconds[: self.anchor_index + 1]
        )

    def actor_states(
        self,
        track_index: int,
    ) -> tuple[scenario_pb2.ObjectState, ...]:
        if not 0 <= track_index < len(self.scenario.tracks):
            raise IndexError(f"Invalid track index: {track_index}")

        return tuple(
            self.scenario.tracks[track_index].states[
                : self.anchor_index + 1
            ]
        )

    def state_at(
        self,
        track_index: int,
        time_index: int,
    ) -> scenario_pb2.ObjectState:
        if time_index > self.anchor_index:
            raise IndexError(
                f"Future access forbidden: time_index={time_index}, "
                f"anchor_index={self.anchor_index}"
            )

        if time_index < 0:
            raise IndexError("Negative time index is not allowed.")

        states = self.actor_states(track_index)

        if time_index >= len(states):
            raise IndexError(f"Invalid time index: {time_index}")

        return states[time_index]
