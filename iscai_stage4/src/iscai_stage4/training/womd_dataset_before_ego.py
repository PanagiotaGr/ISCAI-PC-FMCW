from pathlib import Path
import struct
import math

import torch
from torch.utils.data import Dataset

from waymo_open_dataset.protos import scenario_pb2


class WOMDGRUDataset(Dataset):

    def __init__(
        self,
        root,
        max_files=200,
        history=10,
        future=10,
        actor_type=None,
    ):

        self.files = list(
            Path(root).glob("*tfrecord*")
        )[:max_files]

        self.history = history
        self.future = future

        self.actor_type = (
            actor_type.upper()
            if actor_type is not None
            else None
        )

        self.samples = []

        self._load()


    def _track_type_name(
        self,
        track,
    ):

        enum_desc = (
            track.DESCRIPTOR
            .fields_by_name["object_type"]
            .enum_type
        )

        enum_value = (
            enum_desc
            .values_by_number
            .get(track.object_type)
        )

        if enum_value is None:
            return None

        return (
            enum_value.name
            .replace("TYPE_", "")
            .upper()
        )


    def _load(self):

        for path in self.files:

            with path.open("rb") as f:

                while True:

                    header = f.read(12)

                    if not header:
                        break

                    if len(header) != 12:
                        raise RuntimeError(
                            f"Truncated TFRecord header: {path}"
                        )


                    length = struct.unpack(
                        "<Q",
                        header[:8]
                    )[0]


                    payload = f.read(length)

                    data_crc = f.read(4)


                    if (
                        len(payload) != length
                        or len(data_crc) != 4
                    ):
                        raise RuntimeError(
                            f"Truncated TFRecord record: {path}"
                        )


                    scenario = scenario_pb2.Scenario()

                    scenario.ParseFromString(
                        payload
                    )


                    current = (
                        scenario.current_time_index
                    )


                    history_start = (
                        current
                        -
                        self.history
                        +
                        1
                    )

                    future_start = (
                        current
                        +
                        1
                    )

                    future_end = (
                        future_start
                        +
                        self.future
                    )


                    if history_start < 0:
                        continue


                    for track in scenario.tracks:

                        track_type = (
                            self._track_type_name(
                                track
                            )
                        )


                        if track_type is None:
                            continue


                        if (
                            self.actor_type is not None
                            and
                            track_type != self.actor_type
                        ):
                            continue


                        if (
                            len(track.states)
                            <
                            future_end
                        ):
                            continue


                        past = list(
                            track.states[
                                history_start:
                                current + 1
                            ]
                        )


                        future = list(
                            track.states[
                                future_start:
                                future_end
                            ]
                        )


                        if (
                            len(past)
                            != self.history
                            or
                            len(future)
                            != self.future
                        ):
                            continue


                        # Do NOT compress invalid states.
                        # The real WOMD temporal indices
                        # must remain unchanged.
                        if not all(
                            s.valid
                            for s in past
                        ):
                            continue


                        if not all(
                            s.valid
                            for s in future
                        ):
                            continue


                        # Current/anchor state = WOMD
                        # current_time_index.
                        anchor = past[-1]

                        anchor_x = anchor.center_x
                        anchor_y = anchor.center_y


                        x = []

                        for s in past:

                            heading = s.heading

                            x.append(
                                [
                                    s.center_x
                                    - anchor_x,

                                    s.center_y
                                    - anchor_y,

                                    s.velocity_x,
                                    s.velocity_y,

                                    math.sin(
                                        heading
                                    ),

                                    math.cos(
                                        heading
                                    ),

                                    s.length,
                                    s.width,
                                ]
                            )


                        y = []

                        for s in future:

                            y.append(
                                [
                                    s.center_x
                                    - anchor_x,

                                    s.center_y
                                    - anchor_y,
                                ]
                            )


                        self.samples.append(
                            (
                                torch.tensor(
                                    x,
                                    dtype=torch.float32
                                ),

                                torch.tensor(
                                    y,
                                    dtype=torch.float32
                                ),
                            )
                        )


    def __len__(self):

        return len(
            self.samples
        )


    def __getitem__(
        self,
        idx,
    ):

        return self.samples[idx]
