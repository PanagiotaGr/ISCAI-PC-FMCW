from pathlib import Path
import struct
import math

import torch
from torch.utils.data import Dataset

from waymo_open_dataset.protos import scenario_pb2

from iscai_stage1.contracts.stage1a import (
    HeadlampSurrogateConfig
)

from iscai_stage1.geometry.frames import (
    SdcStateW,
    build_anchor_frames
)



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

        if actor_type is None:
            self.actor_types = None
        elif isinstance(actor_type, (list, tuple, set)):
            self.actor_types = {
                str(x).upper()
                for x in actor_type
            }
        else:
            self.actor_types = {
                str(actor_type).upper()
            }

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
                            self.actor_types is not None
                            and
                            track_type not in self.actor_types
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


                        # Build Stage1 E0 ego frame
                        # using the causal SDC state
                        # at WOMD current_time_index.

                        sdc_track = scenario.tracks[
                            scenario.sdc_track_index
                        ]

                        sdc = sdc_track.states[
                            scenario.current_time_index
                        ]

                        ego_state = SdcStateW(
                            center_w_m=(
                                sdc.center_x,
                                sdc.center_y,
                                0.0,
                            ),
                            heading_rad=sdc.heading,
                            length_m=sdc.length,
                        )

                        frames = build_anchor_frames(
                            ego_state,
                            HeadlampSurrogateConfig()
                        )


                        x = []

                        for s in past:

                            heading = s.heading

                            point_E = (
                                frames.T_E0_from_W.apply_point(
                                    (
                                        s.center_x,
                                        s.center_y,
                                        0.0,
                                    )
                                )
                            )

                            x.append(
                                [
                                    point_E[0],
                                    point_E[1],

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

                            point_E = (
                                frames.T_E0_from_W.apply_point(
                                    (
                                        s.center_x,
                                        s.center_y,
                                        0.0,
                                    )
                                )
                            )

                            y.append(
                                [
                                    point_E[0],
                                    point_E[1],
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
