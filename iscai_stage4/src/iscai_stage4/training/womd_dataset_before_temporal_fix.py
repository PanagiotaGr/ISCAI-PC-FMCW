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


    def _load(self):

        for path in self.files:

            with path.open("rb") as f:

                while True:

                    header = f.read(12)

                    if not header:
                        break


                    length = struct.unpack(
                        "<Q",
                        header[:8]
                    )[0]


                    payload = f.read(length)

                    f.read(4)


                    scenario = scenario_pb2.Scenario()

                    scenario.ParseFromString(
                        payload
                    )


                    for track in scenario.tracks:

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
                            continue

                        track_type = (
                            enum_value.name
                            .replace("TYPE_", "")
                            .upper()
                        )

                        if (
                            self.actor_type is not None
                            and track_type != self.actor_type
                        ):
                            continue


                        states = [
                            s
                            for s in track.states
                            if s.valid
                        ]


                        if len(states) < (
                            self.history+self.future
                        ):
                            continue


                        past = states[:self.history]

                        future = states[
                            self.history:
                            self.history+self.future
                        ]


                        anchor_x = past[-1].center_x
                        anchor_y = past[-1].center_y


                        x = []

                        for s in past:

                            heading = s.heading


                            x.append(
                                [
                                    s.center_x-anchor_x,
                                    s.center_y-anchor_y,
                                    s.velocity_x,
                                    s.velocity_y,
                                    math.sin(heading),
                                    math.cos(heading),
                                    s.length,
                                    s.width
                                ]
                            )



                        y = []

                        for s in future:

                            y.append(
                                [
                                    s.center_x-anchor_x,
                                    s.center_y-anchor_y
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
                                )
                            )
                        )


    def __len__(self):

        return len(self.samples)



    def __getitem__(self,idx):

        return self.samples[idx]
