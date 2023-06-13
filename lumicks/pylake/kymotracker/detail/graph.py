import numpy as np
from scipy.optimize import linear_sum_assignment
from itertools import chain
from dataclasses import dataclass, field
from typing import List


@dataclass
class Vertex:
    frame: int
    position: float
    _parent: "Vertex" = field(default=None, repr=False)
    _child: "Vertex" = field(default=None, repr=False)

    @property
    def coordinate(self):
        return (self.frame, self.position)

    @property
    def parent(self):
        return self._parent

    @parent.setter
    def parent(self, vertex):
        self._parent = vertex

    @property
    def child(self):
        return self._child

    @child.setter
    def child(self, vertex):
        if self.child:
            self.child.parent = None
            self.child.child = None

        self._child = vertex
        if vertex:
            self._child.parent = self

    @property
    def is_head(self):
        return self.parent is None

    @property
    def is_terminal(self):
        return self.child is None

    def walk(self, track=None):
        if track is None:
            track = []

        track.append(self)
        if self.child:
            self.child.walk(track)
        return track


@dataclass
class DiGraph:
    frames: List[Vertex] = field(default_factory=list)

    def __getitem__(self, index):
        return self.frames[index]

    def __len__(self):
        return len(self.frames)

    def add_frame(self, frame, positions):
        self.frames.append([Vertex(frame, p) for p in positions])

    def get_tracks(self):
        tracks = [
            [vertex.walk() for vertex in frame if vertex.is_head] for frame in self
        ]
        return tuple(chain(*tracks))


def calculate_edge_cost(v_prev, v_current):
    gap_length = v_current.frame - v_prev.frame
    cost_diffusion = np.abs(v_current.position - v_prev.position) + (
        (gap_length - 1) * 0.3
    )
    return cost_diffusion


def calculate_cost_matrix(previous_frames, current_frame):
    cost = [
        calculate_edge_cost(v_prev, v_current)
        for v_prev in previous_frames
        for v_current in current_frame
    ]
    return np.reshape(cost, (len(previous_frames), len(current_frame)))


def track_multiframe(frame_positions):
    d = DiGraph()
    for frame, positions in frame_positions:
        d.add_frame(frame, positions)

    # establish initial correspondence with bipartite graph matching
    # cost matrix with previous frames as rows -> current frame as columns
    previous_frame, current_frame = d[0], d[1]
    cost = calculate_cost_matrix(previous_frame, current_frame)

    start_indices, connect_indices = linear_sum_assignment(cost)
    for start_index, connect_index in zip(start_indices, connect_indices):
        previous_frame[start_index].child = current_frame[connect_index]

    # loop through frames, forming extension digraphs
    # TODO: add variable window size, currently hardcoded to 3
    for current_frame_index in range(2, len(d)):
        previous_frames = tuple(
            chain(*[d[j] for j in (current_frame_index - 2, current_frame_index - 1)])
        )
        current_frame = d[current_frame_index]
        cost = calculate_cost_matrix(previous_frames, current_frame)

        start_indices, connect_indices = linear_sum_assignment(cost)
        for start_index, connect_index in zip(start_indices, connect_indices):
            previous_frames[start_index].child = current_frame[connect_index]

        # false hypothesis replacement
        for fhr_prev_idx in range(current_frame_index):
            previous_frame = [v for v in d[fhr_prev_idx] if v.child is None]
            current_frame = [v for v in d[fhr_prev_idx + 1] if v.parent is None]
            cost = calculate_cost_matrix(previous_frame, current_frame)
            start_indices, connect_indices = linear_sum_assignment(cost)
            for start_index, connect_index in zip(start_indices, connect_indices):
                previous_frame[start_index].child = current_frame[connect_index]

    return d
