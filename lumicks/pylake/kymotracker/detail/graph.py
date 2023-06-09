import numpy as np
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
class Digraph:
    frames: List[Vertex] = field(default_factory=list)

    def __getitem__(self, index):
        return self.frames[index]

    def __len__(self):
        return len(self.frames)

    def add_frame(self, frame, positions):
        self.frames.append([Vertex(frame, p) for p in positions])

    def get_tracks(self):
        tracks = [[vertex.walk() for vertex in frame if vertex.is_head] for frame in self]
        return tuple(chain(*tracks))
