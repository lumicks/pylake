import numpy as np
from scipy.optimize import linear_sum_assignment
from itertools import chain
from dataclasses import dataclass, field
from typing import List
from lumicks.pylake.kymotracker.kymotrack import KymoTrack, KymoTrackGroup


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


class SinkVertex(Vertex):
        pass


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
        tracks_vertices = [[vertex.walk() for vertex in frame if vertex.is_head] for frame in self]
        return tuple(chain(*tracks_vertices))

    def get_kymotrackgroup(self, kymo, channel):
        return KymoTrackGroup(
            [
                KymoTrack([int(v.frame) for v in t], [v.position for v in t], kymo, channel)
                for t in self.get_tracks()
            ]
        )


class CostMatrix:
    class SinkVertex(Vertex):
        pass

    def __init__(self, previous_vertices, current_vertices, search_penalty=0.3):
        current_frame_index = current_vertices[0].frame
        first_frame_index = current_frame_index - 2

        self.vertices = {
            "previous": previous_vertices,
            "current": current_vertices,
            "sink": [
                CostMatrix.SinkVertex(
                    current_frame_index,
                    v.position + ((current_frame_index - v.frame) * search_penalty),
                )
                for v in previous_vertices
            ],
            "existing": [v for v in previous_vertices if v.frame != first_frame_index],
        }

        self.start_map = {id(v): (j, v) for j, v in enumerate(previous_vertices)}
        self.end_map = {
            id(v): (j, v)
            for j, v in enumerate(
                current_vertices + self.vertices["existing"] + self.vertices["sink"]
            )
        }
        self.matrix = np.full((len(self.start_map), len(self.end_map)), np.inf)

    def set(self, row_vertex, col_vertex, cost):
        row_idx, _ = self.start_map[id(row_vertex)]
        col_idx, _ = self.end_map[id(col_vertex)]
        self.matrix[row_idx, col_idx] = cost

    def get(self, row, col):
        start_vertex = next(v for idx, v in self.start_map.values() if idx == row)
        end_vertex = next(v for idx, v in self.end_map.values() if idx == col)
        return start_vertex, None if isinstance(end_vertex, CostMatrix.SinkVertex) else end_vertex

    def calculate(self):
        for j, start_vertex in enumerate(self.vertices["previous"]):
            # * extension edges + replacement edges
            for end_vertex in self.vertices["current"]:
                cost = calculate_edge_cost(start_vertex, end_vertex)
                self.set(start_vertex, end_vertex, cost)

            # * sink edges
            end_vertex = self.vertices["sink"][j]
            cost = calculate_edge_cost(start_vertex, end_vertex)
            self.set(start_vertex, end_vertex, cost)

        # * existing edges
        for end_vertex in self.vertices["existing"]:
            # what if child is out of serach window (due to gap)?
            if (start_vertex := end_vertex.parent) and (id(start_vertex) in self.start_map.keys()):
                cost = calculate_edge_cost(start_vertex, end_vertex)
                self.set(start_vertex, end_vertex, cost)


def calculate_edge_cost(v_prev, v_current):
    """
    Adapted cost functions from Chenouard, N. "Objective comparison of particle tracking methods"
    SI. Unclear if position vectors also include temporal coordinate or just spatial (here I've
    included the temporal coordinate as results seemed better, still needs investigating).

    Original from Mubarak, S. "A non-iterative greedy algorithm for multi-frame point
    correspondence" utilizes a similar *gain* function. Need to assess if inequality in Eq. 1 is
    satisfied (ie, "...penalizes the choice of a shorter track when a longer valid track is present").
    """

    gap_length = v_current.frame - v_prev.frame
    cost_diffusion = np.abs(v_current.position - v_prev.position) + ((gap_length - 1) * 0.3)
    # starting vertex, no previous motion information
    # fall back to pure diffusion
    if v_prev.parent is None:
        return cost_diffusion

    slope = (v_prev.position - v_prev.parent.position) / (v_prev.frame - v_prev.parent.frame)
    p_next_expected = v_prev.position + gap_length * slope
    cost_directed = v_current.position - p_next_expected

    # mu_1 = np.exp(-np.abs(v_prev.position - v_prev.parent.position))
    # mu_2 = np.exp(-np.abs(v_prev.position - v_prev.parent.position - 4) ** 2 / 2)
    mu_1 = np.exp(-np.abs(slope))
    mu_2 = np.exp(-np.abs(slope - 4) ** 2 / 2)
    return (mu_1 * cost_diffusion + mu_2 * cost_directed) / (mu_1 + mu_2)


def calculate_cost_matrix(previous_vertices, current_vertices, search_penalty=0.3):
    current_frame_index = current_vertices[0].frame
    first_frame_index = current_frame_index - 2
    sink_vertices = [
        SinkVertex(
            current_frame_index,
            v.position + ((current_frame_index - v.frame) * search_penalty),
        )
        for v in previous_vertices
    ]
    existing_vertices = [v for v in previous_vertices if v.frame != first_frame_index]

    extensions = [[(start, end) for end in current_vertices] for start in previous_vertices]
    # what if child is out of serach window (due to gap)?
    existing = [
        [(start, end) if (start == end.parent) else (None, None) for end in existing_vertices]
        for start in previous_vertices
    ]
    sinks = [
        [(start, end) if j == k else (None, None) for k, end in enumerate(sink_vertices)]
        for j, start in enumerate(previous_vertices)
    ]
    v_matrix = [x + e + s for x, e, s in zip(extensions, existing, sinks)]

    matrix = [
        [calculate_edge_cost(start, end) if start is not None else np.inf for start, end in row]
        for row in v_matrix
    ]
    return np.array(matrix), v_matrix


def track_multiframe(frame_positions, window=3, search_penalty=0.3, inspect_callback=None):
    d = DiGraph()
    for frame, positions in frame_positions:
        d.add_frame(frame, positions)

    # establish initial correspondence with bipartite graph matching
    # cost matrix with previous frames as rows -> current frame as columns
    previous_frame, current_frame = d[0], d[1]
    cmat = CostMatrix(previous_frame, current_frame, search_penalty)
    cmat.calculate()
    rows, cols = linear_sum_assignment(cmat.matrix)
    for r, c in zip(rows, cols):
        start, end = cmat.get(r, c)
        if end is None:
            continue
        start.child = end

    # loop through frames, forming extension digraphs
    for current_frame_index in range(2, len(d)):
        previous_frames = tuple(chain(*[d[j] for j in current_frame_index - np.arange(1, window)]))
        current_frame = d[current_frame_index]

        # cmat = CostMatrix(previous_frames, current_frame, search_penalty)
        # cmat.calculate()
        # rows, cols = linear_sum_assignment(cmat.matrix)
        # for r, c in zip(rows, cols):
        #     start, end = cmat.get(r, c)
        #     if end is None:
        #         continue
        #     start.child = end

        matrix, v_matrix = calculate_cost_matrix(previous_frames, current_frame, search_penalty)
        rows, cols = linear_sum_assignment(matrix)
        for r, c in zip(rows, cols):
            start, end = v_matrix[r][c]
            if isinstance(end, SinkVertex):
                continue
            start.child = end

        # # false hypothesis replacement
        # for fhr_prev_idx in range(current_frame_index):
        #     previous_frame = [v for v in d[fhr_prev_idx] if v.child is None]
        #     current_frame = [v for v in d[fhr_prev_idx + 1] if v.parent is None]
        #     cost = calculate_cost_matrix(previous_frame, current_frame)
        #     start_indices, connect_indices = linear_sum_assignment(cost)
        #     print("false hyp repr")
        #     print(cost)
        #     print(start_indices)
        #     print(connect_indices)
        #     print("=" * 50)
        #     for start_index, connect_index in zip(start_indices, connect_indices):
        #         previous_frame[start_index].child = current_frame[connect_index]

        # if inspect_callback:
        #     inspect_callback(d.get_kymotrackgroup, current_frame_index)

        # TODO: add backtracking when current frame == window size

        # if current_frame_index > 3:
        #     break

    return d


if __name__ == "__main__":
    import numpy as np
    import matplotlib.pyplot as plt
    from lumicks.pylake.simulation import simulate_diffusive_tracks

    np.random.seed(198507102)

    # add some directional motion to diffusion
    vel = lambda n: np.polyval([5e-2, 0], np.arange(n))

    def make_tracks(
        num_tracks,
        num_frames,
        tether_length,
        start_position=None,
        start_times=None,
    ):
        if start_position is None:
            start_position = np.random.uniform(0, tether_length, size=num_tracks)
        if start_times is None:
            start_times = np.random.exponential(1 / 1.8, size=num_tracks)
            start_frames = np.array(start_times // 0.075).astype(int)
        start_frames = start_frames - np.min(start_frames)

        tracks = simulate_diffusive_tracks(
            diffusion_constant=0.001,
            steps=np.max(num_frames),
            dt=0.075,
            observation_noise=0.1,
            num_tracks=num_tracks,
        )

        tracks = KymoTrackGroup(
            [
                KymoTrack(
                    t.time_idx[:n] + start_frame, t.position[:n] + p + vel(n), t._kymo, t._channel
                )
                for t, p, start_frame, n in zip(tracks, start_position, start_frames, num_frames)
            ]
        )
        return tracks

    def extract_coordinates_from_tracks(tracks):
        table = np.hstack(
            [np.vstack((np.full(len(t), j), t._time_idx, t.position)) for j, t in enumerate(tracks)]
        )
        idx = np.argsort(table[1])
        table = table[:, idx]
        track_labels, frame_indices, positions = table
        frames = [(j, positions[frame_indices == j]) for j in np.unique(frame_indices)]
        return frames

    num_frames = [20, 10, 50, 5, 20]
    tracks = make_tracks(len(num_frames), num_frames, 15)
    frames = extract_coordinates_from_tracks(tracks)

    def inspect_callback(gfunc, title):
        new_tracks = gfunc(tracks[0]._kymo, tracks[0]._channel)
        tracks.plot(lw=7, marker=".", color="r", show_outline=False)
        new_tracks.plot(marker=".", color="lightskyblue")
        plt.title(title)
        plt.show()

    d = track_multiframe(frames, window=3, search_penalty=0.1, inspect_callback=inspect_callback)

    new_tracks = d.get_kymotrackgroup(tracks[0]._kymo, tracks[0]._channel)
    tracks.plot(lw=7, marker=".", color="r", show_outline=False)
    new_tracks.plot(marker=".", color="lightskyblue")
    plt.show()
