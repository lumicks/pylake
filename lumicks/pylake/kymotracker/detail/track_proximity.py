from dataclasses import dataclass

import numpy as np


@dataclass(slots=True)
class BoundingBox:
    """Small class to hold a bounding box for a track"""

    time_min: int
    time_max: int
    coord_min: float
    coord_max: float
    _valid: bool = True

    @staticmethod
    def from_track(track, time_slack=1, coord_slack=1.0):
        if len(track.time_idx) == 0:
            return BoundingBox(0, 0, 0, 0, False)

        return BoundingBox(
            np.min(track.time_idx) - time_slack,
            np.max(track.time_idx) + time_slack,
            np.min(track.coordinate_idx) - coord_slack,
            np.max(track.coordinate_idx) + coord_slack,
            True,
        )

    def overlaps_with(self, other):
        """Test whether this bounding box overlaps with another bounding box"""
        if not (self._valid and other._valid):
            return False

        if self.time_max < other.time_min:
            return False

        if other.time_max < self.time_min:
            return False

        if self.coord_max < other.coord_min:
            return False

        if other.coord_max < self.coord_min:
            return False

        return True


def tracks_close(track1, track2, position_window, time_window):
    """Check whether two tracks overlap (with some tolerance)

    Parameters
    ----------
    track1, track2 : lumicks.pylake.kymotracker.kymotrack.KymoTrack
        Kymotracks to compare
    position_window : float
        How much distance is there allowed to be between two detections (in pixels).
    time_window : int
        How much temporal distance is there allowed to be between two detections (in frame indices).

    Returns
    -------
    bool
        True when the tracks are close enough in time and space.
    """

    def dist(coord1, coord2):
        return np.abs(np.asarray(coord2) - np.asarray(coord1)[:, np.newaxis])

    coord_dists = dist(track1.coordinate_idx, track2.coordinate_idx)
    time_dists = dist(track1.time_idx, track2.time_idx)
    if np.any(np.logical_and(coord_dists <= position_window, time_dists <= time_window)):
        return True

    return False


def find_colocalized(
    tracks1,
    tracks2,
    position_window,
    time_window,
    *,
    interpolate=True,
):
    """Returns the list of tracks which have points in close proximity

    Parameters
    ----------
    tracks1, tracks2 : lumicks.pylake.kymotracker.KymoTrack.KymoTrackGroup
        Groups of tracks
    position_window : float
        Spatial distance in pixels above which track points are not considered close anymore.
    time_window : int
        Temporal distance in pixels above which track points are not considered close anymore.
    interpolate : bool
        Interpolate track before starting.

    Returns
    -------
    list of tuple
        Set of pairs of track indices in close enough proximity to be candidates for interaction.
        Note that a track can appear in multiple pairs.
    """
    track_groups = [tracks1, tracks2]

    bboxes = [
        [BoundingBox.from_track(track, time_window, position_window) for track in tracks]
        for tracks in track_groups
    ]

    if interpolate:
        track_groups = [[t.interpolate() for t in tracks] for tracks in track_groups]

    colocalizations = set()
    for idx1, (track1, bbox1) in enumerate(zip(track_groups[0], bboxes[0])):
        for idx2, (track2, bbox2) in enumerate(zip(track_groups[1], bboxes[1])):
            if bbox1.overlaps_with(bbox2):
                if tracks_close(track1, track2, position_window, time_window):
                    colocalizations.add((idx1, idx2))

    sorted_list = list(colocalizations)
    sorted_list.sort()

    return sorted_list
