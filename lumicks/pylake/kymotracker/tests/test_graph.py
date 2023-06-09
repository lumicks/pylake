import numpy as np
from lumicks.pylake.kymotracker.detail.graph import Vertex, Digraph


def test_vertex():
    a = Vertex(0, 1.0)
    b = Vertex(1, 1.0)
    c = Vertex(2, 1.0)
    d = Vertex(3, 1.0)
    e = Vertex(3, 2.0)

    def validate_connections(vertices, links):
        for vertex, (parent, child) in zip(vertices, links):
            assert vertex.parent is parent
            assert vertex.child is child

    def validate_tracks(vertices, ref_tracks):
        tracks = [vertex.walk() for vertex in vertices if vertex.is_head]
        for track, ref_track in zip(tracks, ref_tracks):
            assert track == ref_track

    # a b c d e
    assert a.frame == 0
    assert a.position == 1.0
    assert a.is_head == True
    assert a.is_terminal == True
    assert b.is_head == True
    assert b.is_terminal == True
    validate_connections(
        (a, b, c, d, e), [(None, None), (None, None), (None, None), (None, None), (None, None)]
    )
    validate_tracks((a, b, c, d, e), ([a], [b], [c], [d], [e]))

    # a - b
    # c d e
    a.child = b
    validate_connections(
        (a, b, c, d, e), [(None, b), (a, None), (None, None), (None, None), (None, None)]
    )
    validate_tracks((a, b, c, d, e), ([a, b], [c], [d], [e]))
    assert a.is_head == True
    assert a.is_terminal == False
    assert b.is_head == False
    assert b.is_terminal == True

    # a - b - c
    # d e
    b.child = c
    validate_connections(
        (a, b, c, d, e), [(None, b), (a, c), (b, None), (None, None), (None, None)]
    )
    validate_tracks((a, b, c, d, e), ([a, b, c], [d], [e]))
    assert b.is_head == False
    assert b.is_terminal == False

    # a - b - c - d
    # e
    c.child = d
    validate_connections((a, b, c, d, e), [(None, b), (a, c), (b, d), (c, None), (None, None)])
    validate_tracks((a, b, c, d, e), ([a, b, c, d], [e]))

    # a - b - e
    # c d
    b.child = e
    validate_connections(
        (a, b, c, d, e), [(None, b), (a, e), (None, None), (None, None), (b, None)]
    )
    validate_tracks((a, b, c, d, e), ([a, b, e], [c], [d]))

    # a b c d e
    a.child = None
    validate_connections(
        (a, b, c, d, e), [(None, None), (None, None), (None, None), (None, None), (None, None)]
    )
    validate_tracks((a, b, c, d, e), ([a], [b], [c], [d], [e]))


def test_digraph():
    d = Digraph()
    d.add_frame(0, [1.0])  # a
    d.add_frame(1, [1.0])  # b
    d.add_frame(2, [1.0])  # c
    d.add_frame(3, [1.0, 2.0])  # d, e

    assert len(d) == 4

    frame = d[-1]
    assert len(frame) == 2
    assert frame[0].frame == 3
    assert frame[0].position == 1.0
    assert frame[1].position == 2.0

    def get_track_coordinates(tracks):
        return [np.vstack([v.coordinate for v in track]) for track in tracks]

    track_coords = get_track_coordinates(d.get_tracks())
    assert len(track_coords) == 5
    np.testing.assert_allclose(track_coords[0], [[0, 1.0]])
    np.testing.assert_allclose(track_coords[1], [[1, 1.0]])
    np.testing.assert_allclose(track_coords[2], [[2, 1.0]])
    np.testing.assert_allclose(track_coords[3], [[3, 1.0]])
    np.testing.assert_allclose(track_coords[4], [[3, 2.0]])

    # a - b
    # c d e
    d[0][0].child = d[1][0]
    track_coords = get_track_coordinates(d.get_tracks())
    assert len(track_coords) == 4
    np.testing.assert_allclose(track_coords[0], [[0, 1.0], [1, 1.0]])
    np.testing.assert_allclose(track_coords[1], [[2, 1.0]])
    np.testing.assert_allclose(track_coords[2], [[3, 1.0]])
    np.testing.assert_allclose(track_coords[3], [[3, 2.0]])

    # a - b - c
    # d e
    d[1][0].child = d[2][0]
    track_coords = get_track_coordinates(d.get_tracks())
    assert len(track_coords) == 3
    np.testing.assert_allclose(track_coords[0], [[0, 1.0], [1, 1.0], [2, 1.0]])
    np.testing.assert_allclose(track_coords[1], [[3, 1.0]])
    np.testing.assert_allclose(track_coords[2], [[3, 2.0]])

    # a - b - c - d
    # e
    d[2][0].child = d[3][0]
    track_coords = get_track_coordinates(d.get_tracks())
    assert len(track_coords) == 2
    np.testing.assert_allclose(track_coords[0], [[0, 1.0], [1, 1.0], [2, 1.0], [3, 1.0]])
    np.testing.assert_allclose(track_coords[1], [[3, 2.0]])

    # a - b - e
    # c d
    d[1][0].child = d[3][1]
    track_coords = get_track_coordinates(d.get_tracks())
    assert len(track_coords) == 3
    np.testing.assert_allclose(track_coords[0], [[0, 1.0], [1, 1.0], [3, 2.0]])
    np.testing.assert_allclose(track_coords[1], [[2, 1.0]])
    np.testing.assert_allclose(track_coords[2], [[3, 1.0]])
