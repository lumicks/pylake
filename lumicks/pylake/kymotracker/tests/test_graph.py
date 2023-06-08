from lumicks.pylake.kymotracker.detail.graph import Vertex


def test_vertex():
    a = Vertex("a", 0, 1.0)
    b = Vertex("b", 1, 1.0)
    c = Vertex("c", 2, 1.0)
    d = Vertex("d", 3, 1.0)
    e = Vertex("e", 3, 2.0)

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
