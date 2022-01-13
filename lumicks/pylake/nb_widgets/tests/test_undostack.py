from lumicks.pylake.nb_widgets.detail.undostack import UndoStack


def is_equal(test_list, ref):
    return all(x == y for x, y in zip(test_list, ref))


def test_empty_undo_redo():
    """Test whether empty undo/redo stacks don't change history"""
    state_item1 = [5, 3, 8, 10]
    undostack = UndoStack(state_item1)

    undostack.undo()
    assert is_equal(undostack.state, state_item1)

    undostack.redo()
    assert is_equal(undostack.state, state_item1)


def test_undo_redo():
    state_item1 = [5, 3, 8, 10]
    state_item2 = [2, 1, 8, 10]
    state_item3 = [5, 3, 9, 10]

    undostack = UndoStack(state_item1)
    assert is_equal(undostack.state, state_item1)

    undostack.state = state_item2
    assert is_equal(undostack.state, state_item2)

    undostack.state = state_item3
    assert is_equal(undostack.state, state_item3)

    undostack.undo()
    assert is_equal(undostack.state, state_item2)

    undostack.undo()
    assert is_equal(undostack.state, state_item1)

    # Stack is empty now, not undoing anything anymore
    undostack.undo()
    assert is_equal(undostack.state, state_item1)

    undostack.redo()
    assert is_equal(undostack.state, state_item2)

    undostack.redo()
    assert is_equal(undostack.state, state_item3)

    undostack.redo()
    assert is_equal(undostack.state, state_item3)

    undostack.undo()
    assert is_equal(undostack.state, state_item2)


def test_removal_redo():
    state_item1 = [1, 3, 8, 10]
    state_item2 = [2, 3, 8, 10]
    state_item3 = [3, 3, 3, 10]
    state_item4 = [4, 3, 8, 10]

    undostack = UndoStack(state_item1)
    undostack.state = state_item2
    undostack.state = state_item3
    undostack.undo()
    undostack.undo()

    # Item 2 and 3 are in the redo stack, adding a new item will remove them
    undostack.state = state_item4
    assert is_equal(undostack.state, state_item4)

    undostack.redo()
    assert is_equal(undostack.state, state_item4)

    undostack.undo()
    assert is_equal(undostack.state, state_item1)

    undostack.redo()
    assert is_equal(undostack.state, state_item4)
