from copy import copy


class UndoStack:
    """Keeps a state with an undo and redo list."""

    def __init__(self, initial_state):
        self._state = initial_state
        self._redo_history = []
        self._undo_history = []

    @property
    def state(self):
        return self._state

    @state.setter
    def state(self, new_state):
        self._undo_history.append(self.state)
        self._redo_history = []
        self._state = copy(new_state)

    def undo(self):
        if len(self._undo_history) > 0:
            self._redo_history.append(self.state)
            self._state = self._undo_history.pop()
        return self.state

    def redo(self):
        if len(self._redo_history) > 0:
            self._undo_history.append(self.state)
            self._state = self._redo_history.pop()
        return self.state
