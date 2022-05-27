"""
Base classes for the options framework
"""
from lib2to3.pgen2.token import OP
from barl_simpleoptions import Option
from barl_simpleoptions import State

class FourRoomSubgoalOption(Option):

    def __init__(self) -> None:
        super().__init__()

    def initiation(self, state : State):
        return state.is_action_legal(self.action)

    def policy(self, state : State):
        return self.action

    def termination(self, state : State):
        return True

    def __str__(self):
        return "SubgoalOption({})".format(str(self.subgoal))

    def __repr__(self):
        return str(self)


class EmbodiedAbstractOption(Option):

    def __init__(self) -> None:
        super().__init__()

    def initiation(self, state):
        return state.is_action_legal(self.action)

    def policy(self, state):
        return self.action

    def termination(self, state):
        return True

    def __str__(self):
        return "EmbodiedAbstractOption({})".format(self.action)

    def __repr__(self):
        return str(self)
