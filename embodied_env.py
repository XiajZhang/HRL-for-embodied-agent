"""
Defines state, transition function with student simulator, reward function in envs
"""
from enum import Enum
from aenum import MultiValueEnum
from re import L
from barl_simpleoptions import State
from typing import List, Tuple
import numpy as np

# Define action space
class KeywordKnowledge(Enum):
    Known = 0
    Unknown = 1

class InteractionBehavior(Enum):
    Exploratory = 0
    Querying = 1

EmbodiedActionSpace = [
    (KeywordKnowledge.Known,   InteractionBehavior.Exploratory),
    (KeywordKnowledge.Unknown, InteractionBehavior.Exploratory),
    (KeywordKnowledge.Known,   InteractionBehavior.Querying),
    (KeywordKnowledge.Unknown, InteractionBehavior.Querying)
]

# Define State Space
class StudentEngagementLevel(Enum):
    Low = 0
    Medium = 1
    High = 2

class StudentExploratoryBehavior(Enum):
    Unexplored = 0
    Explored = 1

class UnknownKeywordOnPage(Enum):
    NonePresent = 0
    Present = 1

EmbodiedStateSpace = [
    (StudentEngagementLevel.Low, StudentExploratoryBehavior.Explored, UnknownKeywordOnPage.Present),
    (StudentEngagementLevel.Medium, StudentExploratoryBehavior.Explored, UnknownKeywordOnPage.Present),
    (StudentEngagementLevel.High, StudentExploratoryBehavior.Explored, UnknownKeywordOnPage.Present),
    (StudentEngagementLevel.Low, StudentExploratoryBehavior.Unexplored, UnknownKeywordOnPage.Present),
    (StudentEngagementLevel.Medium, StudentExploratoryBehavior.Unexplored, UnknownKeywordOnPage.Present),
    (StudentEngagementLevel.High, StudentExploratoryBehavior.Unexplored, UnknownKeywordOnPage.Present),
    (StudentEngagementLevel.Low, StudentExploratoryBehavior.Explored, UnknownKeywordOnPage.NonePresent),
    (StudentEngagementLevel.Medium, StudentExploratoryBehavior.Explored, UnknownKeywordOnPage.NonePresent),
    (StudentEngagementLevel.High, StudentExploratoryBehavior.Explored, UnknownKeywordOnPage.NonePresent),
    (StudentEngagementLevel.Low, StudentExploratoryBehavior.Unexplored, UnknownKeywordOnPage.NonePresent),
    (StudentEngagementLevel.Medium, StudentExploratoryBehavior.Unexplored, UnknownKeywordOnPage.NonePresent),
    (StudentEngagementLevel.High, StudentExploratoryBehavior.Unexplored, UnknownKeywordOnPage.NonePresent)
]

class EmbodiedInteractionEnv:

    def __init__(self, student_simulator) -> None:
        self.student_simulator = student_simulator
        self.MAX_LENGTH_PER_EPISODE = 20
        self.current_step = 0
        # self.current_state = self.reset()

    def update_student_simulator(self, student_simulator):
        self.student_simulator = student_simulator

    def reset(self):
        self.current_step = 0
        engage, explore, unknown = self.student_simulator.test_student.sample_student_state(with_explore_state=True)
        state = (engage, explore, unknown)
        values = self.get_state_value(state)
        return values

    def sample(self):
        # Sample student states
        results = list(map(lambda x: self.get_state_value(x), self.student_simulator.sample_student_cohort_state()))
        return results

    def get_all_state_space(self):
        return EmbodiedStateSpace
    
    def get_state_space_by_index(self, idx):
        return EmbodiedStateSpace[idx]
    
    def revert_value_to_state(self, state_vector):
        assert state_vector.shape[0] == 3, ("state vector has more than three values ", state_vector)
        state = [StudentEngagementLevel(state_vector[0]), StudentExploratoryBehavior(state_vector[1]), UnknownKeywordOnPage(state_vector[2])]
        return tuple(state)

    def get_state_value(self, state : Tuple[StudentEngagementLevel, StudentExploratoryBehavior, UnknownKeywordOnPage]):
        if state not in EmbodiedStateSpace:
            print("The state given is not in the EmbodiedStateSpace: ", state)
            return 0
        values = np.asarray(list(map(lambda x: x.value, state)))
        return values

    def get_all_action_space(self):
        return EmbodiedActionSpace
    
    def get_action_by_index(self, idx):
        return EmbodiedActionSpace[idx]

    def get_available_actions(self):
        """
        Returns the list of actions available in this state.
        Returns:
            List[Hashable] -- The list of actions available in this state.
        """
        pass

    def take_action(self, state_vector, action):
        """
        Returns the list of states which can be arrived at by taking the given action in this state.
        """
        # get the current state
        state = self.revert_value_to_state(state_vector)
        # Sample Student Response
        answering, explorations = self.student_simulator.get_new_response(state, action)

        # Calculate Reward
        reward = 0
        if answering != None:
            if answering.value == 1:
                reward += 0.2
                # reward += 0.7

        reward_expl = 0.0
        # print(explorations)
    
        for expl in explorations.keys():
            cnt = explorations[expl] if explorations[expl] <= 3 else 3
            reward_expl += np.log2(cnt + 1)
        if reward_expl > 0.0:
            # Add normalized exploration reward
            # print("Exploration reward", reward_expl * 1/len(explorations.keys()))
            reward += reward_expl * 1/len(explorations.keys())

        # Sample New Student State
        engage, unknown = self.student_simulator.sample_student_state(answered=answering)
        explored = None
        if len(explorations.keys()) == 0:
            explored = StudentExploratoryBehavior.Unexplored
        else:
            explored = StudentExploratoryBehavior.Explored
        next_obs = (engage, explored, unknown)
        assert next_obs in EmbodiedStateSpace, ("The new observation is not in the embodied state space ", next_obs)
        next_obs_vector = self.get_state_value(next_obs)

        done = False
        self.current_step += 1
        if self.current_step >= self.MAX_LENGTH_PER_EPISODE:
            done = True

        return next_obs_vector, reward, done

    def observe_with_student_model(self, student_model, policy, steps):
        """
        Returns the list of states which can be arrived at by taking the given action in this state.
        """
        transitions = []
        state = student_model.sample_student_state(with_explore_state=True)
        for _ in range(steps):
            action_idx = policy.get_next_action(self.get_state_value(state))
            action = EmbodiedActionSpace[action_idx]
            answering, explorations = student_model.get_new_response(state, action)
            # Calculate Reward
            reward = 0
            if answering != None:
                if answering.value == 1:
                    reward += 0.2
            reward_expl = 0.0
            # print(explorations)
            for expl in explorations.keys():
                cnt = explorations[expl] if explorations[expl] <= 3 else 3
                reward_expl += np.log2(cnt + 1)
            if reward_expl > 0.0:
                # Add normalized exploration reward
                # print("Exploration reward", reward_expl * 1/len(explorations.keys()))
                reward += reward_expl * 1/len(explorations.keys())

            # Sample New Student State
            engage, unknown = student_model.sample_student_state(answered=answering)
            explored = None
            if len(explorations.keys()) == 0:
                explored = StudentExploratoryBehavior.Unexplored
            else:
                explored = StudentExploratoryBehavior.Explored

            next_obs = (engage, explored, unknown)
            assert next_obs in EmbodiedStateSpace, ("The new observation is not in the embodied state space ", next_obs)
            next_obs_vector = self.get_state_value(next_obs)
            new_transition = [self.get_state_value(state), action_idx, next_obs_vector, reward]
            transitions.append(new_transition)
            state = next_obs
        return transitions
        

    def is_action_legal(self, action) -> bool:
        """
        Returns whether the given action is legal in this state.
        
        Arguments:
            action {Hashable} -- The action to check for legaility in this state.
        Returns:
            bool -- Whether or not the given action is legal in this state.
        """
        pass

    def is_state_legal(self) -> bool:
        """
        Returns whether or not the current state is legal.
        Returns:
            bool -- Whether or not this state is legal.
        """
        pass

    def is_initial_state(self) -> bool:
        """
        Returns whether or not this state is an initial state.
        Returns:
            bool -- Whether or not this state is an initial state.
        """
        pass

    def is_terminal_state(self) -> bool:
        """
        Returns whether or not this is a terminal state.
        
        Returns:
            bool -- Whether or not this state is terminal.
        """
        pass
        
    def get_successors(self) -> List[State]:
        """
        Returns a list of all states which can be reached from this state in one time step.
        Returns:
            List[State] -- A list of all of the states that can be directly reached from this state.
        """
        pass 