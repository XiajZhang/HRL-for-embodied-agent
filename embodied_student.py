"""
defines a stochastic embodied student with emotive responses
and student simulator
"""
from itertools import count
from multiprocessing.dummy import active_children
from tokenize import Double
from typing import List, Tuple
from enum import Enum
from unittest import TestCase
from embodied_env import InteractionBehavior, StudentEngagementLevel, StudentExploratoryBehavior, KeywordKnowledge, UnknownKeywordOnPage
import numpy as np
import os
import random
import pickle

class Storybook(Enum):
    Easy = 0
    Mid_Low = 1
    Mid_High = 2
    Hard = 3

class AnswerBehavior(Enum):
    NoAnswer = 0
    Answered = 1

class ExplorationBehavior(Enum):
    Word_Definition = 1
    Word_Context = 2
    Word_Pron = 3
    Listen_Audio = 4

# Student Prior Predified : Prior Knowledge, Exploration, Engagement
class KnowledgePrior(Enum):
    Low = 1
    High = 2

class ExplorationPrior(Enum):
    Low = 1
    High = 2

class EngagementPrior(Enum):
    Low = 1
    High = 2

# Prob of not answering is 1 - P(Answered)
# Assuming the questions are always known to the child
STUDENT_ANSWERING_PROB = {
    (StudentEngagementLevel.Low, KeywordKnowledge.Known): 0.2,
    (StudentEngagementLevel.Medium, KeywordKnowledge.Known): 0.5,
    (StudentEngagementLevel.High, KeywordKnowledge.Known): 0.7,
    (StudentEngagementLevel.Low, KeywordKnowledge.Unknown): 0.05,
    (StudentEngagementLevel.Medium, KeywordKnowledge.Unknown): 0.4,
    (StudentEngagementLevel.High, KeywordKnowledge.Unknown): 0.6,
}

# 0.8, 0.6, 0.4
# STUDENT_ANSWERING_PROB = {
#     (StudentEngagementLevel.Low, StudentExploratoryBehavior.Explored, UnknownKeywordOnPage.Present): ,
#     (StudentEngagementLevel.Medium, StudentExploratoryBehavior.Explored, UnknownKeywordOnPage.Present): ,
#     (StudentEngagementLevel.High, StudentExploratoryBehavior.Explored, UnknownKeywordOnPage.Present): ,
#     (StudentEngagementLevel.Low, StudentExploratoryBehavior.Unexplored, UnknownKeywordOnPage.Present): ,
#     (StudentEngagementLevel.Medium, StudentExploratoryBehavior.Unexplored, UnknownKeywordOnPage.Present): ,
#     (StudentEngagementLevel.High, StudentExploratoryBehavior.Unexplored, UnknownKeywordOnPage.Present): ,
#     (StudentEngagementLevel.Low, StudentExploratoryBehavior.Explored, UnknownKeywordOnPage.NonePresent): ,
#     (StudentEngagementLevel.Medium, StudentExploratoryBehavior.Explored, UnknownKeywordOnPage.NonePresent): ,
#     (StudentEngagementLevel.High, StudentExploratoryBehavior.Explored, UnknownKeywordOnPage.NonePresent): 0.8,
#     (StudentEngagementLevel.Low, StudentExploratoryBehavior.Unexplored, UnknownKeywordOnPage.NonePresent): ,
#     (StudentEngagementLevel.Medium, StudentExploratoryBehavior.Unexplored, UnknownKeywordOnPage.NonePresent): ,
#     (StudentEngagementLevel.High, StudentExploratoryBehavior.Unexplored, UnknownKeywordOnPage.NonePresent): ,
# }

# # Six Levels: 0.8, 0.7, 0.6, 0.5, 0.4, 0.3
# STUDENT_EXPLORING_LIKELIHOOD_1 = {
#     (StudentEngagementLevel.Low, StudentExploratoryBehavior.Explored, UnknownKeywordOnPage.Present): 0.4,
#     (StudentEngagementLevel.Medium, StudentExploratoryBehavior.Explored, UnknownKeywordOnPage.Present): 0.6,
#     (StudentEngagementLevel.High, StudentExploratoryBehavior.Explored, UnknownKeywordOnPage.Present): 0.8,

#     (StudentEngagementLevel.Low, StudentExploratoryBehavior.Unexplored, UnknownKeywordOnPage.Present): 0.3,
#     (StudentEngagementLevel.Medium, StudentExploratoryBehavior.Unexplored, UnknownKeywordOnPage.Present): 0.5,
#     (StudentEngagementLevel.High, StudentExploratoryBehavior.Unexplored, UnknownKeywordOnPage.Present): 0.7,

#     (StudentEngagementLevel.Low, StudentExploratoryBehavior.Explored, UnknownKeywordOnPage.NonePresent): 0.4,
#     (StudentEngagementLevel.Medium, StudentExploratoryBehavior.Explored, UnknownKeywordOnPage.NonePresent): 0.5,
#     (StudentEngagementLevel.High, StudentExploratoryBehavior.Explored, UnknownKeywordOnPage.NonePresent): 0.7,

#     (StudentEngagementLevel.Low, StudentExploratoryBehavior.Unexplored, UnknownKeywordOnPage.NonePresent): 0.3,
#     (StudentEngagementLevel.Medium, StudentExploratoryBehavior.Unexplored, UnknownKeywordOnPage.NonePresent): 0.4,
#     (StudentEngagementLevel.High, StudentExploratoryBehavior.Unexplored, UnknownKeywordOnPage.NonePresent): 0.6,
# }

# STUDENT_EXPLORING_LIKELIHOOD_2 = {
#     (StudentEngagementLevel.Low, StudentExploratoryBehavior.Explored, UnknownKeywordOnPage.Present, InteractionBehavior.Querying): 30,
#     (StudentEngagementLevel.Medium, StudentExploratoryBehavior.Explored, UnknownKeywordOnPage.Present, InteractionBehavior.Querying): 50,
#     (StudentEngagementLevel.High, StudentExploratoryBehavior.Explored, UnknownKeywordOnPage.Present, InteractionBehavior.Querying): 100,

#     (StudentEngagementLevel.Low, StudentExploratoryBehavior.Unexplored, UnknownKeywordOnPage.Present, InteractionBehavior.Querying): 25,
#     (StudentEngagementLevel.Medium, StudentExploratoryBehavior.Unexplored, UnknownKeywordOnPage.Present, InteractionBehavior.Querying): 45,
#     (StudentEngagementLevel.High, StudentExploratoryBehavior.Unexplored, UnknownKeywordOnPage.Present, InteractionBehavior.Querying): 20,

#     (StudentEngagementLevel.Low, StudentExploratoryBehavior.Explored, UnknownKeywordOnPage.NonePresent, InteractionBehavior.Querying): 25,
#     (StudentEngagementLevel.Medium, StudentExploratoryBehavior.Explored, UnknownKeywordOnPage.NonePresent, InteractionBehavior.Querying): 45,
#     (StudentEngagementLevel.High, StudentExploratoryBehavior.Explored, UnknownKeywordOnPage.NonePresent, InteractionBehavior.Querying): 50,

#     (StudentEngagementLevel.Low, StudentExploratoryBehavior.Unexplored, UnknownKeywordOnPage.NonePresent, InteractionBehavior.Querying): 25,
#     (StudentEngagementLevel.Medium, StudentExploratoryBehavior.Unexplored, UnknownKeywordOnPage.NonePresent, InteractionBehavior.Querying): 40,
#     (StudentEngagementLevel.High, StudentExploratoryBehavior.Unexplored, UnknownKeywordOnPage.NonePresent, InteractionBehavior.Querying): 80,

#     (StudentEngagementLevel.Low, StudentExploratoryBehavior.Explored, UnknownKeywordOnPage.Present, InteractionBehavior.Exploratory): 70,
#     (StudentEngagementLevel.Medium, StudentExploratoryBehavior.Explored, UnknownKeywordOnPage.Present, InteractionBehavior.Exploratory): 90,
#     (StudentEngagementLevel.High, StudentExploratoryBehavior.Explored, UnknownKeywordOnPage.Present, InteractionBehavior.Exploratory): 10,

#     (StudentEngagementLevel.Low, StudentExploratoryBehavior.Unexplored, UnknownKeywordOnPage.Present, InteractionBehavior.Exploratory): 70,
#     (StudentEngagementLevel.Medium, StudentExploratoryBehavior.Unexplored, UnknownKeywordOnPage.Present, InteractionBehavior.Exploratory): 80,
#     (StudentEngagementLevel.High, StudentExploratoryBehavior.Unexplored, UnknownKeywordOnPage.Present, InteractionBehavior.Exploratory): 50,

#     (StudentEngagementLevel.Low, StudentExploratoryBehavior.Explored, UnknownKeywordOnPage.NonePresent, InteractionBehavior.Exploratory): 70,
#     (StudentEngagementLevel.Medium, StudentExploratoryBehavior.Explored, UnknownKeywordOnPage.NonePresent, InteractionBehavior.Exploratory): 65,
#     (StudentEngagementLevel.High, StudentExploratoryBehavior.Explored, UnknownKeywordOnPage.NonePresent, InteractionBehavior.Exploratory): 85,

#     (StudentEngagementLevel.Low, StudentExploratoryBehavior.Unexplored, UnknownKeywordOnPage.NonePresent, InteractionBehavior.Exploratory): 65,
#     (StudentEngagementLevel.Medium, StudentExploratoryBehavior.Unexplored, UnknownKeywordOnPage.NonePresent, InteractionBehavior.Exploratory): 70,
#     (StudentEngagementLevel.High, StudentExploratoryBehavior.Unexplored, UnknownKeywordOnPage.NonePresent, InteractionBehavior.Exploratory): 30,
# }


STUDENT_EXPLORING_LIKELIHOOD_1 = {
    (StudentEngagementLevel.Low, StudentExploratoryBehavior.Explored, UnknownKeywordOnPage.Present, InteractionBehavior.Querying): 30,
    (StudentEngagementLevel.Medium, StudentExploratoryBehavior.Explored, UnknownKeywordOnPage.Present, InteractionBehavior.Querying): 50,
    (StudentEngagementLevel.High, StudentExploratoryBehavior.Explored, UnknownKeywordOnPage.Present, InteractionBehavior.Querying): 60,

    (StudentEngagementLevel.Low, StudentExploratoryBehavior.Unexplored, UnknownKeywordOnPage.Present, InteractionBehavior.Querying): 25,
    (StudentEngagementLevel.Medium, StudentExploratoryBehavior.Unexplored, UnknownKeywordOnPage.Present, InteractionBehavior.Querying): 45,
    (StudentEngagementLevel.High, StudentExploratoryBehavior.Unexplored, UnknownKeywordOnPage.Present, InteractionBehavior.Querying): 50,

    (StudentEngagementLevel.Low, StudentExploratoryBehavior.Explored, UnknownKeywordOnPage.NonePresent, InteractionBehavior.Querying): 25,
    (StudentEngagementLevel.Medium, StudentExploratoryBehavior.Explored, UnknownKeywordOnPage.NonePresent, InteractionBehavior.Querying): 45,
    (StudentEngagementLevel.High, StudentExploratoryBehavior.Explored, UnknownKeywordOnPage.NonePresent, InteractionBehavior.Querying): 50,

    (StudentEngagementLevel.Low, StudentExploratoryBehavior.Unexplored, UnknownKeywordOnPage.NonePresent, InteractionBehavior.Querying): 25,
    (StudentEngagementLevel.Medium, StudentExploratoryBehavior.Unexplored, UnknownKeywordOnPage.NonePresent, InteractionBehavior.Querying): 40,
    (StudentEngagementLevel.High, StudentExploratoryBehavior.Unexplored, UnknownKeywordOnPage.NonePresent, InteractionBehavior.Querying): 45,

    (StudentEngagementLevel.Low, StudentExploratoryBehavior.Explored, UnknownKeywordOnPage.Present, InteractionBehavior.Exploratory): 70,
    (StudentEngagementLevel.Medium, StudentExploratoryBehavior.Explored, UnknownKeywordOnPage.Present, InteractionBehavior.Exploratory): 90,
    (StudentEngagementLevel.High, StudentExploratoryBehavior.Explored, UnknownKeywordOnPage.Present, InteractionBehavior.Exploratory): 100,

    (StudentEngagementLevel.Low, StudentExploratoryBehavior.Unexplored, UnknownKeywordOnPage.Present, InteractionBehavior.Exploratory): 70,
    (StudentEngagementLevel.Medium, StudentExploratoryBehavior.Unexplored, UnknownKeywordOnPage.Present, InteractionBehavior.Exploratory): 80,
    (StudentEngagementLevel.High, StudentExploratoryBehavior.Unexplored, UnknownKeywordOnPage.Present, InteractionBehavior.Exploratory): 90,

    (StudentEngagementLevel.Low, StudentExploratoryBehavior.Explored, UnknownKeywordOnPage.NonePresent, InteractionBehavior.Exploratory): 70,
    (StudentEngagementLevel.Medium, StudentExploratoryBehavior.Explored, UnknownKeywordOnPage.NonePresent, InteractionBehavior.Exploratory): 75,
    (StudentEngagementLevel.High, StudentExploratoryBehavior.Explored, UnknownKeywordOnPage.NonePresent, InteractionBehavior.Exploratory): 85,

    (StudentEngagementLevel.Low, StudentExploratoryBehavior.Unexplored, UnknownKeywordOnPage.NonePresent, InteractionBehavior.Exploratory): 65,
    (StudentEngagementLevel.Medium, StudentExploratoryBehavior.Unexplored, UnknownKeywordOnPage.NonePresent, InteractionBehavior.Exploratory): 70,
    (StudentEngagementLevel.High, StudentExploratoryBehavior.Unexplored, UnknownKeywordOnPage.NonePresent, InteractionBehavior.Exploratory): 80,
}

STUDENT_UNEXPLORING_LIKELIHOOD_1 = {
    (StudentEngagementLevel.Low, StudentExploratoryBehavior.Explored, UnknownKeywordOnPage.Present, InteractionBehavior.Querying): 90,
    (StudentEngagementLevel.Medium, StudentExploratoryBehavior.Explored, UnknownKeywordOnPage.Present, InteractionBehavior.Querying): 80,
    (StudentEngagementLevel.High, StudentExploratoryBehavior.Explored, UnknownKeywordOnPage.Present, InteractionBehavior.Querying): 70,

    (StudentEngagementLevel.Low, StudentExploratoryBehavior.Unexplored, UnknownKeywordOnPage.Present, InteractionBehavior.Querying): 90,
    (StudentEngagementLevel.Medium, StudentExploratoryBehavior.Unexplored, UnknownKeywordOnPage.Present, InteractionBehavior.Querying): 80,
    (StudentEngagementLevel.High, StudentExploratoryBehavior.Unexplored, UnknownKeywordOnPage.Present, InteractionBehavior.Querying): 70,

    (StudentEngagementLevel.Low, StudentExploratoryBehavior.Explored, UnknownKeywordOnPage.NonePresent, InteractionBehavior.Querying): 90,
    (StudentEngagementLevel.Medium, StudentExploratoryBehavior.Explored, UnknownKeywordOnPage.NonePresent, InteractionBehavior.Querying): 80,
    (StudentEngagementLevel.High, StudentExploratoryBehavior.Explored, UnknownKeywordOnPage.NonePresent, InteractionBehavior.Querying): 70,

    (StudentEngagementLevel.Low, StudentExploratoryBehavior.Unexplored, UnknownKeywordOnPage.NonePresent, InteractionBehavior.Querying): 100,
    (StudentEngagementLevel.Medium, StudentExploratoryBehavior.Unexplored, UnknownKeywordOnPage.NonePresent, InteractionBehavior.Querying): 80,
    (StudentEngagementLevel.High, StudentExploratoryBehavior.Unexplored, UnknownKeywordOnPage.NonePresent, InteractionBehavior.Querying): 70,

    (StudentEngagementLevel.Low, StudentExploratoryBehavior.Explored, UnknownKeywordOnPage.Present, InteractionBehavior.Exploratory): 40,
    (StudentEngagementLevel.Medium, StudentExploratoryBehavior.Explored, UnknownKeywordOnPage.Present, InteractionBehavior.Exploratory): 30,
    (StudentEngagementLevel.High, StudentExploratoryBehavior.Explored, UnknownKeywordOnPage.Present, InteractionBehavior.Exploratory): 20,

    (StudentEngagementLevel.Low, StudentExploratoryBehavior.Unexplored, UnknownKeywordOnPage.Present, InteractionBehavior.Exploratory): 40,
    (StudentEngagementLevel.Medium, StudentExploratoryBehavior.Unexplored, UnknownKeywordOnPage.Present, InteractionBehavior.Exploratory): 30,
    (StudentEngagementLevel.High, StudentExploratoryBehavior.Unexplored, UnknownKeywordOnPage.Present, InteractionBehavior.Exploratory): 20,

    (StudentEngagementLevel.Low, StudentExploratoryBehavior.Explored, UnknownKeywordOnPage.NonePresent, InteractionBehavior.Exploratory): 40,
    (StudentEngagementLevel.Medium, StudentExploratoryBehavior.Explored, UnknownKeywordOnPage.NonePresent, InteractionBehavior.Exploratory): 30,
    (StudentEngagementLevel.High, StudentExploratoryBehavior.Explored, UnknownKeywordOnPage.NonePresent, InteractionBehavior.Exploratory): 20,

    (StudentEngagementLevel.Low, StudentExploratoryBehavior.Unexplored, UnknownKeywordOnPage.NonePresent, InteractionBehavior.Exploratory): 40,
    (StudentEngagementLevel.Medium, StudentExploratoryBehavior.Unexplored, UnknownKeywordOnPage.NonePresent, InteractionBehavior.Exploratory): 30,
    (StudentEngagementLevel.High, StudentExploratoryBehavior.Unexplored, UnknownKeywordOnPage.NonePresent, InteractionBehavior.Exploratory): 20,
}

# STUDENT_UNEXPLORING_LIKELIHOOD_1 = {
#     (StudentEngagementLevel.Low, StudentExploratoryBehavior.Explored, UnknownKeywordOnPage.Present): 0.5,
#     (StudentEngagementLevel.Medium, StudentExploratoryBehavior.Explored, UnknownKeywordOnPage.Present): 0.4,
#     (StudentEngagementLevel.High, StudentExploratoryBehavior.Explored, UnknownKeywordOnPage.Present): 0.3,

#     (StudentEngagementLevel.Low, StudentExploratoryBehavior.Unexplored, UnknownKeywordOnPage.Present): 0.5,
#     (StudentEngagementLevel.Medium, StudentExploratoryBehavior.Unexplored, UnknownKeywordOnPage.Present): 0.4,
#     (StudentEngagementLevel.High, StudentExploratoryBehavior.Unexplored, UnknownKeywordOnPage.Present): 0.3,

#     (StudentEngagementLevel.Low, StudentExploratoryBehavior.Explored, UnknownKeywordOnPage.NonePresent): 0.5,
#     (StudentEngagementLevel.Medium, StudentExploratoryBehavior.Explored, UnknownKeywordOnPage.NonePresent): 0.4,
#     (StudentEngagementLevel.High, StudentExploratoryBehavior.Explored, UnknownKeywordOnPage.NonePresent): 0.3,

#     (StudentEngagementLevel.Low, StudentExploratoryBehavior.Unexplored, UnknownKeywordOnPage.NonePresent): 0.5,
#     (StudentEngagementLevel.Medium, StudentExploratoryBehavior.Unexplored, UnknownKeywordOnPage.NonePresent): 0.4,
#     (StudentEngagementLevel.High, StudentExploratoryBehavior.Unexplored, UnknownKeywordOnPage.NonePresent): 0.3,
# }

class EmbodiedStudent:

    """
        Probabilistic Student Model with Behavioral Parameters that are predefined & calculated
        Predefined:
            1. Prior Knowledge Level
                [Low, High]
            2. Exploration Level
                [Low, High]
            3. Engagement Prior -> Stronger engagement leads to stronger peer-mimicry effects (how to build this into the simulator?)
                [Low, High]
        Generated/Calculated/Fixed:
            1. Answering Behaviors (based on the engagement level)
            2. Explore Behaviors (based on a. robot's behavior, child's knowledge, child's behavior b. student's self-exploration level )

    """

    def __init__(self, random_seed, prior_knowledge, exploration_prior, engagement_prior=0.0) -> None:
        # knowledge level
        self.prior_knowledge_level = prior_knowledge
        # engagement
        self.rng = random.Random(random_seed)
        # self.engagement_prior = [(1-engagement_prior)/2, (1-engagement_prior)/2, engagement_prior]
        self.engagement_prior = [0.5, 1/6]
        # pre-study exploration
        self.exploration_prior = exploration_prior
        # prob of answering the querying based on child's engagement
        self.answering_prior = STUDENT_ANSWERING_PROB
        self.response_rng = random.Random(random_seed+1)

    def construct_prior_knowledge_model(self, storybooks : Tuple[int, Storybook]):
        pass
    
    def construct_priors_for_responses(self):
        pass

    def update_knowledge_growth(self):
        pass
    
    def get_student_posterior_expl_prob(self, state):
        
        p_b_a = round(STUDENT_EXPLORING_LIKELIHOOD_1[state]/np.sum(list(STUDENT_EXPLORING_LIKELIHOOD_1.values())), 4)
        # p_b_a = round(STUDENT_EXPLORING_LIKELIHOOD_2[state]/np.sum(list(STUDENT_EXPLORING_LIKELIHOOD_2.values())), 4)

        
        # p_b_a_1 = round(STUDENT_UNEXPLORING_LIKELIHOOD_1[state]/np.sum(list(STUDENT_UNEXPLORING_LIKELIHOOD_1.values())), 4)
        # p_b = p_b_a*self.exploration_prior + p_b_a_1*(1-self.exploration_prior)
        p_b = round(1/len(STUDENT_UNEXPLORING_LIKELIHOOD_1.keys()), 4)
        bayes_posterior = self.exploration_prior*p_b_a/p_b
        # print("\nPrior", self.exploration_prior)
        # print("Likelihood of exploring", p_b_a)
        # print("Likelihood of unexploring", p_b_a_1)
        # print("Posterior", bayes_posterior)
        return bayes_posterior
    
    def get_new_response(self, state, action):
        answering = None
        if action[1] == InteractionBehavior.Querying:
            # Sample of question answering
            rdn = self.response_rng.random()
            answering_prob = self.answering_prior[(state[0], action[0])]
            if rdn <= answering_prob:
                answering = AnswerBehavior.Answered
            else:
                answering = AnswerBehavior.NoAnswer
                    
        # Four types of exploratory behaviors
        #   student's knowledge, robot's behavior
        #   if the robot asked a question the child answered...
        #       prob to explore after -> 0.7
        #   if robot asked a question the child did not answer...
        #       child knows the word -> 0.2
        #       child does not know the word -> 0.6
        #   if the robot explored
        #       a word the child does not know -> 0.75
        #       a word the child know -> 0.3
        # if action[1] == InteractionBehavior.Querying:
        #     if answering == AnswerBehavior.Answered:
        #         prob_to_explore = 0.7
        #     else:
        #         if action[0] == KeywordKnowledge.Known:
        #             prob_to_explore = 0.2
        #         else:
        #             prob_to_explore = 0.6
        # else:
        #     if action[0] == KeywordKnowledge.Known:
        #         prob_to_explore = 0.3
        #     else:
        #         prob_to_explore = 0.75
        prob_to_explore = self.get_student_posterior_expl_prob(state + (action[1], ))
        # Reduce exploration if not answered:
        # if answering == AnswerBehavior.NoAnswer:
        #     prob_to_explore = prob_to_explore*0.7
        # if action[1] == InteractionBehavior.Exploratory:
        #     prob_to_explore = min(prob_to_explore*1.2, 0.99)
        # Sample exploratory behavior
        exploration = dict(list(zip(ExplorationBehavior, len(ExplorationBehavior)*[0])))
        for _ in range(3*len(ExplorationBehavior)):
            _prob = prob_to_explore
            explored = self.response_rng.random()
            if explored <= _prob:
                explore_bhv = self.response_rng.choice(list(ExplorationBehavior))
                exploration[explore_bhv] += 1
            else:
                break

        return answering, exploration

    def sample_student_state(self, with_explore_state=False, answered=False):
        # Student engagement level when story listening is happening
        engagement_level = list(StudentEngagementLevel)
        if answered:
            engagement_level = engagement_level[1:]
            prob_distri = [0.3, 0.7]
            engagement = self.rng.choices(engagement_level, prob_distri)[0]
        else:
            engagement_var = np.random.normal(*self.engagement_prior)
            if engagement_var >= 0.66:
                engagement = StudentEngagementLevel.High
            elif engagement_var >= 0.33:
                engagement = StudentEngagementLevel.Medium
            else:
                engagement = StudentEngagementLevel.Low

        # Sample Knowledge
        keyword_knowledge = list(UnknownKeywordOnPage)
        keyword_prob_distri = [self.prior_knowledge_level, 1-self.prior_knowledge_level]
        known_keyword = self.rng.choices(keyword_knowledge, keyword_prob_distri)[0]
        
        # Sample explore state
        if with_explore_state:
            explored = list(StudentExploratoryBehavior)
            explored_prob_distri = [self.exploration_prior, 1-self.exploration_prior]
            exploration_behavior = self.rng.choices(explored, explored_prob_distri)[0]
            return engagement, exploration_behavior, known_keyword

        return engagement, known_keyword

class StudentSimulator:
    """
    1. create a cohort of simulated students from a pre-defined normal distribution for their behavior prior
    2. use memory buffer and mini-batches to update regressor until converges
    """
    def __init__(self, num_student : int) -> None:
        # for save & load a particular cohort
        self._dir = os.path.join(os.getcwd(), "student_data")
        self._simulator_loc = os.path.join(self._dir, "student_simulator.pkl")
        if not os.path.isdir(self._dir):
            os.mkdir(self._dir)

        # normal distribution of student with: prior knowledge & exploration 
        self.rnd_seed = 42
        self.num_student = num_student
        self.mu = 0.5
        self.sigma = 1/6
        self.student_cohort = self.generate_student_cohort()
        self.test_student = EmbodiedStudent(self.rnd_seed, 0.5, 0.5)

    def update_test_student(self, student_model):
        self.test_student = student_model

    def sample_student_state(self, answered=False):
        return self.test_student.sample_student_state(answered=answered)
    
    def get_new_response(self, state, action):
        return self.test_student.get_new_response(state, action)

    def generate_student_cohort(self) -> List[EmbodiedStudent]:
        students = []
        
        for _ in range(self.num_student):
            # Sampling two priors from normal distribution
            knowledge_prior = np.random.normal(self.mu, self.sigma)
            exploration_prior = np.random.normal(self.mu, self.sigma)
            new_student = EmbodiedStudent(self.rnd_seed, knowledge_prior, exploration_prior)
            students.append(new_student)
        return students

    def save_student_simulator(self):
        with open(self._simulator_loc, "wb") as f:
            pickle.dump(self, f)
    
    def sample_student_cohort_state(self):
        sampled_states = []
        for stud in self.student_cohort:
            sampled_states.append(tuple(stud.sample_student_state(with_explore_state=True)))
        return sampled_states

    def sample_response_from_student_cohort(self):

        pass

if __name__ == "__main__":

    rnd_seed = 42
    knowledge = 0.7
    engagement = [0.5, 0.3, 0.2]
    student = EmbodiedStudent(rnd_seed, knowledge, engagement)
    
    for i in range(10):
        new_engagement = student.sample_student_state()
        # print(new_engagement, '\n')
        state = (StudentEngagementLevel.Low, 1)
        action = (KeywordKnowledge.Unknown, InteractionBehavior.Exploratory)
        rst = student.get_new_response(state, action)
        print(rst)
