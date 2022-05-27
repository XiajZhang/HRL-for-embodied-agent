"""
Wrapper script for environment and agent
"""
from copy import deepcopy
from email import header, iterators
import itertools
from os import stat
import random
import re
from statistics import mean
from tkinter.tix import Tree
from wsgiref import headers
import numpy as np
import gym
from gym_minigrid.register import env_list
from gym_minigrid.minigrid import Grid, OBJECT_TO_IDX
from gym_minigrid.envs import FourRoomsEnv
from agent import *
from embodied_option import EmbodiedAbstractOption
from embodied_env import *
from embodied_student import *
from tabulate import tabulate
import matplotlib
from local_lib import plotting
import matplotlib.pyplot as plt
from pprint import pprint
import sys
from copy import deepcopy

def train():
    pass

def test_random():

    cohort_number = 50
    student_simulator_data = os.path.join(os.getcwd(), "student_data", "student_simulator.pkl")
    if not os.path.exists(student_simulator_data):
        student_simulator = StudentSimulator(cohort_number)
    else:
        with open(student_simulator_data, "rb") as f:
            student_simulator = pickle.load(f)
    print(len(student_simulator.student_cohort))

def train_agent_with_student_cohort():
    
    # Create or Load Student Cohort
    cohort_number = 50
    student_simulator_data = os.path.join(os.getcwd(), "student_data", "student_simulator.pkl")
    if not os.path.exists(student_simulator_data):
        student_simulator = StudentSimulator(cohort_number)
        student_simulator.save_student_simulator()
    else:
        with open(student_simulator_data, "rb") as f:
            student_simulator = pickle.load(f)
    print(len(student_simulator.student_cohort))

    random.seed(42)
    MAX_TIME_STEP = 500
    MEMORY_BUFFER_SIZE = 2000
    BATCH_SIZE = 200
    gamma = 0.9
    epsilon = 0.1
    lr = 0.01

    action_total_cnt = []
    student_env = EmbodiedInteractionEnv(student_simulator)
    action_space = student_env.get_all_action_space()
    agent = PolynomialLinearValueApproximationAgent(len(action_space), student_env, gamma, epsilon, lr, True)
    agent1 = deepcopy(agent)

    # training_dist = []
    # for t in range(MAX_TIME_STEP):
    #     # For each step, sample students response & add to memory buffer
    #     memory_buffer = []
    #     steps = int(MEMORY_BUFFER_SIZE/len(student_simulator.student_cohort))
    #     for student in student_simulator.student_cohort:
    #         buffered_transitions = student_env.observe_with_student_model(student, agent, steps)
    #         memory_buffer += buffered_transitions
    #         random.shuffle(memory_buffer)
        
    #     # sample mini-batch from memory buffer
    #     mini_batch = random.sample(memory_buffer, BATCH_SIZE)
    #     # calculate agent update
    #     action_cnt = agent.batch_update(mini_batch)
    #     action_total_cnt.append(action_cnt)
    #     # Empty memory
    #     memory_buffer = []

    #     dist = evaluating_two_policies(agent1, agent, student_env)
    #     training_dist.append(dist)
    #     # record agent learning progress
    #     if t%50 == 0:
    #         print("***step at %d***"%t)

    # action_array = np.array(action_total_cnt)
    # # print(action_array)
    # print(action_array.sum(axis=0))
    # # save agent
    # agent.save_policy()

    # y = training_dist/max(training_dist)
    # # y = distance_summary
    # x = list(range(MAX_TIME_STEP))
    # plt.plot(x, y)
    # plt.show()

    with open(os.path.join(os.getcwd(), "model", "polynoLVA_agent5.pkl"), 'rb') as f:
        agent = pickle.load(f)
    
    action_his = []
    obs_list = list(EmbodiedStateSpace)
    for t_obs in obs_list:
        _obs = student_env.get_state_value(t_obs)
        act_idx  = agent.get_next_action(_obs)
        action_his.append(act_idx)
        print(t_obs, EmbodiedActionSpace[act_idx])
    plt.hist(action_his)
    plt.show()

def evaluating_two_policies(agent1, agent2, env):
    ls_distance = 0.0
    for state in list(EmbodiedStateSpace):
        state_vector = env.get_state_value(state)
        for action_idx in range(len(EmbodiedActionSpace)):
            var1 = agent1.get_q_value(state_vector, action_idx)
            var2 = agent2.get_q_value(state_vector, action_idx)
            # print(var1, var2)
            ls_distance += (var1 - var2)**2
    ls_distance = np.sqrt(ls_distance/(len(EmbodiedStateSpace)*len(EmbodiedActionSpace)))
    return ls_distance

def run_embodied_experiment():

    random.seed(42)
    # # Generate Student's Priors
    # #   1. Prior Knowledge Level
    # #   2. Exploration Level
    # #   3. Engagement Prior -> Stronger engagement leads to stronger peer-mimicry effects (how to build this into the simulator?)
    # student_prior_types = [
    #     # (KnowledgePrior.Low, ExplorationPrior.Low),
    #     # (KnowledgePrior.Low, ExplorationPrior.High),
    #     (KnowledgePrior.High, ExplorationPrior.Low),
    #     # (KnowledgePrior.High, ExplorationPrior.High),
    # ]
    # generated_student_priors = dict()
    # student_num_per_type = 1
    # for type_idx in range(len(student_prior_types)):
    #     if type_idx not in generated_student_priors.keys():
    #         generated_student_priors[type_idx] = []
    #     for i in range(student_num_per_type):
    #         student_prior = []
    #         # generate actual prob prior for each: p < 0.5 is Low; p >= 0.5 is High
    #         for prior_type in student_prior_types[type_idx]:
    #             if prior_type.value == 1:
    #                 # Low -> [0, 0.5)
    #                 prob = random.randrange(20, 45)/100
    #             elif prior_type.value == 2:
    #                 # High -> [0.5, 1)
    #                 prob = random.randrange(55, 80)/100
    #             else:
    #                 raise Exception("Unrecognized number for the prior type: ", prior_type)
    #             student_prior.append(prob)
    #         generated_student_priors[type_idx].append(student_prior)
    
    student_simulator_data = os.path.join(os.getcwd(), "student_data", "student_simulator.pkl")
    if not os.path.exists(student_simulator_data):
        student_simulator = StudentSimulator(1)
    else:
        with open(student_simulator_data, "rb") as f:
            student_simulator = pickle.load(f)
    
    # Test script
    episodes = 200
    updating = True
    gamma = 0.9
    epsilon = 0.1
    lr = 0.01
    continual_learning = False
    student_random_seed = 11
    students = [[0.5, 0.5], [0.3, 0.7], [0.8, 0.8], [0.3, 0.3], [0.9, 0.3]]
    # students = [[0.5, 0.5]]
    # students.reverse()
    for priors in students:
        student_random_seed += 1
        student_simulator.update_test_student(EmbodiedStudent(student_random_seed, *priors))
        student_env = EmbodiedInteractionEnv(student_simulator)
        action_space = student_env.get_all_action_space()
        if continual_learning:
            with open(os.path.join(os.getcwd(), "model", "polynoLVA_agent5.pkl"), 'rb') as f:
                agent = pickle.load(f)
        else:
            print("***** creating new agent *****")
            agent = PolynomialLinearValueApproximationAgent(len(action_space), student_env, gamma, epsilon, lr, True)
        agent1 = deepcopy(agent)
        # agent = RandomAgent(len(action_space))

        episode_stats = plotting.EpisodeStats(
            episode_lengths=np.zeros(episodes),
            episode_rewards=np.zeros(episodes)) 
        distance_summary = []
        for epi_i in range(episodes):
            obs = student_env.reset()
            for _step in itertools.count():
                action_idx = agent.get_next_action(obs)
                action = action_space[action_idx]
                # action_history.append(action_idx)
                last_state = obs
                obs, reward, done = student_env.take_action(obs, action)
                # print("Last state", last_state)
                # print("New obs", obs)
                if updating:
                    agent.update_agent(last_state, action_idx, obs, reward)
                episode_stats.episode_rewards[epi_i] += reward
                if done:
                    episode_stats.episode_lengths[epi_i] = _step
                    break
                
                if _step == 9999:
                    episode_stats.episode_lengths[epi_i] = _step

            print("Episode: ", epi_i + 1, " \n\treward: ", episode_stats.episode_rewards[epi_i], " steps: ", episode_stats.episode_lengths[epi_i])
            # print(action_history)
        
            dist = evaluating_two_policies(agent1, agent, student_env)
            distance_summary.append(dist)
        
        y = distance_summary/max(distance_summary)
        # y = distance_summary
        x = list(range(episodes))
        plt.plot(x, y)
        plt.show()

        # Testing scenarios
        # action_his = []
        # obs_list = list(EmbodiedStateSpace)
        # for t_obs in obs_list:
        #     _obs = student_env.get_state_value(t_obs)
        #     act_idx  = agent.get_next_action(_obs)
        #     action_his.append(act_idx)
        # plt.hist(action_his)
        # plt.show()

        print("*"*50)
        print("Summary:")
        print(tabulate(zip(episode_stats.episode_rewards, episode_stats.episode_lengths), headers=["Reward", "Steps"]))
        plotting.plot_episode_stats(episode_stats, smoothing_window=15)

if __name__ == "__main__":
    """
    Here define the type of environment and agent you want to use
    """
    train_agent_with_student_cohort()
    # run_embodied_experiment()
    # test_random()

# def run_standard_experiment():

#     episodes = 200
#     np.random.seed(2)
#     env = gym.make('MountainCar-v0')
#     env._max_episode_steps = 4000
#     # env._max_episode_steps = 10000
#     action_space = env.action_space
#     action_n = action_space.n
    
#     initial_obs = env.reset()
#     # Select the agent: Tabular Vs Polynomial LVA
#     # gamma = 0.9
#     # epsilon = 0.1
#     # lr = 0.05
#     # agent = TabularQAgent(action_space.n, 4, exp, gamma, lr)
#     gamma = 0.9
#     epsilon = 0.1
#     lr = 0.1
#     agent = PolynomialLinearValueApproximationAgent(action_n, env, gamma, epsilon, lr)

#     episode_stats = plotting.EpisodeStats(
#         episode_lengths=np.zeros(episodes),
#         episode_rewards=np.zeros(episodes)) 
#     for epi_i in range(episodes):
#         obs = env.reset()
#         # last_obs = None
#         for _step in itertools.count():
#             # if epi_i >= episodes - 1:
#             #     env.render()

#             # Random agent
#             # action = env.action_space.sample()
#             action = agent.get_next_action(obs)

#             last_state = obs
#             obs, reward, done, info = env.step(action)

#             agent.update_agent(last_state, action, obs, reward)
#             episode_stats.episode_rewards[epi_i] += reward
#             if done:
#                 episode_stats.episode_lengths[epi_i] = _step
#                 break
            
#             if _step == 9999:
#                 episode_stats.episode_lengths[epi_i] = _step

#         print("Episode: ", epi_i + 1, " \n\treward: ", episode_stats.episode_rewards[epi_i], " steps: ", episode_stats.episode_lengths[epi_i])
        

#     print("*"*50)
#     print("Summary:")
#     print(tabulate(zip(episode_stats.episode_rewards, episode_stats.episode_lengths), headers=["Reward", "Steps"]))
#     plotting.plot_episode_stats(episode_stats, smoothing_window=100)







