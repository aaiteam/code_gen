# implemented according to https://github.com/ugo-nama-kun/DQN-chainer/blob/master/DQN-chainer-gym/dqn_agent.py
import chainer
import copy
import gym
import random
import gym_codegen

import pickle
import numpy as np
import scipy.misc as spm

from chainer import cuda, Function, Variable, optimizers, serializers
from chainer import Chain
import chainer.functions as F
import chainer.links as L

import matplotlib.pyplot as plt

#random.seed(42) # for reproducibility

class ActionValue(Chain):
    def __init__(self, n_history, n_act):
        embedding_size = n_act + 1
        super(ActionValue, self).__init__(
            embed=L.EmbedID(n_act + 1, embedding_size),
            lstm=L.LSTM(in_size=embedding_size, out_size=30), 
            q_value=L.Linear(30, n_act)
        )

    def q_function(self, action):
        x = self.embed(Variable(np.asarray([np.int32(action + 1)])))
        h_lstm = self.lstm(x)
        res  = self.q_value(h_lstm)
        return res 

    def reset_state(self):
        self.lstm.reset_state()


class DQN:
    
    # Hyper-Parameters
    gamma = 0.9  # Discount factor
    initial_exploration = 1000  # Initial exploratoin. original: 5x10^4
    replay_size = 100  # Replay (batch) size
    good_replay_size = 70 # Let's replay more good examples
    target_model_update_freq = 10  # Target update frequancy. original: 10^4
    data_size = 10**7  # Data size of history. original: 10^6
    max_steps = 3
    goal_idx = []

    class Episode:

        def __init__(self, actions=[], rewards=[], ended=False):
            self.actions = actions
            self.rewards = rewards
            self.ended = ended

        def custom_print(self):
            print self.actions
            print self.rewards
            print "Ended: {}".format(self.ended)

    def __init__(self, actions, max_steps, n_history=1):
        print "Initializing DQN..."
        self.actions = actions
        self.n_act = len(actions)
        self.code_idx_size = self.n_act
        self.n_history = n_history
        self.max_steps = max_steps
        self.time_stamp = 0

        self.model = ActionValue(self.n_history, self.n_act)
        self.model_target = copy.deepcopy(self.model)

        self.optimizer = optimizers.AdaGrad(lr=0.001)
        self.optimizer.setup(self.model)

        hs = self.n_history
        self.replay_buffer = [np.zeros((self.data_size, hs, self.max_steps, self.n_act), dtype=np.float32),
                  np.zeros(self.data_size, dtype=np.uint8),
                  np.zeros((self.data_size, 1), dtype=np.float32),
                  np.zeros((self.data_size, hs, self.max_steps, self.n_act), dtype=np.float32),
                  np.zeros((self.data_size, 1), dtype=np.bool)]
        self.history = []
        self.xp = self.model.xp

    def action_sample_e_greedy(self, state, epsilon):
        action = [0] * self.n_act
        for a in state[0][0]:
            last = True
            for e in a:
                if e == 1:
                    last = False
            if not last:
                action = a
            else:
                break

        q = self.model.q_function(self.get_action_id(action))
        print "q for action {} is {}".format(self.get_action_id(action), q.data[0])
        q = q.data[0]

        if np.random.rand() < epsilon:
            print "RANDOM"
            action = np.random.randint(0, self.n_act)

        else:
            a = np.argmax(q)
            action = a
        print "next action = {}".format(action)
        return action, q

    # TODO: Use data size
    def stock_experience(self, time, state, action_idx, reward, state_prime, episode_end_flag):
        if self.history and not self.history[-1].ended:
            self.history[-1].actions.append(action_idx)
            self.history[-1].rewards.append(reward)
            if episode_end_flag is True:
                self.history[-1].ended = True
        else:   
            self.history.append(self.Episode(actions=[action_idx], rewards=[reward]))

    def experience_replay(self, time):
        if self.initial_exploration < time:
            replay_all = max(self.replay_size - len(self.goal_idx), self.replay_size - self.good_replay_size)
            replay_good = min(len(self.goal_idx), self.good_replay_size)

            replay_index = random.sample(range(len(self.history)), replay_all) + \
                random.sample(self.goal_idx, replay_good)

            r_episodes = [self.history[id] for id in replay_index]
            for episode in r_episodes:
                self.optimizer.zero_grads()
                self.get_loss(episode).backward()
                self.optimizer.update()
                self.target_model_update(time, soft_update=False)


    def get_action_id(self, action):
        for i in range(len(action)):
            if action[i] == 1:
                return i
        return -1

    def get_loss(self, episode):
        self.model.reset_state()
        self.model_target.reset_state()

        action = -1
        self.model_target.q_function(action)
        loss = 0
        for i in range(len(episode.actions)):
            next_action = episode.actions[i]
            q = self.model.q_function(action)
            q_prime = self.model_target.q_function(next_action) # Q(s',*)

            max_q_prime = q_prime.data.max(axis=1)
            reward = episode.rewards[i]

            tmp = list(map(np.max, q_prime.data))  # max_a Q(s',a)
            max_q_prime = np.asanyarray(tmp, dtype=np.float32)
            target = np.asanyarray(copy.deepcopy(q.data), dtype=np.float32)

            reward = np.sign(episode.rewards[i])

            tmp_ = reward + self.gamma * max_q_prime[0]
            action = next_action
            target[0, action] = tmp_
            td = Variable(target) - q
            td_tmp = td.data + 1000.0 * (abs(td.data) <= 1)  # Avoid zero division
            td_clip = td * (abs(td.data) <= 1) + td/abs(td_tmp) * (abs(td.data) > 1)

            zero_val = Variable(np.zeros((1, self.n_act), dtype=np.float32))
            loss += F.mean_squared_error(td_clip, zero_val)

        return loss

    # always hard update
    # TODO: fix it
    def target_model_update(self, step, soft_update):
        # if soft_update is True:
        #     tau = self.target_update_rate

        #     # Target preference Update
        #     model_params = dict(self.model.namedparams())
        #     model_target_params = dict(self.model_target.namedparams())
        #     for name in model_target_params:
        #         model_target_params[name].data = tau*model_params[name].data\
        #                                 + (1 - tau)*model_target_params[name].data
        # else:
        # if np.mod(step, self.target_model_update_freq) == 0:
        self.model_target = copy.deepcopy(self.model)


class DQNAgent:

    class State:

        def __init__(self, code=""):
            self.code = code


    # TODO: move actions in a better place
    def __init__(self, max_stp, actions=None):
        self.actions = ["print", " ", "x"]
        self.epsilon = 1.0  # Initial exploratoin rate
        self.dqn = DQN(self.actions, max_stp)

    def start(self, code, s_state, policy_frozen=False):
        self.reset_state(code)
        self.dqn.model.reset_state()

        # Generate an Action e-greedy
        state = np.asanyarray(s_state, dtype=np.float32).reshape(1, self.dqn.n_history, self.dqn.max_steps,
            self.dqn.n_act)
        # Exploration decays along the time sequence
        self.policy_frozen = policy_frozen
        if self.policy_frozen is False:  # Learning ON/OFF
            if self.dqn.initial_exploration < self.dqn.time_stamp*3:
                self.epsilon -= 0.2#1.0/10**2
                if self.epsilon < 0.1:
                    self.epsilon = 0.1
                # eps = self.epsilon
            # else:  # Initial Exploation Phase
            #     eps = 1.0
        else:  # Evaluation
            # Freeze it harder :D
            self.epsilon = 0.0
            # eps = self.epsilon #0.05
        print "EPSILON: {}".format(self.epsilon)

        action_idx, Q_now = self.dqn.action_sample_e_greedy(state, self.epsilon)
        self.last_action = self.dqn.actions[action_idx]
        self.last_state = copy.deepcopy(s_state)

        return action_idx

    def act(self, code, s_state, reward):
        self.set_state(code)

        # Exploration should decay along the time sequence
        state = np.asanyarray(s_state, dtype=np.float32).reshape(1, self.dqn.n_history, self.dqn.max_steps,
            self.dqn.n_act)
        action_idx, Q_now = self.dqn.action_sample_e_greedy(state, self.epsilon)
        return action_idx

    def reset_state(self, code):
        self.last_code = code
        self.state = []
        self.state.append(self.State(code=code))

    def set_state(self, code):
        self.last_code = code
        if (len(self.state) >= self.dqn.n_history):
            self.state = [s for s in self.state[1:]]
        self.state.append(self.State(code=code))

    def end(self, reward):
        print "End in agent, episode terminated"
        print "Episode REWARD: {}".format(reward)

