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
        #print "n_history = ", n_history
        super(ActionValue, self).__init__(
            l1=F.Convolution2D(n_history, 9, ksize=1, stride=1, nobias=False, wscale=np.sqrt(2)),
            l2=F.Convolution2D(9, 9, ksize=1, stride=1, nobias=False, wscale=np.sqrt(2)),
            l3=F.Convolution2D(9, 9, ksize=1, stride=1, nobias=False, wscale=np.sqrt(2)),
            l4=F.Linear(135, 512),#, wscale=np.sqrt(2)),
            q_value=F.Linear(512, n_act,
                             initialW=0.0*np.random.randn(n_act, 512).astype(np.float32))
        )

    def q_function(self, state):
        #print state
        #raw_input()
        h1 = F.relu(self.l1(state))
        h2 = F.relu(self.l2(h1))
        h3 = F.relu(self.l3(h2))
        h4 = F.relu(self.l4(h3))
        return self.q_value(h4)

class DQN:
    
    # Hyper-Parameters
    gamma = 0.9  # Discount factor
    initial_exploration = 1000  # Initial exploratoin. original: 5x10^4
    replay_size = 100  # Replay (batch) size
    target_model_update_freq = 10  # Target update frequancy. original: 10^4
    data_size = 10**7  # Data size of history. original: 10^6
    code_idx_size = 3
    n_act = code_idx_size
    
    max_steps = 3
    goal_idx = []
    #img_size = n_act*max_step

    def __init__(self, actions, max_steps, n_history=1):
        print "Initializing DQN..."
        self.actions = actions
        self.n_history = n_history
        self.max_steps = max_steps
        #print "n_history = ", n_history
        self.time_stamp = 0

        print("Model Building")
        self.model = ActionValue(self.n_history, self.n_act)
        self.model_target = copy.deepcopy(self.model)

        print("Initizlizing Optimizer")
        self.optimizer = optimizers.RMSpropGraves(lr=0.00025, alpha=0.95, momentum=0.95, eps=0.01)
        self.optimizer.setup(self.model)


        # History Data :  D=[s, a, r, s_dash, end_episode_flag]
        hs = self.n_history
        #ims = self.img_size
        self.replay_buffer = [np.zeros((self.data_size, hs, self.max_steps, self.n_act), dtype=np.float32),
                  np.zeros(self.data_size, dtype=np.uint8),
                  np.zeros((self.data_size, 1), dtype=np.float32),
                  np.zeros((self.data_size, hs, self.max_steps, self.n_act), dtype=np.float32),
                  np.zeros((self.data_size, 1), dtype=np.bool)]

    # TODO: implement, for now some stupid output
    def action_sample_e_greedy(self, state, epsilon):      
        print state
        s = Variable(state)
        q = self.model.q_function(s)
        q = q.data[0]

        if np.random.rand() < epsilon:
            action = np.random.randint(0, self.n_act)
            print("RANDOM : " + str(action))
        else:
            a = np.argmax(q)
            print("GREEDY  : " + str(a))
            action = a #np.asarray(a, dtype=np.int8)
            print(q)
        return action, q

    #idx = random.randint(0, len(self.actions) - 1)
    #return idx, 0.5

    def stock_experience(self, time,
                        state, action_idx, reward, state_prime,
                        episode_end_flag):
        data_index = time % self.data_size

        if episode_end_flag is True:
            self.replay_buffer[0][data_index] = state
            self.replay_buffer[1][data_index] = action_idx
            self.replay_buffer[2][data_index] = reward
        else:
            self.replay_buffer[0][data_index] = state
            self.replay_buffer[1][data_index] = action_idx
            self.replay_buffer[2][data_index] = reward
            self.replay_buffer[3][data_index] = state_prime
        self.replay_buffer[4][data_index] = episode_end_flag


    def experience_replay(self, time):

        if self.initial_exploration < time:

            rs_t = max(self.replay_size-len(self.goal_idx),self.replay_size/2)
            # Pick up replay_size number of samples from the Data
            if time < self.data_size:  # during the first sweep of the History Data
                replay_index = np.random.randint(0, time, (rs_t, 1))
            else:
                replay_index = np.random.randint(0, self.data_size, (rs_t, 1))


        #print replay_index
        #print rs_t
        #print time
        #raw_input()

            hs = self.n_history
            rs = self.replay_size

            s_replay = np.ndarray(shape=(rs, hs, self.max_steps, self.n_act), dtype=np.float32)
            a_replay = np.ndarray(shape=(rs, 1), dtype=np.uint8)
            r_replay = np.ndarray(shape=(rs, 1), dtype=np.float32)
            s_dash_replay = np.ndarray(shape=(rs, hs, self.max_steps, self.n_act), dtype=np.float32)
            episode_end_replay = np.ndarray(shape=(rs, 1), dtype=np.bool)

            for i in range(rs_t):
                s_replay[i] = np.asarray(self.replay_buffer[0][replay_index[i]], dtype=np.float32)
                a_replay[i] = self.replay_buffer[1][replay_index[i]]
                r_replay[i] = self.replay_buffer[2][replay_index[i]]
                s_dash_replay[i] = np.array(self.replay_buffer[3][replay_index[i]], dtype=np.float32)
                episode_end_replay[i] = self.replay_buffer[4][replay_index[i]]

            for i in range(rs_t, rs):
                s_replay[i] = np.asarray(self.replay_buffer[0][self.goal_idx[i-rs_t]], dtype=np.float32)
                a_replay[i] = self.replay_buffer[1][self.goal_idx[i-rs_t]]
                r_replay[i] = self.replay_buffer[2][self.goal_idx[i-rs_t]]
                s_dash_replay[i] = np.array(self.replay_buffer[3][self.goal_idx[i-rs_t]], dtype=np.float32)
                episode_end_replay[i] = self.replay_buffer[4][self.goal_idx[i-rs_t]]

            # Gradient-based update
            self.optimizer.zero_grads()
            loss, _ = self.get_loss(s_replay, a_replay, r_replay, s_dash_replay, episode_end_replay)
            loss.backward()
            self.optimizer.update()

    def get_loss(self, state, action, reward, state_prime, episode_end):
        s = Variable(state)
        s_dash = Variable(state_prime)

        q = self.model.q_function(s)  # Get Q-value

        # Generate Target Signals
        tmp = self.model_target.q_function(s_dash)  # Q(s',*)
        tmp = list(map(np.max, tmp.data))  # max_a Q(s',a)
        max_q_prime = np.asanyarray(tmp, dtype=np.float32)
        target = np.asanyarray(copy.deepcopy(q.data), dtype=np.float32)

        for i in range(self.replay_size):
            
            rw = np.sign(reward[i])
            if episode_end[i][0] is True:
                    tmp_ = rw
            else:
                #  The sign of reward is used as the reward of DQN!
                tmp_ = rw + self.gamma * max_q_prime[i]

            target[i, action[i]] = tmp_
            #print(tmp_)
        #raw_input('pause!!')

        #print(target)
        # TD-error clipping
        td = Variable(target) - q  # TD error
        # print("TD ")
        # print(td.data)
        td_tmp = td.data + 1000.0 * (abs(td.data) <= 1)  # Avoid zero division
        td_clip = td * (abs(td.data) <= 1) + td/abs(td_tmp) * (abs(td.data) > 1)
        #print(np.round(td.data))

        zero_val = Variable(np.zeros((self.replay_size, self.n_act), dtype=np.float32))
        loss = F.mean_squared_error(td_clip, zero_val)
        return loss, q


    def target_model_update(self, step, soft_update):
        if soft_update is True:
            tau = self.target_update_rate

            # Target preference Update
            model_params = dict(self.model.namedparams())
            model_target_params = dict(self.model_target.namedparams())
            for name in model_target_params:
                model_target_params[name].data = tau*model_params[name].data\
                                        + (1 - tau)*model_target_params[name].data
        else:
            if np.mod(step, self.target_model_update_freq) == 0:
                self.model_target = copy.deepcopy(self.model)

class DQNAgent:

    class State:

        def __init__(self, code=""):
            self.code = code
        #self.idx = [];


    # Action is a string ~ codeword
    ACTIONS = [" ", "print", "x"]

    def __init__(self, max_stp, actions=ACTIONS):
        print "Initializing DQN agent..."
        self.epsilon = 1.0  # Initial exploratoin rate
        self.dqn = DQN(actions, max_stp)

    def start(self, code, s_state):
        print "Starting agent..."
        self.reset_state(code)

        # Generate an Action e-greedy
        #print s_state
        #raw_input()
        state = np.asanyarray(s_state, dtype=np.float32).reshape(1, self.dqn.n_history, self.dqn.max_steps, self.dqn.n_act)
        #print state
        #raw_input()
        # Exploration decays along the time sequence
        self.policyFrozen = False
        if self.policyFrozen is False:  # Learning ON/OFF
            if self.dqn.initial_exploration < self.dqn.time_stamp:
                self.epsilon -= 1.0/10**2
                if self.epsilon < 0.1:
                    self.epsilon = 0.1
                eps = self.epsilon
            else:  # Initial Exploation Phase
                print("Initial Exploration : %d/%d steps" % (self.dqn.time_stamp, self.dqn.initial_exploration))
                eps = 1.0
        else:  # Evaluation
                print("Policy is Frozen")
                eps = 0.05

        action_idx, Q_now = self.dqn.action_sample_e_greedy(state, self.epsilon)
        self.last_action = self.dqn.actions[action_idx]
        self.last_state = copy.deepcopy(s_state)

        return action_idx

    def act(self, code, s_state, reward):
        print "Agent is acting..."
        self.set_state(code)

        # Exploration should decay along the time sequence
        state = np.asanyarray(s_state, dtype=np.float32).reshape(1, self.dqn.n_history, self.dqn.max_steps, self.dqn.n_act)
        action_idx, Q_now = self.dqn.action_sample_e_greedy(state, self.epsilon)

        # Place for learning

        return action_idx

    def reset_state(self, code):
        print "Resetting state..."
        self.last_code = code
        self.state = []
        self.state.append(self.State(code=code))

    def set_state(self, code):
        print "Setting state..."
        self.last_code = code
        if (len(self.state) >= self.dqn.n_history):
            self.state = [s for s in self.state[1:]]
        self.state.append(self.State(code=code))

    def end(self, reward):
        print "End in agent, episode terminated"
        print "Episode REWARD: {}".format(reward)

        # Place for learning
