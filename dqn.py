# implemented according to https://github.com/ugo-nama-kun/DQN-chainer/blob/master/DQN-chainer-gym/dqn_agent.py
import chainer
from chainer import Chain, Variable, optimizers
import chainer.functions as F
import chainer.links as L
import copy
import gym
import random
import numpy as np
# random.seed(42) # for reproducibility

class Neuralnet(Chain):

    def __init__(self, n_in, n_out):
        super(Neuralnet, self).__init__(
            L1 = L.Linear(n_in, 10),
            Q_value = L.Linear(10, n_out, initialW=np.zeros((n_out, 10), dtype=np.float32))
        )

    def Q_func(self, x):
        h = F.leaky_relu(self.L1(x))
        h = self.Q_value(h)
        return h

class DQN:
    # Hyper-Parameters
    gamma = 0.99  # Discount factor
    initial_exploration = 10**2  # Initial exploratoin. original: 5x10^4
    replay_size = 32  # Replay (batch) size
    target_model_update_freq = 10**2  # Target update frequancy. original: 10^4
    data_size = 10 ** 2  # Data size of history. original: 10^6

    def __init__(self, actions, n_history=4, n_max_act_len=10):
        print "Initializing DQN..."
        self.actions = actions
        self.n_history = n_history
        self.n_actions = len(actions)
        self.model = Neuralnet(n_max_act_len, self.n_actions)
        self.model_target = copy.deepcopy(self.model)
        self.n_max_act_len = n_max_act_len

        print("Initizlizing Optimizer")
        self.optimizer = optimizers.RMSpropGraves(lr=0.00025, alpha=0.95, momentum=0.95, eps=0.01)
        self.optimizer.setup(self.model)

        # History Data :  D=[s, a, r, s_dash, end_episode_flag]
        self.replay_buffer = [np.zeros((self.data_size, n_max_act_len), dtype=np.uint8),
                              np.zeros(self.data_size, dtype=np.uint8),
                              np.zeros((self.data_size, 1), dtype=np.float32),
                              np.zeros((self.data_size, n_max_act_len), dtype=np.uint8),
                              np.zeros((self.data_size, 1), dtype=np.bool)]


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
            if episode_end[i][0] is True:
                tmp_ = np.sign(reward[i])
            else:
                #  The sign of reward is used as the reward of DQN!
                tmp_ = np.sign(reward[i]) + self.gamma * max_q_prime[i]

            target[i, action[i]] = tmp_
            # print(tmp_)

        # print(target)
        # TD-error clipping
        td = Variable(target) - q  # TD error
        # print("TD ")
        # print(td.data)
        td_tmp = td.data + 1000.0 * (abs(td.data) <= 1)  # Avoid zero division
        td_clip = td * (abs(td.data) <= 1) + td / abs(td_tmp) * (abs(td.data) > 1)
        # print(np.round(td.data))

        zero_val = Variable(np.zeros((self.replay_size, self.n_act), dtype=np.float32))
        loss = F.mean_squared_error(td_clip, zero_val)
        return loss, q


    def stock_experience(self, time,
                         state, action, reward, state_prime,
                         episode_end_flag):
        data_index = time % self.data_size

        if episode_end_flag is True:
            self.replay_buffer[0][data_index] = state
            self.replay_buffer[1][data_index] = action
            self.replay_buffer[2][data_index] = reward
        else:
            self.replay_buffer[0][data_index] = state
            self.replay_buffer[1][data_index] = action
            self.replay_buffer[2][data_index] = reward
            self.replay_buffer[3][data_index] = state_prime
        self.replay_buffer[4][data_index] = episode_end_flag


    def experience_replay(self, time):
        if self.initial_exploration < time:
            # Pick up replay_size number of samples from the Data
            if time < self.data_size:  # during the first sweep of the History Data
                replay_index = np.random.randint(0, time, (self.replay_size, 1))
            else:
                replay_index = np.random.randint(0, self.data_size, (self.replay_size, 1))

            rs = self.replay_size

            s_replay = np.ndarray(shape=(rs, self.n_max_act_len), dtype=np.float32)
            a_replay = np.ndarray(shape=(rs, 1), dtype=np.int8)
            r_replay = np.ndarray(shape=(rs, 1), dtype=np.float32)
            s_dash_replay = np.ndarray(shape=(rs, self.n_max_act_len), dtype=np.float32)
            episode_end_replay = np.ndarray(shape=(rs, 1), dtype=np.bool)
            for i in range(self.replay_size):
                s_replay[i] = np.asarray(self.replay_buffer[0][replay_index[i]], dtype=np.float32)
                a_replay[i] = self.replay_buffer[1][replay_index[i]]
                r_replay[i] = self.replay_buffer[2][replay_index[i]]
                s_dash_replay[i] = np.array(self.replay_buffer[3][replay_index[i]], dtype=np.float32)
                episode_end_replay[i] = self.replay_buffer[4][replay_index[i]]

            # Gradient-based update
            self.optimizer.zero_grads()
            loss, _ = self.get_loss(s_replay, a_replay, r_replay, s_dash_replay, episode_end_replay)
            loss.backward()
            self.optimizer.update()

    # TODO: implement, for now some stupid output
    def action_sample_e_greedy(self, state, epsilon):
        state.idx
        s = Variable(state)
        q = self.model.q_function(s)
        q = q.data[0]

        if np.random.rand() < epsilon:
            action = np.random.randint(0, self.n_act)
            print("RANDOM : " + str(action))
        else:
            a = np.argmax(q)
            print("GREEDY  : " + str(a))
            action = np.asarray(a, dtype=np.int8)
            print(q)
        return action, q

    def target_model_update(self, soft_update):
        if soft_update is True:
            tau = self.target_update_rate

            # Target preference Update
            model_params = dict(self.model.namedparams())
            model_target_params = dict(self.model_target.namedparams())
            for name in model_target_params:
                model_target_params[name].data = tau*model_params[name].data\
                                        + (1 - tau)*model_target_params[name].data
        else:
            if np.mod(self.step, self.target_model_update_freq) == 0:
                self.model_target = copy.deepcopy(self.model)

    def learn(self, state, action, reward, state_prime, terminal):
        self.stock_experience(self.step,
                         state, action, reward, state_prime,
                         terminal)

        self.experience_replay(self.step)
        self.target_model_update(soft_update=False)

        self.step += 1




class DQNAgent:
    policyFrozen = False

    class State:

        # maybe only observation?
        # observation - code output
        def __init__(self, code="", observation="", idx):
            self.code = code
            self.observation = observation
            self.idx = idx


    # Action is a string ~ codeword
    ACTIONS = [" ", "print", "input"]

    def __init__(self, env, actions=ACTIONS):
        print "Initializing DQN agent..."
        self.epsilon = 1.0  # Initial exploratoin rate
        self.dqn = DQN(actions)

    def start(self, observation):
        print "Starting agent..."
        self.reset_state(observation)

        # Generate an Action e-greedy
        action, Q_now = self.dqn.action_sample_e_greedy(self.state, self.epsilon)
        self.last_action = action
        self.last_state = copy.deepcopy(self.state)

        return action

    def act(self, observation, reward):
        print "Agent is acting..."
        self.set_state(observation)

        # # Exploration should decay along the time sequence
        # action, Q_now = self.dqn.action_sample_e_greedy(self.state, self.epsilon)

        # Place for learning
        # state_ = np.asanyarray(self.state.reshape(1, self.dqn.), dtype=np.float32)
        state_ = np.zeros([self.n_ma])

        # Exploration decays along the time sequence
        if self.policyFrozen is False:  # Learning ON/OFF
            if self.dqn.initial_exploration < self.dqn.step:
                self.epsilon -= 1.0/10**6
                if self.epsilon < 0.1:
                    self.epsilon = 0.1
                eps = self.epsilon
            else:  # Initial Exploation Phase
                print("Initial Exploration : %d/%d steps" % (self.dqn.step, self.dqn.initial_exploration))
                eps = 1.0
        else:  # Evaluation
                print("Policy is Frozen")
                eps = 0.05

        # Generate an Action by e-greedy action selection
        action, Q_now = self.dqn.action_sample_e_greedy(state_, eps)

        # Learning Phase
        if self.policyFrozen is False:  # Learning ON/OFF
            self.dqn.learn(self.last_state, self.last_action, reward, self.state, False)
            self.last_action = copy.deepcopy(action)
            self.last_state = self.state.copy()

        # Simple text based visualization
        print(' Time Step %d /   ACTION  %d  /   REWARD %.1f   / EPSILON  %.6f  /   Q_max  %3f' % (self.dqn.step, action, np.sign(reward), eps, np.max(Q_now)))


        return action

    def reset_state(self, observation):
        print "Resetting state..."
        self.last_observation = observation
        self.state = []
        self.state.append(self.State(code="", observation=observation))

    def set_state(self, observation):
        print "Setting state..."
        self.last_observation = observation
        if (len(self.state) >= self.dqn.n_history):
            self.state = [s for s in self.state[1:]]
        self.state.append(self.State(code="", observation=observation))

    def end(self, reward):
        print "End in agent, episode terminated"
        print "Episode REWARD: {}".format(reward)

        # Place for learning
