# implemented according to https://github.com/ugo-nama-kun/DQN-chainer/blob/master/DQN-chainer-gym/dqn_agent.py
import chainer
import copy
import gym
import random
# random.seed(42) # for reproducibility


class DQN:

    def __init__(self, actions, n_history=4):
        print "Initializing DQN..."
        self.actions = actions
        self.n_history = n_history

    # TODO: implement, for now some stupid output
    def action_sample_e_greedy(self, state, epsilon):
        return self.actions[random.randint(0, len(self.actions) - 1)], 0.5


class DQNAgent:

    class State:

        # maybe only observation?
        # observation - code output
        def __init__(self, code="", observation=""):
            self.code = code
            self.observation = observation


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

        # Exploration should decay along the time sequence
        action, Q_now = self.dqn.action_sample_e_greedy(self.state, self.epsilon)

        # Place for learning

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
