# implementation is based on https://github.com/ugo-nama-kun/DQN-chainer/blob/master/DQN-chainer-gym/dqn_agent.py
import chainer
import copy
import gym
import random
import gym_codegen
import onehot_rep.webpage_scrape as websc
import data_generator as data_gen

import pickle
import numpy as np
import scipy.misc as spm

from chainer import cuda, Function, Variable, optimizers, serializers
from chainer import Chain
from chainer import Link
import chainer.functions as F
import chainer.links as L

from copy import deepcopy
import matplotlib.pyplot as plt

#random.seed(42) # for reproducibility
    

class SimpleLSTM(Chain):

    def __init__(self, n_history, n_act):
        embedding_size = n_act + 1
        super(SimpleLSTM, self).__init__(
            embed=L.EmbedID(n_act + 1, embedding_size),
            lstm=L.LSTM(in_size=embedding_size, out_size=embedding_size)
        )

    def __call__(self, action):
        x = self.embed(Variable(np.asarray([np.int32(action + 1)])))
        y = self.lstm(x)
        return y

    def reset_state(self):
        self.lstm.reset_state()


class ActionValue(Chain):

    # n_act + 1 to incorporate initial state -1 without actions
    def __init__(self, n_history, n_act):
        embedding_size = n_act + 1
        super(ActionValue, self).__init__(
            embed=L.EmbedID(n_act + 1, embedding_size),
            lstm=L.LSTM(in_size=embedding_size, out_size=10),
            # fc=L.Linear(in_size=10, out_size=10),
            q_value=L.Linear(10, n_act)
        )
        self.h_sum = None
        self.seq_len = np.asarray([0])

    def q_function(self, action):
        x = self.embed(Variable(np.asarray([np.int32(action + 1)])))
        lstm_res = self.lstm(x)
        res  = self.q_value(lstm_res)
        # fc_res = F.relu(self.fc(lstm2_res))
        return res

    def reset_state(self):
        self.lstm.reset_state()
        # self.lstm1.reset_state()
        # self.lstm2.reset_state()
        self.h_sum = None
        self.seq_len = np.asarray([0])

    def set_state(self, actions):
        self.reset_state()
        for action in actions:
            self.q_function(action)

class DQN:
    
    # Hyper-Parameters
    gamma = 0.9  # Discount factor
    initial_exploration = 1000  # Initial exploratoin. original: 5x10^4
    replay_size = 100  # Replay (batch) size
    goal_replay_size = 70 # Let's replay more good examples
    target_model_update_freq = 10  # Target update frequancy. original: 10^4
    history_size = 1000
    goal_history_size = 200
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


    class Pretrained:

        def __init__(self, n_history, n_act):
            self.model = ActionValue(n_history, n_act)
            self.optimizer = optimizers.AdaGrad(lr=0.008)
            self.optimizer.setup(self.model)
            self.batchsize = 10
            self.epoch = 2000

    def __init__(self, actions, max_steps, n_history=1):
        print "Prepare Data for pretraining..."
        # 1 : Data from websites
        #websc.code_extraction("onehot_rep/webpage_list.txt", "onehot_rep/output.pkl")
        #_ , data_index , codebook = websc.convert2onehot("onehot_rep/output.pkl")
        #print data_index[0]
        #raw_input()

        # 2 : Synthetic Data
        data_index  = data_gen.Generator(actions).generate_codes_random2(max_steps, False)
        # data_index  = data_gen.Generator(actions).generate_codes_random(max_steps, False)
        # data_index  = data_gen.Generator(actions).generate_codes(max_steps, False)
        codebook = actions
        for k in range(len(codebook)):
            start_k = [x for x in data_index if x[0]==k]
            print "number of Data start with '{}' : {}".format(codebook[k], len(start_k))
        # raw_input()

        self.actions = codebook
        self.n_act = len(self.actions)     
        print "Data Size : ", len(data_index)
        print "codebook: {}".format(codebook)
        print "Data index: "
        for episode in data_index:
            print "".join(actions[i] for i in episode)

        print "LSTM pretraining..."
        self.pretrained = self.Pretrained(n_history, self.n_act)

        loss_old = 1000
        self.pretrained.optimizer.zero_grads()
        for epc in range(self.pretrained.epoch):
            self.pretrained.model.reset_state()
            self.pretrained.optimizer.zero_grads()
            loss = self.compute_loss(self.pretrained.model, data_index, self.pretrained.batchsize)
            
            if epc>=0 :  #or loss.data <= loss_old :
                loss.backward()
                self.pretrained.optimizer.update()
                loss_old = loss.data
            
            if epc % 10 == 0:
                print "Epoch : {} / loss : {}".format(epc, loss_old)



        print  "****Examination****"
        self.pretrained.model.reset_state()        
        test_pred = np.ones(5, dtype=np.int32)*100
        
        #for i in range(5):
        if True:
            test_pred[0] = -1
            code = ""

            for j in range(4):
                pd = self.pretrained.model.q_function(test_pred[j])
                print "pd for {} step : ".format(j)
                print pd.data
                test_pred[j+1] = np.argmax(pd.data)
                code += codebook[test_pred[j+1]]

            print "[output code]"
            print code
            print  "*******************"

        print "pretraining complete!!"
        # raw_input()

        # print "Initializing DQN..."
        # While pretraining is useless
        #######################################################################################
        self.actions = actions
        #######################################################################################
        self.n_act = len(self.actions)
        self.code_idx_size = self.n_act
        self.n_history = n_history
        self.max_steps = max_steps
        self.time_stamp = 0

        #self.model = ActionValue(self.n_history, self.n_act)
        self.model = self.pretrained.model ##Uncomment this if pretraied work!!
        self.model_target = copy.deepcopy(self.model)

        self.optimizer = optimizers.AdaGrad(lr=0.001)
        self.optimizer.setup(self.model)

        self.history = []
        self.goal_history = []
        self.xp = self.model.xp
        self.time = 0

    def compute_loss(self, model, x_list, batch_size):
        loss = 0
        
        minibatch_idx = random.sample(range(len(x_list)), batch_size)
        for i in minibatch_idx:
            #print i#, ":" , cur_word, " " , next_word
            #for cur_word, next_word in zip(x_list[i], x_list[i][1:]):
            for k in range(0, len(x_list[i])):

                if k==0 : 
                    cur_word = -1
                else : 
                    cur_word = x_list[i][k-1]

                next_word = x_list[i][k] 
    
                pred = model.q_function(cur_word)
                #print pred.data
                #pred = F.softmax(pred)
                #print pred.data
                #raw_input()
                gt = np.zeros(self.n_act, dtype=np.float32)
                if k>0:
                    gt = gt-1
                #print gt
                #raw_input()
                gt[next_word] = 1
                grdtru = Variable(np.asarray([gt]))
                #grdtru = Variable(np.asarray(np.int32([next_word])))
                #print grdtru.data
                #raw_input()
                #err = F.softmax_cross_entropy(pred, grdtru)
                err = F.mean_squared_error(pred, grdtru)
                loss += err

        loss = loss/batch_size
        return loss

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

        q = [0] * self.n_act
        if self.initial_exploration < self.time:
            q = self.model.q_function(self.get_action_id(action))
            # print "q for action {} is {}".format(self.get_action_id(action), q.data[0])
            q = q.data[0]

        # if np.random.rand() < epsilon:
        #     print "RANDOM"
        #     action = np.random.randint(0, self.n_act)

        if np.random.rand() < epsilon or self.initial_exploration >= self.time:
            # print "RANDOM"
            ##############################################################################
            action = self.get_action_id(action)
            # print "Sampling action"
            if self.initial_exploration >= self.time or epsilon >= 1.0:
                # print "initial exploration for action {}".format(action)
                if action == -1:
                    # print "Resetting state"
                    self.pretrained.model.reset_state()
                q = self.pretrained.model.q_function(action).data[0]
                if np.random.rand() < 0.5:
                    action = np.argmax(q)
                    # print "LSTM"
                    # print "argmax of {}: {}".format(q, action)
                else:
                    action = np.random.randint(0, self.n_act)
                    # print "random: {}".format(action)
            ##############################################################################
            # action = np.random.randint(0, self.n_act)

        else:
            a = np.argmax(q)
            action = a
        # print "next action = {}".format(action)
        return action, q

    # TODO: Use data size
    def stock_experience(self, time, state, action_idx, reward, state_prime, episode_end_flag):
        self.time = time
        if self.history and not self.history[-1].ended:
            self.history[-1].actions.append(action_idx)
            self.history[-1].rewards.append(reward)
            if episode_end_flag is True:
                self.history[-1].ended = True
                if self.history[-1].rewards[-1] >= 1:
                    for i in range(len(self.history[-1].rewards) - 1):
                        if self.history[-1].rewards[i] < 0:
                            self.history[-1].rewards[i] = 0.1
                    self.goal_history.append(self.history[-1])
        else:   
            self.history.append(self.Episode(actions=[action_idx], rewards=[reward]))

        if len(self.history) > self.history_size:
            self.history = self.history[1:]
        if len(self.goal_history) > self.goal_history_size:
            self.goal_history = self.goal_history[1:]

        # print "history size: {}, goal history size: {}".format(len(self.history), len(self.goal_history))

    def experience_replay(self, time):
        if self.initial_exploration < time:
            replay_goal = min(len(self.goal_history), self.goal_replay_size)
            replay_all = min(replay_goal, self.replay_size - self.goal_replay_size)
            # print "REPLAYING {} good and {} all".format(replay_goal, replay_all)

            replay_index = random.sample(range(len(self.history)), replay_all)
            goal_replay_index = random.sample(range(len(self.goal_history)), replay_goal)
            r_episodes = [deepcopy(self.history[id]) for id in replay_index] + \
                [deepcopy(self.goal_history[id]) for id in goal_replay_index]
            
            # # Can be harmful
            # # randomly decide length of episodes
            # for episode in r_episodes:
            #     length = random.randint(1, len(episode.actions))
            #     episode.actions = episode.actions[:length]
            #     episode.rewards = episode.rewards[:length]

            # update target model
            if self.initial_exploration == time+1 : 
                self.optimizer.zero_grads()
            
            loss = Variable(np.asarray(np.float32(0.0)))
            for episode in r_episodes:
                loss += self.get_loss(episode)
            loss.backward()
            self.optimizer.update()
            self.target_model_update(time, soft_update=False)

            # set model to original state
            if self.history[-1].ended:
                self.model.set_state([-1])
            else:
                self.model.set_state([-1] + self.history[-1].actions)

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
            # reward = episode.rewards[i]

            tmp = list(map(np.max, q_prime.data))  # max_a Q(s',a)
            max_q_prime = np.asanyarray(tmp, dtype=np.float32)
            target = np.asanyarray(copy.deepcopy(q.data), dtype=np.float32)

            reward = np.sign(episode.rewards[i])
            # reward = np.sign(episode.rewards[len(episode.actions)-1])

            tmp_ = reward + self.gamma * max_q_prime[0]
            action = next_action
            target[0, action] = tmp_
            # print "target: {}".format(target)
            # raw_input()
            td = Variable(target) - q
            td_tmp = td.data + 1000.0 * (abs(td.data) <= 1)  # Avoid zero division
            td_clip = td * (abs(td.data) <= 1) + td/abs(td_tmp) * (abs(td.data) > 1)

            zero_val = Variable(np.zeros((1, self.n_act), dtype=np.float32))
            # loss += F.mean_squared_error(td_clip, zero_val) * (i+1)
            loss += F.mean_squared_error(td_clip, zero_val)
            # loss += F.mean_squared_error(td_clip, zero_val)

        # print "last reward: {}, loss: {}".format(reward, loss.data)
        # raw_input()

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
    def __init__(self, max_steps, actions=["print", " ", "x", "+", "1"]):
        self.actions = actions
        self.epsilon = 1.0  # Initial exploratoin rate
        self.dqn = DQN(self.actions, max_steps)

    def start(self, code, s_state, policy_frozen=False):
        self.reset_state(code)
        self.dqn.model.reset_state()

        # Generate an Action e-greedy
        state = np.asanyarray(s_state, dtype=np.float32).reshape(1, self.dqn.n_history, self.dqn.max_steps,
            self.dqn.n_act)
        # Exploration decays along the time sequence
        self.policy_frozen = False if len(self.dqn.goal_idx) < 40 else policy_frozen
        if self.policy_frozen is False:  # Learning ON/OFF
            if len(self.dqn.goal_idx) < 5: #20:
                self.epsilon = 1.0
            elif len(self.dqn.goal_idx) < 10: #40:
                self.epsilon = 0.2
            else:
                self.epsilon = 0.1
        else:  # Evaluation
            self.epsilon = 0.0

        # print "EPSILON: {}".format(self.epsilon)

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
        pass
        # print "End in agent, episode terminated"
        # print "Episode REWARD: {}".format(reward)

