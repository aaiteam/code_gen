import gym
import gym_codegen
import numpy as np
from dqn import DQNAgent

import random
import sys

def set_debugger_org():
    if not sys.excepthook == sys.__excepthook__:
        from IPython.core import ultratb
        sys.excepthook = ultratb.FormattedTB(call_pdb=True)

def set_debugger_org_frc():
    from IPython.core import ultratb
    sys.excepthook = ultratb.FormattedTB(call_pdb=True)

def set_trace():
    from IPython.core.debugger import Pdb
    Pdb(color_scheme='Linux').set_trace(sys._getframe().f_back)
    

def main():
    print "Creating DQN agent..."
    # env = gym.make("codegen-v0")
    set_debugger_org_frc()

    iters = 6300
    n_goal = 0
    n_goal_all = 0
    time_stamp = 0

    max_steps = 5
    agent = DQNAgent(max_steps)
    agent.dqn.initial_exploration = 6000*max_steps

    for iter in range(iters):
        print "\n********Iteration # " , iter, "***********\n"
        # 1 iteration
        env = gym.make("codegen-v0")
        num = random.randrange(1,100)
        print "Goal Number : ", num+1 
        env.my_input = num
    	#env.goal = "['" + env.my_input + "']"
        env.goal = str(num+1)

        code = env._reset()
        step_in_episode = 0
        total_score = 0.0
        reward = 0.0
        mystate = []
        my_state_new = []

        # debug : the sys
        # sss = []
        # for arg in sys.argv[1:]:
        #    sss.append(arg)
        # print "sss = " , sss


        # while True:
        while step_in_episode < max_steps:

            # state = env.code_index_list + [-1]*(max_steps-len(env.code_index_list
            state = env.code_index_list[:]
            state += np.zeros([max_steps - len(env.code_index_list), agent.dqn.code_idx_size], dtype=int).tolist()
            # state = state.tolist()
            # state = 1;
            # print "env = ",env.code_index_list
            # print "state = ",state
            # raw_input()

            if step_in_episode == 0:
                action_idx = agent.start(code, state)
            else:
                action_idx = agent.act(code, state, reward)

            code, reward, terminal, info = env._step(action_idx, agent.dqn.actions)
            state_prime = env.code_index_list[:]
            state_prime += np.zeros([max_steps - len(env.code_index_list), agent.dqn.code_idx_size], dtype=int).tolist()

            # debug : the sys
            # sss = []
            # for arg in sys.argv[1:]:
            #    sss.append(arg)
            # print "sss = " , sss


            print "state : "
            print state
            print "state' : "
            print state_prime

          
            if step_in_episode == max_steps-1:
                agent.dqn.stock_experience(agent.dqn.time_stamp, state, action_idx
                                           , reward, state_prime, 1)
            else:
                agent.dqn.stock_experience(agent.dqn.time_stamp, state, action_idx
                                           , reward, state_prime, 0)

            agent.dqn.experience_replay(agent.dqn.time_stamp)

            agent.dqn.target_model_update(agent.dqn.time_stamp, soft_update=False)

            total_score += reward

            if terminal:

                agent.dqn.goal_idx.append(agent.dqn.time_stamp)

                agent.end(reward)
                agent.dqn.stock_experience(agent.dqn.time_stamp, state, action_idx
                                        , reward, state_prime, 1)

                n_goal_all +=1
                step_in_episode += 1
                agent.dqn.time_stamp += 1

                if iters-iter<=100:
                    n_goal +=1

                break
 
            step_in_episode += 1
            agent.dqn.time_stamp += 1

        if iter == 1 + (agent.dqn.initial_exploration / max_steps):
            print "n_goal_all = ", n_goal_all
            print agent.dqn.goal_idx
            raw_input()

    print "n_goal : ", n_goal
    print "epsilon : ", agent.epsilon
    # print env.code_index_list


if __name__ == "__main__":
    main()
