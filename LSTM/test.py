import gym
import gym_codegen
import numpy as np
import dqn
from dqn import DQNAgent

import io
import os
import random
import sys


def main():
    print "Creating DQN agent..."

    iters = 10000
    n_goal = 0
    n_goal_all = 0
    time_stamp = 0

    ############################################################
    # print x
    # max_steps = 3
    # actions = ["print", " ", "x"]
    ############################################################

    ############################################################
    # print x+1
    max_steps = 5
    actions = ["print", " ", "x", "+", "1"]
    ############################################################

    agent = DQNAgent(max_steps, actions)
    agent.dqn.initial_exploration = iters*0.6

    results = []
    policy_frozen = False
    wins_file = "wins.txt"
    with io.FileIO(wins_file, "w") as file:
        file.write("Winning codes:\n")

    for iter in range(iters):
        print "\n\n::{}::".format(iter)

        if iter == 4300: # 2300:
            policy_frozen = True

        env = gym.make("codegen-v0")
        num = random.randrange(1,100)
        env.my_input = num

        ############################################################
        # print x
        # env.goal = str(num)
        ############################################################

        ############################################################
        # print x+1
        env.goal = str(num + 1)
        ############################################################

        code = env._reset()
        step_in_episode = 0
        total_score = 0.0
        reward = 0.0
        mystate = []
        my_state_new = []

        while step_in_episode < max_steps:
            state = env.code_index_list[:]
            state += np.zeros([max_steps - len(env.code_index_list), agent.dqn.code_idx_size], dtype=int).tolist()

            if step_in_episode == 0:
                action_idx = agent.start(code, state, policy_frozen)
            else:
                action_idx = agent.act(code, state, reward)

            code, reward, terminal, info = env._step(action_idx, agent.dqn.actions)
            state_prime = env.code_index_list[:]
            state_prime += np.zeros([max_steps - len(env.code_index_list), agent.dqn.code_idx_size], dtype=int).tolist()

            agent.dqn.experience_replay(agent.dqn.time_stamp)
            if step_in_episode == max_steps-1 or terminal:
                agent.dqn.stock_experience(agent.dqn.time_stamp, state, action_idx, reward, state_prime, True)
                if terminal:
                    agent.dqn.goal_idx.append(agent.dqn.time_stamp)
                agent.dqn.time_stamp += 1
            else:
                agent.dqn.stock_experience(agent.dqn.time_stamp, state, action_idx, reward, state_prime, False)

            total_score += reward

            if terminal:
                agent.end(reward)

                n_goal_all +=1
                step_in_episode += 1

                if iters-iter<=100:
                    n_goal +=1

            step_in_episode += 1

        if iter >= 100:
            results = results[1:]
        if reward >= 1:
            print "WIN"
            results.append(1.0)
            with io.FileIO(wins_file, "a") as f:
                f.write("\n=====================\n{}\n=====================\n\n".format(code))
                f.flush()
                os.fsync(f)
        else:
            results.append(0.0)
        total_iters = 100 if iter >= 100 else iter + 1
        print "TOTAL {:.2f}% of wins in last {} iters, sum: {}, total good: {}".format(
            100 * sum(results) / total_iters, total_iters, sum(results), len(agent.dqn.goal_idx))

        if iter == 1 + agent.dqn.initial_exploration:
            print "n_goal_all = ", n_goal_all
            print agent.dqn.goal_idx
            raw_input()

    print "n_goal : ", n_goal
    print "epsilon : ", agent.epsilon

if __name__ == "__main__":
    main()
