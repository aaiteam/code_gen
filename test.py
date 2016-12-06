import gym
import gym_codegen

from dqn import DQNAgent

from state import State


def main():
    print "Creating DQN agent..."
    env = gym.make("codegen-v0")
    env.input = "42"
    env.goal = "['42']"

    agent = DQNAgent()

    # 1 iteration
    state = env.reset()
    step_in_episode = 0
    total_score = 0.0
    reward = 0.0

    max_steps = 100000
    # while True:
    cnt = 0
    while step_in_episode < max_steps:

        env.render()

        if step_in_episode == 0:
            state, reward, terminal, info = env.step(agent.start(state))
        else:
            state, reward, terminal, info = env.step(agent.act(state, reward))
        total_score += reward
        step_in_episode += 1

        if terminal:
            agent.end(reward)
            print ("-"*100)
            print "Finished!!!"
            print ("-"*100)
            # break
        cnt += 1
        if cnt % 1000 == 0:
            agent.save()

    agent.save()



if __name__ == "__main__":
    main()
