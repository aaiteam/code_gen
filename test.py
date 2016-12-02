import gym
import gym_codegen

from dqn import DQNAgent



def main():
    print "Creating DQN agent..."
    env = gym.make("codegen-v0")
    env.input = "42"
    env.goal = "['42']"

    agent = DQNAgent(env=env)

    # 1 iteration
    observation = env.reset()
    step_in_episode = 0
    total_score = 0.0
    reward = 0.0

    max_steps = 3
    # while True:
    while step_in_episode < max_steps:

        env.render()

        if step_in_episode == 0:
            observation, reward, terminal, info = env.step(agent.start(observation))
        else:
            observation, reward, terminal, info = env.step(agent.act(observation, reward))

        total_score += reward
        step_in_episode += 1

        if terminal:
            agent.end(reward)
            break


if __name__ == "__main__":
    main()
