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
    code = env.reset()
    step_in_episode = 0
    total_score = 0.0
    reward = 0.0

    max_steps = 7
    # while True:
    while step_in_episode < max_steps:

        env.render()

        if step_in_episode == 0:
            code, reward, terminal, info = env.step(agent.start(code))
        else:
            code, reward, terminal, info = env.step(agent.act(code, reward))

        total_score += reward
        step_in_episode += 1

        if terminal:
            agent.end(reward)
            break


if __name__ == "__main__":
    main()
