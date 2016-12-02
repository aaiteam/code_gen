from contextlib import contextmanager
import StringIO
import subprocess
import sys

import gym

import utils


@contextmanager
def stdoutIO():
    old_stdout = sys.stdout
    stdout = StringIO.StringIO()
    sys.stdout = stdout
    yield stdout
    sys.stdout = old_stdout


class CodegenEnv(gym.Env):
    metadata = {'render.modes': ['human']}

    INITIAL_CODE = """
import sys

input = []
for arg in sys.argv[1:]:
    input.append(arg)

# desired code starts here
"""

    def __init__(self, program_input="", goal=""):
        print "Initializing Codegen Environment..."
        self.code = self.INITIAL_CODE
        self.input = program_input
        self.goal = goal

    # received an action to make from environment
    def _step(self, action):
        print "Codegen is making a step..."
        self.last_action = action
        self.code += action

        observation = self.get_observation()
        reward = 0.0
        if observation == self.goal:
            reward = 1.0
        else:
            reward = -0.5
        self.last_reward = reward
        terminal = True if len(self.code) > 255 else False

        return observation, reward, terminal, {}
    
    def get_observation(self):
        print "Getting observation..."
        observation = utils.Observation(output="", raised_exception=False, exception_type=None)
        # subprocess.call("goal.py")
        with stdoutIO() as s:
            try:
                exec self.code
                # sys.argv = ['42]
                # execfile("goal.py")
            except Exception as e:
                observation.raised_exception = True
                observation.exception_type = type(e).__name__
        observation.output = s.getvalue()
        print "Observation: {}".format(observation)
        return observation

    def _reset(self):
        print "Resetting Codegen environment..."
        print "Input: {}, goal: {}".format(self.input, self.goal)
        self.code = self.INITIAL_CODE
        return self.get_observation()

    # doesn't really supports rendering for now
    # explanations about rendering in https://github.com/openai/gym/blob/master/gym/core.py
    def _render(self, mode='human', close=False):
        print "Rendering Codegen environment..."
        print "Code: \n=====================\n{}\n=====================\n".format(self.code)
