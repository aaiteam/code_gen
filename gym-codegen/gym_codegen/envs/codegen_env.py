import contextlib
import os
import StringIO
import subprocess
import sys

import gym

import utils

from state import State, ACTIONS, INITIAL_CODE


@contextlib.contextmanager
def stdoutIO():
    old_stdout = sys.stdout
    stdout = StringIO.StringIO()
    sys.stdout = stdout
    yield stdout
    sys.stdout = old_stdout


class CodegenEnv(gym.Env):
    metadata = {'render.modes': ['human']}

    INITIAL_CODE = INITIAL_CODE
    FILENAME = "temp.py"

    def __init__(self, program_input="", goal=""):
        print "Initializing Codegen Environment..."

        self.input = program_input
        self.goal = goal
        self.state = State()

    # received an action to make from environment
    def _step(self, action_idx):
        print "Codegen is making a step..."
        self.last_action = ACTIONS[action_idx]
        self.state.set_action_idx(action_idx)

        ex_res = self.get_execution_result()
        reward = 0.
        terminal = False
        if ex_res.output == self.goal:
            print "Got desired result!"
            reward = 1000.0
            terminal = True
        elif ex_res.raised_exception:
            reward = -0.5
        else:
            reward = -0.3
        self.last_reward = reward
        if len(self.state.code) > 255:
            terminal = True 

        return self.state, reward, terminal, {}
    
    def get_execution_result(self):
        print "Getting execution result..."
        ex_res = utils.ExecutionResult(output="", raised_exception=False, exception_type=None)

        with open(self.FILENAME, "w") as f:
            f.write(self.state.code)

        with stdoutIO() as s:
            try:
                sys.argv[1:] = []
                sys.argv.append(self.input)
                execfile(self.FILENAME)
                del sys.argv[-1]
            except Exception as e:
                ex_res.raised_exception = True
                ex_res.exception_type = type(e).__name__

        ex_res.output = s.getvalue()
        # remove newline added by print for convinience
        if len(ex_res.output) > 0:
            ex_res.output = ex_res.output[:-1]
        print "Execution result: {}".format(ex_res)
        return ex_res

    def _reset(self):
        print "Resetting Codegen environment..."
        print "Input: {}, goal: {}".format(self.input, self.goal)
        self.state.code = self.INITIAL_CODE
        self.state.reset_state()

        return self.state

    # doesn't really supports rendering for now
    # explanations about rendering in https://github.com/openai/gym/blob/master/gym/core.py
    def _render(self, mode='human', close=False):
        print "Rendering Codegen environment..."
        print "Code: \n=====================\n{}\n=====================\n".format(self.state.code)
