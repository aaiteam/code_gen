import contextlib
import os
import StringIO
import subprocess
import sys

import gym

import utils

import numpy as np

@contextlib.contextmanager
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

#x = []
#for arg in sys.argv[1:]:
#    x.append(arg)
x = sys.argv[1]

# desired code starts here
"""
    FILENAME = "temp.py"

    def __init__(self, program_input="", goal="", max_len=10000):
        print "Initializing Codegen Environment..."
        self.code = self.INITIAL_CODE
        self.my_input = program_input
        self.goal = goal
        self.code_index_list = []
        self.max_len = max_len

    # received an action to make from environment
    def _step(self, action_idx, actions):
        print "Codegen is making a step..."
        self.last_action = actions[action_idx]
        self.code += actions[action_idx]

        action_st = np.zeros(len(actions), dtype=int).tolist()
        action_st[action_idx] = 1;
        self.code_index_list+= [action_st]
        print "self.code_index_list = ", self.code_index_list

        ex_res = self.get_execution_result()
        reward = 0.
        terminal = False

        if ex_res.output == self.goal:
            print "Got desired result!"
            reward = 1.0
            terminal = True
        elif len(self.code_index_list) == self.max_len:
            reward = -1.0
        elif ex_res.raised_exception:
             if ex_res.exception_type == 'SyntaxError':
                 reward = 0.0
             else:
                 reward = -1.0
        else:
            reward = 0.0
            #terminal = True

        self.last_reward = reward
        if len(self.code) > 255:
            terminal = True 

        return self.code, reward, terminal, {}
    
    def get_execution_result(self):
        print "Getting execution result..."
        ex_res = utils.ExecutionResult(output="", raised_exception=False, exception_type=None)

        with open(self.FILENAME, "w") as f:
            f.write(self.code)

        with stdoutIO() as s:
            try:
                sys.argv[1:] = []
                #print "len = ", len(sys.argv)
                sys.argv.append(self.my_input)
                
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
        print "Input: {}, goal: {}".format(self.my_input, self.goal)
        self.code = self.INITIAL_CODE
        return self.code

    # doesn't really supports rendering for now
    # explanations about rendering in https://github.com/openai/gym/blob/master/gym/core.py
    def _render(self, mode='human', close=False):
        print "Rendering Codegen environment..."
        print "Code: \n=====================\n{}\n=====================\n".format(self.code)
