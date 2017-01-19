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

#     INITIAL_CODE = """
# import sys
# #x = []
# #for arg in sys.argv[1:]:
# #    x.append(arg)
# x = sys.argv[1]
# # desired code starts here
# """
    INITIAL_CODE = """
import sys

x = int(sys.argv[1])

# desired code starts here
"""

    FILENAME = "temp.py"

    def __init__(self, program_input="", goal=""):
        print "Initializing Codegen Environment..."
        self.code = self.INITIAL_CODE
        self.my_input = program_input
        self.goal = goal
        self.code_index_list = []

    # received an action to make from environment
    # TODO: fix the mess with if terminal assignement
    def _step(self, action_idx, actions):
        self.last_action = actions[action_idx]
        self.code += actions[action_idx]

        action_st = np.zeros(len(actions), dtype=int).tolist()
        action_st[action_idx] = 1;
        self.code_index_list+= [action_st]

        ex_res = self.get_execution_result()
        reward = 0.
        terminal = False

        if ex_res.output == self.goal:
            reward = 1.0
            terminal = True
        elif ex_res.raised_exception:
            reward = -1.0
             # if len(self.code_index_list) == 1000:
             #    reward = -1.0
             # elif ex_res.exception_type == 'TypeError':
             #     reward = -1.0             
             # elif ex_res.exception_type == 'IndentationError':
             #     reward = -1.0
             #    #terminal = True
             # elif ex_res.exception_type == 'SyntaxError':
             #     reward = 0.0
             #     #terminal = True
             # elif ex_res.exception_type == 'NameError':
             #    if self.code_index_list[0]==2:
             #        reward = -1.0
             #    else : 
             #        reward = -1.0
             # else:
             #     reward = 0.0
        elif len(ex_res.output) > 0:   
            reward = 0.2
        else:
            reward = 0.1

        self.last_reward = reward
        if len(self.code) > 255:
            terminal = True 

        return self.code, reward, terminal, {}
    
    def get_execution_result(self):
        ex_res = utils.ExecutionResult(output="", raised_exception=False, exception_type=None)

        with open(self.FILENAME, "w") as f:
            f.write(self.code)

        with stdoutIO() as s:
            try:
                sys.argv[1:] = []
                sys.argv.append(self.my_input)
                execfile(self.FILENAME)
            except Exception as e:
                ex_res.raised_exception = True
                ex_res.exception_type = type(e).__name__

        ex_res.output = s.getvalue()
        # remove newline added by print for convinience
        if len(ex_res.output) > 0:
            ex_res.output = ex_res.output[:-1]
        # print "Execution result: {}".format(ex_res)
        return ex_res

    def _reset(self):
        self.code = self.INITIAL_CODE
        return self.code

    def _render(self, mode='human', close=False):
        # print "Rendering Codegen environment..."
        print "Code: \n=====================\n{}\n=====================\n".format(self.code)
