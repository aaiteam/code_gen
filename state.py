import numpy as np

INITIAL_CODE = """
import sys

input = []
for arg in sys.argv[1:]:
    input.append(arg)

# desired code starts here
"""

ACTIONS = [" ", "print", "input"]

class State:
    # maybe only observation?
    # observation - code output
    n_max_act_len = 3
    def __init__(self):
        self.code = INITIAL_CODE
        self.action_idxes = -np.ones(self.n_max_act_len).astype(np.float32)[np.newaxis, :]
        self.cnt = 0


    def reset_state(self):
        self.cnt = 0
        self.action_idxes = -np.ones(self.n_max_act_len).astype(np.float32)[np.newaxis, :]
        self.code = INITIAL_CODE

    def set_action_idx(self, idx):
        if self.cnt < self.n_max_act_len:
            self.action_idxes[0, self.cnt] = idx
            self.cnt += 1
            self.code += ACTIONS[idx]
        else:
           # raise IndexError, "code is too long!"
            print ("code is now too long, so reset the state")
            self.reset_state()

