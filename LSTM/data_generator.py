from copy import deepcopy
import contextlib
import StringIO
import sys


@contextlib.contextmanager
def stdoutIO():
    old_stdout = sys.stdout
    stdout = StringIO.StringIO()
    sys.stdout = stdout
    yield stdout
    sys.stdout = old_stdout


class Generator:

    FILENAME = "temp.py"
    INPUT = 1
    INITIAL_CODE = """
import sys

x = int(sys.argv[1])

# desired code starts here
"""

    def __init__(self, word_dict=["print", " ", "x", "+", "1"]):
        self.word_dict = word_dict

    def raises_exception(self, code):
        with open(self.FILENAME, "w") as f:
            f.write(self.INITIAL_CODE)
            f.write(code)

        with stdoutIO() as s:
            try:
                sys.argv[1:] = []
                sys.argv.append(self.INPUT)
                execfile(self.FILENAME)
            except Exception as e:
                return True
   
        return False

    def check_and_add(self, code, good_codes):
        if not self.raises_exception(''.join([self.word_dict[i] for i in code])):
            good_codes.append(deepcopy(code))


    def append_word(self, code, idx, max_len, good_codes, need_shorter):
        if idx > 0 and need_shorter or idx == max_len:
            self.check_and_add(code, good_codes)

        if idx < max_len:
            idx += 1
            for i in range(len(self.word_dict)):
                code.append(i)
                self.append_word(code, idx, max_len, good_codes, need_shorter)
                del code[-1]

    def generate_codes(self, max_len=5, need_shorter=True):
        code = []
        good_codes = []
        self.append_word(code, 0, max_len, good_codes, need_shorter)
        return good_codes


def main():
    print "Generating examples..."
    good_codes = Generator(["print", " ", "x", "+", "1"]).generate_codes(max_len=5, need_shorter=True) 
    print "good codes: {}".format(good_codes) 
    for code in good_codes:
        code_str = ''.join([["print", " ", "x", "+", "1"][i] for i in code])
        print code_str


if __name__ == "__main__":
    main()
