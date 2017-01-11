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

    def generate_codes(self, word_dict):
        good_codes = []
        good_codes_idx = []
        code = []
        code_idx = []
        for i in range(len(word_dict)):
            code.append(word_dict[i])
            code_idx.append(i)
            for j in range(len(word_dict)):
                code.append(word_dict[j])
                code_idx.append(j)
                for k in range(len(word_dict)):
                    code.append(word_dict[k])
                    code_idx.append(k)
                    for l in range(len(word_dict)):
                        code.append(word_dict[l])
                        code_idx.append(l)
                        for m in range(len(word_dict)):
                            code.append(word_dict[m])
                            code_idx.append(m)
                            code_str = ''.join(code)
                            if not self.raises_exception(code_str):
                                good_codes.append(code_str)
                                good_codes_idx.append(deepcopy(code_idx))
                            del code[-1]
                            del code_idx[-1]
                        del code[-1]
                        del code_idx[-1]
                    del code[-1]
                    del code_idx[-1]
                del code[-1]
                del code_idx[-1]
            del code[-1]
            del code_idx[-1]
        return good_codes_idx, good_codes


def main():
    print "Generating examples..."
    good_codes_idx, good_codes = Generator().generate_codes(["print", " ", "x", "+", "1"])    
    print "Good codes indexies: {}\n, codes: {}".format(good_codes_idx, good_codes)


if __name__ == "__main__":
    main()
