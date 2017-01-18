# coding: utf-8

import os
import keyword
import itertools
import re
import pickle
from progressbar import ProgressBar ## to use: $ pip install progressbar2

def find_all_files(directory):
    for root, dirs, files in os.walk(directory):
        yield root
        for file in files:
            yield os.path.join(root, file)


def get_python_filenames_from_libraries(library_name):
    exec ("import " + library_name)
    library_path, _ = os.path.split(eval(library_name).__file__)
    py_fn_list = []
    for fn in find_all_files(library_path):
        _, extension = os.path.splitext(fn)
        if extension == '.py' and "__init__" not in fn:
            py_fn_list.append(fn)
    return py_fn_list


def keywords_extraction_from_file(input_fn):
    """ split the python code of input_fn to words and change them to keywords
    
    Args:
       input_fn: .py filename
    Return:
       code_refined_list: keyword list
    """

    # all kinds of codeword to extract
    builtin_func = dir(__builtins__)  # builtin functions
    keyword_list = keyword.kwlist  # builtin keywords
    arith_operator = ['+', '-', '*', '/', '%', '//', '**']
    comp_operator = ['>', '<', '!=', '==', '>=', '<=']
    logic_operator = ['or', 'and', 'not']
    assign_operator = ['=', '+=', '-=', '*=', '/=', '%=', '//=', '**=']
    bracket = ['{', '}', '[', ']', '(', ')']
    special_symbol = [':']
    useful_delimiters = [' ', '\n']
    digit_expression = ['y']

    # symbols to be omitted
    omit_list = ['', ',', '"', ">>>", "...", "#", '.']

    filt = list(itertools.chain(builtin_func, keyword_list, arith_operator,
                                comp_operator, logic_operator, assign_operator, bracket, special_symbol,
                                useful_delimiters,
                                digit_expression))

    # delimiters to split string into list
    delimiters = [',', ' ', '(', ')', '[', ']', '{', '}', ':', "...", '\n', '"', '.', '>>>']
    regexPattern = "(" + '|'.join(map(re.escape, delimiters)) + ")"

    f = open(input_fn, 'r')
    code_refine_list = []
    for code_line in f:
        # extract only source code and omit substrings after # 
        code_line = code_line.split("#", 1)[0]
        if code_line == '\n':
            continue
        # replace substring within "" with x
        code_line = re.sub(r'\"(.+?)\"', "x", code_line)
        code_list = re.split(regexPattern, code_line)

        for symbol in code_list:
            if symbol in filt:
                code_refine_list.append(symbol)
            elif symbol.isdigit():
                code_refine_list.append(digit_expression[0])
            elif symbol not in omit_list:
                code_refine_list.append('x')
    f.close()

    return code_refine_list


def code_extraction(library_name_list, output_filename):
    """ read python files from library_name and saved the refined codeword into output_filename
    
    Args:
       library_name_list: target library names like ['numpy', 'sklearn', ...]
       output_filename: .pkl filename that processed list is written
    """
    output_list = []
    for library_name in library_name_list:
        py_fn_list = get_python_filenames_from_libraries(library_name)
        n_file = len(py_fn_list)
        p = ProgressBar(max_value=n_file)
        for i, py_fn in enumerate(py_fn_list):
            output_list.append(keywords_extraction_from_file(py_fn))
            p.update(i + 1)

        print ("{}: {} python files have been processed! ".format(library_name, n_file))

    with open(output_filename, 'wb') as f:
        pickle.dump(output_list, f)


def convert2onehot(input_file):
    """ read codeword examples and convert them to onehot representation,
    save the converted ones into data_onehot as list with each element of a numpy matrix 
    
    Args:
        input_file: from where to read .pkl file
    
    Returns:
        data_onehot: data with onehot representation 
        data_index: data with index representation 
    """
    import pandas as pd
    import numpy as np

    # load .pkl file 
    with open(input_file, "rb") as f:
        data = pickle.load(f)
    # flatten multi-level list into single-level list    
    data_list = [element for lst in data for element in lst]
    # obtain one-hot representation of each codeword
    s = pd.Series(data_list)
    onehot = pd.get_dummies(s)
    codebook = list(onehot.columns.values)
    onehot = onehot.as_matrix()
    # save index of each codeword in index(list)
    index = [int(np.nonzero(row)[0][0]) for row in onehot]
    num_codeword = onehot.shape[1]
    data_onehot = []
    data_index = []
    cnt = 0
    for lst in data:
        lst_onehot = np.empty((0, num_codeword), dtype=int)
        lst_index = []
        for element in lst:
            # lst_onehot = np.concatenate((lst_onehot, onehot[cnt, :][np.newaxis, :]), axis=0)
            lst_index.append(index[cnt])
            cnt += 1
        # data_onehot.append(lst_onehot)
        data_index.append(lst_index)
    return data_onehot, data_index, codebook


if __name__ == '__main__':

    ### Necessary to set these params ###
    pretrain_library_name_list = ['pandas', 'numpy', 'scipy', 'sklearn', 'chainer']
    output_filename = "lib_code_refined_list.pkl"
    ##############################

    code_extraction(pretrain_library_name_list, output_filename)
    _, data_index, codebook = convert2onehot(output_filename)

    res = {
        'pretrain_library_name_list': pretrain_library_name_list,
        'data_index': data_index,
        'codebook': codebook
    }
    with open("python_corpus.pkl", 'wb') as f:
        pickle.dump(res, f)

    print 'data_index\n', data_index
    print 'codebook\n', codebook
    print 'max index', max([max(x) for x in data_index])
