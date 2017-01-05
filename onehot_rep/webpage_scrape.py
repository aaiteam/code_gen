
# coding: utf-8

import urllib2 as urllib
import requests
import keyword
from bs4 import BeautifulSoup
import keyword
import re
import itertools
import cPickle as pickle

def code_extraction(input_file, output_file):

    """ read webpages from input_file, extract python codes from corresponding webpage and saved
    the refined codeword into output_file

    Args:
       input_file: from which txt file to read webpage list
       output_file: to which csv file to write codeword into
    """
    with open(input_file, "rb") as reader:
        output_list = []
        for line in reader:
            # read webpage and scrape python source codes from it
            webpage = line.rstrip()
            web_file = urllib.urlopen(webpage)
            soup = BeautifulSoup(web_file, "lxml")

            # all kinds of codeword to extract
            builtin_func = dir(__builtins__)  # builtin functions
            keyword_list = keyword.kwlist  # builtin keywords
            arith_operator = ['+', '-', '*', '/', '%', '//', '**']
            comp_operator = ['>', '<', '!=', '==', '>=', '<=']
            logic_operator = ['or', 'and', 'not']
            assign_operator = ['=', '+=', '-=', '*=', '/=', '%=', '//=', '**=']
            bracket = ['{', '}', '[', ']', '(', ')']
            special_symbol = [':']

            # symbols to be omitted
            omit_list = ['', ' ', ',', '\n', '"', ">>>", "...", "#", '.']

            filt = list(itertools.chain(builtin_func, keyword_list, arith_operator,
                        comp_operator, logic_operator, assign_operator, bracket, special_symbol))

            # delimiters to split string into list
            delimiters = [',',' ', '(', ')', '[', ']', ':', "...", '\n', '"', '.', '>>>']
            regexPattern = "(" + '|'.join(map(re.escape, delimiters)) + ")"

            code_set = soup.find_all("div", class_="highlight-python")
            for code_block in code_set:
                code_refine = []
                code_txt = code_block.get_text().encode('utf8')
                code_txt = code_txt.split('\n')
                for code_line in code_txt:
                    # extract only source code and omit substrings after #
                    code_line = code_line.split("#",1)[0]
                    # replace substring within "" with x
                    code_line = re.sub(r'\"(.+?)\"', "x", code_line)
                    if code_line.startswith(">>>") or code_line.startswith("..."):
                        code_list = re.split(regexPattern, code_line)
            #             print code_list
                        for symbol in code_list:
                            if symbol in filt:
                                code_refine.append(symbol)
                            elif symbol not in omit_list:
                                code_refine.append('x')
                #  if code_refine is not empty
                if code_refine:
#                     print code_refine
                    output_list.append(code_refine)
    print output_list
    with open(output_file,'wb') as f:
        pickle.dump(output_list,f)

if __name__ == '__main__':
    code_extraction("webpage_list.txt", "output.pkl")

import pandas as pd
s = pd.Series(['one', 'two', 'three', 'four', 'five', 'six'])
onehot = pd.get_dummies(s)
print onehot
