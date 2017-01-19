# Pretrain LSTM for code generation

## Scrape the code from websites or libraries
In this process, some special words or characters (e.g. "def", "print", ":", "OSError", "{", "True") can be extracted from python codes and they are converted to list.

if you use the codes in website, 

    python webpage_scrape.py
or if you use the codes in libraries

    python library_scrape.py
    
Output files are "code refined list" (that is "lib_code_refined_list.pkl" or "webpage_code_refined_list.pkl")
and "id list" converted from "code refined list" (that is "python_corpus.pkl")

In order to get this result, it takes a lot of time. So you can download 
"lib_code_refined_list.pkl" from https://drive.google.com/open?id=0B33vEXpXOGfmSml1d0xyYlhuYm8
and 
"python_corpus.pkl" from https://drive.google.com/open?id=0B33vEXpXOGfmOUJ2eFE0Z3VLaUE
 
Only "python_corpus.pkl"  will be used to train the LSTM Network.
 
## Train LSTM using python corpus
Before you start training, please put the "python_corpus.pkl" in this directory and start training!
    python pretrain_lstm.py
 
Output files are put in "result"(default) directory.

You can download the model (ActionValue(n_vocab, 650)) file from https://drive.google.com/open?id=0B33vEXpXOGfmYkJBRGRUdllBNGM 

If you download, please put the "lstm_model.npz" in "result" directory.

You can check the model by using "play_lstm.py"

If you use one lstm layer, results will be below

    $ python play_lstm.py
    
        loading the data...
        Finished loading!
        >> def
        def:x:yxx
        x = x
        x
        x
        x
        >> print
        print:
         x = [x]
            x
        >> [
        [x]
        x = xx([y])
        x = [
        >> {
        {x: x}
                x = xx
        >> (
        (x x)
        x = [x x x x x
        >>

These are not good results, but the relationship between "(" and ")" , "{" and "}" seems to be learned! 

If I increase the number of lstm layers from one to two, better results can be obtained!

The model is ActionValue2(n_vocab, 650) and can be downloaded from https://drive.google.com/open?id=0B33vEXpXOGfmbDlPNkNfSWVIZUk 

    $ python play_lstm.py
        loading the data...
        Finished loading!
        >> def
        def x(x x x):
            x x
        >> print
        print(x)
        
        >> [
        [y] x x x x x x x x x
        >> {
        {x: x x: x}
        
        >> (
        (x) x x x x x x x x
        
        >>


