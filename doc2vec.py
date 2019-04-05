#!/usr/bin/env python
# -*- coding: UTF-8

# Word Embedding Models: Preprocessing and Doc2Vec Model Training
# Objective: Creating, training, and saving DMM/DBOW Doc2Vec Models.
# Objective 2: Concatenating DMM & DBOW Model for new Doc2Vec model.
# Project title: Charter school identities 
# Creator: Yoon Sung Hong

# Import general packages
import imp, importlib # For working with modules
import nltk # for natural language processing tools
import pandas as pd # for working with dataframes
#from pandas.core.groupby.groupby import PanelGroupBy # For debugging
import numpy as np # for working with numbers
import pickle # For working with .pkl files
from tqdm import tqdm # Shows progress over iterations, including in pandas via "progress_apply"
import sys # For terminal tricks
import _pickle as cPickle # Optimized version of pickle
import gc # For managing garbage collector
import timeit # For counting time taken for a process
import datetime # For working with dates & times

# Import packages for cleaning, tokenizing, and stemming text
import re # For parsing text
from unicodedata import normalize # for cleaning text by converting unicode character encodings into readable format
from nltk import word_tokenize, sent_tokenize # widely used text tokenizer
from nltk.stem.porter import PorterStemmer # an approximate method of stemming words (it just cuts off the ends)
from nltk.stem.porter import PorterStemmer # approximate but effective (and common) method of normalizing words: stems words by implementing a hierarchy of linguistic rules that transform or cut off word endings
stem = PorterStemmer().stem # Makes stemming more accessible
from nltk.corpus import stopwords # for eliminating stop words
import gensim # For word embedding models
from gensim.models.phrases import Phrases # Makes word2vec more robust: Looks not just at  To look for multi-word phrases within word2vec

# Import packages for multiprocessing
import os # For navigation
numcpus = len(os.sched_getaffinity(0)) # Detect and assign number of available CPUs
from multiprocessing import Pool # key function for multiprocessing, to increase processing speed
pool = Pool(processes=numcpus) # Pre-load number of CPUs into pool function
import Cython # For parallelizing word2vec
mpdo = False # Set to 'True' if using multiprocessing--faster for creating words by sentence file, but more complicated
nltk.download('stopwords')
nltk.download('punkt')

# For loading functions from files in data_tools directory:
from clean_text import clean_sentence, stopwords_make, punctstr_make, unicode_make
import helper_func

# ## Create lists of stopwords, punctuation, and unicode characters
stop_words_list = stopwords_make() # Define old vocab file path if you want to remove first, dirty elements
unicode_list = unicode_make()
punctstr = punctstr_make()

print("Sentence cleaning preliminaries complete...")


# ## Prepare to read data

## Did not include data in this repo since file sizes too big ##
## 
##
##

# Define file paths
if mpdo:
    wordsent_path = "data/wem_wordsent_data_train250_nostem_unlapped_clean2.txt"
else:
    wordsent_path = "data/wem_wordsent_data_train250_nostem_unlapped_clean2.pkl"
charters_path = "data/charters_2015.pkl" # All text data; only charter schools (regardless if open or not)
phrasesent_path = "data/wem_phrasesent_data_train250_nostem_unlapped_clean2.pkl"
#wemdata_path = "../data/wem_data.pkl"
model_path = "data/wem_model_train250_nostem_unlapped_300d_clean2.bin"
vocab_path = "data/wem_vocab_train250_nostem_unlapped_300d_clean2.txt"
vocab_path_old = "data/wem_vocab_train250_nostem_unlapped_300d_clean.txt"

# Check if sentences data already exists, to save time
try:
    if (os.path.exists(wordsent_path)) and (os.path.getsize(wordsent_path) > 10240): # Check if file not empty (at least 10K)
        print("Existing sentence data detected at " + str(os.path.abspath(wordsent_path)) + ", skipping preprocessing sentences.")
        sented = True
    else:
        sented = False
except FileNotFoundError or OSError: # Handle common errors when calling os.path.getsize() on non-existent files
    sented = False

# Check if sentence phrases data already exists, to save time
try:
    if (os.path.exists(phrasesent_path)) and (os.path.getsize(phrasesent_path) > 10240): # Check if file not empty (at least 10K)
        print("Existing sentence + phrase data detected at " + str(os.path.abspath(phrasesent_path)) + ", skipping preprocessing sentence phrases.")
        phrased = True
    else:
        phrased = False
except FileNotFoundError or OSError: # Handle common errors when calling os.path.getsize() on non-existent files
    phrased = False


# ## Preprocessing I: Tokenize web text by sentences

df = quickpickle_load(charters_path) # Load charter data into DF
print("DF loaded from " + str(charters_path) + "...")

if phrased: 
    pass # If parsed sentence phrase data exists, don't bother with tokenizing sentences

elif sented: # Check if tokenized sentence data already exists. If so, don't bother reparsing it
    words_by_sentence = []
    
    # Load data back in for parsing phrases and word embeddings model:
    if mpdo:
        words_by_sentence = load_tokslist(wordsent_path) 
    else:
        words_by_sentence = quickpickle_load(wordsent_path) 

else:
    
    words_by_sentence = [] # Initialize variable to hold list of lists of words (sentences)
    pcount=0 # Initialize preprocessing counter
    df["WEBTEXT"] = df["WEBTEXT"].astype(list) # Coerce these to lists in order to avoid type errors

    # Convert DF into list (of lists of tuples) and call preprocess_wem on element each using Pool():
    try:
        tqdm.pandas(desc="Tokenizing sentences") # To show progress, create & register new `tqdm` instance with `pandas`

        # WITH multiprocessing (faster):
        if mpdo:
            weblist = df["WEBTEXT"].tolist() # Convert DF into list to pass to Pool()

            # Use multiprocessing.Pool(numcpus) to run preprocess_wem:
            print("Preprocessing web text into list of sentences...")
            if __name__ == '__main__':
                with Pool(numcpus) as p:
                    p.map(preprocess_wem, tqdm(weblist, desc="Tokenizing sentences")) 

        # WITHOUT multiprocessing (much slower):
        else:
            df["WEBTEXT"].progress_apply(lambda tups: preprocess_wem(tups))

            # Save data for later
            try: # Use quickpickle to dump data into pickle file
                if __name__ == '__main__': 
                    print("Saving list of tokenized sentences to file...")
                    t = timeit.Timer(stmt="quickpickle_dump(words_by_sentence, wordsent_path)", globals=globals())
                    print("Time elapsed saving data: " + str(round(t.timeit(1),4)),'\n')

                '''with open(wordsent_path, 'wb') as destfile:
                    gc.disable() # Disable garbage collector to increase speed
                    cPickle.dump(words_by_sentence, destfile)
                    gc.enable() # Enable garbage collector again'''

            except Exception as e:
                print(str(e), "\nTrying backup save option...")
                try:
                    # Slower way to save data:
                    with open(wordsent_path, 'wb') as destfile:
                        t = timeit.Timer(stmt="pickle.dump(words_by_sentence, destfile)", globals=globals())
                        print("Success! Time elapsed saving data: " + str(round(t.timeit(1),4)),'\n')

                except Exception as e:
                    print("Failed to save sentence data: " + str(e))

    except Exception as e:
        print("Failed to tokenize sentences: " + str(e))
        sys.exit()
        
    
# ## Preprocessing II: Detect and parse common phrases in words_by_sentence

if phrased: # Check if phrased data already exists. If so, don't bother recalculating it
    words_by_sentence = []
    words_by_sentence = quickpickle_load(phrasesent_path) # Load data back in, for word embeddings model

else:
    tqdm.pandas(desc="Parsing phrases") # Change title of tqdm instance

    try:
        print("Detecting and parsing phrases in list of sentences...")
        # Threshold represents a threshold for forming the phrases (higher means fewer phrases). A phrase of words a and b is accepted if (cnt(a, b) - min_count) * N / (cnt(a) * cnt(b)) > threshold, where N is the total vocabulary size. By default this value is 10.0
        phrases = Phrases(words_by_sentence, min_count=3, delimiter=b'_', common_terms=stop_word_list, threshold=8) # Detect phrases in sentences based on collocation counts
        words_by_sentence = [phrases[sent] for sent in tqdm(words_by_sentence, desc="Parsing phrases")] # Apply phrase detection model to each sentence in data

    except Exception as e:
        print("Failed to parse sentence phrases: " + str(e))
        sys.exit()
    
    # Save data for later
    try: # Use quickpickle to dump data into pickle file
        if __name__ == '__main__': 
            print("Saving list of tokenized, phrased sentences to file...")
            t = timeit.Timer(stmt="quickpickle_dump(words_by_sentence, phrasesent_path)", globals=globals())
            print("Time elapsed saving data: " + str(round(t.timeit(1),4)),'\n')
                                         
        '''with open(phrasesent_path, 'wb') as destfile:
            gc.disable() # Disable garbage collector to increase speed
            cPickle.dump(words_by_sentence, destfile)
            gc.enable() # Enable garbage collector again'''

    except Exception as e:
        print(str(e), "\nTrying backup save option...")
        try:
            # Slower way to save data:
            with open(phrasesent_path, 'wb') as destfile:
                t = timeit.Timer(stmt="pickle.dump(words_by_sentence, destfile)", globals=globals())
                print("Success! Time elapsed saving data: " + str(round(t.timeit(1),4)),'\n')

        except Exception as e:
            print("Failed to save parsed sentence phrases: " + str(e))
        

# Take a look at the data 
print("Sample of the first 10 sentences:")
print(words_by_sentence[:10])


docs_tagged = []
for school in df['WEBTEXT']:
    doc = preprocess_wem(school)
    T = gensim.models.doc2vec.TaggedDocument(doc,[''.join(doc)])
    docs_tagged.append(T)
    
#setting up multiprocessing
import multiprocessing
from sklearn import utils
cores = multiprocessing.cpu_count()

# building vocab for doc2vec - dbow model
print("Building dbow model...")
model_dbow = gensim.models.Doc2Vec(dm=0, vector_size=300, window=8, negative=5, hs=0, min_count=50, sample = 0, workers=cores)
model_dbow.build_vocab(docs_tagged)
print("dbow model built successfully!")
#training the model, epoch 30
for epoch in range(30):
    model_dbow.train(utils.shuffle(docs_tagged), total_examples=len(docs_tagged), epochs=1)
    model_dbow.alpha -= 0.002
    model_dbow.min_alpha = model_dbow.alpha
print("dbow model trained successfully!")

#building vocab for doc2vec - dmm model
print("Buildling dmm model...")
model_dmm = gensim.models.Doc2Vec(dm=1, dm_mean=1, vector_size=300, window=8, negative=5, min_count=50, workers=cores, alpha=0.025, min_alpha=0.065)
model_dmm.build_vocab(docs_tagged)
print("dmm model built successfully!")
for epoch in range(30):
    model_dmm.train(utils.shuffle(docs_tagged), total_examples=len(docs_tagged), epochs=1)
    model_dmm.alpha -= 0.002
    model_dmm.min_alpha = model_dmm.alpha
print("dmm model trained successfully!")

#saving the dmm and dbow models
from gensim.test.utils import get_tmpfile
cwd = os.getcwd()
fname_dbow = get_tmpfile(cwd + "/dbow_model")
model_dbow.save(fname_dbow)
fname_dmm = get_tmpfile(cwd + "/dmm_model")
model_dmm.save(fname_dmm)

#concatenating the two models
#deleting temporary training data to free up RAM
model_dbow.delete_temporary_training_data(keep_doctags_vectors=True, keep_inference=True)
model_dmm.delete_temporary_training_data(keep_doctags_vectors=True, keep_inference=True)
from gensim.test.test_doc2vec import ConcatenatedDoc2Vec
new_model = ConcatenatedDoc2Vec([model_dbow, model_dmm])
print("new model created successfully!")


