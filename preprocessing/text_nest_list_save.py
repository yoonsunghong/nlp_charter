#this version of the script uses phrase detection both manually (for dictionary terms) and through the function. Later these may need to be turned off for comparison (e.g. comparing performance of word2vec to GloVe or InferSent, which doesn't implement Phrases detection) 
#the script would originally parse through a folder containing all academic articles, in ocr folder.)
import time
start = time.time()
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
import datetime # For workin g with dates & times

# Import packages for cleaning, tokenizing, and stemming text
import re # For parsing text
from unicodedata import normalize # for cleaning text by converting unicode character encodings into readable format
from nltk import word_tokenize, sent_tokenize # widely used text tokenizer
from nltk.stem.porter import PorterStemmer # an approximate method of stemming words (it just cuts off the ends)
from nltk.stem.porter import PorterStemmer # approximate but effective (and common) method of normalizing words: stems words by implementing a hierarchy of linguistic rules that transform or cut off word endings
stem = PorterStemmer().stem # Makes stemming more accessible
from nltk.corpus import stopwords # for eliminating stop words
import gensim # For word embedding models
from gensim.models.phrases import Phrases, Phraser # Makes word2vec more robust: Looks not just at  To look for multi-word phrases within word2vec

import string # for one method of eliminating punctuation
from nltk.corpus import stopwords # for eliminating stop words
from sklearn.feature_extraction import text
from nltk.stem.porter import PorterStemmer; ps = PorterStemmer() # approximate but effective (and common) method of stemming words

#setting up multiprocessing
import multiprocessing
from sklearn import utils
cores = multiprocessing.cpu_count()

# Import packages for multiprocessing
import os # For navigation
numcpus = len(os.sched_getaffinity(0)) # Detect and assign number of available CPUs
from multiprocessing import Pool # key function for multiprocessing, to increase processing speed
pool = Pool(processes=numcpus) # Pre-load number of CPUs into pool function
import Cython # For parallelizing word2vec
mpdo = False # Set to 'True' if using multiprocessing--faster for creating words by sentence file, but more complicated
nltk.download('stopwords')
nltk.download('punkt')
nltk.download('words')


from os import listdir
from os.path import isfile, join

import sys; sys.path.insert(0, "../functions/")
from clean_text import stopwords_make, punctstr_make, unicode_make, get_common_words, clean_sentence_apache
from quickpickle import quickpickle_dump

cwd = os.getcwd()
ocr_wd = cwd.replace('Computational-Analysis-For-Social-Science/WordEmbedding/other_scripts', 'jstor_data/ocr')
#files = ['../../../jstor_data/ocr/' + f for f in listdir(cwd) if isfile(join(cwd, f))]
colnames = ['file_name']
articles = pd.read_csv("../../../models_storage/word_embeddings_data/filtered_index.csv", names=colnames, header=None)
files_to_be_opened = ["../../../jstor_data/ocr/" + file + '.txt' for file in articles.file_name]
all_files = ['../../../jstor_data/ocr/' + f for f in listdir(ocr_wd) if isfile(join(ocr_wd, f))]

files = [file for file in all_files if file in files_to_be_opened]

#loading the dictionaries for manual Multiple-phrase terms substitution
culture = pd.read_csv("../../../models_storage/word_embeddings_data/Culture_full.csv", sep='\n', header=None)
culture.columns = ["vocab"]
demographic = pd.read_csv("../../../models_storage/word_embeddings_data/Demographic_full.csv", sep='\n', header=None)
demographic.columns = ["vocab"]
relational = pd.read_csv("../../../models_storage/word_embeddings_data/Relational_full.csv", sep='\n', header=None)
relational.columns = ["vocab"]

culture.vocab = culture.vocab.apply(lambda x: re.sub(',', '_', x))
demographic.vocab = demographic.vocab.apply(lambda x: re.sub(',', '_', x))
relational.vocab = relational.vocab.apply(lambda x: re.sub(',', '_', x))
#Merging them to create a master array of multigrams
multigram = np.append(culture.vocab[np.array(["_" in word for word in culture.vocab])],
demographic.vocab[np.array(["_" in word for word in demographic.vocab])])
multigram = np.append(multigram, 
                     relational.vocab[np.array(["_" in word for word in relational.vocab])])


#initializing two lists for strings from files and the filenames
text_ls = []
filename_ls = []
for file in files: #using sample only for cmputational speed purposes, change files_sample --> files for script
    try:
        with open(file, 'r') as myfile:
            data = myfile.read()
        data = data.replace('<plain_text><page sequence="1">', '')
        data = re.sub(r'</page>(\<.*?\>)', ' \n ', data)
        for word in multigram: #iterating through every multigram word and manually substituting
            data = data.replace(word.replace("_", " "), word) 
        text_ls.append(data)
        filename_ls.append(file[40:-4])
    except:
        print(file[40:-4], ", doesn't exist in ocr folder. Passing...")
    
print("Text Files Loading Complete!")

d = {'filename': filename_ls, 'text': text_ls}
df = pd.DataFrame(d)



# Prep dictionaries of English words
from nltk.corpus import words # Dictionary of 236K English words from NLTK
english_nltk = set(words.words()) # Make callable
english_long = set() # Dictionary of 467K English words from https://github.com/dwyl/english-words
fname =  "../../../models_storage/word_embeddings_data/english_words.txt" # Set file path to long english dictionary
with open(fname, "r") as f:
    for word in f:
        english_long.add(word.strip())
        

# ## Create lists of stopwords, punctuation, and unicode characters
stop_words_list = stopwords_make() # Define old vocab file path if you want to remove first, dirty elements
unicode_list = unicode_make()
punctstr = punctstr_make()

print("Stopwords, Unicodes, Punctuations lists creation complete!")


#word2vec computation
whole_text_unnested = []
whole_text_nested = []
tqdm.pandas(desc="Cleaning text")


for school in tqdm(df['text'], desc="Cleaning text"):
    doc = []
    for chunk in school.split("\n"):
        for sent in sent_tokenize(chunk):
            sent = clean_sentence_apache(sent, unhyphenate=True, remove_propernouns=False, remove_acronyms=False)
            sent = [word for word in sent if word != '']
            if len(sent) > 0:
                whole_text_unnested.append(sent)
                doc.append(sent)
    whole_text_nested.append(doc)

print("Saving the Cleaned Sentences as lists...")
print("Saving List 1: Flattened list")
quickpickle_dump(whole_text_unnested, "../../../models_storage/word_embeddings_data/cleaned_text_flat_oct12.pkl")
print("Pickle file 1 saved!")
print("Saving List 2: Nested list")
quickpickle_dump(whole_text_nested, "../../../models_storage/word_embeddings_data/cleaned_text_nested_oct12.pkl")
print("Pickle file 2 saved!")
print("Total Time Taken: ", (time.time() - start)/60, " Minutes.")
