#this script prepares text for infersent model scripting. 
#this script also uses a file under the name filtered_index.csv, which contains all articles names.
#it also uses text pickle files generated from text_nest_list_save.py, which is not included in this repo because of the large file size.
from gensim.test.utils import get_tmpfile
import os
import numpy as np
import pandas as pd
import gensim
import ast
import tqdm
from os import listdir
from os.path import isfile, join
import re
import sys; sys.path.insert(0, "../functions/")
from clean_text import stopwords_make, punctstr_make, unicode_make, get_common_words, clean_sentence_apache
from quickpickle import quickpickle_load

print("Imports completed!")

cwd = os.getcwd()
ocr_wd = cwd.replace('Computational-Analysis-For-Social-Science/WordEmbedding/other_scripts', 'jstor_data/ocr')
#files = ['../../../jstor_data/ocr/' + f for f in listdir(cwd) if isfile(join(cwd, f))]
colnames = ['file_name']
print("Placeholder 1")
articles = pd.read_csv("../../../models_storage/word_embeddings_data/filtered_index.csv", names=colnames, header=None)
print("Placeholder 2")

filename_ls = articles['file_name']


print("Loading pickle file of the nested, cleaned sentences...")
whole_text = quickpickle_load("../../../models_storage/word_embeddings_data/cleaned_text_nested_nov21.pkl")
print("Pickle file loaded as nested list!")

sent_list_collapsed = []
for ls in whole_text:
    article = []
    for sent in ls:
        article.extend(sent)
    sent_list_collapsed.append(article)

sent_list_collapsed = [article[:10000] for article in sent_list_collapsed]  #cutting down to 10000 words max   
    
d = {'filename': filename_ls, 'text': sent_list_collapsed}
df = pd.DataFrame(d)


df["edited_filename"] = df['filename'].apply(lambda x: x[16:])
        
df.to_csv("../../../models_storage/word_embeddings_data/ocr_text_with_tags_10000_jan21.csv")
print("Saved the data-frame with cleaned, truncated texts!")