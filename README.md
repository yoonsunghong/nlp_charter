This repo depicts some of the coding/data science work I did while co-authoring *Applying
Expert-Built Dictionaries for Automated Analysis of Academic Texts.*
This repo contains scripts for helper functions, text preprocessing, modeling, and visualizations.
Data files themselves are not included in this repo as the data sizes are too big; instead, these scripts demonstrate coding steps taken to preprocess the data, train the models, and visualize results.  

**Functions** folder contains python files for helper functions.  
**Preprocessing** folder contains two scripts for text preprocessing: one for creating cleaned text file for doc2vec/word2vec input format (**text_nest_list_save**), and one for formatting the pickled file data specifically for infersent input format (**infersent_text_prepare**).  
**Model_training** folder contains three scripts, for each associated model: **doc2vec_train_save.py**, **infersent.py**, and **word2vec_train_save.py**.  
**Visualization** folder contains jupyter notebooks for visualizations. **correlation_plot.ipynb** contains simple visualizations for average cosine similarity scores for different models and distribution of key-principle words in articles. **negative_binomial.ipynb** contains negative binomial regression computations and its associated pseudo R-squared value plots.