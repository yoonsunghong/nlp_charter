# ## Define helper functions

def quickpickle_load(picklepath):
    '''Very time-efficient way to load pickle-formatted objects into Python.
    Uses C-based pickle (cPickle) and gc workarounds to facilitate speed. 
    Input: Filepath to pickled (*.pkl) object.
    Output: Python object (probably a list of sentences or something similar).'''

    with open(picklepath, 'rb') as loadfile:
        
        gc.disable() # disable garbage collector
        outputvar = cPickle.load(loadfile) # Load from picklepath into outputvar
        gc.enable() # enable garbage collector again
    
    return outputvar
def quickpickle_dump(dumpvar, picklepath):
    '''Very time-efficient way to dump pickle-formatted objects from Python.
    Uses C-based pickle (cPickle) and gc workarounds to facilitate speed. 
    Input: Python object (probably a list of sentences or something similar).
    Output: Filepath to pickled (*.pkl) object.'''

    with open(picklepath, 'wb') as destfile:
        
        gc.disable() # disable garbage collector
        cPickle.dump(dumpvar, destfile) # Dump dumpvar to picklepath
        gc.enable() # enable garbage collector again
    
    
def write_list(file_path, textlist):
    """Writes textlist to file_path. Useful for recording output of parse_school()."""
    
    with open(file_path, 'w') as file_handler:
        
        for elem in textlist:
            file_handler.write("{}\n".format(elem))
    
    return    


def load_list(file_path):
    """Loads list into memory. Must be assigned to object."""
    
    textlist = []
    with open(file_path) as file_handler:
        line = file_handler.readline()
        while line:
            textlist.append(line)
            line = file_handler.readline()
    return textlist
    
    
def write_sentence(sentence, file_path):
    """Writes sentence to file at file_path.
    Useful for recording first row's output of preprocess_wem() one sentence at a time.
    Input: Sentence (list of strings), path to file to save it
    Output: Nothing (saves to disk)"""
    
    with open(file_path, 'w+') as file_handler:
        for word in sentence: # Iterate over words in sentence
            if word == "":
                pass
            else:
                file_handler.write(word + " ") # Write each word on same line, followed by space
            
        file_handler.write("\n") # After sentence is fully written, close line (by inserting newline)
            
    return


def append_sentence(sentence, file_path):
    """Appends sentence to file at file_path. 
    Useful for recording each row's output of preprocess_wem() one sentence at a time.
    Input: Sentence (list of strings), path to file to save it
    Output: Nothing (saves to disk)"""

    with open(file_path, 'a+') as file_handler:
        for word in sentence: # Iterate over words in sentence
            if word == "":
                pass
            else:
                file_handler.write(word + " ") # Write each word on same line, followed by space
            
        file_handler.write("\n") # After sentence is fully written, close line (by inserting newline)
            
    return

    
def load_tokslist(file_path):
    """Loads from file and word-tokenizes list of "\n"-separated, possibly multi-word strings (i.e., sentences). 
    Output must be assigned to object.
    Input: Path to file with list of strings
    Output: List of word-tokenized strings, i.e. sentences"""
    
    textlist = []
    
    with open(file_path) as file_handler:
        line = file_handler.readline() # Read first line
        
        while line: # Continue while there's still a line to read
            textlist.append(word for word in word_tokenize(line)) # Tokenize each line by word while loading in
            line = file_handler.readline() # Read next line
            
    return textlist


def preprocess_wem(tuplist): # inputs were formerly: (tuplist, start, limit)
    
    '''This function cleans and tokenizes sentences, removing punctuation and numbers and making words into lower-case stems.
    Inputs: list of four-element tuples, the last element of which holds the long string of text we care about;
        an integer limit (bypassed when set to -1) indicating the DF row index on which to stop the function (for testing purposes),
        and similarly, an integer start (bypassed when set to -1) indicating the DF row index on which to start the function (for testing purposes).
    This function loops over five nested levels, which from high to low are: row, tuple, chunk, sentence, word.
    Note: This approach maintains accurate semantic distances by keeping stopwords.'''
        
    global mpdo # Check if we're doing multiprocessing. If so, then mpdo=True
    global words_by_sentence # Grants access to variable holding a list of lists of words, where each list of words represents a sentence in its original order (only relevant for this function if we're not using multiprocessing)
    global pcount # Grants access to preprocessing counter
    
    known_pages = set() # Initialize list of known pages for a school

    if type(tuplist)==float:
        return # Can't iterate over floats, so exit
    
    #print('Parsing school #' + str(pcount)) # Print number of school being parsed

    for tup in tuplist: # Iterate over tuples in tuplist (list of tuples)
        if tup[3] in known_pages or tup=='': # Could use hashing to speed up comparison: hashlib.sha224(tup[3].encode()).hexdigest()
            continue # Skip this page if exactly the same as a previous page on this school's website

        for chunk in tup[3].split('\n'): 
            for sent in sent_tokenize(chunk): # Tokenize chunk by sentences (in case >1 sentence in chunk)
                sent = clean_sentence(sent, remove_stopwords=True) # Clean and tokenize sentence
                
                if ((sent == []) or (len(sent) == 0)): # If sentence is empty, continue to next sentence without appending
                    continue
                
                # Save preprocessing sentence to file (if multiprocessing) or to object (if not multiprocessing)
                if mpdo:
                    try: 
                        if (os.path.exists(wordsent_path)) and (os.path.getsize(wordsent_path) > 0): 
                            append_sentence(sent, wordsent_path) # If file not empty, add to end of file
                        else:
                            write_sentence(sent, wordsent_path) # If file doesn't exist or is empty, start file
                    except FileNotFoundError or OSError: # Handle common errors when calling os.path functions on non-existent files
                        write_sentence(sent, wordsent_path) # Start file
                
                else:
                    words_by_sentence.append(sent) # If not multiprocessing, just add sent to object
                    
                    
        known_pages.add(tup[3])
    
    pcount += 1 # Add to counter
    
    return

def preprocess_wem(tuplist): # inputs were formerly: (tuplist, start, limit)
    
    '''This function cleans and tokenizes sentences, removing punctuation and numbers and making words into lower-case stems.
    Inputs: list of four-element tuples, the last element of which holds the long string of text we care about;
        an integer limit (bypassed when set to -1) indicating the DF row index on which to stop the function (for testing purposes),
        and similarly, an integer start (bypassed when set to -1) indicating the DF row index on which to start the function (for testing purposes).
    This function loops over five nested levels, which from high to low are: row, tuple, chunk, sentence, word.
    Note: This approach maintains accurate semantic distances by keeping stopwords.'''
        
    global mpdo # Check if we're doing multiprocessing. If so, then mpdo=True
    global sents_combined # Grants access to variable holding a list of lists of words, where each list of words represents a sentence in its original order (only relevant for this function if we're not using multiprocessing)
    global pcount # Grants access to preprocessing counter
    
    known_pages = set() # Initialize list of known pages for a school
    sents_combined = [] # Initialize list of all school's sentences

    if type(tuplist)==float:
        return # Can't iterate over floats, so exit
    
    #print('Parsing school #' + str(pcount)) # Print number of school being parsed

    for tup in tuplist: # Iterate over tuples in tuplist (list of tuples)
        if tup[3] in known_pages or tup=='': # Could use hashing to speed up comparison: hashlib.sha224(tup[3].encode()).hexdigest()
            continue # Skip this page if exactly the same as a previous page on this school's website

        for chunk in tup[3].split('\n'): 
            for sent in sent_tokenize(chunk): # Tokenize chunk by sentences (in case >1 sentence in chunk)
                sent = clean_sentence(sent) # Clean and tokenize sentence
                
                if ((sent == []) or (len(sent) == 0)): # If sentence is empty, continue to next sentence without appending
                    continue
                
                
                # TO DO: Chunk this by school, not just sentence
                # TO DO: Now that sentences are parsed and cleaned by spaces, 
                # recombine and then parse more accurately using spacy word tokenizer
                
                # Save preprocessing sentence to object (if not multiprocessing)
                #sents_combined.append(sent) # add sent to object #if nested works
                sents_combined.extend(sent) # if nested version doesnt work
                    
        known_pages.add(tup[3])
        
    school_sentslist.append(sents_combined) # add sent to object
    
    #pcount += 1 # Add to counter
    
    return sents_combined
