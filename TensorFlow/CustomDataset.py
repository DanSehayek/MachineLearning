import tensorflow as tf
import numpy as np
import random
import pickle
import nltk
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from collections import Counter

'''
lexicon is a list containing all of the unique words from all of our datafiles
lexicon = [chair,table,spoon,television]
I pulled the chair up to the table => [1 1 0 0]
'''

'''
We will use nltk for our natural language processing.
We are essentially attempting to apply a neural network to our own model.
word_tokenize just converts a sentence into a list of words/strings
WordNetLemmatizer removes "ed" and "ing" and other redundancies and returns
the root word
'''

lemmatizer = WordNetLemmatizer()
hm_lines = 10000000

def create_lexicon(pos,neg):
    lexicon = []
    for wordfile in [pos,neg]:
        with open(wordfile,"r") as f:
            contents = f.readlines()
            for l in contents[:hm_lines]:
                all_words = word_tokenize(l.lower())
                lexicon += list(all_words)

    lexicon = [lemmatizer.lemmatize(i) for i in lexicon]
    word_counts = Counter(lexicon)

    '''
    word_counts will return a dictionary containing each word and the number of
    its occurrences. Example: {"the":20,"and":50}
    We want the lexicon to be as short as possible so that our model does not
    blow up. Thus we will remove very common words such as "and" and "the"..
    '''

    l2 = []
    for word in word_counts:
        if 50 < word_counts[word] < 1000:
            l2.append(word)

    return l2

def sample_handling(sample,lexicon,classification):
    featureset = []

    '''
    Example Feature Set:
    [[[0 1 0 1 1 0],[0 1]],
     [[1 0 0 1 0 1],[0 1]]]

    Each list corresponds to a sentence/line.
    The first sublist shows 0s for words that were not present and 1s for
    words that were present. The second list is [1 0] if the classification
    is positive and [0 1] if the classification is negative.
    '''

    with open(sample,"r") as f:
        contents = f.readlines()
        for l in contents[:hm_lines]:
            current_words = word_tokenize(l.lower())
            current_words = [lemmatizer.lemmatize(i) for i in current_words]
            features = np.zeros(len(lexicon))
            for word in current_words:
                if word.lower() in lexicon:
                    index_value = lexicon.index(word.lower())
                    features[index_value] += 1
            features = list(features)
            featureset.append([features,classification])

    return featureset

def create_feature_sets_and_labels(pos,neg,test_size = 0.1):
    lexicon = create_lexicon(pos,neg)
    features = []
    features += sample_handling("Data/pos.txt",lexicon,[1,0])
    features += sample_handling("Data/neg.txt",lexicon,[0,1])
    random.shuffle(features)
    features = np.array(features)

    testing_size = int(test_size * len(features))
    train_x = list(features[:,0][:-testing_size])
    train_y = list(features[:,1][:-testing_size])
    test_x = list(features[:,0][-testing_size:])
    test_y = list(features[:,1][-testing_size:])

    return train_x,train_y,test_x,test_y

if __name__ == "__main__":
    train_x,train_y,test_x,test_y = create_feature_sets_and_labels("Data/pos.txt","Data/neg.txt")
    with open("sentiment_set.pickle","wb") as f:
        pickle.dump([train_x,train_y,test_x,test_y],f)
