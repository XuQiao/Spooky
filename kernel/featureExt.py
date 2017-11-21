import nltk
import string
from nltk.corpus import stopwords

def unique_word_fraction(text):
    """function to calculate the fraction of unique words on total words of the text"""
    text_splited = text.split(' ')
    text_splited = [''.join(c for c in s if c not in string.punctuation) for s in text_splited]
    text_splited = [s for s in text_splited if s]
    word_count = text_splited.__len__()
    unique_count = list(set(text_splited)).__len__()
    return (unique_count/word_count)


eng_stopwords = set(stopwords.words("english"))
def stopwords_count(text):
    """ Number of stopwords fraction in a text"""
    text = text.lower()
    text_splited = text.split(' ')
    text_splited = [''.join(c for c in s if c not in string.punctuation) for s in text_splited]
    text_splited = [s for s in text_splited if s]
    word_count = text_splited.__len__()
    stopwords_count = len([w for w in text_splited if w in eng_stopwords])
    return (stopwords_count/word_count)


def punctuations_fraction(text):
    """functiopn to claculate the fraction of punctuations over total number of characters for a given text """
    char_count = len(text)
    punctuation_count = len([c for c in text if c in string.punctuation])
    return (punctuation_count/char_count)


def char_count(text):
    """function to return number of chracters """
    return len(text)

def fraction_noun(text):
    """function to give us fraction of noun over total words """
    text_splited = text.split(' ')
    text_splited = [''.join(c for c in s if c not in string.punctuation) for s in text_splited]
    text_splited = [s for s in text_splited if s]
    word_count = text_splited.__len__()
    pos_list = nltk.pos_tag(text_splited)
    noun_count = len([w for w in pos_list if w[1] in ('NN','NNP','NNPS','NNS')])
    return (noun_count/word_count)

def fraction_adj(text):
    """function to give us fraction of adjectives over total words in given text"""
    text_splited = text.split(' ')
    text_splited = [''.join(c for c in s if c not in string.punctuation) for s in text_splited]
    text_splited = [s for s in text_splited if s]
    word_count = text_splited.__len__()
    pos_list = nltk.pos_tag(text_splited)
    adj_count = len([w for w in pos_list if w[1] in ('JJ','JJR','JJS')])
    return (adj_count/word_count)

def fraction_verbs(text):
    """function to give us fraction of verbs over total words in given text"""
    text_splited = text.split(' ')
    text_splited = [''.join(c for c in s if c not in string.punctuation) for s in text_splited]
    text_splited = [s for s in text_splited if s]
    word_count = text_splited.__len__()
    pos_list = nltk.pos_tag(text_splited)
    verbs_count = len([w for w in pos_list if w[1] in ('VB','VBD','VBG','VBN','VBP','VBZ')])
    return (verbs_count/word_count)

import random

class Dictogram(dict):
    def __init__(self, iterable=None):
        """Initialize this histogram as a new dict; update with given items"""
        super(Dictogram, self).__init__()
        self.types = 0  # the number of distinct item types in this histogram
        self.tokens = 0  # the total count of all item tokens in this histogram
        if iterable:
            self.update(iterable)

    def update(self, iterable):
        """Update this histogram with the items in the given iterable"""
        for item in iterable:
            if item in self:
                self[item] += 1
                self.tokens += 1
            else:
                self[item] = 1
                self.types += 1
                self.tokens += 1

    def count(self, item):
        """Return the count of the given item in this histogram, or 0"""
        if item in self:
            return self[item]
        return 0

    def return_random_word(self):
        # Another way:  Should test: random.choice(histogram.keys())
        random_key = random.sample(self, 1)
        return random_key[0]

    def return_weighted_random_word(self):
        # Step 1: Generate random number between 0 and total count - 1
        random_int = random.randint(0, self.tokens-1)
        index = 0
        list_of_keys = self.keys()
        # print 'the random index is:', random_int
        for i in range(0, self.types):
            index += self[list_of_keys[i]]
            # print index
            if(index > random_int):
                # print list_of_keys[i]
                return list_of_keys[i]

# markov chain based features, 3 words memory 
def make_higher_order_markov_model(order, data):
    markov_model = dict()

    for i in range(0, len(data)-order):
        # Create the window
        window = tuple(data[i: i+order])
        # Add to the dictionary
        if window in markov_model:
            # We have to just append to the existing Dictogram
            markov_model[window].update([data[i+order]])
        else:
            markov_model[window] = Dictogram([data[i+order]])
    return markov_model

#train_df.loc[train_df['author']=='EAP'].shape[0]
def tokenixed_list(text):
    """function to calculate the fraction of unique words on total words of the text"""
    text_splited = text.split(' ')
    text_splited = [''.join(c for c in s if c not in string.punctuation) for s in text_splited]
    text_splited = [s for s in text_splited if s]
    return (text_splited)
    
#probablity calculation using markov chain
#print(eap_MM[('no', 'means', 'of')])
#print(sum([x for x in eap_MM[('no', 'means', 'of')].values()]))
def spit_out_prob(tupl , word , mm1 , mm2, mm3):
    """function to spit out a prob of a word given a tuple and a word and a markov model"""
    """fix this function, its working but not calculating probabilities, but providing
    some confidance index, greater than confidance index is, larger is the prob"""
    try:
        a = mm1[tupl][word]
    except:
        a = 0 
    try:
        b = mm2[tupl][word]
    except:
        b = 0
    try:
        c = mm3[tupl][word]
    except:
        c = 0
    try:
        prob_word = a/(a+b+c)#(sum([x for x in mm[tupl].values()]))
        return (prob_word)
    except:
        return (1)

def make_tupl(sentence , n=3):
    """function to make tuples of n size given a sentence and n"""
    list_of_tuple = []
    word_1 = []
    for i in list(range(sentence.__len__()-3)):
        tuple_1 = (sentence[i], sentence[i+1], sentence[i+2])
        list_of_tuple.append(tuple_1)
        word_1.append(sentence[i+3])
    return (list_of_tuple, word_1)


def sent_to_prob_eap(sentence, mm1 , p_eap , mm2, mm3):
    """function to get the markov model to give prob of a author given a sentence """
    list_of_tuples = make_tupl(sentence, n =3)[0]
    words = make_tupl(sentence, n =3)[1]
    #print(list_of_tuples, words)
    p = p_eap 
    for i in list(range(words.__len__())):
        #print(list_of_tuples[i], words[i])
        p = p*spit_out_prob(list_of_tuples[i], words[i], mm1, mm2 , mm3)
    return(p)

def sent_to_prob_hpl(sentence, mm1, p_hpl, mm2, mm3):
    """function to get the markov model to give prob of a author given a sentence """
    list_of_tuples = make_tupl(sentence, n =3)[0]
    words = make_tupl(sentence, n =3)[1]
    #print(list_of_tuples, words)
    p = p_hpl 
    for i in list(range(words.__len__())):
        #print(list_of_tuples[i], words[i])
        p = p*spit_out_prob(list_of_tuples[i], words[i], mm1, mm2 , mm3)
    return(p)

def sent_to_prob_mws(sentence, mm1, p_mws, mm2, mm3):
    """function to get the markov model to give prob of a author given a sentence """
    list_of_tuples = make_tupl(sentence, n =3)[0]
    words = make_tupl(sentence, n =3)[1]
    #print(list_of_tuples, words)
    p = p_mws 
    for i in list(range(words.__len__())):
        #print(list_of_tuples[i], words[i])
        p = p*spit_out_prob(list_of_tuples[i], words[i], mm1 , mm2, mm3)
    return(p)
