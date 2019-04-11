#!/usr/bin/env python
# coding: utf-8

# In[1]:


# import packages
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import nltk
import os
import random
import re


# In[2]:


# read train set
review_train = pd.read_csv('train.tsv', sep='\t')
print(len(review_train))
#review_train.head()
# read test set
review_test = pd.read_csv('test.tsv', sep='\t')
review_test.head()
#print(review_train.Sentiment)


# In[3]:


### Text prosessing and exploratory analysis on training / testing dataset
# train set: drop everything except Phrase column
trainset_phrases = review_train.drop(['PhraseId','SentenceId','Sentiment'], axis = 1)
trainset_phrases = trainset_phrases.astype(str)
#create new text file with reviews
trainset_phrases.to_csv('trainset.txt', sep='\t', index=False)
#load new text file with reviews
f = open('trainset.txt', encoding="utf8")
text1 = f.read()
#print(text)

# test set: drop everything except Phrase column
testset_phrases = review_test.drop(['PhraseId','SentenceId'], axis = 1)
testset_phrases = testset_phrases.astype(str)
#create new text file with reviews
testset_phrases.to_csv('testset.txt', sep='\t', index=False)
#load new text file with reviews
f = open('testset.txt', encoding="utf8")
text2 = f.read()


# In[4]:


##text, tokenize it and choose further pre-processing or filtering
from nltk import FreqDist

#tokenize train set
reviewtokens1 = nltk.word_tokenize(text1)
reviewwords1 = [w.lower( ) for w in reviewtokens1]
print(len(reviewtokens1))
print(reviewtokens1[:50])

ndist1 = FreqDist(reviewwords1)
nitems1 = ndist1.most_common(50)
print(nitems1)
#for item in nitems:
#    print (item[0], '\t', item[1])


# In[5]:


#tokenize test set
reviewtokens2 = nltk.word_tokenize(text2)
reviewwords2 = [w.lower( ) for w in reviewtokens2]
print(len(reviewtokens2))
print(reviewtokens1[:50])

ndist2 = FreqDist(reviewwords2)
nitems2 = ndist2.most_common(50)
print(nitems2)


# In[6]:


import re
#define stop words
#nltkstopwords = nltk.corpus.stopwords.words('english')
#print(nltkstopwords)

stopwords = ['to', 'in', 'on', 
             'that', "that'll", 'these', 'those', 'am', 'is', 'are', 'was', 'were', 'be', 'been', 'being', 'have', 'has', 'had', 'having', 'do', 'does', 'did', 'doing', 'a', 'an', 'the', 'and', 'but', 'if', 'or', 'because', 'as', 'until', 'while', 'of', 'at', 'by', 'for', 'with', 'about','i', 'me', 'my', 'myself', 'we', 'our', 'ours', 'ourselves', 'you', "you're", "you've", "you'll", "you'd", 'your', 'yours', 'yourself', 'yourselves', 'he', 'him', 'his', 'himself', 'she', "she's", 'her', 'hers', 'herself', 'it', "it's", 'its', 'itself', 'they', 'them', 'their', 'theirs', 'themselves', 'what', 'which', 'who', 'whom', 'this', 'sha','wo','y',"'s","'d","'ll","'t","'m","'re","'ve","-lrb-", "-rrb-"]
#stopwords = nltkstopwords + morestopwords


# In[7]:


import re
#define stop words
#nltkstopwords = nltk.corpus.stopwords.words('english')
#morestopwords = ['could','would','might','must','need','sha','wo','y',"'s","'d","'ll","'t","'m","'re","'ve","n't", "-lrb-", "-rrb-"]
#stopwords = nltkstopwords + morestopwords
#print(stopwords)
stoppedwords1 = [w for w in reviewwords1 if not w in stopwords]
#print(len(stoppedorphanswords))

# function that takes a word and returns true if it consists only
#   of non-alphabetic characters  (assumes import re)
def alpha_filter(w):
  # pattern to match word of non-alphabetical characters
  pattern = re.compile('^[^a-z]+$')
  if (pattern.match(w)):
    return True
  else:
    return False

# apply the function to stoppedwords
alphawords1 = [w for w in stoppedwords1 if not alpha_filter(w)]
print(alphawords1[:50])
print(len(alphawords1))

mystring1 = ",".join(alphawords1)
from wordcloud import WordCloud, STOPWORDS 
import matplotlib.pyplot as plt 
wordcloud = WordCloud(max_font_size=50, max_words=100, background_color="white").generate(mystring1)
plt.figure()
plt.imshow(wordcloud, interpolation="bilinear")
plt.axis("off")
plt.show()


# In[8]:


# test set
stoppedwords2 = [w for w in reviewwords2 if not w in stopwords]
#print(len(stoppedorphanswords))

# function that takes a word and returns true if it consists only
#   of non-alphabetic characters  (assumes import re)
def alpha_filter(w):
  # pattern to match word of non-alphabetical characters
  pattern = re.compile('^[^a-z]+$')
  if (pattern.match(w)):
    return True
  else:
    return False

# apply the function to stoppedwords
alphawords2 = [w for w in stoppedwords2 if not alpha_filter(w)]
print(alphawords2[:50])
print(len(alphawords2))

#mystring2 = ",".join(alphawords2)
#from wordcloud import WordCloud, STOPWORDS 
#import matplotlib.pyplot as plt 
#wordcloud2 = WordCloud(max_font_size=50, max_words=100, background_color="white").generate(mystring2)
#plt.figure()
#plt.imshow(wordcloud, interpolation="bilinear")
#plt.axis("off")
#plt.show()


# In[9]:


# train set: frequency distribution after stopwords filter
ndist_after1 = FreqDist(alphawords1)
nitems1 = ndist_after1.most_common(50)
print(nitems1)

# test set: frequency distribution after stopwords filter
ndist_after2 = FreqDist(alphawords2)
nitems2 = ndist_after2.most_common(50)
print(nitems2)


# In[10]:


# Bigrams and Bigram frequency distribution

reviewbigrams1 = list(nltk.bigrams(reviewwords1))

# setup for bigrams and bigram measures
from nltk.collocations import *
bigram_measures = nltk.collocations.BigramAssocMeasures()

# create the bigram finder and score the bigrams by frequency
finder = BigramCollocationFinder.from_words(reviewwords1)
scored = finder.score_ngrams(bigram_measures.raw_freq)

# scores are sorted in decreasing frequency
for bscore in scored[:50]:
    print (bscore)


# In[11]:


# apply a filter to remove non-alphabetical tokens 
finder.apply_word_filter(alpha_filter)
scored = finder.score_ngrams(bigram_measures.raw_freq)
#for bscore in scored[:30]:
#    print (bscore)

# apply a filter to remove stop words
finder.apply_word_filter(lambda w: w in stopwords)
scored = finder.score_ngrams(bigram_measures.raw_freq)
for bscore in scored[:50]:
    print (bscore)
    
## train set
# apply a filter (on a new finder) to remove low frequency words
finder2 = BigramCollocationFinder.from_words(reviewwords1)
finder2.apply_freq_filter(2)
scored = finder2.score_ngrams(bigram_measures.raw_freq)
#for bscore in scored[:50]:
#   print (bscore)


# In[12]:


'''
  This program shell reads phrase data for the kaggle phrase sentiment classification problem.
  The input to the program is the path to the kaggle directory "corpus" and a limit number.
  The program reads all of the kaggle phrases, and then picks a random selection of the limit number.
  It creates a "phrasedocs" variable with a list of phrases consisting of a pair
    with the list of tokenized words from the phrase and the label number from 1 to 4
  It prints a few example phrases.
  In comments, it is shown how to get word lists from the two sentiment lexicons:
      subjectivity and LIWC, if you want to use them in your features
  Your task is to generate features sets and train and test a classifier.

  Usage:  python classifyKaggle.py  <corpus directory path> <limit number>

  This version uses cross-validation with the Naive Bayes classifier in NLTK.
  It computes the evaluation measures of precision, recall and F1 measure for each fold.
  It also averages across folds and across labels.
'''
# open python and nltk packages needed for processing
import os
import sys
import random
import nltk
from nltk.corpus import stopwords

## this code is commented off now, but can be used for sentiment lists
#import sentiment_read_subjectivity
# initialize the positive, neutral and negative word lists
#(positivelist, neutrallist, negativelist) 
#    = sentiment_read_subjectivity.read_three_types('SentimentLexicons/subjclueslen1-HLTEMNLP05.tff')

#import sentiment_read_LIWC_pos_neg_words
# initialize positve and negative word prefix lists from LIWC 
#   note there is another function isPresent to test if a word's prefix is in the list
#(poslist, neglist) = sentiment_read_LIWC_pos_neg_words.read_words()

## define a feature definition function here

# this function define features (keywords) of a document for a BOW/unigram baseline
# each feature is 'V_(keyword)' and is true or false depending
# on whether that keyword is in the document
def document_features(document, word_features):
    document_words = set(document)
    features = {}
    for word in word_features:
        features['V_{}'.format(word)] = (word in document_words)
    return features



# One strategy with negation words is to negate the word following the negation word
#   other strategies negate all words up to the next punctuation
# Strategy is to go through the document words in order adding the word features,
#   but if the word follows a negation words, change the feature to negated word
# Start the feature set with all 2000 word features and 2000 Not word features set to false

# this list of negation words includes some "approximate negators" like hardly and rarely
negationwords = ['no', 'not', 'never', 'none', 'nowhere', 'nothing', 'noone', 'rather', 'hardly', 'scarcely', 
                 'rarely', 'seldom', 'neither', 'nor', 'not', 'n\'t', "don't","never", "nothing", "nowhere", "noone", "none", "not",
                  "hasn't","hadn't","can't","couldn't","shouldn't","won't",
                  "wouldn't","don't","doesn't","didn't","isn't","aren't","ain't", 'bad', 'terrible','useless', 'hate']
def NOT_features(document, word_features, negationwords):
    features = {}
    for word in word_features:
        features['V_{}'.format(word)] = False
        features['V_NOT{}'.format(word)] = False
    # go through document words in order
    for i in range(0, len(document)):
        word = document[i]
        if ((i + 1) < len(document)) and ((word in negationwords) or (word.endswith("n't"))):
            i += 1
            features['V_NOT{}'.format(document[i])] = (document[i] in word_features)
        else:
            features['V_{}'.format(word)] = (word in word_features)
    return features

# define the feature sets
#NOT_featuresets = [(NOT_features(d, word_features, negationwords), c) for (d, c) in documents]
# show the values of a couple of example features
#print(NOT_featuresets[0][0]['V_NOTcare'])
#print(NOT_featuresets[0][0]['V_always'])




## cross-validation ##
# this function takes the number of folds, the feature sets and the labels
# it iterates over the folds, using different sections for training and testing in turn
#   it prints the performance for each fold and the average performance at the end
def cross_validation_PRF(num_folds, featuresets, labels):
    subset_size = int(len(featuresets)/num_folds)
    print('Each fold size:', subset_size)
    # for the number of labels - start the totals lists with zeroes
    num_labels = len(labels)
    total_precision_list = [0] * num_labels
    total_recall_list = [0] * num_labels
    total_F1_list = [0] * num_labels

    # iterate over the folds
    for i in range(num_folds):
        test_this_round = featuresets[(i*subset_size):][:subset_size]
        train_this_round = featuresets[:(i*subset_size)] + featuresets[((i+1)*subset_size):]
        # train using train_this_round
        classifier = nltk.NaiveBayesClassifier.train(train_this_round)
        # evaluate against test_this_round to produce the gold and predicted labels
        goldlist = []
        predictedlist = []
        for (features, label) in test_this_round:
            goldlist.append(label)
            predictedlist.append(classifier.classify(features))

        # computes evaluation measures for this fold and
        #   returns list of measures for each label
        print('Fold', i)
        (precision_list, recall_list, F1_list)                   = eval_measures(goldlist, predictedlist, labels)
       
        print('\tPrecision\tRecall\t\tF1')
        # print measures for each label
        for i, lab in enumerate(labels):
            print(lab, '\t', "{:10.3f}".format(precision_list[i]),               "{:10.3f}".format(recall_list[i]), "{:10.3f}".format(F1_list[i]))
        
        # for each label add to the sums in the total lists
        for i in range(num_labels):
            # for each label, add the 3 measures to the 3 lists of totals
            total_precision_list[i] += precision_list[i]
            total_recall_list[i] += recall_list[i]
            total_F1_list[i] += F1_list[i]

    # find precision, recall and F measure averaged over all rounds for all labels
    # compute averages from the totals lists
    precision_list = [tot/num_folds for tot in total_precision_list]
    recall_list = [tot/num_folds for tot in total_recall_list]
    F1_list = [tot/num_folds for tot in total_F1_list]
    # the evaluation measures in a table with one row per label
    print('\nAverage Precision\tRecall\t\tF1 \tPer Label')
    # print measures for each label
    for i, lab in enumerate(labels):
        print(lab, '\t', "{:10.3f}".format(precision_list[i]),           "{:10.3f}".format(recall_list[i]), "{:10.3f}".format(F1_list[i]))
    
    # print macro average over all labels - treats each label equally
    print('\nMacro Average Precision\tRecall\t\tF1 \tOver All Labels')
    print('\t', "{:10.3f}".format(sum(precision_list)/num_labels),           "{:10.3f}".format(sum(recall_list)/num_labels),           "{:10.3f}".format(sum(F1_list)/num_labels))

    # for micro averaging, weight the scores for each label by the number of items
    #    this is better for labels with imbalance
    # first intialize a dictionary for label counts and then count them
    label_counts = {}
    for lab in labels:
      label_counts[lab] = 0 
    # count the labels
    for (doc, lab) in featuresets:
      label_counts[lab] += 1
    # make weights compared to the number of documents in featuresets
    num_docs = len(featuresets)
    label_weights = [(label_counts[lab] / num_docs) for lab in labels]
    print('\nLabel Counts', label_counts)
    #print('Label weights', label_weights)
    # print macro average over all labels
    print('Micro Average Precision\tRecall\t\tF1 \tOver All Labels')
    precision = sum([a * b for a,b in zip(precision_list, label_weights)])
    recall = sum([a * b for a,b in zip(recall_list, label_weights)])
    F1 = sum([a * b for a,b in zip(F1_list, label_weights)])
    print( '\t', "{:10.3f}".format(precision),       "{:10.3f}".format(recall), "{:10.3f}".format(F1))
    

# Function to compute precision, recall and F1 for each label
#  and for any number of labels
# Input: list of gold labels, list of predicted labels (in same order)
# Output: returns lists of precision, recall and F1 for each label
#      (for computing averages across folds and labels)
def eval_measures(gold, predicted, labels):
    
    # these lists have values for each label 
    recall_list = []
    precision_list = []
    F1_list = []

    for lab in labels:
        # for each label, compare gold and predicted lists and compute values
        TP = FP = FN = TN = 0
        for i, val in enumerate(gold):
            if val == lab and predicted[i] == lab:  TP += 1
            if val == lab and predicted[i] != lab:  FN += 1
            if val != lab and predicted[i] == lab:  FP += 1
            if val != lab and predicted[i] != lab:  TN += 1
        # use these to compute recall, precision, F1
        # for small numbers, guard against dividing by zero in computing measures
        if (TP == 0) or (FP == 0) or (FN == 0):
          recall_list.append (0)
          precision_list.append (0)
          F1_list.append(0)
        else:
          recall = TP / (TP + FP)
          precision = TP / (TP + FN)
          recall_list.append(recall)
          precision_list.append(precision)
          F1_list.append( 2 * (recall * precision) / (recall + precision))

    # the evaluation measures in a table with one row per label
    return (precision_list, recall_list, F1_list)

## function to read kaggle training file, train and test a classifier 
def processkaggle(dirPath,limitStr):
  # convert the limit argument from a string to an int
  limit = int(limitStr)
  
  os.chdir(dirPath)
  
  f = open('./train.tsv', 'r')
  # loop over lines in the file and use the first limit of them
  phrasedata = []
  for line in f:
    # ignore the first line starting with Phrase and read all lines
    if (not line.startswith('Phrase')):
      # remove final end of line character
      line = line.strip()
      # each line has 4 items separated by tabs
      # ignore the phrase and sentence ids, and keep the phrase and sentiment
      phrasedata.append(line.split('\t')[2:4])
  
  # pick a random sample of length limit because of phrase overlapping sequences
  random.shuffle(phrasedata)
  phraselist = phrasedata[:limit]

  print('Read', len(phrasedata), 'phrases, using', len(phraselist), 'random phrases')
  
  # create list of phrase documents as (list of words, label)
  phrasedocs = []
  # add all the phrases

  # each phrase has a list of tokens and the sentiment label (from 0 to 4)
  ### bin to only 3 categories for better performance
  for phrase in phraselist:
    tokens = nltk.word_tokenize(phrase[0])
    phrasedocs.append((tokens, int(phrase[1])))

  # possibly filter tokens
  # lowercase - each phrase is a pair consisting of a token list and a label
  docs = []
  for phrase in phrasedocs:
    lowerphrase = ([w.lower() for w in phrase[0]], phrase[1])
    docs.append (lowerphrase)
  # print a few
  for phrase in docs[:10]:
    print (phrase)

  # continue as usual to get all words and create word features
  all_words_list = [word for (sent,cat) in docs for word in sent]
  all_words = nltk.FreqDist(all_words_list)
  print(len(all_words))

  # get the 1500 most frequently appearing keywords in the corpus
  word_items = all_words.most_common(5000)
  word_features = [word for (word,count) in word_items]

  # feature sets from a feature definition function
  featuresets = [(NOT_features(d, word_features, negationwords), c) for (d, c) in docs]

  # train classifier and show performance in cross-validation
  # make a list of labels
  label_list = [c for (d,c) in docs]
  labels = list(set(label_list))    # gets only unique labels
  num_folds = 5
  cross_validation_PRF(num_folds, featuresets, labels)
  


"""
commandline interface takes a directory name with kaggle subdirectory for train.tsv
   and a limit to the number of kaggle phrases to use
It then processes the files and trains a kaggle movie review sentiment classifier.

"""
if __name__ == '__main__':
    if (len(sys.argv) != 3):
        print ('usage: classifyKaggle.py <corpus-dir> <limit>')
        sys.exit(0)
    processkaggle(sys.argv[1], sys.argv[2])
    #processkaggle('kaggle\\kagglemoviereviews\\corpus\\', '5000')


# In[ ]:




