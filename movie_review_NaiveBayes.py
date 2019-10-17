#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import string
import re
from nltk.stem.porter import PorterStemmer
import random
from sklearn.model_selection import train_test_split
import os
from random import shuffle


# In[ ]:





# In[ ]:


def get_files(files_path, review_type):
    try:
        output = []
        files = os.listdir(files_path) 
        for file in files:
            f = open(files_path + file, 'r', encoding="utf8")
            output.append((f.read(), review_type))
        return output
    except IOError:
        print('Problem opening file')
    finally:
        f.close()


# In[ ]:


# Load training data
train_pos = get_files('../datasets/movie_reviews/data/alle/train/pos/', 0)
train_neg = get_files('../datasets/movie_reviews/data/alle/train/neg/', 1)
#train_pos = get_files('../datasets/movie_reviews/data/subset/train/pos/', 0)
#train_neg = get_files('../datasets/movie_reviews/data/subset/train/neg/', 1)

print('* TRAINING DATA * ')
print('# positives reviews: ', len(train_pos))
print('# negatives reviews', len(train_neg))
train_data = train_pos + train_neg
print('# total reviews: ', len(train_data))
print('-----------------')
# Load test data
test_pos = get_files('../datasets/movie_reviews/data/alle/test/pos/', 0)
test_neg = get_files('../datasets/movie_reviews/data/alle/test/neg/', 1)
#test_pos = get_files('../datasets/movie_reviews/data/subset/test/pos/', 0)
#test_neg = get_files('../datasets/movie_reviews/data/subset/test/neg/', 1)
print('* TEST DATA * ')
print('# positives reviews: ', len(test_pos))
print('# negatives reviews', len(test_neg))
test_data = test_pos + test_neg
print('# total reviews: ', len(test_data))

# Does not want a 50/50 split between training and test
# Therefore creates one big set of data that later will be split into 80/20 train- and testdata
# a = train_data[::2]
# b = train_data[1::2]
# c = test_data[::2]
# d = test_data[1::2]
# all_reviews = a + b + c + d'
all_r = train_data + test_data
shuffle(all_r)


# In[ ]:


all_reviews = all_r


# In[ ]:


stopwords = []
try:
    f = open('../datasets/stopwords.txt', 'r')
    stopwords = f.read().split(',')
except IOError:
    print('Problem opening file')
finally:
    f.close()


# In[ ]:


# * * * PREPROCESSING * * * 

stemmer = PorterStemmer()
preprocessed_reviews = []

for t in all_reviews:
    #print(len(preprocessed_reviews))
    review = t[0]
    review_type = t[1]
    # Remove whitespace and punctutation
    text = re.sub('[' + string.punctuation + ']', ' ', review)
    text = re.sub('[\n\t\r]', '', text)
    
    # Split words into list
    words = text.split()
    new = []
    # Remove stopwords and stem remaining words 
    for word in words:
        stemmed_word = stemmer.stem(word.lower())
        if stemmed_word not in stopwords and len(stemmed_word) > 2:
            new.append(stemmed_word)    
    
    # Add to preproccesed list
    preprocessed_reviews.append((new, review_type))


# In[ ]:


count = 0
for r in preprocessed_reviews:
    words = r[0]
    for w in words:
        if w in stopwords:
            count += 1
a = 191569
count


# In[ ]:


# Splitting data in trainingdata and testdata (80-20 ratio)

total = len(preprocessed_reviews) #Total number of reviews
test_number = int(0.20 * total) # Number of testing reviews
# Picking randomly
print(test_number)
copy = preprocessed_reviews[:]
test_set = []

taken = {}
while len(test_set) < test_number:
    #print(len(train_texts))
    num = random.randint(0, test_number - 1)
    if num not in taken.keys():
        test_set.append(copy.pop(num))
        taken[num] = 1

train_set = copy[:] # Trainset is the remaining reviews
        
len(train_set)/total, len(test_set)/total, len(train_set), len(test_set)


# In[ ]:


# * * * TRAINING THE MODEL * * * 

# meaning: Computing probabilities needed for P(Positive|Word)

def total_goods_and_bads(tset):
    goods = 0
    bads = 0
    for t in tset:
        goods += 1 if t[1] == 0 else 0
        bads += 1 if t[1] == 1 else 0
    return goods, bads

total_positive = total_goods_and_bads(train_set)[0]
total_negative = total_goods_and_bads(train_set)[1]
print(total_positive)
print(total_negative)


# In[ ]:


# First making a word counter for pos and neg reviews
pos_word_counter = {}
neg_word_counter = {}
total_words = 0
for t in train_set:
    review = t[0]
    review_type = t[1]
    already_counted = []
    for word in review:
        total_words += 1
        if review_type == 0:
            if word not in pos_word_counter:
                pos_word_counter[word] = 1
            else:
                if word not in already_counted:
                    pos_word_counter[word] += 1  
        else:
            if word not in neg_word_counter:
                neg_word_counter[word] = 1
            else:
                if word not in already_counted:
                    neg_word_counter[word] += 1
                    
        already_counted.append(word)
total_words


# In[ ]:


# Removes words that are not inluded in at least 0.15% of the reviews
removed_words = 0
for j in range(len(train_set)):
    words = train_set[j][0]
    i = 0
    while i < len(words):
        word = words[i]
        word_removed = False
        if word in pos_word_counter:
            if pos_word_counter[word] < 0.0015*len(train_set):
                train_set[j][0].remove(word)
                word_removed = True
                removed_words += 1
        elif word in neg_word_counter:
            if neg_word_counter[word] < 0.0015*len(train_set):
                train_set[j][0].remove(word)
                word_removed = True
                removed_words += 1
        if not word_removed:
            i += 1
    j += 1
removed_words


# In[ ]:


def sort_dict(dicti, end):
    # Sorterer etter value i dict, gir liste med tupler
    most_common_words = sorted(dicti.items(), key = lambda kv: kv[1])
    most_common_words.reverse()
    most_common_words = most_common_words[:end]
    # Lager dict på formen {word: count, ...}
    # Vil ha dict fremfor liste med tupler, pga. senere søk
    return dict(most_common_words)                

most_used_words_pos = sort_dict(pos_word_counter, 25)
most_used_words_neg = sort_dict(neg_word_counter, 25)
most_used_words_pos


# In[ ]:



# Need these 4 probabilities
# 1) Probability that a word appears in positive reviews
# 2) Probability that a word appears in negative reviews
# 3) Overall probability that any given review is positive
# 4) Overall probability that any given reviews is negative

# # Making a dictionary with probabilities for different words appearing in good and bad reviews
# # Example: {'bad': (0.0881, 0.3226)}
probability_appearing = {}
for t in train_set:
    text = t[0]
    for word in text:
        if word not in probability_appearing:
            if word in pos_word_counter:
                p_appearing_good = pos_word_counter[word]/total_positive
            else:
                p_appearing_good = 0.1
            if word in neg_word_counter:
                p_appearing_bad = neg_word_counter[word]/total_negative
            else:
                p_appearing_bad = 0.1
            probability_appearing[word] = (p_appearing_good, p_appearing_bad)
            

p_pos = total_positive/len(train_set)
p_neg = total_negative/len(train_set)
print(p_good)
print(p_bad)


# Finally we can compute P(Positive | Word)
def p_is_positive_given_word(word):
    return (probability_appearing[word][0]*p_pos)/((probability_appearing[word][0]*p_pos + probability_appearing[word][1]*p_neg))

def p_is_negative_given_word(word):
    return (probability_appearing[word][1]*p_neg)/((probability_appearing[word][1]*p_neg + probability_appearing[word][0]*p_pos))

p_is_positive_given_word('bad'), p_is_negative_given_word('bad')


# In[ ]:


probabilities = {}
for t in train_set:
    text = t[0]
    for word in text:
        if word not in probabilities:
            p_pos = p_is_positive_given_word(word)
            p_neg = p_is_negative_given_word(word)
            if p_pos == 0:
                p_pos = 0.1 # tweaking this value
            if p_pos == 1:
                p_pos = 0.98
            if p_neg == 0:
                p_neg = 0.1
            if p_neg == 1:
                p_neg = 0.98
                
            probabilities[word] = (p_pos, p_neg)


# In[ ]:


# Filter out words that are not informative (probabilities between 0.45 and 0.55)
print(len(probabilities))
for word in list(probabilities):
    probs = probabilities[word]
    if 0.40 < probs[0] and probs[0] < 0.60 and 0.40 < probs[1] and probs[1] < 0.60:
        del probabilities[word]
print(len(probabilities))


# In[ ]:


probabilities


# In[ ]:


# COMBINING INDIVIDUAL PROBABILITIES
# Determining whether a message is spam or ham based only on the presence of one word is error-prone,
# must try to consider all the words (or the most interesting) in the message

from functools import reduce
def p_is_type(words):
    words = list(filter(lambda x: x in probabilities, words)) # Filter out words not met during training-fase
    pos_probs = []
    neg_probs = []
    for word in words:
        pos_probs.append(probabilities[word][0])
        neg_probs.append(probabilities[word][1])
        #else:
         #   probs.append(0.5) # tweaking this value
    pos_probs_not = list(map(lambda prob: 1-prob, pos_probs))
    neg_probs_not = list(map(lambda prob: 1-prob, neg_probs))
    
    pos_product = reduce(lambda x, y: x * y, pos_probs, 1)
    neg_product = reduce(lambda x, y: x * y, neg_probs, 1) 
    
    pos_product_not = reduce(lambda x, y: x * y, pos_probs_not, 1)
    neg_product_not = reduce(lambda x, y: x * y, neg_probs_not, 1)
    return pos_product/(pos_product + pos_product_not), neg_product/(neg_product + neg_product_not) 

p_is_type(['good', 'enjoy', 'well']), p_is_type(['terribl', 'hate'])


# In[ ]:


# * * * TESTING THE MODEL * * * 
total_correct = 0

true_good_as_good = 0
true_good_as_bad = 0

true_bad_as_bad = 0
true_bad_as_good = 0
count = 0
for t in test_set:
    guess = -1
    words = t[0]
    answer = t[1]
    try:
        p_positive = p_is_type(words)[0]
        p_negative = p_is_type(words)[1]
    except:
        count += 1
        #print(words)

    guess = 0 if p_positive > p_negative else 1
    if guess == answer:
        total_correct += 1
        if answer == 0: # true negative
            true_good_as_good += 1
        else: # true positive
            true_bad_as_bad += 1 
    else:
        #print(words, answer)
        if answer == 0: # false positive
            true_good_as_bad += 1
        else: # true negative
            true_bad_as_good += 1

            
true_positives = total_goods_and_bads(test_set)[0]
true_negatives = total_goods_and_bads(test_set)[1]

print('Total test texts: ', len(test_set))
print('Number of correct: ', total_correct)
print('Accuracy: ', total_correct*100/(true_positives+true_negatives))
print('-------------------------------')
print('Positives precision: ', true_good_as_good/(true_good_as_good + true_bad_as_good))
print('Positives recall: ', true_good_as_good/(true_good_as_good + true_good_as_bad))
print('Negatives precision: ', true_bad_as_bad/(true_bad_as_bad + true_good_as_bad))
print('Negatives recall: ', true_bad_as_bad/(true_bad_as_bad + true_bad_as_good))
print('-------------------------------')


# In[ ]:


# * * * VISUALISATIONS * * * 
from wordcloud import WordCloud

pos_reviews = ""
neg_reviews = ""
revs = all_reviews[:100]

for t in revs:
    review = t[0].split()
    s = ""
    for word in review:
        if len(word) > 2:
            s += word + ' '
    text = re.sub('[' + string.punctuation + ']', ' ', s)
    text = re.sub('[\n\t\r]', '', text)
    if t[1] == 0:
        pos_reviews += text
    else:
        neg_reviews += text

# Generate a word cloud image
pos_wordcloud = WordCloud(width=600, height=400).generate(pos_reviews)
#neg_wordcloud = WordCloud(width=600, height=400).generate(neg_reviews)


# In[ ]:


#Spam Word cloud
plt.figure(figsize=(10,8), facecolor='k')
plt.imshow(pos_wordcloud)
plt.axis("off")
plt.tight_layout(pad=0)


# In[ ]:




