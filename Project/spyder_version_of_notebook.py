# SageMaker deployment -- Final project (PyTorch and SageMaker)
#
# Pure Python version (no Jupyter notebook) corresponding
# to SageMakekr Project.ipynb.
# Non AWS version for quick prototyping.
#
#

# %% Get the data

# Only execute that once.
# !wget -O ../data/aclImdb_v1.tar.gz http://ai.stanford.edu/~amaas/data/sentiment/aclImdb_v1.tar.gz
# !tar -zxf ../data/aclImdb_v1.tar.gz -C ../data

# %% Prepare and process the data (1/6)

import os
import glob

def read_imdb_data(data_dir='../data/aclImdb'):
    data = {}
    labels = {}
    
    for data_type in ['train', 'test']:
        data[data_type] = {}
        labels[data_type] = {}
        
        for sentiment in ['pos', 'neg']:
            data[data_type][sentiment] = []
            labels[data_type][sentiment] = []
            
            path = os.path.join(data_dir, data_type, sentiment, '*.txt')
            files = glob.glob(path)
            
            for f in files:
                with open(f) as review:
                    data[data_type][sentiment].append(review.read())
                    # Here we represent a positive review by '1' and a negative review by '0'
                    labels[data_type][sentiment].append(1 if sentiment == 'pos' else 0)
                    
            assert len(data[data_type][sentiment]) == len(labels[data_type][sentiment]), \
                    "{}/{} data size does not match labels size".format(data_type, sentiment)
                
    return data, labels

# %% Prepare and process (2/6)

data, labels = read_imdb_data()
print("IMDB reviews: train = {} pos / {} neg, test = {} pos / {} neg".format(
            len(data['train']['pos']), len(data['train']['neg']),
            len(data['test']['pos']), len(data['test']['neg'])))

# %% Prepare and process (3/6)

from sklearn.utils import shuffle

def prepare_imdb_data(data, labels):
    """Prepare training and test sets from IMDb movie reviews."""
    
    #Combine positive and negative reviews and labels
    data_train = data['train']['pos'] + data['train']['neg']
    data_test = data['test']['pos'] + data['test']['neg']
    labels_train = labels['train']['pos'] + labels['train']['neg']
    labels_test = labels['test']['pos'] + labels['test']['neg']
    
    #Shuffle reviews and corresponding labels within training and test sets
    data_train, labels_train = shuffle(data_train, labels_train)
    data_test, labels_test = shuffle(data_test, labels_test)
    
    # Return a unified training data, test data, training labels, test labets
    return data_train, data_test, labels_train, labels_test


# %% Prepare and process (4/6)

train_X, test_X, train_y, test_y = prepare_imdb_data(data, labels)
print("IMDb reviews (combined): train = {}, test = {}".format(len(train_X), len(test_X)))


# %% Prepare and process (5/6)

print(train_X[100])
print(train_y[100])


# %% Prepare and process (6/6)

import nltk
from nltk.corpus import stopwords
from nltk.stem.porter import *

import re
from bs4 import BeautifulSoup

def review_to_words(review):
    nltk.download("stopwords", quiet=True)
    stemmer = PorterStemmer()
    
    text = BeautifulSoup(review, "html.parser").get_text() # Remove HTML tags
    text = re.sub(r"[^a-zA-Z0-9]", " ", text.lower()) # Convert to lower case
    words = text.split() # Split string into words
    words = [w for w in words if w not in stopwords.words("english")] # Remove stopwords
    words = [PorterStemmer().stem(w) for w in words] # stem
    
    return words



# %% TODO: Apply review_to_words to a review (train_X[100] or any other review)

print(review_to_words(train_X[100]))

# Answer to question:
# Apart from removing the HTML tags and performing stemming,
# the review_to_words also removes
# special characters such as punctuation and English
# stopwords such as "I", "me", "they", "but", etc.
# Furthermore, it converts the entire text to lowercase.

# %% Perform actual preprocessing and store locally in the cache

import pickle

cache_dir = os.path.join("../cache", "sentiment_analysis")  # where to store cache files
os.makedirs(cache_dir, exist_ok=True)  # ensure cache directory exists

def preprocess_data(data_train, data_test, labels_train, labels_test,
                    cache_dir=cache_dir, cache_file="preprocessed_data.pkl"):
    """Convert each review to words; read from cache if available."""

    # If cache_file is not None, try to read from it first
    cache_data = None
    if cache_file is not None:
        try:
            with open(os.path.join(cache_dir, cache_file), "rb") as f:
                cache_data = pickle.load(f)
            print("Read preprocessed data from cache file:", cache_file)
        except:
            pass  # unable to read from cache, but that's okay
    
    # If cache is missing, then do the heavy lifting
    if cache_data is None:
        # Preprocess training and test data to obtain words for each review
        #words_train = list(map(review_to_words, data_train))
        #words_test = list(map(review_to_words, data_test))
        words_train = [review_to_words(review) for review in data_train]
        words_test = [review_to_words(review) for review in data_test]
        
        # Write to cache file for future runs
        if cache_file is not None:
            cache_data = dict(words_train=words_train, words_test=words_test,
                              labels_train=labels_train, labels_test=labels_test)
            with open(os.path.join(cache_dir, cache_file), "wb") as f:
                pickle.dump(cache_data, f)
            print("Wrote preprocessed data to cache file:", cache_file)
    else:
        # Unpack data loaded from cache file
        words_train, words_test, labels_train, labels_test = (cache_data['words_train'],
                cache_data['words_test'], cache_data['labels_train'], cache_data['labels_test'])
    
    return words_train, words_test, labels_train, labels_test

# Preprocess data
train_X, test_X, train_y, test_y = preprocess_data(train_X, test_X, train_y, test_y)


# %% Transform the data
import numpy as np

def build_dict(data, vocab_size = 5000):
    """Construct and return a dictionary mapping each of the most frequently appearing words to a unique integer."""
    
    # TODO: Determine how often each word appears in `data`. Note that `data` is a list of sentences and that a
    #       sentence is a list of words.
    
    all_words = [word for sentence in data for word in sentence]
    all_words_set = set(all_words)
    
    # A dict storing the words that appear in the reviews along with how often they occur
    word_count = {w:0 for w in all_words_set}
    for word in all_words:
        word_count[word] += 1
   
    
    # TODO: Sort the words found in `data` so that sorted_words[0] is the most frequently appearing word and
    #       sorted_words[-1] is the least frequently appearing word.
    
    word_count_pairs = [(k,v) for k,v in word_count.items()]
    word_count_pairs_sorted = sorted(word_count_pairs, key=lambda pair: pair[1], reverse=True)
    sorted_words = [k for k, v in word_count_pairs_sorted]
    
    print(sorted_words[:10])
    
    word_dict = {} # This is what we are building, a dictionary that translates words into integers
    for idx, word in enumerate(sorted_words[:vocab_size - 2]): # The -2 is so that we save room for the 'no word'
        word_dict[word] = idx + 2                              # 'infrequent' labels
        
    return word_dict



# %% Build the dictionary

word_dict = build_dict(train_X)

# Answer to question
word_dict_reverse={idx:w for w,idx in word_dict.items()}

for ii in range(5):
    print(f'{word_dict_reverse[ii++2]}')
# The five most frequently appearing words are
# movi
# film
# one
# like
# time
#
# Given that we are dealing with movie (or film) reviews, this
# seems to make perfect sense.
# It seems to cover typical reviews such as
# 'This movie is one of the best in all time', or
# 'That film is like one of the worst ever'.
#
# As we removed stopwords, we do not find common words such
# as "the" leading this list.
#


# %% Save word dict

data_dir = '../data/pytorch' # The folder we will use for storing data
if not os.path.exists(data_dir): # Make sure that the folder exists
    os.makedirs(data_dir)


with open(os.path.join(data_dir, 'word_dict.pkl'), "wb") as f:
    pickle.dump(word_dict, f)



# %% Transform the reviews

def convert_and_pad(word_dict, sentence, pad=500):
    NOWORD = 0 # We will use 0 to represent the 'no word' category
    INFREQ = 1 # and we use 1 to represent the infrequent words, i.e., words not appearing in word_dict
    
    working_sentence = [NOWORD] * pad
    
    for word_index, word in enumerate(sentence[:pad]):
        if word in word_dict:
            working_sentence[word_index] = word_dict[word]
        else:
            working_sentence[word_index] = INFREQ
            
    return working_sentence, min(len(sentence), pad)

def convert_and_pad_data(word_dict, data, pad=500):
    result = []
    lengths = []
    
    for sentence in data:
        converted, leng = convert_and_pad(word_dict, sentence, pad)
        result.append(converted)
        lengths.append(leng)
        
    return np.array(result), np.array(lengths)

# %% Actual transforming

train_X, train_X_len = convert_and_pad_data(word_dict, train_X)
test_X, test_X_len = convert_and_pad_data(word_dict, test_X)

# %% Sanity check

# Use this cell to examine one of the processed reviews to make sure
# everything is working as intended.
review_reconstructed = [word_dict_reverse[i] for i in train_X[20,:] if i >= 2]
print('Reconstructed review')
print(review_reconstructed)
print(f'Sentiment is ' + ('pos' if train_y[20] else 'neg'))

# %% Question (same methods for training and testing data set)

# This is not a problem and in fact makes sense.
# We need to prepare the (user-written) input in a form that we
# can pass to our model. That actually needs to be done in the
# same way for training and testing.
# We do not augment data (which we would only do with training data),
# therefore it is not a problem if we apply it to the test data.
# As we convert all data separately, we are also not at risk
# of mixing any test data into our training data set.


# %% Upload the data to S3

# This cannot be done outside of AWS, but we can
# nevertheless save the data locally.
import pandas as pd
    
pd.concat([pd.DataFrame(train_y), pd.DataFrame(train_X_len), pd.DataFrame(train_X)], axis=1) \
        .to_csv(os.path.join(data_dir, 'train.csv'), header=False, index=False)

# %% Build and train the PyTorch model

import torch
import torch.utils.data
import torch.nn as nn

# Read in only the first 5000 rows
train_sample = pd.read_csv(os.path.join(data_dir, 'train.csv'), header=None, names=None, nrows=5000)

# Turn the input pandas dataframe into tensors
train_sample_y = torch.from_numpy(train_sample[[0]].values).float().squeeze()
train_sample_X = torch.from_numpy(train_sample.drop([0], axis=1).values).long()

# Build the dataset
train_sample_ds = torch.utils.data.TensorDataset(train_sample_X, train_sample_y)
# Build the dataloader
train_sample_dl = torch.utils.data.DataLoader(train_sample_ds, batch_size=50)


# %% Write the training method

def train(model, train_loader, epochs, optimizer, loss_fn, device):
    # Hidden state
    # h = None
    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = 0
        for batch in train_loader:         
            batch_X, batch_y = batch
            
            batch_X = batch_X.to(device)
            batch_y = batch_y.to(device)
            
            # TODO: Complete this train method to train the model provided.
            
            # Note: Most of this code has been taken from
            # the RNN exercise of chapter 3.
            
            # Avoid backprop through the entire training history
            # by creating a new variable.
            # if h is not None:
            #     h = tuple([each.data for each in h])
    
            # zero accumulated gradients
            model.zero_grad()
    
            # get the output from the model
            # output, h = model(batch_X, h)
            output = model(batch_X)
    
            # calculate the loss and perform backprop
            loss = loss_fn(output.squeeze(), batch_y.float())
            loss.backward()
            # `clip_grad_norm` helps prevent the exploding gradient problem in RNNs / LSTMs.
            # nn.utils.clip_grad_norm_(model.parameters(), clip)
            optimizer.step()
            
            total_loss += loss.data.item()
        print("Epoch: {}, BCELoss: {}".format(epoch, total_loss / len(train_loader)))

# %% Train the model

import torch.optim as optim
from train.model import LSTMClassifier

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = LSTMClassifier(32, 200, 5000).to(device)
optimizer = optim.Adam(model.parameters())
loss_fn = torch.nn.BCELoss()

train(model, train_sample_dl, 20, optimizer, loss_fn, device)


# %% Test the model

from sklearn.metrics import accuracy_score

pd.concat([pd.DataFrame(test_y), pd.DataFrame(test_X_len), pd.DataFrame(test_X)], axis=1) \
        .to_csv(os.path.join(data_dir, 'test.csv'), header=False, index=False)
        
        
# Read in only the first 250 rows
test_sample = pd.read_csv(os.path.join(data_dir, 'test.csv'), header=None, names=None, nrows=250)

# Turn the input pandas dataframe into tensors
test_sample_y = torch.from_numpy(test_sample[[0]].values).float().squeeze()
test_sample_X = torch.from_numpy(test_sample.drop([0], axis=1).values).long()

# Build the dataset
test_sample_ds = torch.utils.data.TensorDataset(test_sample_X, test_sample_y)
# Build the dataloader
test_sample_dl = torch.utils.data.DataLoader(test_sample_ds, batch_size=50)


for batch_X, batch_y in test_sample_dl:
    batch_X = batch_X.to(device)
    batch_y = batch_y.to(device)
    output_test_y = model(batch_X)
    output_test_y = np.round(output_test_y.cpu().detach().numpy())
    print(accuracy_score(batch_y.cpu().detach().numpy(), output_test_y))
    


# %% Prepare testing of the predict_fn

model.word_dict = word_dict

def predict_fn(input_data, model):
    print('Inferring sentiment of input data.')

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    if model.word_dict is None:
        raise Exception('Model has not been loaded properly, no word_dict.')
    
    # TODO: Process input_data so that it is ready to be sent to our model.
    #       You should produce two variables:
    #         data_X   - A sequence of length 500 which represents the converted review
    #         data_len - The length of the review
    
    input_data_as_words = review_to_words(input_data)
    data_X, data_len = convert_and_pad(model.word_dict, input_data_as_words, pad=500)

    # Using data_X and data_len we construct an appropriate input tensor. Remember
    # that our model expects input data of the form 'len, review[500]'.
    data_pack = np.hstack((data_len, data_X))
    data_pack = data_pack.reshape(1, -1)
    
    data = torch.from_numpy(data_pack)
    data = data.to(device)

    # Make sure to put the model into evaluation mode
    model.eval()

    # TODO: Compute the result of applying the model to the input data. The variable `result` should
    #       be a numpy array which contains a single integer which is either 1 or 0

    result = model(data)
    result = result.cpu().detach().numpy()
    result = np.round(result).astype('int')

    return result


# %% Test predict_fn

my_review = 'This movie is awesome! -- I really like it a lot'
my_review = 'This film sucks so much, worst movie ever.'
my_sentiment = predict_fn(my_review, model)

print('pos' if my_sentiment else 'neg')















