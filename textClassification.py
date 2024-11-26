import pandas as pd
import numpy as np
import re
import string
# import this for storing our BOW format
import scipy
from scipy import sparse
from sklearn import svm
import random

with open('train.txt', encoding="latin-1") as file:
    training_data = ''.join(file.readlines()[1:])  # Skip the first line

with open('ttds_2024_cw2_test.txt', encoding="latin-1") as file:
    test_data = ''.join(file.readlines()[1:])  # Skip the first line


def preprocess_data(data):
    
    chars_to_remove = re.compile(f'[{string.punctuation}]')
    
    documents = []
    categories = []
    vocab = set([])
    
    lines = data.split('\n')
    
    for line in lines:
        # make a dictionary for each document
        # word_id -> count (could also be tf-idf score, etc.)
        line = line.strip()
        if line:
            # split on tabs, we have 3 columns in this tsv format file
            tweet_id, sentiment, tweet = line.split('\t')

            # process the words
            words = chars_to_remove.sub('',tweet).lower().split()
            for word in words:
                vocab.add(word)
            # add the list of words to the documents list
            documents.append(words)
            # add the sentiment to the categories list
            categories.append(sentiment)
            
    return documents, categories, vocab

# Function to shuffle and split the data
def shuffle_and_split_with_vocab(documents, categories, vocab, split_ratio=0.8):
    # Combine documents and categories into pairs
    combined = list(zip(documents, categories))
    
    # Shuffle the data
    random.shuffle(combined)
    
    # Split the shuffled data
    split_point = int(len(combined) * split_ratio)
    train_data = combined[:split_point]
    dev_data = combined[split_point:]
    
    # Unzip the data back into separate lists
    train_docs, train_cats = zip(*train_data)
    dev_docs, dev_cats = zip(*dev_data)
    
    # Update vocabulary for training and development sets
    train_vocab = set(word for doc in train_docs for word in doc)
    dev_vocab = set(word for doc in dev_docs for word in doc)
    
    return list(train_docs), list(train_cats), train_vocab, list(dev_docs), list(dev_cats), dev_vocab


preprocessed_training_data, training_categories, train_vocab = preprocess_data(training_data)
preprocessed_test_data, test_categories, test_vocab = preprocess_data(test_data)


# Apply the shuffle_and_split_with_vocab function
train_docs, train_cats, train_vocab, dev_docs, dev_cats, dev_vocab = shuffle_and_split_with_vocab(
    preprocessed_training_data, 
    training_categories, 
    train_vocab)



word2id = {}
for word_id,word in enumerate(train_vocab):
    word2id[word] = word_id
        
# and do the same for the categories
cat2id = {}
for cat_id,cat in enumerate(set(train_cats)):
    cat2id[cat] = cat_id


# build a BOW representation of the files: use the scipy 
# data is the preprocessed_data
# word2id maps words to their ids
def convert_to_bow_matrix(preprocessed_data, word2id):
    
    # matrix size is number of docs x vocab size + 1 (for OOV)
    matrix_size = (len(preprocessed_data),len(word2id)+1)
    oov_index = len(word2id)
    # matrix indexed by [doc_id, token_id]
    X = scipy.sparse.dok_matrix(matrix_size)

    # iterate through all documents in the dataset
    for doc_id,doc in enumerate(preprocessed_data):
        for word in doc:
            # default is 0, so just add to the count for this word in this doc
            # if the word is oov, increment the oov_index
            X[doc_id,word2id.get(word,oov_index)] += 1
    
    return X


X_train = convert_to_bow_matrix(train_docs, word2id)

y_train = [cat2id[cat] for cat in train_cats]

X_dev = convert_to_bow_matrix(dev_docs, word2id)

y_dev = [cat2id[cat] for cat in dev_cats]

X_test = convert_to_bow_matrix(preprocessed_test_data, word2id)

y_test = [cat2id[cat] for cat in test_categories]

def CalAcc(y_pred, y_true, system, split):
    f = open("classification.csv", "a")
    if system == 'baseline' and split == 'train':
        hdr = "system,split,p-pos,r-pos,f-pos,p-neg,r-neg,f-neg,p-neu,r-neu,f-neu,p-macro,r-macro,f-macro\n"
        f.write(hdr)

    dfpred = pd.DataFrame(y_pred)
    dftrue = pd.DataFrame(y_true)
    labels = [0, 1, 2]
    precision = []
    recall = []
    F1 = []

    for label in labels:
        pred = dfpred[dfpred[0] == label]
        index_pred = pred.index.tolist()
        true = dftrue[dftrue[0] == label]
        index_true = dftrue.reindex(index=index_pred)

        precision.append(sum(np.array(pred) == np.array(index_true)) / len(pred))
        recall.append(sum(np.array(pred) == np.array(index_true)) / len(true))
        F1.append(2*precision[label]*recall[label] / (precision[label]+recall[label]))

    macro_P = np.mean(precision)
    macro_R = np.mean(recall)
    macro_F1 = 2*macro_P*macro_R / (macro_P+macro_R)

    line = system+','+split+','+"{:.3f},{:.3f},{:.3f},{:.3f},{:.3f},{:.3f},{:.3f},{:.3f},{:.3f},{:.3f},{:.3f},{:.3f}\n".format(precision[0][0], recall[0][0], F1[0][0], precision[1][0], recall[1][0], F1[1][0], precision[2][0], recall[2][0], F1[2][0], macro_P, macro_R, macro_F1)
    f.write(line)


def BaseLine(X_train,y_train,X_dev,y_dev,X_test,y_test):
    
    model = svm.SVC(C=1000, kernel ="linear")
    # then train the model!
    model.fit(X_train,y_train)

    ytrn_pred = model.predict(X_train)
    ydev_pred = model.predict(X_dev)
    ytest_pred = model.predict(X_test)

    CalAcc(ytrn_pred, y_train, 'baseline', 'train')
    CalAcc(ydev_pred, y_dev, 'baseline', 'dev')
    CalAcc(ytest_pred, y_test, 'baseline', 'test')

BaseLine(X_train,y_train,X_dev,y_dev,X_test,y_test)
