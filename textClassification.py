import pandas as pd
import numpy as np
import re
from nltk.stem import PorterStemmer
from nltk.tokenize import TweetTokenizer
import string
# import this for storing our BOW format
import scipy
from scipy import sparse
from sklearn import svm
import random

#import torch
#from transformers import BertTokenizer, BertForSequenceClassification
#from torch.utils.data import DataLoader, Dataset
from sklearn.metrics import accuracy_score

tokenizer=TweetTokenizer()
stemmer = PorterStemmer()

with open('train.txt', encoding="latin-1") as file:
    training_data = ''.join(file.readlines()[1:])  # Skip the first line

with open('ttds_2024_cw2_test.txt', encoding="latin-1") as file:
    test_data = ''.join(file.readlines()[1:])  # Skip the first line

stopwordlist = ['a', 'about', 'above', 'after', 'again', 'ain', 'all', 'am', 'an',
             'and','any','are', 'as', 'at', 'be', 'because', 'been', 'before',
             'being', 'below', 'between','both', 'by', 'can', 'd', 'did', 'do',
             'does', 'doing', 'down', 'during', 'each','few', 'for', 'from',
             'further', 'had', 'has', 'have', 'having', 'he', 'her', 'here',
             'hers', 'herself', 'him', 'himself', 'his', 'how', 'i', 'if', 'in',
             'into','is', 'it', 'its', 'itself', 'just', 'll', 'm', 'ma',
             'me', 'more', 'most','my', 'myself', 'now', 'o', 'of', 'on', 'once',
             'only', 'or', 'other', 'our', 'ours','ourselves', 'out', 'own', 're','s', 'same', 'she', "shes", 'should', "shouldve",'so', 'some', 'such',
             't', 'than', 'that', "thatll", 'the', 'their', 'theirs', 'them',
             'themselves', 'then', 'there', 'these', 'they', 'this', 'those',
             'through', 'to', 'too','under', 'until', 'up', 've', 'very', 'was',
             'we', 'were', 'what', 'when', 'where','which','while', 'who', 'whom',
             'why', 'will', 'with', 'won', 'y', 'you', "youd","youll", "youre",
             "youve", 'your', 'yours', 'yourself', 'yourselves']

# Process document content: tokenization, stemming, stopword removal
def tokenize_and_clean(content, stopwords, stemmer):
    #chars_to_remove = re.compile(f'[{string.punctuation}]')
    #remPunc = chars_to_remove.sub('',content)
    #remURL = re.sub('((www.[^s]+)|(https?://[^s]+))',' ',remPunc)
    final_tokens = tokenizer.tokenize(content.lower())
    stemmed_tokens = [stemmer.stem(token) for token in final_tokens]
    return stemmed_tokens

def preprocess_data(data,model):
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
            if model =='baseline':
                # process the words
                words = chars_to_remove.sub('',tweet).lower().split()
                for word in words:
                    vocab.add(word)
                # add the list of words to the documents list
                documents.append(words)
                # add the sentiment to the categories list
                categories.append(sentiment)
            else:
                # process the words
                words = tokenize_and_clean(tweet,stopwordlist,stemmer)
                for word in words:
                    vocab.add(word)
                # add the list of words to the documents list
                documents.append(words)
                # add the sentiment to the categories list
                categories.append(sentiment)    
    return documents, categories, vocab

# Function to shuffle and split the data
def shuffle_and_split_with_vocab(documents, categories, vocab, split_ratio=0.7):
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


def createDict(train_vocab,train_cats):
    word2id = {}
    for word_id,word in enumerate(train_vocab):
        word2id[word] = word_id
            
    # and do the same for the categories
    cat2id = {}
    for cat_id,cat in enumerate(set(train_cats)):
        cat2id[cat] = cat_id
    
    return word2id,cat2id


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

def convert_to_tfidf_matrix(preprocessed_data, word2id):
    # Number of documents and vocabulary size
    num_docs = len(preprocessed_data)
    vocab_size = len(word2id)
    oov_index = vocab_size  # Out-of-Vocabulary index

    # Calculate Document Frequency (DF) for each word
    doc_freq = np.zeros(vocab_size + 1)  # +1 for OOV words
    for doc in preprocessed_data:
        unique_words = set(doc)
        for word in unique_words:
            doc_freq[word2id.get(word, oov_index)] += 1

    # Calculate IDF for each word
    idf = np.log(num_docs / (1 + doc_freq))  # Smoothing by adding 1 to DF

    # Create a sparse matrix for TF-IDF
    tfidf_matrix = scipy.sparse.dok_matrix((num_docs, vocab_size + 1))

    # Iterate through all documents
    for doc_id, doc in enumerate(preprocessed_data):
        # Calculate Term Frequency (TF) for the current document
        word_counts = {}
        for word in doc:
            word_id = word2id.get(word, oov_index)
            word_counts[word_id] = word_counts.get(word_id, 0) + 1
        
        total_words = len(doc)
        
        # Calculate TF-IDF for each word in the document
        for word_id, count in word_counts.items():
            tf = count / total_words
            tfidf_matrix[doc_id, word_id] = tf * idf[word_id]

    return tfidf_matrix

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


def baseLine():

    preprocessed_training_data, training_categories, train_vocab = preprocess_data(training_data,'baseline')
    preprocessed_test_data, test_categories, test_vocab = preprocess_data(test_data,'baseline')

    train_docs, train_cats, train_vocab, dev_docs, dev_cats, dev_vocab = shuffle_and_split_with_vocab(
    preprocessed_training_data, 
    training_categories, 
    train_vocab)

    word2id,cat2id = createDict(train_vocab,train_cats)

    X_train = convert_to_bow_matrix(train_docs, word2id)
    y_train = [cat2id[cat] for cat in train_cats]
    X_dev = convert_to_bow_matrix(dev_docs, word2id)
    y_dev = [cat2id[cat] for cat in dev_cats]
    X_test = convert_to_bow_matrix(preprocessed_test_data, word2id)
    y_test = [cat2id[cat] for cat in test_categories]

    model = svm.SVC(C=1000, kernel ="linear")
    # then train the model!
    model.fit(X_train,y_train)

    ytrn_pred = model.predict(X_train)
    ydev_pred = model.predict(X_dev)
    ytest_pred = model.predict(X_test)
    print(compute_accuracy(ytest_pred,y_test),'baseline')
    CalAcc(ytrn_pred, y_train, 'baseline', 'train')
    CalAcc(ydev_pred, y_dev, 'baseline', 'dev')
    CalAcc(ytest_pred, y_test, 'baseline', 'test')

def improved():
    
    preprocessed_training_data, training_categories, train_vocab = preprocess_data(training_data,'improved')
    preprocessed_test_data, test_categories, test_vocab = preprocess_data(test_data,'improved')

    train_docs, train_cats, train_vocab, dev_docs, dev_cats, dev_vocab = shuffle_and_split_with_vocab(
    preprocessed_training_data, 
    training_categories, 
    train_vocab)

    word2id,cat2id = createDict(train_vocab,train_cats)

    X_train = convert_to_tfidf_matrix(train_docs, word2id)
    y_train = [cat2id[cat] for cat in train_cats]
    X_dev = convert_to_tfidf_matrix(dev_docs, word2id)
    y_dev = [cat2id[cat] for cat in dev_cats]
    X_test = convert_to_tfidf_matrix(preprocessed_test_data, word2id)
    y_test = [cat2id[cat] for cat in test_categories]
    #model = svm.LinearSVC(C=1, random_state=0, max_iter=1e5)
    model = svm.SVC(C=100, kernel ="rbf")
    # then train the model!
    model.fit(X_train,y_train)

    ytrn_pred = model.predict(X_train)
    ydev_pred = model.predict(X_dev)
    ytest_pred = model.predict(X_test)
    print(compute_accuracy(ytest_pred,y_test),'improved')
    CalAcc(ytrn_pred, y_train, 'improved', 'train')
    CalAcc(ydev_pred, y_dev, 'improved', 'dev')
    CalAcc(ytest_pred, y_test, 'improved', 'test')

def compute_accuracy(predictions, true_values):
    num_correct = 0
    num_total = len(predictions)
    for predicted,true in zip(predictions,true_values):
        if predicted==true:
            num_correct += 1
    return num_correct / num_total

#baseLine()
#improved()

'''
tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")

class SentimentDataset(Dataset):
    def __init__(self, documents, categories, tokenizer, max_length=128):
        self.documents = documents
        self.categories = categories
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.documents)

    def __getitem__(self, index):
        text = " ".join(self.documents[index])  # Join words into a single string
        label = int(self.categories[index])  # Convert sentiment to int
        encoding = self.tokenizer(
            text,
            truncation=True,
            padding="max_length",
            max_length=self.max_length,
            return_tensors="pt"
        )
        return {
            "input_ids": encoding["input_ids"].squeeze(),
            "attention_mask": encoding["attention_mask"].squeeze(),
            "labels": torch.tensor(label, dtype=torch.long)
        }

# Prepare the datasets
def prepare_bert_datasets(documents, categories):
    dataset = SentimentDataset(documents, categories, tokenizer)
    return DataLoader(dataset, batch_size=16, shuffle=True)


# Preprocess data using your preprocessing function
train_docs, train_cats, _ = preprocess_data(training_data, model="bert")
test_docs, test_cats, _ = preprocess_data(test_data, model="bert")

# Create data loaders
train_loader = prepare_bert_datasets(train_docs, train_cats)
test_loader = prepare_bert_datasets(test_docs, test_cats)

# Load the BERT model for sequence classification
model = BertForSequenceClassification.from_pretrained("bert-base-uncased", num_labels=3)  # Assuming binary classification
model.to("cuda")  # Move to GPU if available

# Define optimizer and loss function
optimizer = torch.optim.AdamW(model.parameters(), lr=5e-5)
loss_fn = torch.nn.CrossEntropyLoss()


# Training loop
def train_model(model, train_loader, epochs=3):
    model.train()
    for epoch in range(epochs):
        total_loss = 0
        for batch in train_loader:
            optimizer.zero_grad()
            input_ids = batch["input_ids"].to("cuda")
            attention_mask = batch["attention_mask"].to("cuda")
            labels = batch["labels"].to("cuda")
            outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
            loss = outputs.loss
            total_loss += loss.item()
            loss.backward()
            optimizer.step()
        print(f"Epoch {epoch + 1}, Loss: {total_loss / len(train_loader)}")

# Evaluation function
def evaluate_model(model, data_loader):
    model.eval()
    predictions, true_labels = [], []
    with torch.no_grad():
        for batch in data_loader:
            input_ids = batch["input_ids"].to("cuda")
            attention_mask = batch["attention_mask"].to("cuda")
            labels = batch["labels"].to("cuda")
            outputs = model(input_ids, attention_mask=attention_mask)
            logits = outputs.logits
            preds = torch.argmax(logits, axis=1).cpu().numpy()
            predictions.extend(preds)
            true_labels.extend(labels.cpu().numpy())
    return accuracy_score(true_labels, predictions)

# Train the model
train_model(model, train_loader)

# Evaluate the model
accuracy = evaluate_model(model, test_loader)
print(f"Test Accuracy: {accuracy}")
'''
#Accuracy
#0.5669240669240669 baseline
#0.6271986271986272 improved
