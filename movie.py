# -*- coding: utf-8 -*-
"""
Dataset: Movie Reviews

"""
import pandas as pd
import numpy as np
import os
import re
from nltk.corpus import stopwords
from nltk.stem.snowball import SnowballStemmer
from time import time

def read_file():
    labels = {'pos':1, 'neg':0}
    df = pd.DataFrame()
    for i in ('test', 'train'):
        for j in ('pos', 'neg'):
            for file in os.listdir('./aclImdb/%s/%s' % (i,j)):
                with open('./aclImdb/%s/%s/%s' % (i,j,file), 'r', encoding="utf8") as f:
                    txt = f.read()
                df = df.append([[txt, labels[j]]], ignore_index= True)
    df= df.reindex(np.random.permutation(df.index))
    df.columns = ['review','sentiment']
    return df  
          
# Import
movie = read_file()
movie.head(4)

#Cleaning text from punctuation,abbreviation, stopwords, stemmed or lemmatized and tokenized
def clean(text, using_stem = True, using_stop = False):
    stopword = stopwords.words('english')
    stemmer = SnowballStemmer('english')
    abbrev = {r"don't": "do not", r"doesn't": "does not",
              r"didn't": "did not", r"hasn't": "has not",
              r"haven't": "have not", r"hadn't": "had not",
              r"won't": "will not", r"wouldn't": "would not",
              r"can't": "can not", r"cannot": "can not"
              }
    #Substitute abbreviations with words
    for k in abbrev:
        text = text.replace(k,abbrev[k])
    emoji = re.findall(r'(?::|;|=)(?:-)?(?:\)|\(|D|P)', text)
    # Remove hyperlinks and lowercase all
    text = re.sub('<.*?>','', text).lower()
    # Remove all punctuations
    text = re.sub('[\W]+', ' ', text) + ' '.join(emoji)
    if using_stop == True:
        clean = [ i for i in text.split() if i not in stopword]
    else:
        clean = [ i for i in text.split() if len(i) > 1]
    if using_stem == True:
        clean = [stemmer.stem(i) for i in clean]
    return clean

#Testing
print(clean(movie['review'][242]))
from sklearn.metrics import precision_recall_curve, average_precision_score
from sklearn.cross_validation import train_test_split
 
def evaluate_classifier(data, clf):
    X, y= data['review'], data['sentiment']
    X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.5, random_state=1)
    
    clf = clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    precision, recall = dict(),dict()
    auc = dict()
    precision, recall, _ = precision_recall_curve(y_test,y_pred)
    auc = average_precision_score(y_test,y_pred)
    return precision, recall, auc

#Using MultiNomial Naive Bayes with Tfidfvectorizer
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_selection import SelectKBest, chi2

tfidf = TfidfVectorizer(ngram_range=(1,1), lowercase=False, preprocessor=None, tokenizer=clean, stop_words=None)
nb = MultinomialNB(alpha=0.1)
selection = SelectKBest(chi2, k=10000)

tfidf_nb = Pipeline([('tfidf', tfidf),('selection', selection), ('nb',nb)])

print("Using TfidfVectorizer with Multinomial Naive Bayes")
t0 = time()
precision1, recall1, auc1 = evaluate_classifier(movie, tfidf_nb)
duration = (time() - t0)/60
print('It takes %.2f minutes to train' % duration)

# Using TFidfVectorized  + Support Vector Machine
from sklearn.svm import LinearSVC
svm = LinearSVC(penalty='l2', C=1.0, random_state=1)
tfidf_svm = Pipeline([("tfidf",tfidf),('selection', selection), ('svm', svm)])

print("Using TfidfVectorizer with Linear Support Vector Machine")
t0 = time()
precision2, recall2, auc2 = evaluate_classifier(movie, tfidf_svm)
t = float(time() - t0)/60
print("Take %.2f minutes to run" % t)

# Logistic Regression + TfidfVectorizer 
from sklearn.linear_model import LogisticRegression
logreg = LogisticRegression(penalty='l2', C=1.0, random_state=0)
tfidf_logreg = Pipeline([('vect', tfidf), ('clf', logreg)])

print("Using TfidfVectorizer with Logistic Regression")
t0 = time()
precision3, recall3, auc3 = evaluate_classifier(movie, tfidf_logreg)
t = float(time() - t0)/60
print("Take %.2f minutes to run" % t)

import matplotlib.pyplot as plt
print('Plotting result')
plt.style.use('seaborn-whitegrid')
plt.figure(figsize=(12,8))
plt.plot(recall1, precision1, label='Multinomial NB: AUC={0:0.2f}'.format(auc1), color='royalblue')
plt.plot(recall2, precision2, label='Linear SVC: AUC={0:0.2f}'.format(auc2), color='salmon')
plt.plot(recall3, precision3, label='Logistic Regression: AUC={0:0.2f}'.format(auc3), color='darkcyan')
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.ylim([0.0, 1.05])
plt.xlim([0.0, 1.0])
plt.title('Precision Recall Curve')
plt.legend(loc="lower left")
plt.show()

''' ESTIMATE PERFOMANCE WITH SHUFFLESPLIT''' #(more accuracte perfomance)
from sklearn.cross_validation import ShuffleSplit

def estimate_shuffle(data, clf):
 
    X, y= data['review'].values, data['sentiment'].values
    cv = ShuffleSplit(n=X.shape[0], n_iter=5, test_size=0.4, random_state=0)
    scores = []
    for train, test in cv:
        try:
            X_test, y_test =X[test], y[test]
            X_train, y_train = X[train], y[train]
            clf.fit(X_train, y_train)
            scores.append(clf.score(X_test, y_test))
        except StopIteration:
            return None
    return np.mean(scores), np.std(scores)
        
print(' Starting classification : Multinomial NB and TfidfVectorizer')        
t0 = time()
mean , std = estimate_shuffle(movie, tfidf_nb)
t = float(time() - t0)/60
print("Take %.2f minutes to run" % t)
print(' Accuracy score is %.3f +/- %.3f' %(mean, std))
'''
 Accuracy score is 0.856 +/- 0.002
 '''
print(' Starting classification : LogRegression and TfidfVectorizer')        
t0 = time()
mean , std = estimate_shuffle(movie, tfidf_logreg)
t = float(time() - t0)/60
print("Take %.2f minutes to run" % t)
print(' Accuracy score is %.3f +/- %.3f' %(mean, std))
'''
 Accuracy score is 0.893 +/- 0.002
 '''
print(' Starting classification : Linear Support Vector Machine and TfidfVectorizer')        
t0 = time()
mean , std = estimate_shuffle(movie, tfidf_svm)
t = float(time() - t0)/60
print("Take %.2f minutes to run" % t)
print(' Accuracy score is %.3f +/- %.3f' %(mean, std))
'''
 Accuracy score is 0.895 +/- 0.003
 '''


''' OUT-OF-CORE LEARNING'''
# HashingVectorizer + SGD LogReg
from sklearn.feature_extraction.text import HashingVectorizer
from sklearn.linear_model import SGDClassifier

sgd = SGDClassifier(loss='log', random_state=1, n_iter=1, n_jobs=-1)
vect = HashingVectorizer(decode_error='ignore',n_features=2**21, preprocessor= None,tokenizer=clean)

def get_minibatch(size):
    X , y = [], []
    with open('movie_data.csv','r') as f:
        next(f)
        for i, line in enumerate(f):
            if i < size:
                review, sentiment = line[:-3], int(line[-2])
                X.append(review)
                y.append(sentiment)
            else:
                break
    return X, y

X_train, y_train = get_minibatch(10000)    
X_train = vect.transform(X_train)
sgd.partial_fit(X_train, y_train, classes=np.array([0,1]))
X_test, y_test = get_minibatch(5000)
X_test = vect.transform(X_test)
print('Accuracy: %.3f' % sgd.score(X_test, y_test))
# Accuracy :0.853
   





    
    