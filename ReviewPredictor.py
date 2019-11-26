# -*- coding: utf-8 -*-
"""
Created on Tue Sep 10 16:13:09 2019

@author: Shivam-PC
"""
from nltk.tokenize import sent_tokenize, word_tokenize
import string
import numpy as np
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
from keras.models import load_model
import pickle
from twitterApi import getTweets

stopwords = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()

def text_process(sent):
   all_words=[]
   sent=sent.translate(str.maketrans("","",string.punctuation))
   tokenized=word_tokenize(sent)
   for w in tokenized:
       word=w.lower()
       if word.isalpha() and word not in stopwords:
           word=lemmatizer.lemmatize(word)
           all_words.append(word)
   str2=" ".join(all_words)
   return str2


def predict(tst, classification, cv):
    tstl=[]
    tstl.append(text_process(tst))
    
    Xt = cv.transform(tstl).toarray()
    y=classification.predict(Xt)
    
    return y

    
def predictKey(text):
    tweets = getTweets(text)
    classification = load_model("ReviewModel.h5")
    cv=pickle.load(open("CountVector.pkl", 'rb'))
    data=[]
    positive = 0
    negative = 0
    neutral = 0
    for tweet in tweets:
        print(tweet)
        pred = predict(tweet,classification,cv)
        if pred>0.6:
            pred='positive'
            positive+=1
        elif pred<0.4:
            pred = 'negative'
            negative+=1
        elif pred>=0.4 and pred<=0.6:
            pred = 'neutral'
            neutral+=1
        data.append([tweet,pred])
    return data
#d = predictKey('delhi')

