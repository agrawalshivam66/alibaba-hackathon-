import nltk
from nltk.corpus import movie_reviews
from nltk.classify.scikitlearn import SklearnClassifier
from sklearn.naive_bayes import MultinomialNB, BernoulliNB
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.svm import SVC, LinearSVC, NuSVC
from nltk.classify import ClassifierI
from statistics import mode
from nltk.tokenize import sent_tokenize, word_tokenize
import string
import numpy as np
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords

training_file=open("indiahackathon_train_20180928.txt", "r",encoding='utf-8')

test_file=open("indiahackathon_test_20180928.txt", "r",encoding='utf-8')

stopwords = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()

def text_process(sent):
   all_words=[]
   str1=''
   sent=sent.translate(str.maketrans("","",string.punctuation))
   tokenized=word_tokenize(sent)
   for w in tokenized:
       word=w.lower()
       if word.isalpha() and word not in stopwords:
           word=lemmatizer.lemmatize(word)
           all_words.append(word)
   str1 = ' '.join(all_words)
   return str1

train_list=[]
with training_file as inputfile1:
   for line in inputfile1:
       train_list.append(line)


feature_test=[]
with test_file as inputfile2:
   for line in inputfile2:
       feature_test.append(text_process(line))

label_train=[]
feature_train=[]
final_feature=[]


for elements in train_list:
  feature=elements.split(" ",1)
  feature_train.append(feature[1])
  if feature[0]=='__label__1':
     label_train.append(0)
  else:
     label_train.append(1) 

for elements in feature_train:
   processed=text_process(elements)
   final_feature.append(processed)
   
from sklearn.feature_extraction.text import CountVectorizer
cv = CountVectorizer(max_features = 3500)
X = cv.fit_transform(final_feature).toarray()

X_test = cv.transform(feature_test).toarray()

#X=X.reshape(-1, 1)
from sklearn.naive_bayes import GaussianNB
clf = GaussianNB()
clf.fit(X[:9900], label_train[:9900])
arr=clf.predict(X_test)
print(clf.score(X[9001:],label_train[9001:]))
file=open("SPS.txt", "w",encoding='utf-8')
for i in arr:
    if i==1:
        file.write("__label__2")
        file.write("\n")
    else:
        file.write("__label__1")
        file.write("\n")
    
file.close()