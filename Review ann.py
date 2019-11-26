from nltk.tokenize import sent_tokenize, word_tokenize
import string
import numpy as np
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
from keras.models import load_model
import pickle

training_file=open("indiahackathon_train_20180928.txt", "r",encoding='utf-8')
test_file=open("indiahackathon_test_20180928.txt", "r",encoding='utf-8')

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
cv = CountVectorizer(max_features = 1000)
X = cv.fit_transform(final_feature).toarray()
X_test = cv.transform(feature_test).toarray()

#Saving the countvector

filename = 'CountVector.pkl'
pickle.dump(cv, open(filename, 'wb'))


# Part 2 - Now let's make the ANN!

# Importing the Keras libraries and packages
from keras.models import Sequential
from keras.layers import Dense

# Initialising the ANN
classifier = Sequential()

# Adding the input layer and the first hidden layer
classifier.add(Dense(units = 20, kernel_initializer = 'uniform', activation = 'sigmoid', input_dim = 1000))

# Adding the second hidden layer
classifier.add(Dense(units = 20, kernel_initializer = 'uniform', activation = 'sigmoid'))

# Adding the output layer
classifier.add(Dense(units = 1, kernel_initializer = 'uniform', activation = 'sigmoid'))

# Compiling the ANN
classifier.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])

# Fitting the ANN to the Training set
classifier.fit(X[:9000], label_train[:9000], batch_size = 10, epochs = 100)

# Part 3 - Making predictions and evaluating the model

# Predicting the Test set results
y_pred = classifier.predict(X[9001:])
for i in range(len(y_pred)): 
    if y_pred[i] >0.5:
        y_pred[i]=1
    else:
        y_pred[i]=0

from sklearn.metrics import f1_score
y_true = label_train[9001:]
print(f1_score(y_true, y_pred, average='macro'))
print(f1_score(y_true, y_pred, average='micro')) 
print(f1_score(y_true, y_pred, average='weighted'))  
print(f1_score(y_true, y_pred, average=None))

#Saving the model
#The below package is used for loading the saved model. Did not use it, just for information.

#To save the model which we have trained.
classifier.save("ReviewModel.h5")





