import nltk
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
   
   #all_words = nltk.FreqDist(all_words)
   #word_feature=list(all_words.most_common(100))
   #for i in word_feature:
       #str1=str1+" "+i[0]
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
cv = CountVectorizer(max_features = 5000)
X = cv.fit_transform(final_feature).toarray()
X_test = cv.transform(feature_test).toarray()

#Implementing Neural networks
from keras.utils import to_categorical
from keras.models import Sequential
from keras.layers import Dense,Dropout
# Initialising the ANN
classification = Sequential()

# Adding the input layer and the first hidden layer
classification.add(Dense(units = 1250, kernel_initializer = 'uniform', activation = 'relu', input_dim = 2500))
classification.add(Dropout(0.5))

# Adding the second hidden layer
classification.add(Dense(units = 1250, kernel_initializer = 'uniform', activation = 'relu'))
classification.add(Dropout(0.5))

# Adding the second hidden layer
classification.add(Dense(units = 1250, kernel_initializer = 'uniform', activation = 'relu'))
classification.add(Dropout(0.5))

# Adding the output layer
classification.add(Dense(units = 10, kernel_initializer = 'uniform', activation = 'softmax'))

# Compiling the ANN
classification.compile(optimizer = 'adam', loss = 'categorical_crossentropy', metrics = ['accuracy'])

y_train2 = to_categorical(label_train[:9000])
#y_train2 = y_train2[:,1:] dont do this

# Fitting the ANN to the Training set
classification.fit(X[:9000], label_train[:9000], batch_size = 40, epochs = 100)

from sklearn.linear_model import LogisticRegression
clf = LogisticRegression(random_state=0, solver='lbfgs',
                          multi_class='multinomial').fit(X[:9000], label_train[:9000])
print(clf.score(X[9001:],label_train[9001:]))

from sklearn.pipeline import Pipeline
from sklearn.linear_model import SGDClassifier
text_clf = Pipeline([
                      ('clf', SGDClassifier(loss='hinge', penalty='l2',
                                            alpha=1e-3, random_state=42,
                                            max_iter=5, tol=None)),
 ])
text_clf.fit(X[:9000], label_train[:9000]) 
print(text_clf.score(X[9001:],label_train[9001:]))

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import AdaBoostClassifier #84.5 for 3500features
classifier9 = AdaBoostClassifier(LogisticRegression(),n_estimators=500)
classifier9.fit(X[:9000], label_train[:9000])
print(classifier9.score(X[9001:],label_train[9001:]))


from sklearn.naive_bayes import GaussianNB
clf = GaussianNB()
clf.fit(X[:9000], label_train[:9000])
arr=clf.predict(X_test)
print(clf.score(X[9001:],label_train[9001:]))

from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
clf2 = AdaBoostClassifier()
clf2.fit(X[:9000], label_train[:9000])
arr=clf2.predict(X_test)
print(clf2.score(X[9001:],label_train[9001:]))

from sklearn import tree
clf4 = tree.DecisionTreeClassifier(min_samples_split=40)
clf4.fit(X[:9000], label_train[:9000])
arr=clf4.predict(X[9001:],label_train[9001:])
print(clf4.score(X[9001:],label_train[9001:]))


from sklearn.metrics import f1_score
y_true = label_train[9001:]
y_pred = arr
print(f1_score(y_true, y_pred, average='macro'))
print(f1_score(y_true, y_pred, average='micro')) 
print(f1_score(y_true, y_pred, average='weighted'))  
print(f1_score(y_true, y_pred, average=None))


file=open("SPS.txt", "w",encoding='utf-8')
for i in arr:
    if i==1:
        file.write("__label__2")
        file.write("\n")
    else:
        file.write("__label__1")
        file.write("\n")
    
file.close()
