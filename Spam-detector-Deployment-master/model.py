#importing libraries important for NLP and Machine Learning
import numpy as np
import pandas as pd
import random as rnd
import math

from sklearn.naive_bayes import GaussianNB

# visualization
import seaborn as sns
import matplotlib.pyplot as plt
#%matplotlib inline

#NLP
import nltk
from nltk.corpus import stopwords
import re
import string
from nltk.tokenize import sent_tokenize
from nltk import word_tokenize
from nltk.stem.porter import PorterStemmer
from nltk.stem import WordNetLemmatizer
import pickle

data_train = pd.read_csv('SMSSpamCollection', sep='\t',
                           names=["label", "message"])

nltk.download('stopwords')
nltk.download('wordnet')

stop = set(stopwords.words('english'))

ps = PorterStemmer()   # //Stemming we will do lemitization as well.
lemmatizer = WordNetLemmatizer()
corpus = []
for i in range(0, len(data_train)):
    review = re.sub('[^a-zA-Z]', ' ', data_train['message'][i])
    review = review.lower()
    review = review.split()
    review = [lemmatizer.lemmatize(word) for word in review if word not in set(stopwords.words('english'))]
    #review = [ps.stem(word) for word in review if not word in stopwords.words('english')]
    review = ' '.join(review)
    corpus.append(review)
    
# Creating the TF-IDF model
from sklearn.feature_extraction.text import TfidfVectorizer
cv = TfidfVectorizer(max_features=2500)
X = cv.fit_transform(corpus).toarray()    

pickle.dump(cv,open('transform.pkl','wb'))

combine = [data_train]
titlemapping = {'ham':0, 'spam':1}
for row in combine:
    row["label"] = row["label"].map(titlemapping).astype(int) 
    

y=data_train.iloc[:,0].values

# Train Test Split

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.20, random_state = 0)

# Training model using Naive bayes classifier

from sklearn.naive_bayes import MultinomialNB
spam_detect_model = MultinomialNB().fit(X_train, y_train)


y_pred=spam_detect_model.predict(X_test)
from sklearn.metrics import confusion_matrix 
con =confusion_matrix(y_test,y_pred)

from sklearn.metrics import accuracy_score
acc = accuracy_score(y_test,y_pred)

pickle.dump(spam_detect_model,open('nlp_model.pkl','wb'))










    
    
    
    
    
    
    
    
    
    