import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
raw_mail_data = pd.read_csv('mail_data.csv')
mail_data = raw_mail_data.where((pd.notnull(raw_mail_data)), '')
mail_data.shape
mail_data.head()
#label spam mail as 0; ham mail as 1
mail_data.loc[mail_data['Category'] == 'spam', 'Category',] = 0
mail_data.loc[mail_data['Category'] == 'ham', 'Category',] = 1
# separating the data as texts and label
X = mail_data['Message']
Y = mail_data['Category']
#splitting the data into training data and testing data
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=3)
#transfrom  the text data to feature vectors that can be used as input to the logistic regression
feature_extraction = TfidfVectorizer(min_df=1, stop_words='english', lowercase=True)

X_train_features = feature_extraction.fit_transform(X_train)
X_test_features = feature_extraction.transform(X_test)

#convert Y_train and Y_test values as integers
Y_train = Y_train.astype('int')
Y_test = Y_test.astype('int')
# Training the Logistic Regression model
model = LogisticRegression()
model.fit(X_train_features, Y_train)

# # Evaluating the trained model
# prediction_on_train_data = model.predict(X_train_features)
# accuracy_on_train_data = accuracy_score(Y_train, prediction_on_train_data)
# # print(accuracy_on_train_data)
# prediction_on_test_data = model.predict(X_test_features)
# accuracy_on_test_data = accuracy_score(Y_test, prediction_on_test_data)
# # print(accuracy_on_test_data)

import joblib

# Save the model to disk
joblib.dump(model, 'spam_classifier_model.pkl')
joblib.dump(feature_extraction, 'vectorizer_fe.pkl')
print('saved')

#Building a Predictive System

