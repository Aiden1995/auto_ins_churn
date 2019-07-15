# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""


import pandas as pd


dataset = pd.read_csv('Auto Insurance.csv',sep=',') 

# Droppping unnecessaru columns

dataset= dataset.drop(['customer_id'], axis=1)
dataset= dataset.drop(['first_name'], axis=1)
dataset= dataset.drop(['last_name','effective_date'],axis=1)
dataset= dataset.drop(['other_drivers_relationship','payment_method','monthly_charges','Yearly_charges','last_rate_change_date'],axis=1)
dataset=dataset.drop(['avgInsurance_rates_zipcode','last_coverage_enquiry_date','initial_sentiment','initial_sentiment'], axis=1)
dataset= dataset.drop(['customers_preferred_value','state','tenure','engineHP_bucket','total_No_claims'], axis = 1)

dataset=dataset.drop([1],axis=0)


#url="https://app.box.com/s/yxm1vhhg9fnnpeflnaanil4znvb47n6b"

X = dataset.iloc[:,1:6]   #Independent variable
y = dataset.iloc[:, 6]      #Dependent variabl


from sklearn.preprocessing import LabelEncoder
labelencoder_X = LabelEncoder()
X.iloc[:, 4] = labelencoder_X.fit_transform(X.iloc[:, 4])  #Change the column value accordingly
#onehotencoder = OneHotEncoder(categorical_features = [0])
#X = onehotencoder.fit_transform(X).toarray()
from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25, random_state = 0)

# 1.Logistic Regression

# Fitting Logistic Regression to the Training set
from sklearn.linear_model import LogisticRegression
classifier = LogisticRegression(random_state = 0)
classifier.fit(X_train, y_train)
                    


# Fitting Logistic Regression to the Training set
from sklearn.linear_model import LogisticRegression
classifier = LogisticRegression(random_state = 0)
classifier.fit(X_train, y_train)

# Predicting the Test set results
y_pred = classifier.predict(X_test)
test1= [[0,2,2,1,0]]
y_pred1 = classifier.predict(test1)
confidence = classifier.predict_proba(test1)
# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)

# Saving the model
from sklearn.externals import joblib
joblib.dump(classifier,'classification_model.pkl')
print('Model Dumped!!!!!!')


joblib.load('classification_model.pkl')

model_columns = list(X.columns)
joblib.dump(model_columns, 'model_columns.pkl')
print("Models columns dumped!")






























