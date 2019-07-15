# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""


import pandas as pd
from sklearn import preprocessing


dataset = pd.read_csv('Auto Insurance.csv',sep=',') 


# Droppping unnecessaru columns

dataset= dataset.drop(['customer_id'], axis=1)
dataset= dataset.drop(['first_name'], axis=1)
dataset= dataset.drop(['last_name','effective_date'],axis=1)
dataset= dataset.drop(['other_drivers_relationship','payment_method','monthly_charges','Yearly_charges','last_rate_change_date'],axis=1)
dataset= dataset.drop(['avgInsurance_rates_zipcode','last_coverage_enquiry_date','initial_sentiment','initial_sentiment'], axis=1)
dataset= dataset.drop(['customers_preferred_value','customer_since ','state','tenure','engineHP_bucket','total_No_claims'], axis = 1)
#dataset.columns


testData= dataset.iloc[0]
testData= testData.transpose()
dataset=dataset.drop([0],axis=0)




#url="https://app.box.com/s/yxm1vhhg9fnnpeflnaanil4znvb47n6b"


#Not using label encoder for the logic issue .
#from sklearn.preprocessing import LabelEncoder
#labelencoder_X = LabelEncoder()
#X.iloc[:, 4] = labelencoder_X.fit_transform(X.iloc[:, 4])  #Change the column value accordingly


#pd.get_dummies(dataset['gender'], prefix='Gender_')
data=dataset

columns_data=[ 'gender', 'age_bucket', 'coverage', 'contract_type',
        'employment_status',  'dependents',
       'marital_Status',  'final-sentiment',
       'auto_and_home_owners_policy_bundle_status',
       'auto_and_renters_policy_bundle_Status',
       'has_customer_dropped_any_coverage',
       'home_owner_status', 'customer_zip_code_change',
        'bought_new_car', 'bought_new_house',
       'vehicle_type', 'vehicle_size', 'miles_driven_annually_bucket'  ]

for each_coulmn in columns_data:
   encoded_data= pd.get_dummies(data[each_coulmn],prefix=each_coulmn+'_')
   data=data.join(encoded_data)

data=data.drop(columns_data,axis=1)
data.columns

np_data = data.values #returns a numpy array
min_max_scaler = preprocessing.MinMaxScaler()
x_scaled = min_max_scaler.fit_transform(np_data)
df = pd.DataFrame(x_scaled)

X = df.iloc[:,1:]   #Independent variable
y = df.iloc[:,0]      #Dependent variabl





from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25, random_state = 0)

# 1.Logistic Regression
# Fitting Logistic Regression to the Training set
from sklearn.linear_model import LogisticRegression
classifier_logistic = LogisticRegression(random_state = 2)
classifier_logistic.fit(X_train, y_train)

from sklearn.neighbors import KNeighborsClassifier
classifier_knn = KNeighborsClassifier(n_neighbors=10)
classifier_knn.fit(X_train, y_train)

X_train.describe()
# Predicting the Test set results
#y_pred = classifier.predict(X_test)
y_pred_neighbour= classifier_knn.predict(X_test)
test1= [[0,2,2,1,0]]
y_pred1 = classifier_knn.predict(test1)
#confidence = classifier.predict_proba(test1)
# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred_neighbour)

# Saving the model
from sklearn.externals import joblib
joblib.dump(classifier_knn,'classification_model.pkl')
print('Model Dumped!!!!!!')


joblib.load('classification_model.pkl')

model_columns = list(X.columns)
joblib.dump(model_columns, 'model_columns.pkl')
print("Models columns dumped!")






























