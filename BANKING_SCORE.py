from sklearn.ensemble import RandomForestClassifier
rfc=RandomForestClassifier()
import numpy as np
import streamlit as st 

import zipfile

chemin_zip=(r'train_credit_score.zip')
with zipfile.ZipFile(chemin_zip, 'r') as zip_ref:
    zip_ref.extractall('predictions')

import pandas as pd
df=pd.read_csv('predictions/train_credit_score.csv')
dfb=df.drop_duplicates(subset=['Name'])
dfa=dfb.drop('ID', axis=1)
dfa=dfa.drop('Customer_ID', axis=1)
dfa=dfa.drop('Name', axis=1)
dfa=dfa.drop('SSN', axis=1)
dfa=dfa.drop('Type_of_Loan', axis=1)
dfa=dfa.drop('Delay_from_due_date', axis=1)
dfa=dfa.drop('Changed_Credit_Limit', axis=1)
dfa=dfa.drop('Occupation', axis=1)
dfa=dfa.drop('Payment_Behaviour', axis=1)
dfa=dfa.drop('Payment_of_Min_Amount', axis=1)
dfa['Credit_Mix']=dfa['Credit_Mix'].map({'Good':'2',
                                         'Standard':'1',
                                         'Bad':'0'})
dfa['Credit_Mix']=pd.to_numeric(dfa['Credit_Mix'], errors='coerce')


dfa['Credit_Score']=dfa['Credit_Score'].map({'Good':'2',
                                             'Standard':'1',
                                             'Poor':'0'})
dfa['Credit_Score']=pd.to_numeric(dfa['Credit_Score'], errors='coerce')
y=dfa['Credit_Score']
x=dfa.drop('Credit_Score', axis=1)

from sklearn.model_selection import train_test_split
xtrain ,xtest, ytrain, ytest=train_test_split(x, y, test_size=0.2)
rfc.fit(xtrain, ytrain)
rfc.score(xtest, ytest)

import joblib 
joblib.dump(rfc, 'rf_model.sav')