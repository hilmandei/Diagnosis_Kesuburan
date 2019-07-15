import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC

df = pd.read_csv('fertility.csv')

print(len(df.columns))
for i in df.columns:
    print(i, df[i].unique(), '\n')

dfNew = df[['Age', 'Number of hours spent sitting per day']].copy()
# print(dfNew)

dfdummy = pd.get_dummies(df[['Childish diseases', 'Accident or serious trauma', 'Surgical intervention',
                             'Frequency of alcohol consumption', 'Smoking habit']])

dfNew = pd.concat([dfNew, dfdummy], axis='columns')

# print((dfNew.columns))

# model spliting =================
from sklearn.model_selection import train_test_split
xtrain, xtest, ytrain, ytest = train_test_split(dfNew, df['Diagnosis'], test_size=0.1, random_state=3)

print(xtrain.head())
print((xtrain.columns))


# Model Fitting =================
modelLR = LogisticRegression(solver='liblinear', multi_class='auto')
modelKNN = KNeighborsClassifier()
modelSVC = SVC(gamma='auto')


modelLR.fit(xtrain, ytrain)
print(modelLR.score(xtest, ytest)*100)

modelKNN.fit(xtrain, ytrain)
print(modelKNN.score(xtest, ytest)*100)

modelSVC.fit(xtrain, ytrain)
print(modelSVC.score(xtest, ytest)*100)

# Model Prediction
'''
[
'Age', 
'Number of hours spent sitting per day', 
'Childish diseases_no',
'Childish diseases_yes', 
'Accident or serious trauma_no',
'Accident or serious trauma_yes', 
'Surgical intervention_no',
'Surgical intervention_yes',
'Frequency of alcohol consumption_every day',
'Frequency of alcohol consumption_hardly ever or never',
'Frequency of alcohol consumption_once a week',
'Frequency of alcohol consumption_several times a day',
'Frequency of alcohol consumption_several times a week',
'Smoking habit_daily', 
'Smoking habit_never',
'Smoking habit_occasional'
]

'''
#     1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13,14, 15, 16

arin = [29, 5, 1, 0, 1, 0, 1, 0, 1, 0, 0, 0, 0, 1, 0, 0]

bebi = [31, 24, 1, 0, 0, 1, 0, 1, 1, 0, 0, 0, 1, 0, 1, 0]

caca = [25, 7, 0, 1, 1, 0, 1, 0, 0, 1, 0, 0, 0, 0, 1, 0]

dini = [28, 24, 1, 0, 0, 1, 0, 1, 0, 1, 0, 0, 0, 1, 0, 0]

enno = [42, 8, 0, 1, 1, 0, 1, 0, 0, 1, 0, 0, 0, 0, 1, 0]


wanita = [arin, bebi, caca, dini, enno]
nama = ['arin', 'bebi', 'caca', 'dini', 'enno']


for i in range(len(wanita)):
    print(nama[i], 'prediksi kesuburan : ', modelLR.predict([wanita[i]])[0], '(LogisticRegression)')
    print(nama[i], 'prediksi kesuburan : ', modelKNN.predict([wanita[i]])[0], '(KNeighborsClassifier')
    print(nama[i], 'prediksi kesuburan : ', modelSVC.predict([wanita[i]])[0], '(SupportVectorClassifier)')
    print()




