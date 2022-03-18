import pandas as pd 
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
# Importing the classification report and confusion matrix
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.metrics import accuracy_score
import pickle
df = pd.read_csv('drug200.csv')
nom_cols=['Sex','BP', 'Cholesterol', 'Drug']
ord_cols=[]
#encode
from sklearn.preprocessing import OneHotEncoder, LabelEncoder
le=LabelEncoder()
for column in nom_cols:
    
    df[column]=le.fit_transform(df[column])

#print(df.head)
# split test and train
X = df.drop('Drug', axis=1)
y = df.Drug



def test_model(c,kernel,gamma):
    avg_score = list()
    for i in range (0,10):
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=i)
        if(kernel=='linear'):
            model = SVC(C = c,kernel= kernel)
        else:
            model = SVC(C = c,kernel= kernel,gamma=gamma)   
        model.fit(X_train, y_train.values.ravel())
        y_pred = model.predict(X_test)
        avg_score.append(accuracy_score(y_test, y_pred))
    avg_score = pd.Series(avg_score)    
    return avg_score.mean()*100 

print('model c = 0.1 and accuracy_score',test_model(0.1,'linear',0))
print('model c = 0.5 and accuracy_score',test_model(0.5,'linear',0))
print('model c = 1 and accuracy_score',test_model(1,'linear',0))
print('model c = 10 and accuracy_score : ',test_model(10,'linear',0))

print('model c = 1 & gamma = 0.2  and accuracy_score : ',test_model(1,'rbf',0.2))
print('model c = 1 & gamma = 0.01 and accuracy_score : ',test_model(1,'rbf',0.01))
print('model c = 10 & gamma =0.01 and accuracy_score : ',test_model(10,'rbf',0.01))


clf_model = SVC(kernel='linear')
clf_model.fit(X, y)
y_pred = clf_model.predict(X)
pickle.dump(clf_model, open('model.pkl', 'wb' ))

