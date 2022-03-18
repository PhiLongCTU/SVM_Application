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

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

from sklearn.svm import SVC

clf_model = SVC(kernel='linear')
clf_model.fit(X, y)

pickle.dump(clf_model, open('model.pkl', 'wb' ))

