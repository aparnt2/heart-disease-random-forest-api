import pandas as pd 
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix,accuracy_score
import joblib
df=pd.read_csv('heart.csv')
X=df.drop('target' ,axis=1)
y=df['target']
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=42)
model=RandomForestClassifier(
    n_estimators=100,
    max_depth=3,
    random_state=42
)
model.fit(X_train,y_train)
print(df.info())
prediction=model.predict(X_test)
print("actual value:",y_test.values)
print("prediction :",prediction)
print("accuracy:",accuracy_score(y_test,prediction))
print("confusion matrix:",confusion_matrix(y_test,prediction))
joblib.dump(model,'models/heart_model.joblib')


