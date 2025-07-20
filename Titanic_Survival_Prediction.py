import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score,classification_report
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder


df=pd.read_csv('train.csv')


df['Age'].fillna(df['Age'].median() ,inplace=True)
df['Embarked'].fillna(df['Embarked'].mode()[0], inplace=True)


label= LabelEncoder()
df['Sex']= label.fit_transform(df["Sex"])
df['Embarked']=label.fit_transform(df['Embarked'])


df = df.drop(['Name', 'Ticket', 'Cabin'], axis=1)


X= df.drop('Survived', axis=1)
y=df['Survived']


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)


y_pred = model.predict(X_test)


accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy * 100:.2f}%")


print(classification_report(y_test, y_pred))

features = X.columns
importances = model.feature_importances_

sns.barplot(x=importances, y=features)
plt.title('Feature Importance')
plt.show()

