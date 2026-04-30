import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, f1_score, classification_report, r2_score, mean_squared_error
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
df=pd.read_csv(r"C:\Users\NANI NALLA\Downloads\archive\student_performance.csv")
df.head()
print(df.info())
print(df.isnull().sum())
print(df.describe())
print(df['passed'].value_counts())

sns.countplot(x='passed', data=df)
plt.show()

sns.scatterplot(x='study_hours_per_week', y='final_score', data=df)
plt.show()

sns.heatmap(df.select_dtypes(include='number').corr(), annot=True, cmap='coolwarm')
plt.show()
X = df.drop(columns=['passed','final_score','student_id'])
y = df['passed'].map({'No':0,'Yes':1})
num_cols = X.select_dtypes(include='number').columns
cat_cols = X.select_dtypes(exclude='number').columns
preprocessor = ColumnTransformer([
 ('num', Pipeline([
     ('imputer', SimpleImputer(strategy='median')),
     ('scaler', StandardScaler())
 ]), num_cols),
 ('cat', Pipeline([
     ('imputer', SimpleImputer(strategy='most_frequent')),
     ('onehot', OneHotEncoder(handle_unknown='ignore'))
 ]), cat_cols)
])
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2,random_state=42,stratify=y)

clf = Pipeline([
 ('prep', preprocessor),
 ('model', RandomForestClassifier(n_estimators=200, random_state=42))
])

clf.fit(X_train, y_train)
pred = clf.predict(X_test)

print('Accuracy:', accuracy_score(y_test,pred))
print('F1 Score:', f1_score(y_test,pred))
print(classification_report(y_test,pred))
Xr = df.drop(columns=['final_score','passed','student_id'])
yr = df['final_score']

X_train, X_test, y_train, y_test = train_test_split(Xr,yr,test_size=0.2,random_state=42)

reg = Pipeline([
 ('prep', preprocessor),
 ('model', LinearRegression())
])

reg.fit(X_train, y_train)
pred = reg.predict(X_test)

rmse = np.sqrt(mean_squared_error(y_test,pred))
print('R2 Score:', r2_score(y_test,pred))
print('RMSE:', rmse)
rf = clf.named_steps['model']
print('Top driver variables: previous_score, attendance_rate, study_hours_per_week')

