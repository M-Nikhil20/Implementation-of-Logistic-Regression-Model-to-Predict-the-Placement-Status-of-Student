# Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student

## AIM:
To write a program to implement the the Logistic Regression Model to Predict the Placement Status of Student.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. Import the python library pandas
2. Read the dataset of Placement_Data
3. Copy the dataset in data1
4. Remove the columns which have null values using drop()
5. Import the LabelEncoder for preprocessing of the dataset
6. Assign x and y as status column values
7. From sklearn library select the model to perform Logistic Regression
8. Print the accuracy, confusion matrix and classification report of the dataset

## Program:
```
/*
Program to implement the the Logistic Regression Model to Predict the Placement Status of Student.
Developed by: M Nikhil
RegisterNumber:  212222230095
*/
```
```python
import pandas as pd
data = pd.read_csv('Placement_Data.csv')
data.head()

data1 = data.copy()
data1 = data1.drop(["sl_no","salary"],axis = 1) # Removes the specified row or column
data1.head()

data1.isnull().sum()
data1.duplicated().sum()

from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
data1["gender"] = le.fit_transform(data1["gender"])
data1["ssc_b"] = le.fit_transform(data1["ssc_b"])
data1["hsc_b"] = le.fit_transform(data1["hsc_b"])
data1["hsc_s"] = le.fit_transform(data1["hsc_s"])
data1["degree_t"] = le.fit_transform(data1["degree_t"])
data1["workex"] = le.fit_transform(data1["workex"])
data1["specialisation"] = le.fit_transform(data1["specialisation"])
data1["status"] = le.fit_transform(data1["status"])
data1

x = data1.iloc[:,:-1]
x
y = data1["status"]
y

from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size = 0.2, random_state = 0)

from sklearn.linear_model import LogisticRegression
lr = LogisticRegression (solver ='liblinear') # A Library for Large Linear Classification
lr.fit(x_train,y_train)
y_pred = lr.predict(x_test)
y_pred

from sklearn.metrics import accuracy_score
accuracy = accuracy_score(y_test,y_pred) # Accuracy Score = (TP+TN)/ (TP+FN+TN+FP) ,True +ve/
#accuracy_score (y_true,y_pred, normalize = false)
# Normalize : It contains the boolean value (True/False). If False, return the number of correct
accuracy

from sklearn.metrics import confusion_matrix
confusion = confusion_matrix(y_test,y_pred)
confusion

from sklearn.metrics import classification_report
classification_report1 = classification_report(y_test,y_pred)
print(classification_report1)

lr.predict([[1,80,1,90,1,1,90,1,0,85,1,85]])
```

## Output:
## PLACEMENT DATA 
![image](https://github.com/M-Nikhil20/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/118707852/d41b5588-6140-46bb-81ac-87fb9eb5251b)
## SALARY DATA
![image](https://github.com/M-Nikhil20/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/118707852/5b4b9934-b410-4733-8f55-83652759dfa8)
## CHECKING THE NULL() FUNCTION
![image](https://github.com/M-Nikhil20/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/118707852/6307bf9b-cb8b-4945-9f26-b6842ff34f84)
## DATA DUPLICATE
![image](https://github.com/M-Nikhil20/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/118707852/53c9bccf-76ef-4a44-8540-6513698813c1)
## PRINT DATA
![image](https://github.com/M-Nikhil20/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/118707852/a73dca76-b844-4751-bcad-71ce0aa7d733)
## DATA-STATUS
![image](https://github.com/M-Nikhil20/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/118707852/3514a0bc-120b-44f4-bcbe-cd78957d003b)
## Y_PREDICTION ARRAY
![image](https://github.com/M-Nikhil20/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/118707852/a195ec11-af73-45d1-a825-7201d155bc90)
## ACCURACY VALUE
![image](https://github.com/M-Nikhil20/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/118707852/3075ebdc-a403-4a20-a464-4a86cd4aad7c)
## CONFUSION ARRAY
![image](https://github.com/M-Nikhil20/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/118707852/062f2d98-9eee-4549-860d-51b17b5e7056)
## CLASSIFICATION REPORT
![image](https://github.com/M-Nikhil20/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/118707852/1ca46854-c192-43d1-b11d-14ac8e086545)
## PREDICTION OF LR
![image](https://github.com/M-Nikhil20/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/118707852/a37c4340-9dae-4756-a78f-f8a25d248f74)
![image](https://github.com/M-Nikhil20/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/118707852/3f7e3163-0d44-44c2-a60b-ee724efe34b3)

## Result:
Thus the program to implement the the Logistic Regression Model to Predict the Placement Status of Student is written and verified using python programming.
