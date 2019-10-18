#Import packages
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np


pd.set_option('display.max_columns', 500)

#Load the training data
df_train=pd.read_csv('train.csv')

#Check data shape
print(df_train.shape)

#check first five rows
df_train.head()

#Check missing data
df_train.isnull().sum()

df_train = df_train.drop('Name', axis=1)
df_train = df_train.drop('Ticket', axis=1)
df_train = df_train.drop('Fare', axis=1)


#Fill in the missing age with median age number
df_train['Age'] = df_train['Age'].fillna(df_train['Age'].median())

#Another way is to replace these two with the most frequent one 'S'
df_train['Embarked'].fillna('S',inplace=True)

embarked_mapping = {"S": 0, "C": 1, "Q":2}
df_train['Embarked'] = df_train['Embarked'].map(embarked_mapping)

#There are too many missing values in Cabin, instead of removing this variable, we use 'Missing' to replace it
df_train.loc[df_train['Pclass']==1,'Cabin'] = df_train.loc[df_train['Pclass']==1,'Cabin'].fillna('C')
df_train.loc[df_train['Pclass']==2,'Cabin'] = df_train.loc[df_train['Pclass']==2,'Cabin'].fillna('E')
df_train.loc[df_train['Pclass']==3,'Cabin'] = df_train.loc[df_train['Pclass']==3,'Cabin'].fillna('F')


#Also, the cabin contains numbers, but we care more about the Cabin class, so we will remove numbers.
# mapping each Cabin value with the cabin letter
df_train['Cabin'] = df_train['Cabin'].map(lambda x: x[0])
df_train['Cabin'].unique()

cabin_mapping = {"A": 0, "B": 1,"C": 2, "D": 3,"E": 4, "F": 5,"G": 6, "T": 7}
df_train['Cabin'] = df_train['Cabin'].map(cabin_mapping)

sex_mapping = {"male": 0, "female": 1}
df_train['Sex'] = df_train['Sex'].map(sex_mapping)

df_train["Age"] = df_train["Age"].fillna(-0.5)
bins = [-1, 0, 5, 12, 18, 24, 35, 60, np.inf]
labels = ['0', '1', '2', '3', '4', '5', '6', '7']
df_train['AgeGroup'] = pd.cut(df_train["Age"], bins, labels = labels)

df_train = df_train.drop('Age', axis=1)

from sklearn.model_selection import train_test_split

predictors = df_train.drop(['Survived', 'PassengerId'], axis=1)
target = df_train["Survived"]
x_train, x_val, y_train, y_val = train_test_split(predictors, target, test_size = 0.22, random_state = 0)

from sklearn.metrics import accuracy_score

from sklearn.tree import DecisionTreeClassifier

decisiontree = DecisionTreeClassifier()
decisiontree.fit(x_train, y_train)
y_pred = decisiontree.predict(x_val)
acc_decisiontree = round(accuracy_score(y_pred, y_val) * 100, 2)
print(acc_decisiontree)

df_test=pd.read_csv('test.csv')

df_test = df_test.drop('Name', axis=1)
df_test = df_test.drop('Ticket', axis=1)

#Fill in the missing age with median age number
df_test['Age'] = df_test['Age'].fillna(df_test['Age'].median())

df_test = df_test.drop('Fare', axis=1)

#Another way is to replace these two with the most frequent one 'S'
df_test['Embarked'].fillna('S',inplace=True)

embarked_mapping = {"S": 0, "C": 1, "Q":2}
df_test['Embarked'] = df_test['Embarked'].map(embarked_mapping)

#There are too many missing values in Cabin, instead of removing this variable, we use 'Missing' to replace it
df_test.loc[df_test['Pclass']==1,'Cabin'] = df_test.loc[df_test['Pclass']==1,'Cabin'].fillna('C')
df_test.loc[df_test['Pclass']==2,'Cabin'] = df_test.loc[df_test['Pclass']==2,'Cabin'].fillna('E')
df_test.loc[df_test['Pclass']==3,'Cabin'] = df_test.loc[df_test['Pclass']==3,'Cabin'].fillna('F')

#Also, the cabin contains numbers, but we care more about the Cabin class, so we will remove numbers.
# mapping each Cabin value with the cabin letter
df_test['Cabin'] = df_test['Cabin'].map(lambda x: x[0])
df_test['Cabin'].unique()

cabin_mapping = {"A": 0, "B": 1,"C": 2, "D": 3,"E": 4, "F": 5,"G": 6, "T": 7}
df_test['Cabin'] = df_test['Cabin'].map(cabin_mapping)

sex_mapping = {"male": 0, "female": 1}
df_test['Sex'] = df_test['Sex'].map(sex_mapping)

df_test["Age"] = df_test["Age"].fillna(-0.5)
bins = [-1, 0, 5, 12, 18, 24, 35, 60, np.inf]
labels = ['0', '1', '2', '3', '4', '5', '6', '7']
df_test['AgeGroup'] = pd.cut(df_test["Age"], bins, labels = labels)

df_test = df_test.drop('Age', axis=1)

ids = df_test['PassengerId']
predictions = decisiontree.predict(df_test.drop('PassengerId', axis=1))

#set the output as a dataframe and convert to csv file named submission.csv
output = pd.DataFrame({ 'PassengerId' : ids, 'Survived': predictions })
output.to_csv('submission.csv', index=False)