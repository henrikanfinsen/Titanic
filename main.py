# %%
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, precision_score
=======
from sklearn.svm import LinearSVC

# %% Read data
train = pd.read_csv('train.csv')
test = pd.read_csv('test.csv')

# Drop irrelevant info
train = train.drop('PassengerId', axis=1)
train = train.drop('Cabin', axis=1)
test = test.drop('Cabin', axis=1)
train = train.drop('Embarked', axis=1)
test = test.drop('Embarked', axis=1)
train = train.drop('Ticket', axis=1)
test = test.drop('Ticket', axis=1)
train['Sex'].replace({'male': 0, 'female': 1}, inplace=True)
test['Sex'].replace({'male': 0, 'female': 1}, inplace=True)

# %% Use title to interpolate the missing ages
all_data = [train, test]
for data in all_data:
    data['Status'] = data['Name'].str.extract(' ([A-Za-z]+)\.', expand=False)
<<<<<<< HEAD

# %%
Status_mapping = {"Mr": 0, "Miss": 1, "Mrs": 2,
                  "Master": 3, "Dr": 3, "Rev": 3, "Col": 3, "Major": 3, "Mlle": 3, "Countess": 3,
                  "Ms": 3, "Lady": 3, "Jonkheer": 3, "Don": 3, "Dona": 3, "Mme": 3, "Capt": 3, "Sir": 3}
for dataset in all_data:
    dataset['Status'] = dataset['Status'].map(Status_mapping)

train['Age'].fillna(train.groupby("Status")["Age"].transform("median"), inplace=True)
test['Age'].fillna(test.groupby("Status")["Age"].transform("median"), inplace=True)

train = train.drop('Name', axis=1)
test = test.drop('Name', axis=1)


# %% Interpolate the missing "Fare"
train['Fare'].fillna(train['Fare'].mean(), inplace=True)
test['Fare'].fillna(train['Fare'].mean(), inplace=True)

# %% Family size
train['FamilySize'] = train['SibSp'] + train['Parch'] + 1
test['FamilySize'] = test['SibSp'] + test['Parch'] + 1


# %% New feature: 'Minor'
train['Minor'] = train['Age'] < 16
test['Minor'] = test['Age'] < 16

# %%
y = train["Survived"]

features = ["Pclass", "Minor", "Age", "Sex", "FamilySize", "Status", "Fare"]
X = pd.get_dummies(train[features])
X_test = pd.get_dummies(test[features])

model = RandomForestClassifier(n_estimators=120, max_depth=3, random_state=3)
# model = LinearSVC()
model.fit(X, y)
predictions = model.predict(X_test)

predictions2 = model.predict(X)
print(confusion_matrix(y, predictions2))
print(precision_score(y, predictions2))
print(1 - sum(np.abs(y - predictions2)) / len(y))

# %%
output = pd.DataFrame({'PassengerId': test.PassengerId, 'Survived': predictions})
output.to_csv('my_submission.csv', index=False)
