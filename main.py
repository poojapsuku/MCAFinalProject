
import pandas as pd
import numpy as np

train = pd.read_csv('dat.csv')


import pandas as pd # our main data management package
from sklearn import model_selection
from sklearn.linear_model import LogisticRegression
import pickle
from sklearn import metrics
from sklearn.neighbors import KNeighborsClassifier


train.head()
train = train.drop(['name', 'gender', 'caste'], axis = 1)

#print(train.count())

data = train.values

X = data[:, 1:8]
Y = data[:, 0]

test_size = 0.33
seed = 7
X_train, X_test, Y_train, Y_test = model_selection.train_test_split(X, Y, test_size=test_size, random_state=seed)
# Fit the model on training set
model = LogisticRegression()
model.fit(X_train, Y_train)

filename = 'finalized_model.sav'
pickle.dump(model, open(filename, 'wb'))


#print("--------------------------------")
#print("printing model")

# load the model from disk
loaded_model = pickle.load(open(filename, 'rb'))
result = loaded_model.score(X_test, Y_test)



#print(result)

#print("--------------------------------")
print("--------------------------------")

Y_pred = model.predict(X_test)

LogReg = round(model.score(X_test, Y_test), 2)

mae_lr = round(metrics.mean_absolute_error(Y_test, Y_pred), 4)
mse_lr = round(metrics.mean_squared_error(Y_test, Y_pred), 4)

# KNN
knn = KNeighborsClassifier(n_neighbors=2)
knn.fit(X_train, Y_train)

Y_pred = knn.predict(X_test)
print(X_test)
KNN = round(knn.score(X_test, Y_test), 2)

mae_knn = metrics.mean_absolute_error(Y_test, Y_pred)
mse_knn = metrics.mean_squared_error(Y_test, Y_pred)

compare_models = pd.DataFrame(
    {'Model': ['LogReg', 'KNN'],
     'Score': [LogReg, KNN],
     'MAE': [mae_lr, mae_knn],
     'MSE': [mse_lr, mse_knn]
     })

print(compare_models)