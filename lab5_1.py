import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.ensemble import BaggingRegressor
from sklearn import preprocessing
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import GridSearchCV

# Open the datafile 'housing.csv'
df=pd.read_csv('housing.csv')

# Preprocessing the data
  # Encode
def ordinalEncode_category(arr,str):
    enc=preprocessing.OrdinalEncoder()
    encodedData=enc.fit_transform(arr[[str]])
    column_name = str + " ordinalEncoded"
    arr[column_name] = encodedData
    return arr
df=ordinalEncode_category(df,"ocean_proximity")
df=df.drop(columns="ocean_proximity")

  # filling nan value
df.fillna(df.mean(),inplace=True)
  # scaling by MinMax
mm_df = pd.DataFrame(MinMaxScaler().fit_transform(df),columns = df.columns)


# preprocessing finished. The result of data (top 20)
print(df.head(20))

#First set train set, test set, validation set and hyperparameters
x=mm_df[["population","longitude","latitude","housing_median_age","total_rooms","total_bedrooms","households","median_income","ocean_proximity ordinalEncoded"]]
y=mm_df[["median_house_value"]]
  # set x, y values, and split them to test and training (0.8,0.2)
x_train,x_test,y_train,y_test=train_test_split(x,y,train_size=0.8,test_size=0.2,random_state=1)
  # set x_train, y_train values, and split them to validation and training (0.8,0.2)
x_train,x_valid,y_train,y_valid=train_test_split(x_train,y_train,train_size=0.8,test_size=0.2,random_state=1)
  # set Hyperparameters for bagging and decision tree
bagging_params = {
    "n_estimators" : [50,100,150],
    "max_samples" : [0.25,0.5,1.0],
    "max_features" : [0.25,0.5,1.0],
    "bootstrap" : [True,False],
    "bootstrap_features" : [True,False]
}

tree_params = {
    "criterion":["mse", "friedman_mse", "mae", "poisson"],
    "splitter" : ["best","random"],
    "max_depth" : [2,6,10,14],
    "max_features" : ["auto","sqrt","log2",5,None]
}

# Learning 1) DecisionTreeRegressor
  # make a decision tree model
tree_model = DecisionTreeRegressor(random_state=12).fit(x_train,y_train)
pred = tree_model.predict(x_test)
RSME = np.sqrt(mean_squared_error(pred,y_test))
  # calculate RSME
print(RSME)
  # gridSearchCV for finding best decision tree model
tree_grid = GridSearchCV(DecisionTreeRegressor(random_state = 122),
                         tree_params,
                         scoring="neg_mean_squared_error",
                         verbose = 1,
                         n_jobs = -1)
  # make a gridSearchCV model
tree_grid.fit(x_train,y_train)
  # Best decision tree - parameters & score
print("Best parameter(Decision Tree) : ")
print(tree_grid.best_params_)
print("Best score(Decision Tree) : ")
print(tree_grid.best_score_)

  # make decision tree with best parameters
best_tree = DecisionTreeRegressor(criterion="mse",max_depth=10,max_features="auto",splitter="best",random_state=121)

  # predict with train data and get score
best_tree.fit(x_train,y_train)
best_pred = best_tree.predict(x_train)
mse = mean_squared_error(best_pred,y_train)
print("------------train_mse---------------")
print(mse)
print("------------train_score---------------")
print(best_tree.score(x_train,y_train))


  # predict with valid data and get score
best_tree.fit(x_valid,y_valid)
best_valid_pred = best_tree.predict(x_valid)
valid_mse = mean_squared_error(best_valid_pred,y_valid)
print("------------valid_mse---------------")
print(valid_mse)
print("------------valid_score---------------")
print(best_tree.score(x_valid,y_valid))


  # predict with test data and get score
best_tree.fit(x_test,y_test)
best_test_pred = best_tree.predict(x_test)
test_mse = mean_squared_error(best_test_pred,y_test)
print("------------test_mse---------------")
print(test_mse)
print("------------test_root_mse---------------")
print(np.sqrt(test_mse))
print("------------test_score---------------")
print(best_tree.score(x_test,y_test))


  # gridSearchCV for finding best bagging decision tree model
bagging_grid = GridSearchCV(BaggingRegressor(DecisionTreeRegressor()),
                             bagging_params,
                             cv=4,
                            verbose = 1,
                            n_jobs=-1)

  # convert data
y_train=y_train.values.ravel()
y_valid=y_valid.values.ravel()
y_test=y_test.values.ravel()

bagging_grid.fit(x_train,y_train)
print("Best parameter(Bagging) : ")
print(bagging_grid.best_params_)
print("Best score(Bagging) : ")
print(bagging_grid.best_score_)

 # make best bagging decision tree model by bagging parameters & best decision tree parameters
best_bagging = BaggingRegressor(DecisionTreeRegressor(criterion="mse",max_depth=10,max_features="auto",splitter="best",random_state=121),
                               bootstrap = False, bootstrap_features = True, max_features = 1.0,max_samples = 1.0,
                               n_estimators = 150)
  # predict with train data and get score
best_bagging.fit(x_train,y_train)
best_pred = best_bagging.predict(x_train)
mse = mean_squared_error(best_pred,y_train)
print("------------train_mse with bagging---------------")
print(mse)
print("------------train_score with bagging---------------")
print(best_bagging.score(x_train,y_train))


  # predict with valid data and get score
best_bagging.fit(x_valid,y_valid)
best_valid_pred = best_bagging.predict(x_valid)
valid_mse = mean_squared_error(best_valid_pred,y_valid)
print("------------valid_mse with bagging---------------")
print(valid_mse)
print("------------valid_score with bagging---------------")
print(best_bagging.score(x_valid,y_valid))


# predict with test data and get score
best_bagging.fit(x_test,y_test)
best_test_pred = best_bagging.predict(x_test)
test_mse = mean_squared_error(best_test_pred,y_test)
print("------------test_mse with bagging---------------")
print(test_mse)
print("------------test_root_mse with bagging---------------")
print(np.sqrt(test_mse))
print("------------test_score with bagging---------------")
print(best_bagging.score(x_test,y_test))

#
#
#
# print(y_train.values.ravel())
#
# model=BaggingRegressor(n_estimators=100,max_samples=20,max_features=3)
# model.fit(x_train,y_train.ravel())
#
# prediction=model.predict(x_test)
# score=model.score(x_test,y_test.ravel())
#
# bg=BaggingRegressor()
# grid_param={'n_estimators': [100, 300, 500, 800, 1000]}
# gd_sr=GridSearchCV(estimator=bg, param_grid=grid_param,scoring='accuracy',cv=5,n_jobs=1)
# gd_sr.fit(x_train,y_train.values.ravel())
# best_parameters=gd_sr.best_params_
# print(best_parameters)
# best_result=gd_sr.best_score_
# print("Best parameter : "+best_parameters+"\nBest score : "+best_result)
