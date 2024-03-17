import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.ensemble import BaggingRegressor, GradientBoostingRegressor
from sklearn import preprocessing
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
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

  # set Hyperparameters for random forest
gb_params = {
    "n_estimators" : [50,100,150],
    "loss" : ["ls","lad","quantile"],
    "max_features" : [0.25,0.5,1.0],"criterion" : ["mse","friedman_mse"],
    "max_depth" : [3,4,11,14],
}


# Learning 4) GradientBoosting
  # convert data
y_train=y_train.values.ravel()
y_valid=y_valid.values.ravel()
y_test=y_test.values.ravel()

#   # make a GradientBoosting model
# gb_model = GradientBoostingRegressor(random_state=12).fit(x_train,y_train)
# pred = gb_model.predict(x_test)
# RSME = np.sqrt(mean_squared_error(pred,y_test))
#   # calculate RSME
# print(RSME)
#   # calculate score
# print(gb_model.score(x_test,y_test))
  # gridSearchCV for finding best gradient boosting regressor model
# gb_grid = GridSearchCV(GradientBoostingRegressor(random_state=10),gb_params,cv=3, verbose = 1,n_jobs=-1)
#
# gb_grid.fit(x_train,y_train)
# print("Best parameter(Ada Boost) : ")
# print(gb_grid.best_params_)
# print("Best score(Ada Boost) : ")
# print(gb_grid.best_score_)

 # make best ada boost model by best ada boost parameters
best_gb = GradientBoostingRegressor(random_state=12,criterion="mse",loss="ls",max_depth=11,max_features=0.5,n_estimators=150)
  # predict with train data and get score
best_gb.fit(x_train,y_train)
best_pred = best_gb.predict(x_train)
mse = mean_squared_error(best_pred,y_train)
print("------------train_mse with gradient boosting---------------")
print(mse)
print("------------train_score with gradient boosting---------------")
print(best_gb.score(x_train,y_train))


  # predict with valid data and get score
best_gb.fit(x_valid,y_valid)
best_valid_pred = best_gb.predict(x_valid)
valid_mse = mean_squared_error(best_valid_pred,y_valid)
print("------------valid_mse with gradient boosting---------------")
print(valid_mse)
print("------------valid_score with gradient boosting---------------")
print(best_gb.score(x_valid,y_valid))


# predict with test data and get score
best_gb.fit(x_test,y_test)
best_test_pred = best_gb.predict(x_test)
test_mse = mean_squared_error(best_test_pred,y_test)
print("------------test_mse with gradient boosting---------------")
print(test_mse)
print("------------test_root_mse with gradient boosting---------------")
print(np.sqrt(test_mse))
print("------------test_score with gradient boosting---------------")
print(best_gb.score(x_test,y_test))