# -DS-3-1-LAB5
written by 2021.05.26


### Purpose
---
To predict Median house value


### Data
---
Housing.csv
<https://www.kaggle.com/datasets/camnugent/california-housing-prices>


### Preprocessing
---
1. Oridinal Encoding
2. MinMax Scaling
3. Fill NaN value with mean value

### Algorithms (ALL Regressor)
---
1. Bagging
   
+DecisionTreeRegressor & grid search =>Find best decision tree

    best_tree = DecisionTreeRegressor(criterion="mse",max_depth=10,max_features="auto",splitter="best",random_state=121)

+BaggingRegressor & grid search =>Find best decision tree

    best_bagging = BaggingRegressor(DecisionTreeRegressor(criterion="mse",max_depth=10,max_features="auto",splitter="best",random_state=121),
                               bootstrap = False, bootstrap_features = True, max_features = 1.0,max_samples = 1.0,
                               n_estimators = 150)

3. RandomForest
4. AdaBoost
5. GradientBoost

