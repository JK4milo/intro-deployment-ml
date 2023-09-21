import pandas as pd
from sklearn.model_selection import train_test_split, cross_validate,GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.ensemble import GradientBoostingRegressor
from utils import update_model, save_simple_metrics_report, get_model_performance_test_set

import prepare
import numpy as np


data = pd.read_csv('./dataset/full_data.csv')

model = Pipeline([
    ('imputer',SimpleImputer(strategy='mean',missing_values=np.nan)),
    ('core_model',GradientBoostingRegressor())
])
print(data.columns)
X =data.drop(['worldwide_gross'],axis=1)
Y = data['worldwide_gross']

X_train,X_test,Y_train,Y_test = train_test_split(X,Y,test_size=0.35,random_state=42)

param_tuning = {'core_model__n_estimators':range(20,301,20)}

grid_search =GridSearchCV(model,param_grid=param_tuning,scoring='r2',cv=5)

grid_search.fit(X_train,Y_train)

final_result = cross_validate(grid_search.best_estimator_,X_train,Y_train,return_train_score=True,cv=5)

train_score = np.mean(final_result['train_score'])
test_score = np.mean(final_result['test_score'])

print(train_score,test_score)

assert train_score >0.7
assert test_score >0.65

update_model(grid_search.best_estimator_)

validation_score = grid_search.best_estimator_.score(X_test,Y_test)

save_simple_metrics_report(train_score,test_score,validation_score,grid_search.best_estimator_)

y_test_predict = grid_search.best_estimator_.predict(X_test)

get_model_performance_test_set(Y_test,y_test_predict)