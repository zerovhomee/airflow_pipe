import pandas as pd
try:
    from sklearn.model_selection import train_test_split
except ImportError:
    from pip._internal import main as pip
    pip(['install', 'scikit-learn'])
    from sklearn.model_selection import train_test_split

from sklearn.preprocessing import StandardScaler, OrdinalEncoder, OneHotEncoder, PowerTransformer
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer # т.н. преобразователь колонок
from sklearn.linear_model import SGDRegressor
from sklearn.metrics import root_mean_squared_error
import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GridSearchCV

from datetime import datetime
from airflow.models.dag  import DAG
from airflow.operators.python import PythonOperator
import requests
from pathlib import Path
import os
from datetime import timedelta


# на своем датасете изменил исходную функцию и убрал функцию препроцессинга
def scale_frame(frame):
    df = frame.copy()
    
    X = df.drop('target', axis=1)
    y = df['target']

    
    scaler = StandardScaler()
    power_trans = PowerTransformer()
    X_scale = scaler.fit_transform(X.values)
    Y_scale = power_trans.fit_transform(y.values.reshape(-1,1))
    return X_scale, Y_scale, power_trans

def read_data():
    df = pd.read_csv('https://raw.githubusercontent.com/Alexey3250/Start_ML/refs/heads/main/Machine_learning/4_gradient_descend/hw/data.csv')
def clear_data():
    ### Датасет уже подготовленный, так как в airflow препроцессинг работал на другом датасете неправильно
    df = pd.read_csv('https://raw.githubusercontent.com/Alexey3250/Start_ML/refs/heads/main/Machine_learning/4_gradient_descend/hw/data.csv')

    return True

# метрики оставил те же, что и в примере
def eval_metrics(actual, pred):
    rmse = np.sqrt(mean_squared_error(actual, pred))
    mae = mean_absolute_error(actual, pred)
    r2 = r2_score(actual, pred)
    return rmse, mae, r2

def train():
    df = pd.read_csv('https://raw.githubusercontent.com/Alexey3250/Start_ML/refs/heads/main/Machine_learning/4_gradient_descend/hw/data.csv')

    X,Y, power_trans = scale_frame(df)
    X_train, X_val, y_train, y_val = train_test_split(X, Y,
                                                    test_size=0.3,
                                                    random_state=42)
    

    params = {'alpha': [0.0001, 0.001, 0.01, 0.05, 0.1 ],
            'l1_ratio': [0.001, 0.05, 0.01, 0.2],
            "penalty": ["l1","l2","elasticnet"],
            "loss": ['squared_error', 'huber', 'epsilon_insensitive'],
            "fit_intercept": [False, True],
            }
    
    
    lr = SGDRegressor(random_state=42)
    clf = GridSearchCV(lr, params, cv = 3, n_jobs = 4)
    clf.fit(X_train, y_train.reshape(-1))
    best = clf.best_estimator_
    y_pred = best.predict(X_val)
    y_price_pred = power_trans.inverse_transform(y_pred.reshape(-1,1))
    (rmse, mae, r2)  = eval_metrics(power_trans.inverse_transform(y_val), y_price_pred)
    alpha = best.alpha
    l1_ratio = best.l1_ratio
    penalty = best.penalty
    eta0 = best.eta0
    predictions = best.predict(X_train)

    return rmse, mae, r2


 
dag_check = DAG(
    dag_id="train_pipelines",
    start_date=datetime(2025, 3, 26),
    concurrency=4,
    schedule_interval=timedelta(minutes=5),
#    schedule="@hourly",
    max_active_runs=1,
    catchup=False,
)
read_task = PythonOperator(python_callable=read_data, task_id = "read_some", dag = dag_check)
clear_task = PythonOperator(python_callable=clear_data, task_id = "clear_some", dag = dag_check)
train_task = PythonOperator(python_callable=train, task_id = "train_some", dag = dag_check)
read_task >> clear_task >> train_task
