from sklearn.linear_model import ElasticNet, BayesianRidge
from sklearn.svm import SVR
from sklearn.metrics import root_mean_squared_error, mean_absolute_error, r2_score
import joblib

def train_baseline_model(model, X_train, y_train):
    model.fit(X_train, y_train)
    return model

def evaluate_model(model, X_test, y_test):
    predictions = model.predict(X_test)
    metrics = {
        "RMSE": root_mean_squared_error(y_test, predictions, squared=False),
        "MAE": mean_absolute_error(y_test, predictions),
        "R2": r2_score(y_test, predictions)
    }
    return metrics, predictions

def save_model(model, path):
    joblib.dump(model, path) 