import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import joblib
from sklearn.metrics import root_mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import ShuffleSplit, GridSearchCV, KFold
from sklearn.feature_selection import r_regression
from sklearn.preprocessing import StandardScaler
from pathlib import Path

# Define a function for data preprocessing (cleaning) that includes all the data tranformations
# necessary for the winner model to be evaluated

def data_preprocessing(path):
    df = pd.read_csv(path)
    metadata_columns = ['Project ID','Experiment type', 'Sex', 'Host age','Disease MESH ID']
    df = df.drop(columns = metadata_columns)
    df = df.iloc[:, 1:]
    X = df.drop(columns=['BMI'])
    y = df['BMI']
    return X, y

# Define a load_data function to load the cleaned dataset path and split it in X and y

def load_data(path):
    df = pd.read_csv(path)
    X = df.drop(columns=['BMI']) # features
    y = df['BMI'] # target
    return X, y

# Create a BaseRegressor class to contain all the function that are the same for 
# the baseline, feature selection and tuning
class BaseRegressor:
    def __init__(self, X, y, eval_X, eval_y, model, model_name, tuning_params_name, save_dir = "models", project_root=Path.cwd().parent):
        self.X = X
        self.y = y
        self.eval_X = eval_X
        self.eval_y = eval_y
        self.model = model
        self.model_name = model_name
        self.tuning_params_name = tuning_params_name
        self.save_dir = save_dir
        self.scaler = StandardScaler() 
        self.project_root = project_root # this is used for the notebook

    def evaluate(self, n_repeats=50, test_size=0.2, random_state=42): # seed = 42 ensures reproducibility
        # list to store metrics for each repetition
        rmse_list, mae_list, r2_list = [], [], []
        
        # splitter for repeated random train-test splits
        splitter = ShuffleSplit(n_splits=n_repeats, test_size=test_size, random_state=random_state)

        # evaluation over multiple test splits
        for _, test_idx in splitter.split(self.eval_X):
            X_test = self.eval_X.iloc[test_idx]
            y_test = self.eval_y.iloc[test_idx]

            # standarize features
            scaler = StandardScaler()
            X_test = scaler.fit_transform(X_test)

            # predict on the scaled test data
            y_pred = self.model.predict(X_test)

            # compute and store evaluation metrics
            rmse_list.append(root_mean_squared_error(y_test, y_pred))
            mae_list.append(mean_absolute_error(y_test, y_pred))
            r2_list.append(r2_score(y_test, y_pred))
        
        # function to compute mean, median and 95% CI
        def summary_stats(metric_list):
            return {
                'mean': np.mean(metric_list),
                'median': np.median(metric_list),
                '95%_CI': (
                    np.percentile(metric_list, 2.5),
                    np.percentile(metric_list, 97.5)
                )
            }
        
        # compute summary statistics for all metrics
        rmse_stats = summary_stats(rmse_list)
        mae_stats = summary_stats(mae_list)
        r2_stats = summary_stats(r2_list)

        # print all the summary statistics
        print("Evaluation Summary ({} Repeats)".format(n_repeats))
        print("=" * 50)
        print("RMSE")
        print("  Mean:   {:.3f}".format(rmse_stats['mean']))
        print("  Median: {:.3f}".format(rmse_stats['median']))
        print("  95% CI: ({:.3f}, {:.3f})".format(*rmse_stats['95%_CI']))
        print("MAE")
        print("  Mean:   {:.3f}".format(mae_stats['mean']))
        print("  Median: {:.3f}".format(mae_stats['median']))
        print("  95% CI: ({:.3f}, {:.3f})".format(*mae_stats['95%_CI']))
        print("R² Score")
        print("  Mean:   {:.3f}".format(r2_stats['mean']))
        print("  Median: {:.3f}".format(r2_stats['median']))
        print("  95% CI: ({:.3f}, {:.3f})".format(*r2_stats['95%_CI']))
        print("=" * 50)

        # create boxplots for all metrics
        metrics = {
                'RMSE': rmse_list,
                'MAE': mae_list,
                'R²': r2_list
                }
        for metric, values in metrics.items():
            mean_val = np.mean(values)
            median_val = np.median(values)

            plt.figure(figsize=(8, 6))
            sns.boxplot(y=values, color='skyblue', showmeans=True,
            meanprops={"marker":"o", "markerfacecolor":"green", "markeredgecolor":"black", "markersize":"8"},
            medianprops={"color": "red", "linewidth": 2})

            plt.title(f"{metric.upper()} Distribution")
            plt.ylabel(metric.upper())

            # add mean and median annotations
            plt.text(0.02, 0.1, f"Mean: {mean_val:.3f}", color='green', transform=plt.gca().transAxes)
            plt.text(0.02, 0.03, f"Median: {median_val:.3f}", color='red', transform=plt.gca().transAxes)

            plt.grid(True, linestyle='--', alpha=0.6)
            plt.tight_layout()
            plt.show()

        return {
            'RMSE': rmse_stats,
            'MAE': mae_stats,
            'R2': r2_list,
        }

    def save(self):
        # create the directory path the model will be saved
        dir_path = self.project_root / self.save_dir
        filepath = dir_path / f"{self.model_name}.pkl"
        
        # create the directory if it doesn't exist
        dir_path.mkdir(parents=True, exist_ok=True)
        
        # save the class instance using joblib
        joblib.dump(self, filepath)

# Baseline model with no feature selection or hyperparameter tuning
class BaselineRegressor(BaseRegressor):
    def train(self):
        # standarize all feautures
        self.X = self.scaler.fit_transform(self.X)
        self.model.fit(self.X, self.y) 
        

# Model with simple filter-based feature selection
class FeatureSelectionRegressor(BaseRegressor):
    def train(self):
        # compute correlations between each feature and target
        correlations = pd.Series(r_regression(self.X, self.y), index=self.X.columns)
        self.feature_correlations = correlations
        
        # select featutes with correlation of at least 0.1
        self.selected_features = correlations[correlations.abs() >= 0.1].index

        # filter training and evaluatoon data to use only selected features
        X_selected = self.X[self.selected_features]
        self.eval_X = self.eval_X[self.selected_features]

        # standarize the selected training features
        X_selected = self.scaler.fit_transform(X_selected)

        self.model.fit(X_selected, self.y)


# Model with both feature selection and hyperparameter tuning
class TuningRegressor(BaselineRegressor):
    def train(self):
        # feature selection process (same as above)
        correlations = pd.Series(r_regression(self.X, self.y), index=self.X.columns)
        self.feature_correlations = correlations
        self.selected_features = correlations[correlations.abs() >= 0.1].index
        X_selected = self.X[self.selected_features]
        self.eval_X = self.eval_X[self.selected_features]

        # standarize selected feautures
        X_scaled = self.scaler.fit_transform(X_selected)

        # define hyperparameter grids
        param_grids = {
            'elastic_net': {
                'alpha': [0.0001, 0.001, 0.01, 0.1, 1.0, 10.0],
                'l1_ratio': [0.1, 0.3, 0.5, 0.7, 0.9, 1.0],
                'max_iter': [1000, 5000],
                'fit_intercept': [True, False],
                'selection': ['cyclic', 'random']
            },
            'svr': {
                'C': [0.01, 0.1, 1, 10, 100],
                'epsilon': [0.001, 0.01, 0.1, 1],
                'kernel': ['rbf', 'linear', 'poly'],
                'degree': [2, 3, 4], 
                'gamma': ['scale', 'auto'],
                'shrinking': [True, False]
            },
            'bayesian_ridge': {
                'alpha_1': [1e-7, 1e-6, 1e-5],
                'alpha_2': [1e-7, 1e-6, 1e-5],
                'lambda_1': [1e-7, 1e-6, 1e-5],
                'lambda_2': [1e-7, 1e-6, 1e-5],
                'fit_intercept': [True],
                'compute_score': [True],
                'tol': [1e-4, 1e-3, 1e-2],
            }
        }   
        
        # Define 5-fold cross-validation strategy with shuffling for better generalization
        cv = KFold(n_splits=5, shuffle=True, random_state=42)
        
        # Set up GridSearchCV with appropriate param grid and scoring metric
        grid = GridSearchCV(estimator=self.model, param_grid=param_grids[self.tuning_params_name],
                            scoring='neg_root_mean_squared_error',
                            cv=cv, n_jobs=-1)

        # Fit the grid search on scaled training data
        self.model = grid.fit(X_scaled, self.y)

        # Update model with the best estimator from grid search
        self.model = grid.best_estimator_
        self.best_params = grid.best_params_
        
        print(f"Best params for {self.tuning_params_name}: {grid.best_params_}")
        print(f"Best RMSE from CV: {-grid.best_score_:.4f}")
        



    

    
