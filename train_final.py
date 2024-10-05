import os
import sys
import logging
import numpy as np
import pandas as pd
from scipy.stats import norm
from sklearn.model_selection import train_test_split, GridSearchCV, KFold
from sklearn.metrics import roc_curve, f1_score, recall_score, precision_score, accuracy_score, roc_auc_score
from sklearn.utils import resample
from options.train_options import TrainOptions
from imblearn.under_sampling import RandomUnderSampler
from imblearn.over_sampling import SMOTE
from imblearn.over_sampling import ADASYN
from imblearn.pipeline import Pipeline
import joblib  # 用于保存和加载模型

def read_data(data_path, target, test_size, cols, random_state=None):
    """
    Read data from the data_path
    :param data_path: the path of the data
    :param target: the target variable
    :param test_size: the ratio of test dataset
    :param cols: the variables of the data need to be normalized
    :param random_state: the random state
    :return: x_train, x_test, y_train, y_test
    """
    df = pd.read_csv(data_path)
    y = df[target]  # Child is alive (1=yes,0=no)
    x = df[[i for i in df.columns if i != target]]  # all variables left
    normalize_data(x, cols)
    return train_test_split(x, y, test_size=test_size, random_state=random_state)


def normalize_data(data, columns):
    """
    Normalize the data
    :param data: a dataframe need to be normalized
    :return: None, the data will be normalized in place
    """
    data[columns] = data[columns].apply(lambda x: (x - x.min()) / (x.max() - x.min()))


def train_model(model, x_train, y_train, param_grid, random_state=None, k=5):
    """
    Train the model with kfold cross validation and grid search
    :param model: the model
    :param x_train: the training data
    :param y_train: the training target
    :param param_grid: the parameters of the model
    :param random_state: the random state
    :param k: the number of folds
    :return: the best parameters and the best estimator
    """
    # use kfold cross validation to find the best parameters of the model
    kf = KFold(n_splits=5, shuffle=True, random_state=0)
    print('start search param')
    grid_search = GridSearchCV(model, param_grid, cv=kf, scoring='accuracy',n_jobs = 10)
    grid_search.fit(x_train, y_train)

    return grid_search.best_params_, grid_search.best_estimator_


def AUC_CI(y_true, y_pred, n_bootstrap=1000, ci_level=0.95, random_state=None):
    """
    Calculate the confidence interval of AUC
    :param y_true: the true label
    :param y_pred: the predicted label
    :param n_bootstrap: the number of bootstrap
    :param ci_level: the confidence level
    :param random_state: the random state
    :return: the confidence interval of AUC
    """
    y_true_arr = np.array(y_true)
    n_samples = len(y_true_arr)
    auc_scores = []
    np.random.seed(random_state)
    for _ in range(n_bootstrap):
        idx = np.random.choice(n_samples, size=n_samples, replace=True)
        auc_scores.append(roc_auc_score(y_true_arr[idx], y_pred[idx]))
    
    auc_scores.sort()
    lower = int(n_bootstrap * (1 - ci_level) / 2)
    upper = int(n_bootstrap * (1 + ci_level) / 2)

    return auc_scores[lower], auc_scores[upper]
    
def ACC_CI(y_true, y_pred, n_bootstrap=1000, ci_level=0.95, random_state=None):
    """
    Calculate the confidence interval of Accuracy (ACC)
    :param y_true: the true label
    :param y_pred: the predicted label
    :param n_bootstrap: the number of bootstrap samples
    :param ci_level: the confidence level
    :param random_state: the random state for reproducibility
    :return: the confidence interval of ACC
    """
    y_true_arr = np.array(y_true)
    n_samples = len(y_true_arr)
    acc_scores = []
    np.random.seed(random_state)
    for _ in range(n_bootstrap):
        idx = np.random.choice(n_samples, size=n_samples, replace=True)
        acc_scores.append(accuracy_score(y_true_arr[idx], y_pred[idx]))

    acc_scores.sort()
    lower = int(n_bootstrap * (1 - ci_level) / 2)
    upper = int(n_bootstrap * (1 + ci_level) / 2)

    return acc_scores[lower], acc_scores[upper]

def bootstrap_metric(y_true, y_pred, metric_func, n_bootstrap=1000, random_state=None):
    """
    Bootstrap to compute the mean and standard deviation of a metric.
    :param y_true: True labels
    :param y_pred: Predicted labels (or probabilities for AUC)
    :param metric_func: Metric function (e.g., f1_score, recall_score, precision_score, accuracy_score)
    :param n_bootstrap: Number of bootstrap samples
    :param random_state: Random state for reproducibility
    :return: mean and std of the metric across bootstrap samples
    """
    np.random.seed(random_state)
    metric_values = []
    n_samples = len(y_true)
    
    for _ in range(n_bootstrap):
        # Randomly sample indices with replacement
        indices = np.random.choice(range(n_samples), size=n_samples, replace=True)
        y_true_sample = np.array(y_true)[indices]  # Convert to NumPy array and subset the true values
        
        # Special case for AUC: we need to pass the probabilities (y_pred_prob)
        if metric_func == roc_auc_score:
            y_pred_sample = y_pred[indices]  # Use probability scores for AUC
        else:
            y_pred_sample = np.array(y_pred)[indices]  # Use NumPy indexing for other metrics
        
        # Compute the metric for this bootstrap sample
        metric_values.append(metric_func(y_true_sample, y_pred_sample))
    
    return np.mean(metric_values), np.std(metric_values)

def caculate_p(auc, ci, baseline=0.5):
    """
    Calculate the p value of AUC
    :param auc: the auc value
    :param ci: the confidence interval
    :param baseline: the baseline
    :return: the p value
    """
    z_score = (auc - baseline) / ((ci[1] - ci[0]) / (2 * norm.ppf(0.975)))
    p_value = 2 * (1 - norm.cdf(np.abs(z_score)))
    return p_value

if __name__ == '__main__':
    # Setting logging
    logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
    
    # Load options
    opt = TrainOptions().parse()

    # Read feature columns
    with open(os.path.join(opt.dataroot, opt.saveroot, "final_data_continuous_vars.txt")) as f:
        continuous_vars = eval(f.read())

    # Initialize the model based on the options
    if opt.model == 'XGBoost':
        from xgboost.sklearn import XGBClassifier
        model = XGBClassifier(n_jobs=14)
        param_grid = {'max_depth': [5], 'n_estimators': [3000], 'learning_rate': [0.3]}
    elif opt.model == 'RandomForest':
        from sklearn.ensemble import RandomForestClassifier
        model = RandomForestClassifier()
        param_grid = {'n_estimators': [1000]}
    elif opt.model == 'NaiveBayes':
        from sklearn.naive_bayes import GaussianNB
        model = GaussianNB()
        param_grid = {'var_smoothing': [1e-9]}
    elif opt.model == "LogisticRegression":
        from sklearn.linear_model import LogisticRegression
        model = LogisticRegression()
        param_grid = {'C': [1]}
    elif opt.model == "NeuralNetwork":
        from sklearn.neural_network import MLPClassifier
        model = MLPClassifier()
        param_grid = {'hidden_layer_sizes': [(300,)], 'alpha': [0.001], 'max_iter': [300]}
    elif opt.model == "Bagging":
        from sklearn.ensemble import BaggingClassifier
        model = BaggingClassifier()
        param_grid = {'n_estimators': [200], 'max_samples': [1.0], 'max_features': [0.5]}
    else:
        logging.warning("Model not found")
        sys.exit(0)

    # Read data and split into training and test sets
    logging.info("Start Reading Data")
    x_train, x_test, y_train, y_test = read_data(
        data_path=os.path.join(opt.dataroot, opt.saveroot, "imputed_data.csv"),
        target=opt.target,
        test_size=opt.test_size,
        cols=continuous_vars,
        random_state=opt.randomstate
    )
    logging.info("Finish Reading Data")

    # Apply over/under-sampling
    over = ADASYN(sampling_strategy=0.07, random_state=opt.randomstate)
    under = RandomUnderSampler(sampling_strategy=0.5, random_state=opt.randomstate)
    pipeline = Pipeline(steps=[('o', over), ('u', under)])
    x_train1, y_train1 = pipeline.fit_resample(x_train, y_train)

    # Train the model
    logging.info("Start Training Model")
    best_params, best_model = train_model(
        model, 
        x_train1, 
        y_train1, 
        param_grid,
        random_state=opt.randomstate,
        k=5
    )
    logging.info(f"Best Parameters: {best_params}")
    logging.info("Finish Training Model")

    # Predict on the test set
    y_pred = best_model.predict(x_test)
    y_pred_prob = best_model.predict_proba(x_test)[:, 1]

    # Calculate AUC, F1-score, Recall, Precision, Accuracy using Bootstrap

    auc_mean, auc_sd = bootstrap_metric(y_test, y_pred_prob, roc_auc_score, random_state=opt.randomstate)
    f1_mean, f1_sd = bootstrap_metric(y_test, y_pred, f1_score, random_state=opt.randomstate)
    recall_mean, recall_sd = bootstrap_metric(y_test, y_pred, recall_score, random_state=opt.randomstate)
    precision_mean, precision_sd = bootstrap_metric(y_test, y_pred, precision_score, random_state=opt.randomstate)
    accuracy_mean, accuracy_sd = bootstrap_metric(y_test, y_pred, accuracy_score, random_state=opt.randomstate)
    
    # Output the mean and SD for each metric
    logging.info(f"Accuracy: {accuracy_mean:.4f} (SD: {accuracy_sd:.4f})")
    logging.info(f"AUC: {auc_mean:.4f} (SD: {auc_sd:.4f})")
    logging.info(f"F1-score: {f1_mean:.4f} (SD: {f1_sd:.4f})")
    logging.info(f"Recall: {recall_mean:.4f} (SD: {recall_sd:.4f})")
    logging.info(f"Precision: {precision_mean:.4f} (SD: {precision_sd:.4f})")

    # Calculate AUC confidence interval
    CI = AUC_CI(y_test, y_pred_prob, random_state=opt.randomstate)
    p = caculate_p(auc_mean, CI)
    logging.info(f"AUC of {opt.model} with CI {CI} and p value {p}")

    # Calculate Accuracy confidence interval (ACC CI)
    acc_CI = ACC_CI(y_test, y_pred, random_state=opt.randomstate)
    logging.info(f"Accuracy CI of {opt.model} is {acc_CI}")

    # Save the trained model
    joblib.dump(best_model, os.path.join(opt.dataroot, opt.saveroot, f'{opt.model}_model.pkl'))

    # Compute ROC curve
    fpr, tpr, thresholds = roc_curve(y_test, y_pred_prob)
    logging.info(f"False Positive Rate (FPR): {fpr}")
    logging.info(f"True Positive Rate (TPR): {tpr}")
    
    # Save ROC curve data
    roc_data = pd.DataFrame({
        'False Positive Rate': fpr,
        'True Positive Rate': tpr,
        'Thresholds': thresholds
    })
    roc_data.to_csv(os.path.join(opt.dataroot, opt.saveroot, f'{opt.model}_roc_curve.csv'), index=False)


    #Calculate the feature importance (only for Bagging, NeuralNetwork and NaiveBayes)
    from sklearn.inspection import permutation_importance
    
    if opt.model in ['Bagging', 'NeuralNetwork', 'NaiveBayes']:
        result = permutation_importance(best_model, x_test, y_test, n_repeats=10, random_state=2024)
        
        feature_importance = pd.DataFrame({
            'Feature': x_test.columns,
            'Importance Mean': result.importances_mean,
            #'Importance Std': result.importances_std
        }).sort_values(by='Importance Mean', ascending=False)
        min_importance = feature_importance['Importance Mean'].min()
        max_importance = feature_importance['Importance Mean'].max()
        if max_importance != min_importance:
            feature_importance['Scaled Importance'] = (feature_importance['Importance Mean'] - min_importance) / (max_importance - min_importance)
        
        feature_importance['Standard Importance'] = feature_importance['Scaled Importance']/feature_importance['Scaled Importance'].sum()
        feature_importance.to_csv(os.path.join(opt.dataroot, opt.saveroot, f'{opt.model}_feature_importance.csv'),index = False)
        print('feature_importance has saved')
        
        
    # feature importance (only for RandomForest, XGBoost, LogisticRegression)
    if opt.model == 'RandomForest':
        importance = best_model.feature_importances_
        feature_names = best_model.feature_names_in_
        importance_df = pd.DataFrame({
            'Feature': feature_names,
            'Importance': importance
        })
        importance_df = importance_df.sort_values(by='Importance', ascending=False)
        
        # print and save FI
        print("Feature Importances for RandomForest:")
        print(importance_df)
        importance_df.to_csv(os.path.join(opt.dataroot, opt.saveroot, "feature_importance_RandomForest.csv"), index=False)

    if opt.model == 'XGBoost':
        logging.info("Feature Importance (XGBoost):")
        feature_importances = best_model.feature_importances_
        
        importance_df = pd.DataFrame({
            'Feature': x_train.columns,
            'Importance': feature_importances
        })
        # print and save FI
        importance_df = importance_df.sort_values(by='Importance', ascending=False)
        print("Feature Importances for XGBoost:")
        print(importance_df)
        importance_df.to_csv(os.path.join(opt.dataroot, opt.saveroot, "feature_importance_XGBoost.csv"), index=False)     
        
    if opt.model == 'LogisticRegression':
        importance = best_model.coef_[0]
        feature_names = best_model.feature_names_in_
        importance_df = pd.DataFrame({
            'Feature': feature_names,
            'Importance': importance
        })
        importance_df = importance_df.sort_values(by='Importance', ascending=False)
        
        # print and save FI
        print("Feature Coefficients for LogisticRegression:")
        print(importance_df)
        importance_df.to_csv(os.path.join(opt.dataroot, opt.saveroot, "feature_importance_LogisticRegression.csv"), index=False)