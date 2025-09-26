import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, f1_score, roc_auc_score, precision_score, recall_score, roc_curve, precision_recall_curve, average_precision_score
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import FunctionTransformer
import pandas as pd
import numpy as np
from sklearn.base import clone
import plots as p
from imblearn.pipeline import Pipeline
from sklearn.model_selection import cross_validate



def evaluate_regression(y_test, y_pred, model_name="Model"):
    mae = mean_absolute_error(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_test, y_pred)

    results = pd.DataFrame({
        "Model": [model_name],
        "MAE": [mae],
        "MSE": [mse],
        "RMSE": [rmse],
        "R²": [r2]
    })
    return results

def plot_regression_results(y_test, y_pred, model_name="Model"):
    residuals = y_test - y_pred

    # 1. scatter y_test vs y_pred
    plt.figure(figsize=(6, 6))
    plt.scatter(y_test, y_pred, alpha=0.5)
    plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--')
    plt.xlabel("True values")
    plt.ylabel("Predicted values")
    plt.title(f"{model_name} — y_test vs y_pred")
    plt.show()

    # 2. residual plot
    plt.figure(figsize=(6, 4))
    plt.scatter(y_pred, residuals, alpha=0.5)
    plt.axhline(0, color="red", linestyle="--")
    plt.xlabel("Predicted values")
    plt.ylabel("Residuals")
    plt.title(f"{model_name} — Residual plot")
    plt.show()

    # 3. histogram of residuals
    plt.figure(figsize=(6, 4))
    plt.hist(residuals, bins=30, edgecolor="black", alpha=0.7)
    plt.axvline(0, color="red", linestyle="--")
    plt.xlabel("Residuals")
    plt.ylabel("Frequency")
    plt.title(f"{model_name} — Residual distribution")
    plt.show()



def divide_data(data, target_column):
    X = data.drop(columns=[target_column])
    y = data[target_column]
    return X, y


def calculate_classification_metrics(y_test, y_pred, y_probs=None, use_pr_curve=False):
    metrics = {
        'ROC AUC': roc_auc_score(y_test, y_probs),
        'Average Precision': average_precision_score(y_test, y_probs),
        'F1 Score': f1_score(y_test, y_pred, average='macro'),
        'Precision': precision_score(y_test, y_pred, average='macro'),
        'Recall': recall_score(y_test, y_pred, average='macro'),
        'Accuracy': (y_pred == y_test).mean(),
        'Confusion Matrix': confusion_matrix(y_test, y_pred)
    }

    metrics['Classification Report'] = {
        'Class': ['Positive', 'Negative'],
        'Precision': [
            precision_score(y_test, y_pred, pos_label=1),
            precision_score(y_test, y_pred, pos_label=0)
        ],
        'Recall': [
            recall_score(y_test, y_pred, pos_label=1),
            recall_score(y_test, y_pred, pos_label=0)
        ]
    }

    if y_probs is not None:
        if use_pr_curve:
            precision, recall, thresholds = precision_recall_curve(y_test, y_probs)
            metrics['Precision-Recall Curve'] = {
                'precision': precision,
                'recall': recall,
                'thresholds': thresholds
            }
        else:
            fpr, tpr, thresholds = roc_curve(y_test, y_probs)
            metrics['ROC Curve'] = {
                'fpr': fpr,
                'tpr': tpr,
                'thresholds': thresholds
            }

    return metrics



# def calculate_classification_metrics(y_test, y_pred, y_probs=None):
#     """
#     Calculate classification performance metrics

#     Parameters:
#     -----------
#     y_test : array-like
#         True labels
#     y_pred : array-like
#         Predicted labels
#     y_probs : array-like, optional
#         Predicted probabilities for positive class

#     Returns:
#     --------
#     dict: Dictionary containing all calculated metrics
#     """
#     metrics = {
#         'ROC AUC': roc_auc_score(y_test, y_probs) if y_probs is not None else None,
#         'F1 Score': f1_score(y_test, y_pred, average='macro'),
#         'Precision': precision_score(y_test, y_pred, average='macro'),
#         'Recall': recall_score(y_test, y_pred, average='macro'),
#         'Accuracy': (y_pred == y_test).mean(),
#         'Confusion Matrix': confusion_matrix(y_test, y_pred)
#     }

#     # Additional metrics for classification report
#     metrics['Classification Report'] = {
#         'Class': ['Positive', 'Negative'],
#         'Precision': [
#             precision_score(y_test, y_pred, pos_label=1),
#             precision_score(y_test, y_pred, pos_label=0)
#         ],
#         'Recall': [
#             recall_score(y_test, y_pred, pos_label=1),
#             recall_score(y_test, y_pred, pos_label=0)
#         ]
#     }

#     if y_probs is not None:
#         fpr, tpr, thresholds = roc_curve(y_test, y_probs)
#         metrics['ROC Curve'] = {
#             'fpr': fpr,
#             'tpr': tpr,
#             'thresholds': thresholds
#         }

#     return metrics


def evaluate_classification(y_test, y_pred, y_probs=None, model_name="Model", enable_plot=True, use_pr_curve=False):
    """
    Evaluate classification performance with comprehensive metrics and visualizations

    Parameters:
    -----------
    y_test : array-like
        True labels
    y_pred : array-like
        Predicted labels
    y_probs : array-like, optional
        Predicted probabilities for positive class (required for ROC AUC)
    model_name : str, optional
        Name of the model for display purposes
    enable_plot : bool, optional
        Whether to display plots and detailed reports

    Returns:
    --------
    dict: Dictionary containing all calculated metrics
    """
    # Calculate all metrics
    metrics = calculate_classification_metrics(y_test, y_pred, y_probs, use_pr_curve)

    if enable_plot:
        # Generate plots
        p.plot_classification_results(metrics, model_name)

        # Print detailed report
        p.print_classification_report(metrics, model_name)

    # Return metrics dictionary (excluding plot data for cleaner output)
    if use_pr_curve:
        curve = 'Precision-Recall'
    else:
        curve = 'ROC'
    return {k: v for k, v in metrics.items() if k not in ['Confusion Matrix', '{curve} Curve', 'Classification Report']}


def train_evaluate_model(model, model_name, X_train, y_train, X_test, y_test, seed=None):
    # Set random seed if provided and model has the parameter
    if seed is not None:
        if hasattr(model, 'random_state'):
            model.set_params(random_state=seed)
        if hasattr(model, 'seed'):
            model.set_params(seed=seed)

    # Train the model
    model.fit(X_train, y_train)

    # Get predictions
    y_pred = model.predict(X_test)
    y_probs = model.predict_proba(X_test)[:, 1]  # For ROC curve

    # Evaluate
    metrics = evaluate_classification(
        y_test=y_test,
        y_pred=y_pred,
        y_probs=y_probs,
        model_name=model_name,
        enable_plot=False
    )

    return metrics


def train_evaluate_model_cv(model, model_name, X, y,
                            preprocessor=None, cv=5, seed=None):
    """
    Train and evaluate a model using cross-validation and optional preprocessing.

    Args:
        model: The model to train and evaluate
        model_name: Name of the model for reporting
        X_train: Training features
        y_train: Training labels
        X_test: Test features
        y_test: Test labels
        preprocessor: Preprocessing pipeline (e.g., StandardScaler, OneHotEncoder)
        cv: Number of cross-validation folds
        seed: Random seed for reproducibility

    Returns:
        Dictionary containing evaluation metrics
    """
    # Set random seed if provided and model has the parameter
    if seed is not None:
        if hasattr(model, 'random_state'):
            model.set_params(random_state=seed)
        if hasattr(model, 'seed'):
            model.set_params(seed=seed)

    if isinstance(preprocessor, Pipeline):
        # If preprocessor is already a pipeline, append the model to it
        preprocessor.steps.append(('model', model))
        pipeline = preprocessor
    elif preprocessor is not None:
        # Create new pipeline with preprocessor and model
        pipeline = Pipeline([
            ('preprocessor', preprocessor),
            ('model', model)
        ])
    else:
        # No preprocessor, just use the model
        pipeline = model


    # Scoring metrics for cross-validation (using macro averaging)
    scoring = {
        'accuracy': 'accuracy',
        'precision': 'precision_macro',
        'recall': 'recall_macro',
        'f1': 'f1_macro',
        'roc_auc': 'roc_auc'
    }

    # Perform cross-validation on training data
    cv_results = cross_validate(
        pipeline,
        X,
        y,
        cv=cv,
        scoring=scoring,
        return_train_score=False
    )

    metrics = {
        'ROC AUC': cv_results['test_roc_auc'].mean(),
        'F1 Score': cv_results['test_f1'].mean(),
        'Precision': cv_results['test_precision'].mean(),
        'Recall': cv_results['test_recall'].mean(),
        'Accuracy': cv_results['test_accuracy'].mean(),
    }

    p.plot_classification_results(metrics, model_name)

    return metrics


def train_evaluate_models_cv(models: list, X, y, preprocessor=None, cv=5, seed=None):
    # Dictionary to store all metrics
    all_metrics = {}

    for model_name, model in models:
        # Работаем с копией модели, чтобы не изменять исходные модели, переданные в качестве аргументов
        current_model = clone(model)
        if isinstance(preprocessor, Pipeline):
            current_preprocessor = clone(preprocessor)
        else:
            current_preprocessor = None
        # Store metrics
        all_metrics[model_name] = train_evaluate_model_cv(
            current_model, model_name, X, y, current_preprocessor, cv, seed)

    # Convert metrics to DataFrame
    metrics_df = pd.DataFrame.from_dict(all_metrics, orient='index')

    # Plot heatmap
    plt.figure(figsize=(8, 4))
    sns.heatmap(metrics_df, cmap='RdBu_r', annot=True, fmt=".2f")
    plt.title('Model Evaluation Metrics Comparison')
    plt.tight_layout()
    plt.show()

    return metrics_df


def train_evaluate_models(models: list, X_train, y_train, X_test, y_test, seed=None):
    """
    Train and evaluate multiple classification models, then display a heatmap of the metrics.

    Parameters:
    -----------
    models : list
        List of tuples containing (model_name, model_instance) where model_instance is a scikit-learn compatible classifier
    X_train : array-like
        Training features
    y_train : array-like
        Training labels
    X_test : array-like
        Test features
    y_test : array-like
        Test labels
    preprocessor : Pipeline or Transformer, optional
        Preprocessing pipeline to apply to the data before training
    seed : int, optional
        Random seed for reproducibility

    Returns:
    --------
    pd.DataFrame
        DataFrame containing all evaluation metrics for all models
    """

    # Dictionary to store all metrics
    all_metrics = {}

    for model_name, model in models:
        # Работаем с копией модели, чтобы не изменять исходные модели, переданные в качестве аргументов
        current_model = clone(model)

        # Store metrics
        all_metrics[model_name] = train_evaluate_model(
            current_model, model_name, X_train, y_train, X_test, y_test, seed)

    # Convert metrics to DataFrame
    metrics_df = pd.DataFrame.from_dict(all_metrics, orient='index')

    # Plot heatmap
    plt.figure(figsize=(8, 4))
    sns.heatmap(metrics_df, cmap='RdBu_r', annot=True, fmt=".2f")
    plt.title('Model Evaluation Metrics Comparison')
    plt.tight_layout()
    plt.show()

    return metrics_df


def winsorize_outliers(df, column_name, lower_bound=None, upper_bound=None):
    df = df.copy()

    if lower_bound is not None:
        df.loc[df[column_name] < lower_bound, column_name] = lower_bound
    if upper_bound is not None:
        df.loc[df[column_name] > upper_bound, column_name] = upper_bound

    return df


def train_evaluate_models_cv_regression(models, X, y, cv, preprocessor=None, seed=None):
    import numpy as np
    import pandas as pd
    import math
    from sklearn.pipeline import Pipeline
    from sklearn.model_selection import StratifiedKFold, KFold, cross_validate
    from sklearn.base import clone

    X_arr = X
    y_arr = y
    use_custom_splits = False
    splits = None

    try:
        is_stratified_cv = isinstance(cv, StratifiedKFold)
    except Exception:
        is_stratified_cv = False

    if is_stratified_cv:
        y_series = pd.Series(y_arr).reset_index(drop=True)
        n_splits = cv.n_splits
        try:
            strat_labels = pd.qcut(y_series, q=n_splits, labels=False, duplicates='drop')
            if strat_labels.nunique() < n_splits:
                kf = KFold(n_splits=n_splits, shuffle=cv.shuffle, random_state=cv.random_state or seed)
                splits = list(kf.split(X_arr))
                use_custom_splits = True
            else:
                skf = StratifiedKFold(n_splits=n_splits, shuffle=cv.shuffle, random_state=cv.random_state or seed)
                splits = list(skf.split(X_arr, strat_labels))
                use_custom_splits = True
        except Exception:
            kf = KFold(n_splits=n_splits, shuffle=cv.shuffle, random_state=cv.random_state or seed)
            splits = list(kf.split(X_arr))
            use_custom_splits = True

    results_list = []
    scoring = {'mae': 'neg_mean_absolute_error', 'mse': 'neg_mean_squared_error', 'r2': 'r2'}

    for name, estimator in models:
        if preprocessor is not None:
            pipeline = Pipeline([('preprocessor', preprocessor), ('estimator', clone(estimator))])
        else:
            pipeline = clone(estimator)

        if use_custom_splits:
            cv_arg = splits
        else:
            cv_arg = cv

        cv_results = cross_validate(
            pipeline,
            X_arr,
            y_arr,
            cv=cv_arg,
            scoring=scoring,
            n_jobs=-1,
            return_train_score=False
        )
        mae = -np.mean(cv_results['test_mae'])
        mse = -np.mean(cv_results['test_mse'])
        rmse = math.sqrt(mse)
        r2 = np.mean(cv_results['test_r2'])
        results_list.append({'model': name, 'mae': mae, 'mse': mse, 'rmse': rmse, 'r2': r2})

    df = pd.DataFrame(results_list).set_index('model')
    return df


# import numpy as np
# import pandas as pd
# from sklearn.model_selection import train_test_split
# from sklearn.metrics import roc_auc_score, f1_score, precision_score, recall_score
# from catboost import CatBoostClassifier, Pool

# def prune_features_catboost(X, y, cat_features=None, test_size=0.3, seed=42,
#                              iterations=1000, early_stopping_rounds=50,
#                              min_change_pct=10.0, verbose=True, random_state=42):
#     if not isinstance(X, pd.DataFrame):
#         X = pd.DataFrame(X)
#     y = pd.Series(y).reset_index(drop=True)
#     classes = np.unique(y)
#     if len(classes) != 2:
#         raise ValueError("Только бинарная классификация поддерживается.")
#     mapping = {classes[0]: 0, classes[1]: 1}
#     y_bin = y.map(mapping).astype(int)
#     X_train, X_val, y_train, y_val = train_test_split(X, y_bin, test_size=test_size,
#                                                       stratify=y_bin, random_state=seed)
#     current_features = list(X.columns)
#     if cat_features is None:
#         cat_feat_names = None
#     else:
#         if all(isinstance(cf, int) for cf in cat_features):
#             cat_feat_names = [X.columns[i] for i in cat_features]
#         else:
#             cat_feat_names = [str(cf) for cf in cat_features]
#         cat_feat_names = [f for f in cat_feat_names if f in current_features]
#     history = []
#     while True:
#         train_pool = Pool(X_train[current_features], y_train, cat_features=cat_feat_names)
#         val_pool = Pool(X_val[current_features], y_val, cat_features=cat_feat_names)
#         model = CatBoostClassifier(iterations=iterations, random_seed=random_state,
#                                    early_stopping_rounds=early_stopping_rounds, verbose=False)
#         model.fit(train_pool, eval_set=val_pool)
#         proba = model.predict_proba(X_val[current_features])[:, 1]
#         preds = model.predict(X_val[current_features])
#         metrics_before = {
#             'roc_auc': float(roc_auc_score(y_val, proba)),
#             'f1': float(f1_score(y_val, preds)),
#             'precision': float(precision_score(y_val, preds)),
#             'recall': float(recall_score(y_val, preds))
#         }
#         importances = model.get_feature_importance(train_pool)
#         print(importances)
#         print('_'*50)
#         if len(importances) != len(current_features):
#             raise RuntimeError("Feature importance length mismatch.")
#         idx_least = int(np.argmin(importances))
#         feature_to_drop = current_features[idx_least]
#         if len(current_features) == 1:
#             if verbose:
#                 print("Остался один признак — прекращаю.")
#             break
#         next_features = [f for f in current_features if f != feature_to_drop]
#         next_cat_feat_names = None if cat_feat_names is None else [f for f in cat_feat_names if f in next_features]
#         train_pool_next = Pool(X_train[next_features], y_train, cat_features=next_cat_feat_names)
#         val_pool_next = Pool(X_val[next_features], y_val, cat_features=next_cat_feat_names)
#         model_next = CatBoostClassifier(iterations=iterations, random_seed=random_state,
#                                         early_stopping_rounds=early_stopping_rounds, verbose=False)
#         model_next.fit(train_pool_next, eval_set=val_pool_next)
#         proba_next = model_next.predict_proba(X_val[next_features])[:, 1]
#         preds_next = model_next.predict(X_val[next_features])
#         metrics_after = {
#             'roc_auc': float(roc_auc_score(y_val, proba_next)),
#             'f1': float(f1_score(y_val, preds_next)),
#             'precision': float(precision_score(y_val, preds_next)),
#             'recall': float(recall_score(y_val, preds_next))
#         }
#         eps = 1e-8
#         pct_changes = {}
#         for k in metrics_before.keys():
#             denom = max(abs(metrics_before[k]), eps)
#             pct_changes[k] = 100.0 * (metrics_after[k] - metrics_before[k]) / denom
#         max_abs_pct = max(abs(v) for v in pct_changes.values())
#         history.append({
#             'dropped_feature': feature_to_drop,
#             'n_features_before': len(current_features),
#             'metrics_before': metrics_before,
#             'metrics_after': metrics_after,
#             'pct_change': pct_changes,
#             'max_abs_pct_change': max_abs_pct
#         })
#         if verbose:
#             print(f"Удалён: {feature_to_drop} | max_abs_pct_change = {max_abs_pct:.2f}%")
#         if max_abs_pct > min_change_pct:
#             if verbose:
#                 print(f"Остановка: при удалении '{feature_to_drop}' максимальное изменение {max_abs_pct:.2f}% < {min_change_pct}%")
#             break
#         current_features = next_features
#         cat_feat_names = next_cat_feat_names
#     history_df = pd.DataFrame([{
#         'dropped_feature': h['dropped_feature'],
#         'n_features_before': h['n_features_before'],
#         'roc_before': h['metrics_before']['roc_auc'],
#         'roc_after': h['metrics_after']['roc_auc'],
#         'roc_pct_change': h['pct_change']['roc_auc'],
#         'f1_before': h['metrics_before']['f1'],
#         'f1_after': h['metrics_after']['f1'],
#         'f1_pct_change': h['pct_change']['f1'],
#         'precision_before': h['metrics_before']['precision'],
#         'precision_after': h['metrics_after']['precision'],
#         'precision_pct_change': h['pct_change']['precision'],
#         'recall_before': h['metrics_before']['recall'],
#         'recall_after': h['metrics_after']['recall'],
#         'recall_pct_change': h['pct_change']['recall'],
#         'max_abs_pct_change': h['max_abs_pct_change']
#     } for h in history])
#     result = {
#         'final_features': current_features,
#         'history': history_df,
#         'last_metrics': history[-1]['metrics_after'] if len(history) > 0 else None
#     }
#     return result
