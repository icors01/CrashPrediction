"""
Modeling utilities for crash prediction pipeline.
Contains functions for data preparation, feature engineering, and model evaluation.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold
from sklearn.metrics import roc_auc_score, classification_report, confusion_matrix
from sklearn.metrics import average_precision_score, precision_recall_curve


def cap_rare_categories(df, categorical_cols, top_n=30):
    """
    Cap infrequent categories to 'OTHER' to reduce one-hot encoding dimensionality.
    
    Parameters:
    -----------
    df : pd.DataFrame
        Input dataframe
    categorical_cols : list
        List of categorical column names
    top_n : int
        Keep top N categories per column, rest become 'OTHER'
    
    Returns:
    --------
    pd.DataFrame
        DataFrame with capped categories
    """
    df_capped = df.copy()
    
    for col in categorical_cols:
        vc = df_capped[col].value_counts(dropna=False)
        keep = set(vc.nlargest(top_n).index)
        df_capped[col] = df_capped[col].where(df_capped[col].isin(keep), 'OTHER')
    
    return df_capped


def get_orig_feature_name(encoded_name: str) -> str:
    """
    Extract original feature name from one-hot encoded column name.
    
    Examples:
        'num__TAVG_MEDIAN' -> 'TAVG_MEDIAN'
        'cat__Region_Metro' -> 'Region'
        'cat__CountyNameTxt_WABASHA' -> 'CountyNameTxt'
    
    Parameters:
    -----------
    encoded_name : str
        Encoded feature name from preprocessor
    
    Returns:
    --------
    str
        Original feature name
    """
    if '__' in encoded_name:
        prefix, rest = encoded_name.split('__', 1)
    else:
        prefix, rest = None, encoded_name

    if prefix == 'num':
        return rest

    if prefix == 'cat':
        # one-hot columns: '<orig>_<category>'
        base = rest.split('_', 1)[0]
        return base

    return rest


def build_preprocessor(numeric_cols, categorical_cols, sparse=True):
    """
    Create a ColumnTransformer for preprocessing features.
    
    Parameters:
    -----------
    numeric_cols : list
        List of numeric feature columns
    categorical_cols : list
        List of categorical feature columns
    sparse : bool
        Whether to use sparse output for one-hot encoding
    
    Returns:
    --------
    ColumnTransformer
        Fitted preprocessor pipeline
    """
    return ColumnTransformer(
        transformers=[
            ('num', StandardScaler(with_mean=False), numeric_cols),
            ('cat', OneHotEncoder(handle_unknown='ignore', drop='if_binary', 
                                  sparse_output=sparse), categorical_cols),
        ]
    )


def build_rf_model(n_estimators=400, max_depth=None, min_samples_leaf=2, 
                   max_features='sqrt', **kwargs):
    """
    Create a balanced RandomForestClassifier.
    
    Parameters:
    -----------
    n_estimators : int
        Number of trees
    max_depth : int or None
        Max tree depth
    min_samples_leaf : int
        Min samples per leaf
    max_features : str or float
        Feature split strategy
    **kwargs : dict
        Additional arguments to RandomForestClassifier
    
    Returns:
    --------
    RandomForestClassifier
        Configured model
    """
    return RandomForestClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        min_samples_leaf=min_samples_leaf,
        max_features=max_features,
        random_state=42,
        class_weight='balanced_subsample',
        n_jobs=-1,
        **kwargs
    )


def build_pipeline(preprocessor, rf_model):
    """
    Combine preprocessor and RF model into a pipeline.
    
    Parameters:
    -----------
    preprocessor : ColumnTransformer
        Feature preprocessing pipeline
    rf_model : RandomForestClassifier
        Random forest model
    
    Returns:
    --------
    Pipeline
        Combined preprocessing + model pipeline
    """
    return Pipeline([
        ('prep', preprocessor),
        ('rf', rf_model)
    ])


def evaluate_model(y_true, y_prob, y_pred=None):
    """
    Evaluate model performance with multiple metrics.
    
    Parameters:
    -----------
    y_true : array-like
        True labels
    y_prob : array-like
        Predicted probabilities
    y_pred : array-like, optional
        Hard predictions (if None, use 0.5 threshold)
    
    Returns:
    --------
    dict
        Dictionary with ROC AUC, sensitivity, specificity
    """
    if y_pred is None:
        y_pred = (y_prob >= 0.5).astype(int)
    
    auc = roc_auc_score(y_true, y_prob)
    cm = confusion_matrix(y_true, y_pred)
    tn, fp, fn, tp = cm.ravel()
    
    sensitivity = tp / (tp + fn)
    specificity = tn / (tn + fp)
    
    return {
        'auc': auc,
        'sensitivity': sensitivity,
        'specificity': specificity,
        'confusion_matrix': cm,
        'classification_report': classification_report(y_true, y_pred)
    }


def compute_pr_auc(y_true, y_scores):
    """
    Compute Precision-Recall AUC (Average Precision) and return curve points.

    Parameters:
    -----------
    y_true : array-like
        Binary ground truth labels
    y_scores : array-like
        Predicted probabilities or decision scores

    Returns:
    --------
    dict
        'average_precision': float, 'precision': np.ndarray, 'recall': np.ndarray, 'thresholds': np.ndarray
    """
    ap = average_precision_score(y_true, y_scores)
    precision, recall, thresholds = precision_recall_curve(y_true, y_scores)
    return {
        'average_precision': ap,
        'precision': precision,
        'recall': recall,
        'thresholds': thresholds
    }


def precision_recall_at_k_daily(df, date_col, cell_col, y_col, score_col, k=100):
    """
    Compute Precision@K and Recall@K per day by ranking cells by predicted score.

    For each date, sort rows by `score_col` descending, take top-K, compute:
      - precision@K = (# positives in top-K) / K
      - recall@K = (# positives in top-K) / (total positives that day)

    Parameters:
    -----------
    df : pd.DataFrame
        Dataframe containing predictions and labels
    date_col : str
        Date column name
    cell_col : str
        Cell identifier column name (used only for clarity)
    y_col : str
        Binary label column name
    score_col : str
        Predicted probability/score column name
    k : int
        Top-K to evaluate

    Returns:
    --------
    dict
        'daily': pd.DataFrame with columns [date, precision_at_k, recall_at_k, positives_in_top_k, total_positives]
        'overall': {'precision_at_k': float, 'recall_at_k': float}
    """
    records = []
    for dt, group in df.groupby(date_col):
        group_sorted = group.sort_values(score_col, ascending=False)
        top_k = group_sorted.head(k)
        positives_top_k = int(top_k[y_col].sum())
        total_pos = int(group[y_col].sum())
        precision_k = positives_top_k / max(k, 1)
        recall_k = positives_top_k / max(total_pos, 1) if total_pos > 0 else 0.0
        records.append({
            'date': dt,
            'precision_at_k': precision_k,
            'recall_at_k': recall_k,
            'positives_in_top_k': positives_top_k,
            'total_positives': total_pos
        })

    daily_df = pd.DataFrame(records).sort_values('date')
    overall_precision = daily_df['precision_at_k'].mean() if len(daily_df) else 0.0
    overall_recall = daily_df['recall_at_k'].mean() if len(daily_df) else 0.0

    return {
        'daily': daily_df,
        'overall': {
            'precision_at_k': overall_precision,
            'recall_at_k': overall_recall
        }
    }


def extract_feature_importances(pipeline, feature_cols):
    """
    Extract and aggregate feature importances from fitted pipeline.
    
    Parameters:
    -----------
    pipeline : Pipeline
        Fitted pipeline with preprocessor and RF model
    feature_cols : list
        Original feature column names
    
    Returns:
    --------
    pd.DataFrame
        DataFrame with original feature names and importance scores
    """
    rf_model = pipeline.named_steps['rf']
    preprocessor_obj = pipeline.named_steps['prep']
    
    importances = rf_model.feature_importances_
    feature_names = []
    
    # Get names from numeric and categorical transformers
    for name, _, cols in preprocessor_obj.transformers_:
        if name == 'num':
            feature_names.extend([f"num__{c}" for c in cols])
        elif name == 'cat':
            encoder = preprocessor_obj.named_transformers_['cat']
            cat_features = encoder.get_feature_names_out()
            feature_names.extend(cat_features)
    
    # Create dataframe with feature importance
    fi_df = pd.DataFrame({
        'feature': feature_names,
        'importance': importances
    }).sort_values('importance', ascending=False)
    
    # Extract original feature names
    fi_df['orig_feature'] = fi_df['feature'].apply(get_orig_feature_name)
    
    # Aggregate by original feature
    grouped_importance = (
        fi_df.groupby('orig_feature')['importance']
        .sum()
        .sort_values(ascending=False)
    )
    
    return fi_df, grouped_importance


def plot_feature_importance(grouped_importance, top_n=15, figsize=(10, 6)):
    """
    Visualize top features by importance.
    
    Parameters:
    -----------
    grouped_importance : pd.Series
        Aggregated feature importances
    top_n : int
        Number of top features to display
    figsize : tuple
        Figure size (width, height)
    """
    fig, ax = plt.subplots(figsize=figsize)
    grouped_importance.head(top_n).plot(kind='barh', ax=ax, color='steelblue')
    ax.set_xlabel('Total Importance')
    ax.set_title(f'Top {top_n} Features by Importance')
    ax.invert_yaxis()
    plt.tight_layout()
    plt.show()


def aggregate_predictions_by_cell(df, cell_col, predictions):
    """
    Aggregate model predictions by cell ID.
    
    Parameters:
    -----------
    df : pd.DataFrame
        Input dataframe with cell IDs
    cell_col : str
        Name of cell ID column
    predictions : array-like
        Model predictions for each row
    
    Returns:
    --------
    pd.DataFrame
        Dataframe with cell_id and mean prediction
    """
    result = pd.DataFrame({
        cell_col: df[cell_col],
        'pred_crash_prob': predictions
    })
    
    return (
        result.groupby(cell_col)['pred_crash_prob']
        .mean()
        .reset_index()
        .sort_values('pred_crash_prob', ascending=False)
    )
