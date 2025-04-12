# CLASSIFYING CREDIT WORTHINESS
# AI and ML 


from IPython import get_ipython
get_ipython().magic('reset -sf')
get_ipython().magic('clear')
#%%
import pandas as pd
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.filterwarnings("ignore", category=pd.errors.SettingWithCopyWarning)
warnings.filterwarnings("ignore", message=".*KMeans is known to have a memory leak on Windows with MKL.*")
import yfinance as yf
from arch import arch_model
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import datasets
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV, KFold, StratifiedKFold
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import mean_absolute_error,accuracy_score, roc_curve, classification_report, confusion_matrix, make_scorer, roc_auc_score, RocCurveDisplay, ConfusionMatrixDisplay, precision_recall_curve, average_precision_score, auc
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.impute import KNNImputer
from sklearn.inspection import permutation_importance
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from yahooquery import Ticker
import os
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
# Plotting style
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette("viridis")
sns.set_context("talk")


#%%  Uploading data
# Load data from online source
url = "https://raw.githubusercontent.com/Rickeagle/Credit-Worthiness.-Study-case-in-the-UK/main/EM.csv"
df = pd.read_csv(url)
print(df.head())

# Show shape
print(f"Data loaded with {df.shape[0]} rows and {df.shape[1]} columns")

# Preview missing values
print("\nTop columns with missing values:")
print(df.isna().sum().sort_values(ascending=False).head(10))

# Preview the data
df.head()

# Define a function for loading and initial preprocessing
def load_and_preprocess_data(file_path):
    """
    Load data and perform initial preprocessing steps
    """
    print("Loading data...")
    df = pd.read_csv(file_path)
    print(f"Data loaded with {df.shape[0]} rows and {df.shape[1]} columns")

    # Display basic info
    print("\nBasic information about the dataset:")
    print(f"Number of missing values per column (top 10):\n{df.isna().sum().sort_values(ascending=False).head(10)}")

    return df

#%% ALL ANALYSIS STARTS HERE

############
# Define a function for exploring features
def explore_features(df, feature_list, target=None):
    """
    Perform exploratory data analysis on specified features
    """
    print(f"\nExploratory Data Analysis for selected features:")

    # Summary statistics
    summary_stats = df[feature_list].describe().T
    print("\nSummary statistics:")
    print(summary_stats)

    # Distribution plots
    n_features = len(feature_list)
    fig_rows = (n_features + 1) // 2  # Calculate number of rows needed

    plt.figure(figsize=(15, 5 * fig_rows))
    for i, feature in enumerate(feature_list):
        plt.subplot(fig_rows, 2, i+1)
        sns.histplot(df[feature].dropna(), kde=True, bins=30)
        plt.title(f'Distribution of {feature}')
        plt.xlabel(feature)
        plt.ylabel('Frequency')

        # Add a vertical line for the mean
        plt.axvline(df[feature].mean(), color='red', linestyle='--', label=f'Mean: {df[feature].mean():.2f}')
        plt.axvline(df[feature].median(), color='green', linestyle='-.', label=f'Median: {df[feature].median():.2f}')
        plt.legend()

    plt.tight_layout()
    plt.show()

    # If target is provided, explore relationship with target
    if target and target in df.columns:
        plt.figure(figsize=(15, 5 * fig_rows))
        for i, feature in enumerate(feature_list):
            plt.subplot(fig_rows, 2, i+1)
            if df[target].nunique() <= 5:  # If target is categorical with few categories
                sns.boxplot(x=target, y=feature, data=df)
                plt.title(f'Distribution of {feature} by {target}')
            else:
                sns.scatterplot(x=feature, y=target, data=df, alpha=0.5)
                plt.title(f'Relationship between {feature} and {target}')

        plt.tight_layout()
        plt.show()

    # Correlation matrix for numerical features
    if len(feature_list) > 1:
        plt.figure(figsize=(12, 10))
        correlation_matrix = df[feature_list].corr()
        mask = np.triu(correlation_matrix)
        sns.heatmap(correlation_matrix, annot=True, fmt=".2f", cmap="coolwarm", mask=mask)
        plt.title('Correlation Matrix of Features')
        plt.show()

# Create target variable function
def create_target_variable(df, source_column, method='mean', threshold=None):
    """
    Create a binary target variable based on a source column

    Parameters:
    df (DataFrame): Input dataframe
    source_column (str): Column used to create the target
    method (str): Method to determine threshold ('mean', 'median', or 'custom')
    threshold (float): Custom threshold value if method is 'custom'

    Returns:
    tuple: (Modified dataframe, threshold value used)
    """
    if method == 'mean':
        threshold_value = df[source_column].mean()
    elif method == 'median':
        threshold_value = df[source_column].median()
    elif method == 'custom' and threshold is not None:
        threshold_value = threshold
    else:
        raise ValueError("Invalid method or missing threshold for custom method")

    # Create target variable
    target_column = f"{source_column}_Level"
    df[target_column] = (df[source_column] > threshold_value).astype(int)

    print(f"Created target variable '{target_column}' using threshold: {threshold_value}")
    print(f"Class distribution:\n{df[target_column].value_counts(normalize=True) * 100}")

    return df, threshold_value

# Imputation function
def impute_missing_values(X, strategy='knn', n_neighbors=5):
    """
    Impute missing values using various strategies

    Parameters:
    X (DataFrame): Features dataframe
    strategy (str): Imputation strategy ('mean', 'median', 'knn')
    n_neighbors (int): Number of neighbors for KNN imputation

    Returns:
    DataFrame: Imputed dataframe
    """
    X_copy = X.copy()

    if strategy == 'mean':
        return X_copy.fillna(X_copy.mean())

    elif strategy == 'median':
        return X_copy.fillna(X_copy.median())

    elif strategy == 'knn':
        imputer = KNNImputer(n_neighbors=n_neighbors)
        imputed_array = imputer.fit_transform(X_copy)
        return pd.DataFrame(imputed_array, columns=X_copy.columns, index=X_copy.index)

    else:
        raise ValueError("Invalid imputation strategy")

# Feature importance analysis
def analyze_feature_importance(model, X, y, method='built_in', n_repeats=10):
    """
    Analyze feature importance using different methods

    Parameters:
    model: Trained model
    X (DataFrame): Features
    y (Series): Target variable
    method (str): Method to use ('built_in' or 'permutation')
    n_repeats (int): Number of repeats for permutation importance

    Returns:
    DataFrame: Feature importance dataframe
    """
    if method == 'built_in':
        if hasattr(model, 'feature_importances_'):
            importances = model.feature_importances_
        elif hasattr(model, 'coef_'):
            importances = np.abs(model.coef_[0])
        else:
            raise ValueError("Model doesn't have built-in feature importance")

    elif method == 'permutation':
        result = permutation_importance(model, X, y, n_repeats=n_repeats, random_state=42)
        importances = result.importances_mean

    else:
        raise ValueError("Invalid feature importance method")

    # Create and return DataFrame
    importance_df = pd.DataFrame({
        'Feature': X.columns,
        'Importance': importances
    }).sort_values(by='Importance', ascending=False)

    return importance_df

# Model training and evaluation
def train_and_evaluate_models(X_train, X_test, y_train, y_test, models_dict, cv=5, scoring='accuracy'):
    """
    Train multiple models and evaluate their performance

    Parameters:
    X_train, X_test, y_train, y_test: Train-test split data
    models_dict (dict): Dictionary of models to train
    cv (int): Number of cross-validation folds
    scoring (str): Scoring metric for cross-validation

    Returns:
    dict: Dictionary of trained models and their performance metrics
    """
    results = {}

    for name, model in models_dict.items():
        print(f"\nTraining and evaluating {name}...")

        # Cross-validation
        cv_scores = cross_val_score(model, X_train, y_train, cv=cv, scoring=scoring)
        cv_mean, cv_std = cv_scores.mean(), cv_scores.std()

        # Fit model
        model.fit(X_train, y_train)

        # Predictions
        y_pred = model.predict(X_test)
        if hasattr(model, "predict_proba"):
            y_proba = model.predict_proba(X_test)[:, 1]
        else:
            y_proba = None

        # Calculate metrics
        report = classification_report(y_test, y_pred, output_dict=True)
        cm = confusion_matrix(y_test, y_pred)

        # ROC curve and AUC if probability estimates are available
        if y_proba is not None:
            fpr, tpr, _ = roc_curve(y_test, y_proba)
            roc_auc = auc(fpr, tpr)

            # Precision-Recall curve
            precision, recall, _ = precision_recall_curve(y_test, y_proba)
            pr_auc = average_precision_score(y_test, y_proba)
        else:
            fpr, tpr, roc_auc = None, None, None
            precision, recall, pr_auc = None, None, None

        # Store results
        results[name] = {
            'model': model,
            'cv_mean': cv_mean,
            'cv_std': cv_std,
            'report': report,
            'confusion_matrix': cm,
            'roc_curve': (fpr, tpr),
            'roc_auc': roc_auc,
            'pr_curve': (precision, recall),
            'pr_auc': pr_auc
        }

        print(f"{name} CV {scoring}: {cv_mean:.4f} ± {cv_std:.4f}")
        print(f"{name} Test Accuracy: {report['accuracy']:.4f}")

    return results

# Hyperparameter tuning
def tune_hyperparameters(X, y, model, param_grid, cv=5, scoring='accuracy'):
    """
    Perform hyperparameter tuning for a model

    Parameters:
    X, y: Features and target variable
    model: Model to tune
    param_grid (dict): Parameter grid for tuning
    cv (int): Number of cross-validation folds
    scoring (str): Scoring metric

    Returns:
    object: Best model
    """
    print(f"\nPerforming hyperparameter tuning...")

    grid_search = GridSearchCV(model, param_grid, cv=cv, scoring=scoring, n_jobs=-1)
    grid_search.fit(X, y)

    print(f"Best parameters: {grid_search.best_params_}")
    print(f"Best cross-validation score: {grid_search.best_score_:.4f}")

    return grid_search.best_estimator_

# Visualize results
def visualize_results(results):
    """
    Visualize model results with various plots

    Parameters:
    results (dict): Dictionary of model results from train_and_evaluate_models
    """
    # Model performance comparison
    model_names = list(results.keys())
    cv_scores = [results[name]['cv_mean'] for name in model_names]
    test_scores = [results[name]['report']['accuracy'] for name in model_names]

    plt.figure(figsize=(12, 6))
    bar_width = 0.35
    index = np.arange(len(model_names))

    plt.bar(index, cv_scores, bar_width, label='Cross-Validation Score', color='skyblue')
    plt.bar(index + bar_width, test_scores, bar_width, label='Test Score', color='lightcoral')

    plt.xlabel('Model')
    plt.ylabel('Accuracy')
    plt.title('Model Performance Comparison')
    plt.xticks(index + bar_width / 2, model_names)
    plt.legend()
    plt.tight_layout()
    plt.show()

    # Confusion matrices
    n_models = len(model_names)
    fig, axes = plt.subplots(1, n_models, figsize=(5 * n_models, 5))
    if n_models == 1:
        axes = [axes]

    for i, name in enumerate(model_names):
        cm = results[name]['confusion_matrix']
        disp = ConfusionMatrixDisplay(confusion_matrix=cm)
        disp.plot(cmap='Blues', ax=axes[i], values_format='d')
        axes[i].set_title(f"Confusion Matrix - {name}")

    plt.tight_layout()
    plt.show()

    # ROC curves
    plt.figure(figsize=(12, 8))

    for name in model_names:
        if results[name]['roc_curve'][0] is not None:
            fpr, tpr = results[name]['roc_curve']
            roc_auc = results[name]['roc_auc']
            plt.plot(fpr, tpr, lw=2, label=f'{name} (AUC = {roc_auc:.4f})')

    plt.plot([0, 1], [0, 1], color='gray', lw=1, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC) Curves')
    plt.legend(loc="lower right")
    plt.grid(True, alpha=0.3)
    plt.show()

    # Precision-Recall curves
    plt.figure(figsize=(12, 8))

    for name in model_names:
        if results[name]['pr_curve'][0] is not None:
            precision, recall = results[name]['pr_curve']
            pr_auc = results[name]['pr_auc']
            plt.plot(recall, precision, lw=2, label=f'{name} (AP = {pr_auc:.4f})')

    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall Curves')
    plt.legend(loc="lower left")
    plt.grid(True, alpha=0.3)
    plt.show()

#%% Main execution
def main():
    # Load data
    file_path = url
    df = load_and_preprocess_data(file_path) 

    # Explore key features for understanding credit status
    important_features = ['Creditscore', 'Likelihoodoffailure', 'CreditlimitGBPGBP',
                          'ReturnonTotalAssets2019', 'Currentratiox2019', 'SolvencyratioLiabilitybased2019']
    
    explore_features(df, important_features)
    # Create target variable
    df, threshold = create_target_variable(df, 'Creditscore', method='mean')

    # Keep only numerical features and prepare data
    df_numeric = df.select_dtypes(include=[np.number])
    print(f"\nWorking with {df_numeric.shape[1]} numerical features")

    # Drop target from features (and source of target)
    X = df_numeric.drop(columns=['Creditscore', 'Creditscore_Level'])
    y = df_numeric['Creditscore_Level']

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=123, stratify=y)
    print(f"Training set: {X_train.shape[0]} samples")
    print(f"Testing set: {X_test.shape[0]} samples")
    

    # Handle missing values with KNN imputation
    print("\nImputing missing values using KNN...")
    X_train_imputed = impute_missing_values(X_train, strategy='knn', n_neighbors=5)
    X_test_imputed = impute_missing_values(X_test, strategy='knn', n_neighbors=5)

    # Scaling
    print("Scaling features...")
    scaler = RobustScaler()  # More robust to outliers than StandardScaler
    X_train_scaled = scaler.fit_transform(X_train_imputed)
    X_test_scaled = scaler.transform(X_test_imputed)

    # Define models to train
    models = {
        'Decision Tree': DecisionTreeClassifier(random_state=123),
        'Random Forest': RandomForestClassifier(random_state=123),
        'SVM': SVC(kernel='rbf', random_state=123, probability=True),
        'Logistic Regression': LogisticRegression(random_state=123, max_iter=1000)
    }


    # Train and evaluate models
    results = train_and_evaluate_models(
        X_train_scaled, X_test_scaled, y_train, y_test,
        models, cv=5, scoring='accuracy'
    )

    # Visualize results
    visualize_results(results)

    # Find best performing model
    best_model_name = max(results, key=lambda k: results[k]['cv_mean'])
    best_model = results[best_model_name]['model']
    print(f"\nBest performing model: {best_model_name}")

    # Analyze feature importance of best model
    importance_df = analyze_feature_importance(best_model, pd.DataFrame(X_train_scaled, columns=X_train.columns), y_train)
    print("\nTop 15 most important features:")
    print(importance_df.head(15))

    # Visualize feature importance
    plt.figure(figsize=(12, 8))
    sns.barplot(x='Importance', y='Feature', data=importance_df.head(15))
    plt.title(f'Top 15 Feature Importances - {best_model_name}')
    plt.tight_layout()
    plt.show()

    # Experiment: Removing 'Likelihoodoffailure' if it's too dominant
    if 'Likelihoodoffailure' in importance_df['Feature'].values:
        if importance_df.loc[importance_df['Feature'] == 'Likelihoodoffailure', 'Importance'].values[0] > 0.5:
            print("\nExperiment: Removing dominant feature 'Likelihoodoffailure'...")

            # Drop Likelihoodoffailure
            X_train_reduced = X_train.drop(columns=['Likelihoodoffailure'])
            X_test_reduced = X_test.drop(columns=['Likelihoodoffailure'])

            # Impute and scale
            X_train_reduced_imputed = impute_missing_values(X_train_reduced, strategy='knn')
            X_test_reduced_imputed = impute_missing_values(X_test_reduced, strategy='knn')

            scaler_reduced = RobustScaler()
            X_train_reduced_scaled = scaler_reduced.fit_transform(X_train_reduced_imputed)
            X_test_reduced_scaled = scaler_reduced.transform(X_test_reduced_imputed)

            # Train and evaluate with reduced feature set
            reduced_results = train_and_evaluate_models(
                X_train_reduced_scaled, X_test_reduced_scaled, y_train, y_test,
                models, cv=5, scoring='accuracy'
            )

            # Visualize reduced results
            visualize_results(reduced_results)

            # Feature importance without Likelihoodoffailure
            best_reduced_model_name = max(reduced_results, key=lambda k: reduced_results[k]['cv_mean'])
            best_reduced_model = reduced_results[best_reduced_model_name]['model']

            # Analyze feature importance
            importance_reduced_df = analyze_feature_importance(
                best_reduced_model,
                pd.DataFrame(X_train_reduced_scaled, columns=X_train_reduced.columns),
                y_train
            )

            print("\nTop 15 most important features (without Likelihoodoffailure):")
            print(importance_reduced_df.head(15))

            # Visualize feature importance
            plt.figure(figsize=(12, 8))
            sns.barplot(x='Importance', y='Feature', data=importance_reduced_df.head(15))
            plt.title(f'Top 15 Feature Importances - {best_reduced_model_name} (without Likelihoodoffailure)')
            plt.tight_layout()
            plt.show()

    # Hyperparameter tuning for best model
    if best_model_name == 'Decision Tree':
        param_grid = {
            'max_depth': [None, 5, 10, 15, 20],
            'min_samples_split': [2, 5, 10],
            'min_samples_leaf': [1, 2, 4]
        }
    elif best_model_name == 'Random Forest':
        param_grid = {
            'n_estimators': [50, 100, 200],
            'max_depth': [None, 10, 20],
            'min_samples_split': [2, 5, 10]
        }
    elif best_model_name == 'SVM':
        param_grid = {
            'C': [0.1, 1, 10, 100],
            'gamma': ['scale', 'auto', 0.1, 0.01]
        }
    elif best_model_name == 'Logistic Regression':
        param_grid = {
            'C': [0.01, 0.1, 1, 10, 100],
            'solver': ['liblinear', 'lbfgs', 'saga']
        }

    # Tune hyperparameters for best model
    tuned_model = tune_hyperparameters(
        X_train_scaled, y_train,
        models[best_model_name], param_grid,
        cv=5, scoring='accuracy'
    )

    # Final evaluation of tuned model
    y_pred_tuned = tuned_model.predict(X_test_scaled)
    final_report = classification_report(y_test, y_pred_tuned)
    print("\nFinal model evaluation:")
    print(final_report)

    print("\nAnalysis complete!")

if __name__ == "__main__":
    main()
    
#%%  EXTRA: CALIBRATING SVM 

    file_path = url
    df = load_and_preprocess_data(file_path) 

    important_features = ['Creditscore', 'Likelihoodoffailure', 'CreditlimitGBPGBP',
                          'ReturnonTotalAssets2019', 'Currentratiox2019', 'SolvencyratioLiabilitybased2019']
    
    explore_features(df, important_features)
    df, threshold = create_target_variable(df, 'Creditscore', method='mean')

    df_numeric = df.select_dtypes(include=[np.number])
    print(f"\nWorking with {df_numeric.shape[1]} numerical features")

    X = df_numeric.drop(columns=['Creditscore', 'Creditscore_Level'])
    X=X.fillna(X.mean())
    y = df_numeric['Creditscore_Level']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=123, stratify=y)
    print(f"Training set: {X_train.shape[0]} samples")
    print(f"Testing set: {X_test.shape[0]} samples")


    print("Scaling features...")
    scaler = RobustScaler()  # More robust to outliers than StandardScaler
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Define the parameter grid for C and gamma
param_grid = {'C': [0.1, 1, 10, 100],
              'gamma': [0.001, 0.01, 0.1, 1, 'scale', 'auto']}
print("Evaluating SVM...")
# Create an SVC object with the rbf kernel
svc = SVC(kernel='rbf', random_state=123, probability=True)

# Set up grid search with 5-fold cross-validation
grid_search = GridSearchCV(svc, param_grid, cv=5)

# Fit the model using the scaled training data and training labels
grid_search.fit(X_train_scaled, y_train)

# Display the best parameters found
print("Best parameters:", grid_search.best_params_)
 

#%% LETS FOCUS ON SVM
folder_path=r"C:\Users\Utente\Desktop\WESTMINSTER\AI and ML\Second part\Images"

creditscore_mean = df['Creditscore'].mean()
df['CreditLevel'] = (df['Creditscore'] > creditscore_mean).astype(int)
df_numeric = df.select_dtypes(include=[np.number])

X = df_numeric.drop(columns=['Creditscore', 'CreditLevel'])

X = X.fillna(X.mean())

y = df_numeric['CreditLevel']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=123)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
X.shape, y.shape, X_train_scaled.shape, X_test_scaled.shape

# Initialise Support Vector Machine (SVM)
svm_model = SVC(kernel='rbf', random_state=123, probability=True)

# Cross-validation (5-fold) for SVM
svm_scores = cross_val_score(svm_model, X_train_scaled, y_train, cv=5, scoring='accuracy')
svm_mean, svm_std = svm_scores.mean(), svm_scores.std()

{"SVM CV Accuracy": f"{svm_mean:.4f} ± {svm_std:.4f}"}

svm_model.fit(X_train_scaled, y_train)
# Predict on test set and get probability estimates
y_pred_svm = svm_model.predict(X_test_scaled)
y_proba_svm = svm_model.predict_proba(X_test_scaled)[:, 1]


# Classification report
print(classification_report(y_test, y_pred_svm))
report_svm = classification_report(y_test, y_pred_svm, output_dict=True)
report_svm_df = pd.DataFrame(report_svm).transpose()
print(report_svm_df)

# Confusion matrix
cm_svm = confusion_matrix(y_test, y_pred_svm)
disp_svm = ConfusionMatrixDisplay(confusion_matrix=cm_svm)
disp_svm.plot(cmap='Blues')
plt.title("Confusion Matrix - SVM")
plt.savefig(os.path.join(folder_path, "ConfusionMatrix-SVM.png"))

plt.show()

# ROC Curve
fpr_svm, tpr_svm, _ = roc_curve(y_test, y_proba_svm)
roc_auc_svm = auc(fpr_svm, tpr_svm)

plt.figure(figsize=(6, 5))
plt.plot(fpr_svm, tpr_svm, label=f"AUC = {roc_auc_svm:.4f}")
plt.plot([0, 1], [0, 1], 'k--')
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve - SVM")
plt.legend(loc="lower right")
plt.grid()
plt.savefig(os.path.join(folder_path, "ROC-SVM.png"))
plt.show()

report_svm_df.round(3)

#%% APPENDIX  A
     #TESTING IF ORIGINAL TRAIN AND TEST SAMPLES HAVE SAME DISTRIBUTION without considering the missing values!
    url = "https://raw.githubusercontent.com/Rickeagle/Credit-Worthiness.-Study-case-in-the-UK/main/EM.csv"
    file_path = url
    df = load_and_preprocess_data(file_path) 
    important_features = ['Creditscore', 'Likelihoodoffailure', 'CreditlimitGBPGBP',
                           'ReturnonTotalAssets2019', 'Currentratiox2019', 'SolvencyratioLiabilitybased2019']
    df2=df.dropna()
    explore_features(df2, important_features)
    df2, threshold2 = create_target_variable(df2, 'Creditscore', method='mean')
    df2_numeric = df2.select_dtypes(include=[np.number])
    print(f"\nWorking with {df2_numeric.shape[1]} numerical features")
    X2 = df2_numeric.drop(columns=['Creditscore', 'Creditscore_Level'])
    y2 = df2_numeric['Creditscore_Level']
    X_train2, X_test2, y_train2, y_test2 = train_test_split(X2, y2, test_size=0.3, random_state=123, stratify=y2)
    print(f"Training set: {X_train2.shape[0]} samples")
    print(f"Testing set: {X_test2.shape[0]} samples")

    X = np.concatenate([X_train2, X_test2], axis=0)
    y = np.concatenate([np.zeros(len(X_train2)), np.ones(len(X_test2))])

    clf2 = LogisticRegression(max_iter=1000)
    accuracy2 = cross_val_score(clf2, X, y, cv=5, scoring='accuracy').mean()
    print("Mean accuracy:", accuracy2)
    print("Given that the accuracy is close to 50% we can conclude that the two samples have the same distribution!")
    ###########################################################
#%% Appendix B - UNSUPERVISED ML for EDA
# Scale the full feature set (using the X defined earlier)
scaler_full = StandardScaler()
X_scaled = scaler_full.fit_transform(X)

# Apply KMeans clustering (set n_clusters as needed, here we use 2)
kmeans = KMeans(n_clusters=2, random_state=123)
clusters = kmeans.fit_predict(X_scaled)

# Evaluate clustering quality with Silhouette Score
sil_score = silhouette_score(X_scaled, clusters)
print("Silhouette Score:", sil_score)

# Visualize clusters with PCA (reducing dimensions to 2 for plotting)
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)

plt.figure(figsize=(6, 5))
plt.scatter(X_pca[:, 0], X_pca[:, 1], c=clusters, cmap='viridis', alpha=0.6)
plt.xlabel("Principal Component 1")
plt.ylabel("Principal Component 2")
plt.title("KMeans Clustering Visualization")
plt.show()















