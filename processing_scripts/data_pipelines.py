# import dependencies
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_curve, auc, roc_auc_score, accuracy_score, log_loss, confusion_matrix
from sklearn.pipeline import make_pipeline, Pipeline
from sklearn.base import TransformerMixin, BaseEstimator
import matplotlib.pyplot as plt
import os


def impute_columns_udf(col, imputation_type):
    """
    UDF to specify imputation method for a given variable.
    col is the pd.Series for which you want to impute, imputation_type should be chosen from
    ['mode', 'mean', 'zero', 'none']
    """
    if imputation_type == 'mode':
        val = col.mode()[0]
        return col.fillna(val)

    elif imputation_type == 'mean':
        val = col.mean()
        return col.fillna(val)

    elif imputation_type == 'zero':
        return col.fillna(0)

    elif imputation_type == 'none':
        return col.fillna('None')

    else:
        raise ValueError('imputation_type argument not valid')


class ImputeMissingData(BaseEstimator, TransformerMixin):
    """ Pipeline step that impute missing values, given impute methods specified.
        Note that numerical features with NaN's and no impute methods will raise an error,
        categorical features on the other hand will be filled with 'None' by default unless specified otherwise.
    """

    def __init__(self, impute_methods, num_features, cat_features):
        self.impute_methods = impute_methods
        self.num_features = num_features
        self.cat_features = cat_features

    def fit(self, X, y=None):
        return self

    def transform(self, input_data):
        output_data = input_data.copy()

        # check which features have missing values but imputation methods not specified
        features_nulls = output_data.isnull().sum()
        features_nulls = features_nulls[features_nulls > 0].index.tolist()
        features_need_impute = [f for f in features_nulls if f not in self.impute_methods.keys()]
        num_features_need_impute = [f for f in features_need_impute if f in self.num_features]
        cat_features_need_impute = [f for f in features_need_impute if f in self.cat_features]

        # raise exception for numerical features with missing values and no imputation method specified
        if num_features_need_impute:
            raise Exception("""These numerical features have missing values: {}. 
            Please specify their impute methods.""".format(num_features_need_impute))

            # set imputation method as 'none' for cat features with missing values with no imputation method specified
        if cat_features_need_impute:
            for cat_f in cat_features_need_impute:
                self.impute_methods[cat_f] = 'none'
            print("""{} have missing values with no imputation method specified. 
            By default, they have been filled with 'None'.""".format(cat_features_need_impute))

            # apply imputations
        for col, imp_method in self.impute_methods.items():
            output_data[col] = impute_columns_udf(output_data[col], imp_method)

        return output_data


class ConvertDataTypes(BaseEstimator, TransformerMixin):
    """ Coerce data types to specifications defined by CAT_FEATURES and NUM_FEATURES """

    def __init__(self, num_features, cat_features):
        self.num_features = num_features
        self.cat_features = cat_features

    def fit(self, X, y=None):
        return self

    def transform(self, input_data):
        output_data = input_data.copy()
        output_data[self.num_features] = output_data[self.num_features].astype(float)
        output_data[self.cat_features] = output_data[self.cat_features].astype(str)
        return output_data


class Dummify(BaseEstimator, TransformerMixin):
    """ Pipeline step that dummifies all categorical variables """

    def __init__(self, cat_feature_values):
        self.cat_feature_values = cat_feature_values

    def fit(self, X, y=None):
        return self

    def transform(self, input_data):
        output_data = input_data.copy()
        # specify values categorical features can take on, to ensure train/test DF have same cols
        cat_features = self.cat_feature_values.keys()
        for col in cat_features:
            output_data[col] = pd.Categorical(output_data[col], categories=self.cat_feature_values[col])
        output_data = pd.get_dummies(output_data, columns=cat_features)
        return output_data


# method to subset data for training model
def get_model_data(df, label, cat_features, num_features, impute_methods):
    """ Subsets variables used for model, runs it through pipeline to output data for model """

    # subsets features we are interested in
    cat_feature_values = dict(
        [(f, df[f].dropna().unique().tolist()) for f in cat_features])  # do EDA and limit to smaller list
    data = df[cat_features + num_features + [label]].copy()

    # train test split
    X, y = data.drop(label, axis=1).copy(), data[label].copy()
    x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # make pipeline
    data_pipeline = Pipeline([
        ('impute', ImputeMissingData(impute_methods, num_features, cat_features)),
        ('convert_dtypes', ConvertDataTypes(num_features, cat_features)),
        ('dummify', Dummify(cat_feature_values))
    ])

    # run pipeline
    X_train = data_pipeline.fit_transform(x_train)
    X_test = data_pipeline.transform(x_test)
    print("Training Data: {} | Test Data: {}".format(X_train.shape, X_test.shape))

    return X_train, X_test, y_train, y_test


# method to output evaluation metrics and plot ROC curve

def evaluate_model(truth, pred, plot_auc):
    """ Takes in arrays of truth and pred y values and return accuracy, logloss, roc_auc, and plot ROC """
    accuracy = accuracy_score(truth, (pred > 0.5).astype(int))
    logloss = log_loss(truth, pred)
    fpr, tpr, thresholds = roc_curve(truth, pred)
    roc_auc = auc(fpr, tpr)
    metrics = {'Accuracy': accuracy, 'ROC AUC': roc_auc, 'Log Loss': logloss}
    if plot_auc:
        plt.plot(fpr, tpr, label='AUC = {0:.3f}'.format(roc_auc))
        plt.plot([0, 1], [0, 1], 'k--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.0])
        plt.xlabel('FPR')
        plt.ylabel('TPR')
        plt.title('ROC')
        plt.legend(loc="lower right")
    else:
        pass
    return metrics


# method to train a model, output results, and plot AUC

def train_model(model, X_train, y_train, X_test, y_test, plot_auc=True):
    model.fit(X_train, y_train.values.ravel())
    y_pred = model.predict_proba(X_test)[:, 1]
    metrics = evaluate_model(y_test.values.ravel(), y_pred, plot_auc)
    return metrics
