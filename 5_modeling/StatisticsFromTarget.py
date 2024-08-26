import numpy as np
import scipy
from sklearn.base import BaseEstimator, OneToOneFeatureMixin
from sklearn.utils.validation import check_is_fitted
import pandas as pd

# statistical functions to apply over numerical columns 
def mean(x):
    return np.nanmean(x)

def std(x):
    return np.nanstd(x)

def median(x):
    return np.nanmedian(x)

def first_quartile(x):
    return np.nanquantile(x, .25)

def third_quartile(x):
    return np.nanquantile(x, .75)

def robust_min(x):
    return np.nanquantile(x, .05)

def robust_max(x):
    return np.nanquantile(x, .95)

def robust_range(x):
    return robust_max(x) - robust_min(x)

def skew(x):
    skew_value = scipy.stats.skew(x, nan_policy="omit")
    if np.isnan(skew_value):
        return 0
    else:
        return skew_value

def kurtosis(x):
    kurtosis_value = scipy.stats.kurtosis(x, nan_policy="omit")
    if np.isnan(kurtosis_value):
        return 0
    else:
        return kurtosis_value


# get statistics from numerical features
class StatisticsFromTarget(OneToOneFeatureMixin, BaseEstimator):

    def __init__(self,
                 categorical_cols):
        
        self.categorical_cols = categorical_cols

    def fit(self, X, y):
        
        if not isinstance(X, pd.DataFrame):
            raise TypeError("X must be a DataFrame.")
        
        func_list = [mean, std, median, first_quartile, third_quartile, robust_min, robust_max, robust_range, skew, kurtosis]
        
        X = X.copy()
        X["target"] = y
        self.reference_dict_ = {}
        for categorical_col in self.categorical_cols:
            self.reference_dict_[categorical_col] = X.groupby(categorical_col)[["target"]].agg(func_list)
            self.reference_dict_[categorical_col].columns = self.reference_dict_[categorical_col].columns.map(f'_{categorical_col}_'.join)
            
        return self

    def transform(self, X):
        
        if not isinstance(X, pd.DataFrame):
            raise TypeError("X must be a DataFrame.")
        
        check_is_fitted(self, 'reference_dict_')
        
        for categorical_col in self.categorical_cols:
            X = X.merge(self.reference_dict_[categorical_col], how="left", left_on=categorical_col, right_index=True)

        X.drop(columns=self.categorical_cols, inplace=True)

        return X
    
    def fit_transform(self, X, y):
        
        if not isinstance(X, pd.DataFrame):
            raise TypeError("X must be a DataFrame.")
        
        self.fit(X, y)
        return self.transform(X)