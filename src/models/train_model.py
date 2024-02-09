import pandas as pd
import numpy as np
from sklearn.linear_model import (LinearRegression, Ridge, ElasticNet, Lasso,
                                  TheilSenRegressor, RANSACRegressor,
                                  HuberRegressor, SGDRegressor, Lars,
                                  RidgeCV)
from sklearn.model_selection import GridSearchCV


class EstimatorSelectionHelper:
    """
    Iterate through dictionaries of models and returning sum of results.

    Class used for iterating through two dictionaries - one with list of
    models, and one with list of parameters (describing hyperparameter space).
    For each combination of two dictionaries class instantiates model object,
    trains it on provided dataset using k-fold linear regression to estimate
    average score per #parameters. Applicable for regression.

    Parameters
    ----------
    models : str
        Dictionary of model object instances.
    params : str
        Dictionary of hyperparameters for each model object from models dict.

    Notes
    -------------------
    Required libraries: \n
    import pandas as pd \n
    import numpy as np \n
    from sklearn.linear_model import LinearRegression \n
    from sklearn.linear_model import Ridge \n
    from sklearn.linear_model import ElasticNet \n
    from sklearn.linear_model import Lasso \n
    from sklearn.linear_model import TheilSenRegressor \n
    from sklearn.linear_model import RANSACRegressor \n
    from sklearn.linear_model import HuberRegressor \n
    from sklearn.linear_model import SGDRegressor \n
    from sklearn.linear_model import Lars \n
    from sklearn.linear_model import RidgeCV \n
    from sklearn.model_selection import GridSearchCV

    Methods
    -------
    score_summary(self, sort_by='mean_score')
        Return summary of scores.
    fit(self, X, y, cv=3, n_jobs=3, verbose=1, scoring=None, refit=False))
        Fit models from dictionaries to data using GridSearchCV.
    __init__(self, models, params)
        Constructor method.

    References
    ----------
    Source materials: \n
    1. Blog <https://www.davidsbatista.net/blog/2018/02/23/model_optimization/>
    """
    def __init__(self, models, params):
        """
        Constructor method.
        """
        # DQ check - raise ValueError if param dict doesn't have corresponding
        # items in models dict
        if not set(models.keys()).issubset(set(params.keys())):
            missing_params = list(set(models.keys()) - set(params.keys()))
            raise ValueError("Some estimators are missing parameters: %s"
                             % missing_params)
        self.models = models
        self.params = params
        self.keys = models.keys()
        self.grid_searches = {}

    def fit(self, X, y, cv=3, n_jobs=3, verbose=1, scoring=None, refit=False):
        """
        Fit models from dictionaries on given dataset using cross-validation.

        Using each model from models dictionary and set of hyperparameters
        from params dictionary method fits model to the dataset and estimates
        parameters.

        Parameters
        ----------
        X : str
            DataFrame with independent variables to analyze.
        y : str
            Series with dependent variable.
        cv : int, default = 3
            Cross-validation parameter - splits X dataset into cv independent
            samples.
        n_jobs : int, default = 3
            Multi-threading parameter - runs n_jobs in parallel.
        verbose : int, default = 1
            Controlling output - values from 0 (no messages) to 3 (all
            messages and times of computation).
        scoring : str, default = None
            Test evaluation strategy.
        refit : bool, default = False
            Refit an estimator using the best found parameters on the whole
            dataset.
        """
        for key in self.keys:
            print("Running GridSearchCV for %s." % key)
            model = self.models[key]
            params = self.params[key]
            gs = GridSearchCV(model, params, cv=cv, n_jobs=n_jobs,
                              verbose=verbose, scoring=scoring, refit=refit,
                              return_train_score=True)
            gs.fit(X, y)
            self.grid_searches[key] = gs

    def score_summary(self, sort_by='mean_score'):
        """
        Create model fit summary scores.

        Creates DataFrame containing information about model (estimator) and
        cross-validation fit scores on provided dataset: minimum, maximum,
        mean and standard deviation.

        Parameters
        ----------
        sort_by : str, default = 'mean_score'
            Variable used to sort resulting dataframe (descending).

        Returns
        -------
        df: DataFrame
            Pandas Dataframe with cross-validation estimation results for each
            model.
        """
        def row(key, scores, params):
            d = {
                 'estimator': key,
                 'min_score': min(scores),
                 'max_score': max(scores),
                 'mean_score': np.mean(scores),
                 'std_score': np.std(scores),
            }
            return pd.Series({**params, **d})

        rows = []
        for k in self.grid_searches:
            print(k)
            params = self.grid_searches[k].cv_results_['params']
            scores = []
            for i in range(self.grid_searches[k].cv):
                key = "split{}_test_score".format(i)
                r = self.grid_searches[k].cv_results_[key]
                scores.append(r.reshape(len(params), 1))

            all_scores = np.hstack(scores)
            for p, s in zip(params, all_scores):
                rows.append((row(k, s, p)))

        df = pd.concat(rows, axis=1).T.sort_values([sort_by], ascending=False)

        columns = ['estimator', 'min_score', 'mean_score', 'max_score',
                   'std_score']
        columns = columns + [c for c in df.columns if c not in columns]

        return df[columns]
