import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import (KFold, GridSearchCV)
from sklearn.linear_model import LinearRegression
from sklearn.feature_selection import (RFE, SequentialFeatureSelector, SelectFromModel)
from sklearn.linear_model import RidgeCV


def kfold_nFeaturesSelector(data_in, features_in, target, random_state,
                            max_features, n_splits=5, shuffle=True):
    """
    Use k-fold linear regression to estimate average score per #parameters.

    Estimate average R-squared adjustment on kfold cross-validated sample
    grouped by number of explanatory variables used. Applicable for regression
    tasks. Using cross-validation tests average. Estimated by linear model.

    Parameters
    ----------
    data_in : str
        DataFrame with independent variables to analyze.
    features_in : str
        List of variables to be chosen from. Must come from data_in DataFrame.
    target : str
        Series with dependent variable. Must be continuous.
    random_state : int, default = 123
        Random number generator seed. used for KFold sampling.
    max_features : int, default = 10
        Limit of features in iteration. Algorithm will compute for models from
        i = 1 feature to max_features.
    n_splits : int, default = 5
        Cross-validation parameter - will split data_in to n_splits equal
        parts. VarClusHi library).
    shuffle : bool, default = True
        Whether to shuffle the data before splitting into batches. Note that
        the samples within each split will not be shuffled.

    Returns
    -------
    table: top10 scores
        Top 10 R-squared scores by mean test-set score and corresponding 
        number of features category.
    plot: mean scores plot
        Line plot of number of features selected versus average train & test
        sample R-squared scores.

    Notes
    -------------------
    Required libraries: \n
    import pandas as pd: \n
    import matplotlib.pyplot as plt: \n
    import numpy as np: \n
    from sklearn.model_selection import (KFold, GridSearchCV): \n
    from sklearn.linear_model import LinearRegression: \n
    from sklearn.feature_selection import RFE
    """
    # cross-validation configuration
    folds = KFold(n_splits=5, shuffle=True, random_state=random_state)

    # Removing low variance to avoid problems with cross_validation
    features_exclude = list(data_in[features_in].std()[data_in[features_in].std() < 0.1].index)
    features_fin = list(set(features_in)-set(features_exclude))

    # hyperparameters configuration
    hyper_params = [{'n_features_to_select': list(range(1, max_features))}]

    # grid search
    # Model specification
    # Further improvement: handling multiple types of models
    lm = LinearRegression()
    lm.fit(data_in[features_fin], target)  # y_train_valid
    rfe = RFE(lm)

    # executing GridSearchCV()
    model_cv = GridSearchCV(estimator=rfe,
                            param_grid=hyper_params,
                            scoring='r2',
                            cv=folds,
                            verbose=1,
                            return_train_score=True)
    # fitting model on train_valid sample
    model_cv.fit(data_in[features_fin], target)
    # cv results
    cv_results = pd.DataFrame(model_cv.cv_results_)

    # print(cv_results)
    # plotting cv results
    plt.figure(figsize=(16, 6))
    plt.plot(cv_results["param_n_features_to_select"],
             cv_results["mean_test_score"])
    plt.plot(cv_results["param_n_features_to_select"],
             cv_results["mean_train_score"])
    plt.xlabel('number of features')
    plt.ylabel('R^2')
    plt.xticks(np.arange(1, max_features, 1.0))
    plt.ylim(ymax=1.0, ymin=0.5)
    plt.yticks(np.arange(0.5, 1, 0.05))
    plt.title("Optimal Number of Features")
    plt.legend(['valid score', 'train score'], loc='upper left')
    print(cv_results[["param_n_features_to_select", "mean_train_score",
                      "mean_test_score"]].sort_values("mean_test_score",
                                                      ascending=False).head(10))


def varSelect_fwdBckw(data_in, features_in, target, n_features_space,
                      variable_dictionary):
    """
    Select subsets of n best predictors and save in a variable dictionary.

    Creates dictionary of variable sets for modelling using RidgeCV 
    regression. Variables are chosen using 3 algorithms: Select from model,
    forward selection, backwards selection with constraint to no more 
    variables than indicated in n_features_space list.

    Parameters
    ----------
    data_in : str
        DataFrame with independent variables to analyze.
    features_in : str
        List of variables to be chosen from. Must come from data_in DataFrame.
    target : str
        Series with dependent variable. Must be continuous.
    n_features_space : list
        List of limit of selected features. Passed value of [1,2,3] will
        select 3 sets of features for each method.
    variable_dictionary : str
        Pointer at variable dictionary that will be updated with
        variable_dictionary[value] = [list of selected features].

    Returns
    -------
    variable_dictionary["SFM_"+str(n)] : list
        List of variables stored as dictionary entry. Chosen by select from
        model method.
    variable_dictionary["FWD_"+str(n)] : list
        List of variables stored as dictionary entry. Chosen by forward
        variable selection method.
    variable_dictionary["BKWD_"+str(n)] : list
        List of variables stored as dictionary entry. Chosen by backward
        variable selection method.

    Notes
    -------------------
    Required libraries: \n
    * from sklearn.linear_model import RidgeCV \n
    * from sklearn.feature_selection import (SequentialFeatureSelector, SelectFromModel)

    References
    ----------
    Source materials: \n
    1. Diabetes use-case <https://scikit-learn.org/stable/auto_examples/feature_selection/plot_select_from_model_diabetes.html>
    """

    # Estimate coefficients of parameters with Ridge Cross-Validation estimator
    ridge = RidgeCV(alphas=np.logspace(-6, 6, num=5)).fit(data_in[features_in],
                                                          target)
    importance = np.abs(ridge.coef_)
    feature_names = np.array(data_in[features_in].columns)

    for n in n_features_space:
        n_features = n
        threshold = np.sort(importance)[-n_features] + 0.01
        sfm = SelectFromModel(ridge,
                              threshold=threshold).fit(data_in[features_in],
                                                       target)
        sfs_forward = SequentialFeatureSelector(ridge,
                                                n_features_to_select=n_features,
                                                direction="forward").fit(data_in[features_in],
                                                                         target)
        sfs_backward = SequentialFeatureSelector(ridge,
                                                 n_features_to_select=n_features,
                                                 direction="backward").fit(data_in[features_in],
                                                                          target)
        variable_dictionary["SFM_"+str(n)] = list(feature_names[sfm.get_support()])
        variable_dictionary["FWD_"+str(n)] = list(feature_names[sfs_forward.get_support()])
        variable_dictionary["BKWD_"+str(n)] = list(feature_names[sfs_backward.get_support()])
