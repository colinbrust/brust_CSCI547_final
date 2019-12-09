import pickle
import pandas as np
import numpy as np
import pydot
from pprint import pprint
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error,  r2_score
from sklearn.tree import export_graphviz


# Look at model accuracy
def evaluate(model, test_features, test_labels):
    # Created with help from https://towardsdatascience.com/random-forest-in-python-24d0893d51c0
    predictions = model.predict(test_features)
    errors = abs(predictions - test_labels)
    mape = 100 * np.mean(errors / test_labels)
    rmse = np.sqrt(mean_squared_error(test_labels, predictions))
    accuracy = 100 - mape
    r2 = r2_score(test_labels, predictions)
    print('Average Error: {:0.4f}'.format(np.mean(errors)))
    print('Accuracy = {:0.2f}%'.format(accuracy))
    print('RMSE: {:0.4f}'.format(rmse))
    print('R-Squared: {:0.4f}'.format(r2))


# sort variables based on importance.
def get_variable_importance(rf_model, feat_names):

    rf_model = rf_model.best_estimator_
    # Created with help from https://towardsdatascience.com/random-forest-in-python-24d0893d51c0
    imp = list(rf_model.feature_importances_)
    feature_imp = [(feature, round(importance, 2)) for feature, importance in zip(feat_names, imp)]
    feature_imp = sorted(feature_imp, key=lambda x: x[1], reverse=True)
    [print('Variable: {:20} Importance: {}'.format(*pair)) for pair in feature_imp]


# Get variable importance for different models
def evaluate_model(model_pickle):

    dat = pickle.load(open(model_pickle, 'rb'))

    x_train, x_test, y_train, y_test, feat_names = dat['train_test']
    opt_model = dat['opt_model']
    opt_params = dat['opt_params']

    print('Variable Importance:')
    get_variable_importance(opt_model, feat_names)

    print('\nModel Performance:')
    evaluate(opt_model, x_test, y_test)

    print('\nOptimal Hyperparameters:')
    pprint(opt_params)


