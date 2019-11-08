import pandas as pd
import numpy as np
import pydot
import pickle
from pprint import pprint
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error,  r2_score
from sklearn.tree import export_graphviz


# Read in data, convert to np arrays, and then split into train and test sets
def gather_split_data(df, cols=None, rz=True):

    df = pd.read_csv(df)
    df = df.sample(frac=1)
    df = df.head(10000)

    if rz:
        labels = df['rzMean'].values
    else:
        labels = df['surfMean'].values

    if cols is not None:
        df = df[cols]
    else:
        df = df.drop(['rzMean', 'surfMean'], axis=1)
    feat_names = list(df.columns)

    x_train, x_test, y_train, y_test = train_test_split(df.values, labels, test_size=0.25, random_state=1)

    return [x_train, x_test, y_train, y_test, feat_names]


# run baseline model
def run_rf_model(train_test, n_estimators=2000, min_samples_split=10, min_samples_leaf=5, max_features='auto',
                 max_depth=None, bootstrap=True):

    x_train, x_test, y_train, y_test, feat_names = train_test

    rf = RandomForestRegressor(n_estimators=n_estimators, min_samples_split=min_samples_split,
                               min_samples_leaf=min_samples_leaf, max_features=max_features,
                               max_depth=max_depth, bootstrap=bootstrap)
    rf.fit(x_train, y_train)

    return rf


# Visualize the tree created.
def visualize_tree(rf_model, feat_names):
    # Created with help from https://towardsdatascience.com/random-forest-in-python-24d0893d51c0
    tree = rf_model.estimators_[5]

    export_graphviz(tree, out_file='tree.dot', feature_names=feat_names, rounded=True, precision=1)
    (graph,) = pydot.graph_from_dot_file('tree.dot')
    graph.write_png('tree.png')


# sort variables based on importance.
def get_variable_importance(rf_model, feat_names):
    # Created with help from https://towardsdatascience.com/random-forest-in-python-24d0893d51c0
    imp = list(rf_model.feature_importances_)
    feature_imp = [(feature, round(importance, 2)) for feature, importance in zip(feat_names, imp)]
    feature_imp = sorted(feature_imp, key=lambda x: x[1], reverse=True)
    [print('Variable: {:20} Importance: {}'.format(*pair)) for pair in feature_imp];


# Create a random grid of hyperparameters to figure out what works the best
def tune_hyperparameters(train_test):

    x_train, x_test, y_train, y_test, feat_names = train_test

    max_d = [int(x) for x in np.linspace(10, 110, num=11)]
    max_d.append(None)

    random_grid = {'n_estimators': [int(x) for x in np.linspace(start=200, stop=2000, num=10)],
                   'max_features': ['auto', 'sqrt'],
                   'max_depth': max_d,
                   'min_samples_split': [2, 5, 10],
                   'min_samples_leaf': [1, 5, 10],
                   'bootstrap': [True, False]}

    rf = RandomForestRegressor()
    rf_grid = RandomizedSearchCV(estimator=rf, param_distributions=random_grid,
                                 n_iter=200, cv=5, n_jobs=10)

    rf_grid.fit(x_train, y_train)

    return rf_grid


# Look at model accuracy
def evaluate(model, test_features, test_labels):
    # Created with help from https://towardsdatascience.com/random-forest-in-python-24d0893d51c0
    predictions = model.predict(test_features)
    errors = abs(predictions - test_labels)
    mape = 100 * np.mean(errors / test_labels)
    rmse = np.sqrt(mean_squared_error(test_labels, predictions))
    accuracy = 100 - mape
    r2 = r2_score(test_labels, predictions)
    print('Model Performance')
    print('Average Error: {:0.4f} degrees.'.format(np.mean(errors)))
    print('Accuracy = {:0.2f}%.'.format(accuracy))

    return {'mae': errors,
            'accuracy': accuracy,
            'rmse': rmse,
            'r2': r2}


# Create a model for range of scenarios, tune hyperparameters, run the best model.
def train_model(df, method='rz'):

    if method == 'rz':
        train_test = gather_split_data(df, cols=['tmmn', 'tmmx', 'vpd'], rz=True)
    elif method == 'surf':
        train_test = gather_split_data(df, cols=['rmax', 'rmin', 'vpd'], rz=False)
    elif method == 'rz_full':
        train_test = gather_split_data(df, cols=None, rz=True)
    elif method == 'surf_full':
        train_test = gather_split_data(df, cols=None, rz=False)
    else:
        raise ValueError("'method' argument must either be 'rz', 'surf', 'rz_full', or 'surf_full'.")

    model_tuned = tune_hyperparameters(train_test)
    params = model_tuned.best_params_

    opt_model = run_rf_model(train_test, **params)

    out_dict = {'train_test': train_test,
                'opt_model': opt_model,
                'opt_params': params}

    pickle.dump(out_dict, open('/home/colin.brust/workspace/data/sm_ml_data/'+method+'_trained.pickle', 'wb'))

    return out_dict


# df = '/mnt/e/Data/sm_ml_data/smap_gm_data.csv'
dat = '/home/colin.brust/workspace/data/sm_ml_data/smap_gm_data.csv'

rz_model = train_model(dat, 'rz')
surf_model = train_model(dat, 'surf')
rz_full_model = train_model(dat, 'rz_full')
surf_full_model = train_model(dat, 'surf_full')
