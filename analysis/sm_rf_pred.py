import pandas as pd
import numpy as np
import pydot
from pprint import pprint
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
from sklearn.tree import export_graphviz


# Created with help from https://towardsdatascience.com/random-forest-in-python-24d0893d51c0
def run_rf_model(df, cols=None, rz=True):
    df = pd.read_csv(df)
    df = df.sample(frac=1)
    df = df.head(10000)

    if rz:
        labels = df['rzMean'].values
    else:
        labels = df['surfMean'].values

    if cols is not None:
        df = df[cols]

    df = df.drop(['rzMean', 'surfMean'], axis=1)
    feat_names = list(df.columns)

    x_train, x_test, y_train, y_test = train_test_split(df.values, labels, test_size=0.25, random_state=1)
    rf = RandomForestRegressor(n_estimators=1000, random_state=1)
    rf.fit(x_train, y_train)

    out_dict = {'feat_names': feat_names,
                'x_train': x_train,
                'x_test': x_test,
                'y_train': y_train,
                'y_test': y_test,
                'model': rf}

    return out_dict


def visualize_tree(rf_model, feat_names):

    tree = rf_model.estimators_[5]

    export_graphviz(tree, out_file='tree.dot', feature_names=feat_names, rounded=True, precision=1)
    (graph,) = pydot.graph_from_dot_file('tree.dot')
    graph.write_png('tree.png')


def get_variable_importance(rf_model, feat_names):

    imp = list(rf_model.feature_importances_)
    feature_imp = [(feature, round(importance, 2)) for feature, importance in zip(feat_names, imp)]
    feature_imp = sorted(feature_imp, key=lambda x: x[1], reverse=True)
    [print('Variable: {:20} Importance: {}'.format(*pair)) for pair in feature_imp];


def tune_hyperparameters(df, rz=True, cols=None):

    df = pd.read_csv(df)
    df = df.sample(frac=1)
    df = df.head(10000)

    if rz:
        labels = df['rzMean'].values
    else:
        labels = df['surfMean'].values

    if cols is not None:
        df = df[cols]

    df = df.drop(['rzMean', 'surfMean'], axis=1)
    feat_names = list(df.columns)

    x_train, x_test, y_train, y_test = train_test_split(df.values, labels, test_size=0.25, random_state=1)

    # n_estimators = [int(x) for x in np.linspace(start=200, stop=2000, num=10)]
    n_estimators = [500, 1000, 2000]
    max_features = ['auto', 'sqrt']
    # max_depth = [int(x) for x in np.linspace(10, 110, num=11)]
    # max_depth.append(None)
    max_depth = [3, None]
    min_samples_split = [2, 5, 10]
    min_samples_leaf = [1, 5, 10]
    bootstrap = [True, False]

    random_grid = {'n_estimators': n_estimators,
                   'max_features': max_features,
                   'max_depth': max_depth,
                   'min_samples_split': min_samples_split,
                   'min_samples_leaf': min_samples_leaf,
                   'bootstrap': bootstrap}

    rf = RandomForestRegressor()
    rf_grid = RandomizedSearchCV(estimator=rf, param_distributions=random_grid,
                                 n_iter=200, cv=5, n_jobs=10)

    rf_grid.fit(x_train, y_train)






df = '/mnt/e/Data/sm_ml_data/smap_gm_data.csv'
model = run_rf_model(df)

get_variable_importance(model['model'], model['feat_names'])
# pred = rf.predict(x_test)
# rmse = np.sqrt(mean_squared_error(y_test, pred))
# mae = abs(pred - y_test)
# acc = 100 - np.mean(100 * (mae / y_test))
