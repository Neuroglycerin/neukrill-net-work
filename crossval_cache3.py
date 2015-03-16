#!/usr/bin/env python

from __future__ import print_function

from pprint import pprint
from time import time
import logging

import sys
import numpy as np
import sklearn
import neukrill_net.utils
import neukrill_net.highlevelfeatures
import neukrill_net.stacked
import time
from sklearn.externals import joblib
import sklearn.ensemble
import sklearn.pipeline
import sklearn.feature_selection
import sklearn.grid_search

# Display progress logs on stdout
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s %(levelname)s %(message)s')

settings = neukrill_net.utils.Settings('settings.json')

train_split = 0.8

X = joblib.load('/disk/data1/s1145806/cached_hlf_train3_data_raw_ranged.pkl')
X_paths,y = settings.flattened_train_paths(settings.classes)
y = np.array(y)

n_augments = X.shape[0]

li_train = neukrill_net.utils.train_test_split_bool(settings.image_fnames, 'train', train_split, classes=settings.classes)
li_validate = neukrill_net.utils.train_test_split_bool(settings.image_fnames, 'validation', train_split, classes=settings.classes)

X_train = X[:,li_train,:]
X_validate = X[:,li_validate,:]
y_train = y[li_train]
y_validate = y[li_validate]
XX_train = X_train.reshape((X_train.shape[0]*X_train.shape[1],X_train.shape[2]))
XX_validate = X_validate.reshape((X_validate.shape[0]*X_validate.shape[1],X_validate.shape[2]))
yy_train = np.tile(y_train, n_augments)
yy_validate = np.tile(y_validate, n_augments)


clf = sklearn.ensemble.RandomForestClassifier(n_estimators=1000, max_depth=25, min_samples_leaf=3, n_jobs=16, random_state=42)
pcfilter = sklearn.feature_selection.SelectPercentile(sklearn.feature_selection.f_classif, percentile=100)

pipeline = sklearn.pipeline.Pipeline([('filter', pcfilter), ('clf', clf)])

parameters = {'filter__percentile': [50, 60, 40, 70, 30, 80, 20]}

scorer = sklearn.metrics.make_scorer(sklearn.metrics.log_loss, greater_is_better=False, needs_proba=True)

grid_search = sklearn.grid_search.GridSearchCV(pipeline, parameters, scoring=scorer, n_jobs=1, cv=4, verbose=1)

print("Performing grid search...")
print("pipeline:", [name for name, _ in pipeline.steps])
print("parameters:")
pprint(parameters)
t0 = time.time()
grid_search.fit(XX_train, yy_train)
print("done in %0.3fs" % (time.time() - t0))
print()

print("Best score: %0.3f" % grid_search.best_score_)
print("Best parameters set:")
best_parameters = grid_search.best_estimator_.get_params()
for param_name in sorted(parameters.keys()):
    print("\t%s: %r" % (param_name, best_parameters[param_name]))

try:
    joblib.dump(grid_search, '/disk/data1/s1145806/train3_model.pkl')
except:
    pass


