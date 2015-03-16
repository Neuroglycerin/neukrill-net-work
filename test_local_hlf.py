#!/usr/bin/env python

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

# Make sure we don't just quit on error
def info(type, value, tb):
    if hasattr(sys, 'ps1') or not sys.stderr.isatty():
    # we are in interactive mode or we don't have a tty-like
    # device, so we call the default hook
        sys.__excepthook__(type, value, tb)
    else:
        import traceback, pdb
        # we are NOT in interactive mode, print the exception...
        traceback.print_exception(type, value, tb)
        print
        # ...then start the debugger in post-mortem mode.
        # pdb.pm() # deprecated
        pdb.post_mortem(tb) # more "modern"
sys.excepthook = info

print "Setup..."

max_num_kp = 200
train_cache_path = '/disk/data1/s1145806/cached_hlf_train_data_raw_ranged.pkl'
test_cache_path = '/disk/data1/s1145806/cached_hlf_test_data_raw_ranged.pkl'


settings = neukrill_net.utils.Settings('settings.json')
X,y = settings.flattened_train_paths(settings.classes)
X_test = settings.image_fnames['test']

X = np.array(X)
y = np.array(y)

detector_list = [lambda image: neukrill_net.image_features.get_ORB_keypoints(image, n=max_num_kp, patchSize=9),
                 lambda image: neukrill_net.image_features.get_BRISK_keypoints(image, n=max_num_kp),
                 lambda image: neukrill_net.image_features.get_MSER_keypoints(image, n=max_num_kp)]

describer_list = [neukrill_net.image_features.get_ORB_descriptions,
                  neukrill_net.image_features.get_BRISK_descriptions,
                  neukrill_net.image_features.get_ORB_descriptions]

kprf_base = sklearn.ensemble.RandomForestClassifier(n_estimators=500, max_depth=15,
                                                            min_samples_leaf=20, n_jobs=16, random_state=42)

hlf_list = []
for index,detector in enumerate(detector_list):
    hlf_list += [neukrill_net.highlevelfeatures.KeypointEnsembleClassifier(detector, describer_list[index], kprf_base,
                                                                     return_num_kp=True, summary_method='vote')]

hlf = neukrill_net.highlevelfeatures.MultiHighLevelFeature(hlf_list)

print "Partitioned the training data"

inner, outer = sklearn.cross_validation.train_test_split(np.arange(len(y)), test_size=0.2, random_state=42)

print "Fitting keypoint predictions"

hlf.fit(X[inner], y[inner])

joblib.dump(hlf, '/disk/data1/s1145806/keypoint_transformer.pkl')

print "Transforming to keypoint prediction basis"

X_train_local = hlf.transform(X[outer])
X_train_local = X_train_local.squeeze(0)
X_train_local = X_train_local.asdtype(np.float32)

X_test_local = hlf.transform(X_test)
X_test_local = X_test_local.squeeze(0)

print "Training second tier classifier"

clf = sklearn.ensemble.RandomForestClassifier(n_estimators=1500, max_depth=20,
                                              min_samples_leaf=3, n_jobs=16, random_state=42)

clf.fit(X_train_local)

print "Getting predictions"

p = clf.predict_proba(X_test_local)

names = [os.path.basename(path) for path in settings.image_fnames['test']]

print "Saving to file"

neukrill_net.utils.write_predictions('local_keypoint_predicitons.csv', p, names, settings.classes)


# Now add global features on top of this

print "Getting global features from path"

X_train_global = joblib.load(train_cache_path)
X_train_global = X_train_global.squeeze(0)
X_train_global = X_train_global[outer,:]

X_test_global = joblib.load(test_cache_path)
X_test_global = X_test_global.squeeze(0)

X_train_combo = np.concatenate((X_train_local, X_train_global),1)
X_test_combo = np.concatenate((X_test_local, X_test_global),1)

print "Training classifier on the combined dataset"

clf2 = sklearn.ensemble.RandomForestClassifier(n_estimators=1500, max_depth=20,
                                              min_samples_leaf=3, n_jobs=16, random_state=42)

clf2.fit(X_train_combo)

print "Getting predicitons"

p = clf2.predict_proba(X_test_combo)

names = [os.path.basename(path) for path in settings.image_fnames['test']]

print "Saving to file"

neukrill_net.utils.write_predictions('local_and_global_keypoint_predicitons.csv', p, names, settings.classes)


