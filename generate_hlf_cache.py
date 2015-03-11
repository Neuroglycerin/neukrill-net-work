#!/usr/bin/env python
"""
Generate a cache of all the
"""

from __future__ import division

import neukrill_net.highlevelfeatures
import neukrill_net.utils
import copy
from sklearn.externals import joblib

# Define output path
pkl_path1 = '/disk/scratch/s1145806/cached_hlf_train_data_raw.pkl'
pkl_path2 = '/disk/scratch/s1145806/cached_hlf_train_raw.pkl'
pkl_path3 = '/disk/scratch/s1145806/cached_hlf_train_data_ranged.pkl'
pkl_path4 = '/disk/scratch/s1145806/cached_hlf_train_ranged.pkl'
pkl_path5 = '/disk/scratch/s1145806/cached_hlf_train_data_posranged.pkl'
pkl_path6 = '/disk/scratch/s1145806/cached_hlf_train_posranged.pkl'

# Define which basic attributes to use
attrlst = ['height','width','numpixels','sideratio','mean','std','stderr',
           'propwhite','propnonwhite','propbool']

# Parse the data
settings = neukrill_net.utils.Settings('settings.json')
X,y = neukrill_net.utils.load_rawdata(settings.image_fnames, settings.classes)

# Combine all the features we want to use
hlf  = neukrill_net.highlevelfeatures.BasicAttributes(attrlst)
hlf += neukrill_net.highlevelfeatures.ContourMoments()
hlf += neukrill_net.highlevelfeatures.ContourHistogram()
hlf += neukrill_net.highlevelfeatures.ThresholdAdjacency()
hlf += neukrill_net.highlevelfeatures.ZernikeMoments()
hlf += neukrill_net.highlevelfeatures.Haralick()
# hlf += neukrill_net.highlevelfeatures.CoocurProps()


# Save the raw values of every feature
X = hlf.generate_cache(X)
hlf_raw = copy.deepcopy(hlf)
# Save the feature matrix to disk
joblib.dump(X, pkl_path1)


# the +ve squashed values
X = hlf.generate_cache(X, lambda x: x/x.max(0), squash_for_postproc=True)
hlf_range = copy.deepcopy(hlf)
# Save the feature matrix
joblib.dump(X, pkl_path3)


# the range squashed values
X = hlf.generate_cache(X, lambda x: (x-x.min(0))/(x.max(0)-x.min(0)), squash_for_postproc=True)
hlf_posrange = copy.deepcopy(hlf)
# Save the feature matrix
joblib.dump(X, pkl_path5)


# Save the feature object with cache to path
joblib.dump(hlf_raw, pkl_path2)
# Save the feature object with cache to path
joblib.dump(hlf_range, pkl_path4)
# Save the feature object with cache to path
joblib.dump(hlf_posrange, pkl_path6)
