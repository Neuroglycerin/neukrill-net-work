#!/usr/bin/env python
"""
Generate a cache of all the
"""

from __future__ import division

import neukrill_net.highlevelfeatures
import neukrill_net.utils
import copy
import numpy as np
from sklearn.externals import joblib

# Define output path
pkl_path1 = '/disk/data1/s1145806/cached_hlf_test_data_raw.pkl'
pkl_path2 = '/disk/data1/s1145806/cached_hlf_test_raw.pkl'
pkl_path3 = '/disk/data1/s1145806/cached_hlf_test_data_ranged.pkl'
pkl_path4 = '/disk/data1/s1145806/cached_hlf_test_ranged.pkl'
pkl_path5 = '/disk/data1/s1145806/cached_hlf_test_data_posranged.pkl'
pkl_path6 = '/disk/data1/s1145806/cached_hlf_test_posranged.pkl'

# Define which basic attributes to use
attrlst = ['height','width','numpixels','sideratio','mean','std','stderr',
           'propwhite','propnonwhite','propbool']

# Parse the data
settings = neukrill_net.utils.Settings('settings.json')
X,y = neukrill_net.utils.load_rawdata(settings.image_fnames)

# Combine all the features we want to use
hlf  = neukrill_net.highlevelfeatures.BasicAttributes(attrlst)
hlf += neukrill_net.highlevelfeatures.ContourMoments()
hlf += neukrill_net.highlevelfeatures.ContourHistogram()
hlf += neukrill_net.highlevelfeatures.ThresholdAdjacency()
hlf += neukrill_net.highlevelfeatures.ZernikeMoments()
hlf += neukrill_net.highlevelfeatures.Haralick()
# hlf += neukrill_net.highlevelfeatures.CoocurProps()


# Save the raw values of every feature
X_raw = hlf.generate_cache(X)
hlf_raw = copy.deepcopy(hlf)
# Save the feature matrix to disk
joblib.dump(X_raw, pkl_path1)


# the [-1,1] squashed values
X_range = X / np.amax(np.absolute(X),0)
# Save the feature matrix
joblib.dump(X_range, pkl_path3)


# the [0,1] squashed values
X_posrange = (X-X.min(0))/(X.max(0)-X.min(0))
# Save the feature matrix
joblib.dump(X_posrange, pkl_path5)


