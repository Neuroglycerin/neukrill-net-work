#!/usr/bin/env python
"""
Generate a cache of all the ComputerVision highlevelfeatures to send to Pylearn2
"""

from __future__ import division

import neukrill_net.augment
import neukrill_net.highlevelfeatures
import neukrill_net.utils
import copy
import numpy as np
from sklearn.externals import joblib

# Define output path
pkl_path1 = '/disk/data1/s1145806/cached_hlf_test15_data_raw.pkl'
pkl_path2 = '/disk/data1/s1145806/cached_hlf_test15_raw.pkl'
pkl_path3 = '/disk/data1/s1145806/cached_hlf_test15_data_ranged.pkl'
pkl_path4 = '/disk/data1/s1145806/cached_hlf_test15_ranged.pkl'
pkl_path5 = '/disk/data1/s1145806/cached_hlf_test15_data_posranged.pkl'
pkl_path6 = '/disk/data1/s1145806/cached_hlf_test15_posranged.pkl'

# Define which basic attributes to use
attrlst = ['height','width','numpixels','sideratio','mean','std','stderr',
           'propwhite','propnonwhite','propbool']

# Parse the data
settings = neukrill_net.utils.Settings('settings.json')
X,y = neukrill_net.utils.load_rawdata(settings.image_fnames)

# Combine all the features we want to use
hlf_list  = []
hlf_list.append( neukrill_net.highlevelfeatures.BasicAttributes(attrlst) )
hlf_list.append( neukrill_net.highlevelfeatures.ContourMoments()         )
hlf_list.append( neukrill_net.highlevelfeatures.ContourHistogram()       )
hlf_list.append( neukrill_net.highlevelfeatures.ThresholdAdjacency()     )
hlf_list.append( neukrill_net.highlevelfeatures.ZernikeMoments()         )
hlf_list.append( neukrill_net.highlevelfeatures.Haralick()               )
# hlf_list.append( neukrill_net.highlevelfeatures.CoocurProps()            )

augs = {'units': 'uint8',
        'rotate': 3,
        'rotate_is_resizable': 1,
        'crop': 1}

aug_fun = neukrill_net.augment.augmentation_wrapper(**augs)
hlf = neukrill_net.highlevelfeatures.MultiHighLevelFeature(hlf_list, augment_func=aug_fun)

# Save the raw values of every feature
X_raw = hlf.generate_cache(X)
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

