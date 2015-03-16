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
pkl_path1 = '/disk/data1/s1145806/cached_hlf_train15alt_data_raw.pkl'
pkl_path2 = '/disk/data1/s1145806/cached_hlf_train15alt_raw.pkl'
pkl_path3 = '/disk/data1/s1145806/cached_hlf_train15alt_data_ranged.pkl'
pkl_path4 = '/disk/data1/s1145806/cached_hlf_train15alt_ranged.pkl'
pkl_path5 = '/disk/data1/s1145806/cached_hlf_train15alt_data_posranged.pkl'
pkl_path6 = '/disk/data1/s1145806/cached_hlf_train15alt_posranged.pkl'

# Define which basic attributes to use
attrlst = ['height','width','numpixels','sideratio','mean','std','stderr',
           'propwhite','propnonwhite','propbool']

# Parse the data
settings = neukrill_net.utils.Settings('settings.json')
X,y = neukrill_net.utils.load_rawdata(settings.image_fnames, settings.classes)

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
        'rotate': 5,
        'rotate_is_resizable': 1,
        'shear': [0,5,10]}

aug_fun = neukrill_net.augment.augmentation_wrapper(**augs)
hlf = neukrill_net.highlevelfeatures.MultiHighLevelFeature(hlf_list, augment_func=aug_fun)

# Save the raw values of every feature
X_raw = hlf.generate_cache(X)
# Save the feature matrix to disk
joblib.dump(X_raw, pkl_path1)

# the [-1,1] squashed values
X_max = np.amax(np.absolute(X.reshape((X.shape[0]*X.shape[1],X.shape[2]))),0)
X_range = X / X_max
# Save the feature matrix
joblib.dump(X_range, pkl_path3)

# the [0,1] squashed values
X_posrange = (X-X.min(0).min(1))/(X.max(0).max(1)-X.min(0).min(1))
# Save the feature matrix
joblib.dump(X_posrange, pkl_path5)

