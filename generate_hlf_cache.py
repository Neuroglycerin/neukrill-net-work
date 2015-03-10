#!/usr/bin/env python
"""
Generate a cache of all the
"""

import neukrill_net.highlevelfeatures
import neukrill_net.utils

from sklearn.externals import joblib

# Define output path
pkl_path1 = 'cached_hlf_train_data.pkl'
pkl_path2 = 'cached_hlf_train.pkl'

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

# Save the Z-score of every feature
X = hlf.generate_cache(X, lambda x: (x-x.mean(0))/x.std(0), squash_for_postproc=True)

# Save the feature object with cache to path
joblib.dump(X, pkl_path1)

# Save the feature object with cache to path
joblib.dump(hlf, pkl_path2)
