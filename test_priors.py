#!/usr/bin/env python

import neukrill_net.utils as utils

import csv
import os
import gzip
import numpy as np
import sklearn.preprocessing
import sklearn.metrics

def main():
    out_fname = 'submission_priorprobs.csv'
    settings = utils.Settings('settings.json')
    
    # Get names of test data files
    names = [os.path.basename(fpath) for fpath in settings.image_fnames['test']]
    
    
    # Score expected from training data (not a CV score because no folds)
    labels = []
    for class_index, class_name in enumerate(settings.classes):
        num_images = len(settings.image_fnames['train'][class_name])
        # generate the class labels and add them to the list
        labels += num_images * [class_name]
    
    p = settings.class_priors[np.newaxis,:]
    p = np.tile(p, (len(labels),1))
    
    label_encoder = sklearn.preprocessing.LabelEncoder()
    y = label_encoder.fit_transform(labels)
    
    cv = sklearn.metrics.log_loss(y, p)
    print('CV = {}'.format(cv))
    
    
    # Write output
    with open(out_fname, 'w') as csv_out:
        out_writer = csv.writer(csv_out, delimiter=',')
        out_writer.writerow(['image'] + list(settings.classes))
        for index in range(len(names)):
            out_writer.writerow([names[index]] + list(settings.class_priors))
    
    with open(out_fname, 'rb') as f_in:
        f_out = gzip.open(out_fname + '.gz', 'wb')
        f_out.writelines(f_in)
        f_out.close()
    
if __name__ == '__main__':
    main()

