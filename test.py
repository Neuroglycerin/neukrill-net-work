#!/usr/bin/env python

import neukrill_net.utils as utils
import neukrill_net.image_processing as image_processing
import neukrill_net.augment as augment

import csv
import pickle
from sklearn.externals import joblib
import numpy as np
import glob
import os

def main():
    settings = utils.Settings('settings.json')

    # this should be parsed from json, but hardcoded for now
    augment_settings = {'resize':(48,48)}
    processing = augment.augmentation_wrapper(augment_settings)

    image_fname_dict = settings.image_fnames

    X, names = utils.load_data(image_fname_dict, processing=processing,
                               verbose=True)

    clf = joblib.load('model.pkl')
    p = clf.predict_proba(X)

    with open('submission.csv', 'w') as csv_out:
        out_writer = csv.writer(csv_out, delimiter=',')
        out_writer.writerow(['image'] + list(settings.classes))
        for index in range(len(names)):
            out_writer.writerow([names[index]] + list(p[index,]))

if __name__ == '__main__':
    main()

