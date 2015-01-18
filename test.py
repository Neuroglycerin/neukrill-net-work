#!/usr/bin/env python

import neukrill_net.utils as utils
import neukrill_net.image_processing as image_processing

import csv
import pickle
from sklearn.externals import joblib
import numpy as np
import glob
import os

def main():
    settings = utils.Settings('settings.json')

    image_fname_dict = settings.image_fnames

    processing = lambda image: image_processing.resize_image(image, (48,48))

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

