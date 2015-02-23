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
import argparse
import six.moves

def main(run_settings_path):
    # this should just run either function depending on the run settings
    settings = utils.Settings('settings.json')
    run_settings = utils.load_run_settings(run_settings_path, 
            settings,
            settings_path='settings.json')
    # HELLO BOILERPLATE
    if run_settings['model type'] == 'sklearn':
        train_sklearn(run_settings)
    elif run_settings['model type'] == 'pylearn2':
        #train_pylearn2(run_settings)
        raise NotImplementedError("Unsupported model type.")
    else:
        raise NotImplementedError("Unsupported model type.")


def train_sklearn(run_settings, verbose=False):
    # some more boilerplate here
    # unpack settings
    settings = run_settings['settings']

    # get all training file paths and class names
    image_fname_dict = settings.image_fnames

    # parsed from json, preproc settings are dict
    augment_settings = run_settings["preprocessing"]
    processing = augment.augmentation_wrapper(augment_settings)

    image_fname_dict = settings.image_fnames

    X, names = utils.load_data(image_fname_dict, processing=processing,
                               verbose=verbose)
    
    # load the model from where it's _expected_ to be saved
    clf = joblib.load(run_settings['pickle abspath'])
    p = clf.predict_proba(X)
   
    utils.write_predictions(run_settings['submissions abspath'], p, names, settings)

def test_pylearn2(run_settings, batch_size=400, verbose=False):
    # Based on the script found at:
    #   https://github.com/zygmuntz/pylearn2-practice/blob/master/predict.py
  
    import pylearn2.utils
    import pylearn2.config
    import neukrill_net.dense_dataset
    import theano.tensor
    import theano.function

    # first load the model
    model = pylearn2.utils.serial.load(run_settings['pickle abspath'])

    # then load the dataset
    dataset = neukrill_net.dense_dataset.DensePNGDataset(
            settings_path=run_settings['settings_path'],
            run_settings=run_settings['run_settings_path'],
            train_or_predict='predict')
    
    # then set batches:
    model.set_batch_size(batch_size)
            
    # then check batch size
    m = dataset.X.shape[0]
    # see how much extra we're going to have
    extra = batch_size - m%batch_size
    # check that worked
    assert (m+extra)%batch_size == 0

    # if we have extra, then we're going to have to pad with zeros
    # this might be a problem, with the massive array we're going 
    # to end up with
    if extra > 0:
        dataset.X = np.concatenate((dataset.X, 
            np.zeros((extra, dataset.X.shape[1]), dtype=dataset.X.dtype)), 
            axis=0)

        #then check that worked:
        assert dataset.X.shape[0]%batch_size == 0

    import pdb
    # make a function to perform the forward propagation in our network
    X = model.get_input_space().make_batch_theano()
    Y = model.fprop(X)
    f = theano.function([X],Y)

    # didn't want to use xrange explicitly
    for i in six.moves.range(dataset.X.shape[0]/batch_size):
        # grab a row
        x_arg = dataset.X[i*batch_size:(i+1)*batch_size,:]
        # check if we're dealing with images:
        if X.ndim > 2:
            # if so redefine to topological view
            x_arg = dataset.get_topological_view(x_arg)
        # and append the resulting value to y
        y.append(f(x_arg.astype(X.dtype)))
        pdb.set_trace()

    


if __name__ == '__main__':
    # copied code from train.py here instead of making a function
    # because this code may diverge
    # need to argparse for run settings path
    parser = argparse.ArgumentParser(description='Train a model and store a'
                                                 'pickled model file.')
    # nargs='?' will look for a single argument but failover to default
    parser.add_argument('run_settings', metavar='run_settings', type=str, 
            nargs='?', default=os.path.join("run_settings","default.json"),
            help="Path to run settings json file.")
    args = parser.parse_args()
    main(args.run_settings)
