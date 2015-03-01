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

def main(run_settings_path, verbose=False):
    # this should just run either function depending on the run settings
    settings = utils.Settings('settings.json')
    # test script won't overwrite the pickle, so always force load
    run_settings = utils.load_run_settings(run_settings_path, 
            settings,
            settings_path='settings.json',
            force=True)
    # HELLO BOILERPLATE
    if run_settings['model type'] == 'sklearn':
        test_sklearn(run_settings, verbose=verbose)
    elif run_settings['model type'] == 'pylearn2':
        #train_pylearn2(run_settings)
        test_pylearn2(run_settings, verbose=verbose)
    else:
        raise NotImplementedError("Unsupported model type.")


def test_sklearn(run_settings, verbose=False):
    # some more boilerplate here
    # unpack settings
    settings = run_settings['settings']

    # get all training file paths and class names
    image_fname_dict = settings.image_fnames

    # parsed from json, preproc settings are dict
    augment_settings = run_settings["preprocessing"]
    processing = augment.augmentation_wrapper(**augment_settings)

    image_fname_dict = settings.image_fnames

    X, names = utils.load_data(image_fname_dict, processing=processing,
                               verbose=verbose)
    
    # load the model from where it's _expected_ to be saved
    clf = joblib.load(run_settings['pickle abspath'])
    p = clf.predict_proba(X)
   
    utils.write_predictions(run_settings['submissions abspath'], p, 
            names, settings.classes)

def test_pylearn2(run_settings, batch_size=4075, verbose=False):
    # Based on the script found at:
    #   https://github.com/zygmuntz/pylearn2-practice/blob/master/predict.py
  
    import pylearn2.utils
    import pylearn2.config
    import neukrill_net.dense_dataset
    import theano

    # unpack settings
    settings = run_settings['settings']

    # first load the model
    if verbose:
        print("Loading model...")
    model = pylearn2.utils.serial.load(run_settings['pickle abspath'])

    # then load the dataset
    if verbose:
        print("Loading data...")
    dataset = neukrill_net.dense_dataset.DensePNGDataset(
            settings_path=run_settings['settings_path'],
            run_settings=run_settings['run_settings_path'],
            train_or_predict='test')
    
    # then set batches:
    model.set_batch_size(batch_size)
            
    # then check batch size
    N_images = dataset.X.shape[0]
    # see how much extra we're going to have
    extra = batch_size - N_images%batch_size
    # check that worked
    assert (N_images+extra)%batch_size == 0

    # if we have extra, then we're going to have to pad with zeros
    # this might be a problem, with the massive array we're going 
    # to end up with
    if extra > 0:
        if verbose:
            print("Extra detected, padding dataset with"
                    " zeros for batch processing")
        dataset.X = np.concatenate((dataset.X, 
            np.zeros((extra, dataset.X.shape[1]), dtype=dataset.X.dtype)), 
            axis=0)

        #then check that worked:
        assert dataset.X.shape[0]%batch_size == 0

    # make a function to perform the forward propagation in our network
    if verbose:
        print("Compiling Theano function...")
    X = model.get_input_space().make_batch_theano()
    Y = model.fprop(X)
    f = theano.function([X],Y)

    # initialise our results array
    y = np.zeros((N_images, len(settings.classes)))
    # didn't want to use xrange explicitly
    n_batches = int(dataset.X.shape[0]/batch_size)
    for i in range(n_batches):
        if verbose:
            print("Processing batch {0} of {1}".format(i+1,n_batches))
        # grab a row
        x_arg = dataset.X[i*batch_size:(i+1)*batch_size,:]
        # check if we're dealing with images:
        if X.ndim > 2:
            # if so redefine to topological view
            x_arg = dataset.get_topological_view(x_arg)
        # and append the resulting value to y
        # don't understand why I have to transpose here, but I do
        y[i*batch_size:(i+1)*batch_size,:] = (f(x_arg.astype(X.dtype).T))

    # stupidest solution to augmentation problem
    # just collapse adjacent predictions until we
    # have the right number
    if len(dataset.names) < y.shape[0]:
        y_collapsed = np.zeros((len(dataset.names),y.shape[1]))
        augmentation_factor = int(y.shape[0]/len(dataset.names))
        # collapse every <augmentation_factor> predictions by averaging
        for i,(low,high) in enumerate(zip(range(0,
                        y.shape[0]-augmentation_factor, augmentation_factor),
                        range(augmentation_factor,y.shape[0],
                                                    augmentation_factor))):
            # confused yet?
            pdb.set_trace()
            # slice from low to high and take average down columns
            y_collapsed[i,:] = np.mean(y[low:high,:], axis=0)
        y = y_collapsed

    # then write our results to csv 
    if verbose:
        print("Writing csv")
    utils.write_predictions(run_settings['submissions abspath'], y, 
            dataset.names, settings.classes)

if __name__ == '__main__':
    # copied code from train.py here instead of making a function
    # because this code may diverge
    # need to argparse for run settings path
    parser = argparse.ArgumentParser(description='Train a model and store a '
                                                 'pickled model file.')
    # nargs='?' will look for a single argument but failover to default
    parser.add_argument('run_settings', metavar='run_settings', type=str, 
            nargs='?', default=os.path.join("run_settings","default.json"),
            help="Path to run settings json file.")
    # add verbose option
    parser.add_argument('-v', action="store_true", help="Run verbose.")
    args = parser.parse_args()
    main(args.run_settings, verbose=args.v)
