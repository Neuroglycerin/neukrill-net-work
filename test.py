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

def main(run_settings_path, verbose=False, altdata=None, augment=1):
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
        test_pylearn2(run_settings, verbose=verbose,altdata=altdata,
                augment=augment)
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

def test_pylearn2(run_settings, batch_size=4075, verbose=False, 
        augment=1, altdata=None):
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
    if altdata:
        if verbose:
            print("   Loading alternative data file from {0}".format(altdata))
        dataset = neukrill_net.dense_dataset.DensePNGDataset(
            settings_path=run_settings['settings_path'],
            run_settings=altdata,
            train_or_predict='test')
    else:
        # format the YAML
        yaml_string = neukrill_net.utils.format_yaml(run_settings, settings)
        # load proxied objects
        proxied = pylearn2.config.yaml_parse.load(yaml_string, instantiate=False)
        # pull out proxied dataset
        proxdata = proxied.keywords['dataset']
        # force loading of dataset and switch to test dataset
        proxdata.keywords['force'] = True
        proxdata.keywords['train_or_predict'] = 'test'
        proxdata.keywords['verbose'] = verbose
        # then instantiate the dataset
        dataset = pylearn2.config.yaml_parse._instantiate(proxdata)
    
    # find a good batch size 
    if verbose:
        print("Finding batch size...")
    if hasattr(dataset.X, 'shape'):
        N_examples = dataset.X.shape[0]
    else:
        N_examples = len(dataset.X)
    batch_size = batch_size
    while N_examples%batch_size != 0:
        batch_size += 1
    n_batches = int(N_examples/batch_size)
    if verbose:
        print("    chosen batch size {0}"
                " for {1} batches".format(batch_size,n_batches))

    #then check that worked (paranoid)
    assert N_examples%batch_size == 0

    # make a function to perform the forward propagation in our network
    if verbose:
        print("Compiling Theano function...")
    X = model.get_input_space().make_batch_theano()
    Y = model.fprop(X)
    if type(X) == tuple:
        f = theano.function(X,Y)
    else:
        f = theano.function([X],Y)

    if verbose:
        print("Making predictions...")
    # initialise our results array
    y = np.zeros((N_examples*augment, len(settings.classes)))
    # didn't want to use xrange explicitly
    pcost = proxied.keywords['algorithm'].keywords['cost']
    cost = pylearn2.config.yaml_parse._instantiate(pcost)
    data_specs = cost.get_data_specs(model)
    i = 0 
    for _ in range(augment):
        # make sequential iterator
        iterator = dataset.iterator(batch_size=batch_size,num_batches=n_batches,
                            mode='even_sequential', data_specs=data_specs)
        for batch in iterator:
            if verbose:
                print("    Batch {0} of {1}".format(i+1,n_batches*augment))
            if type(X) == tuple:
                y[i*batch_size:(i+1)*batch_size,:] = f(batch[0],batch[1])
            else:
                y[i*batch_size:(i+1)*batch_size,:] = f(batch[0])
            i += 1

    # stupidest solution to augmentation problem
    # just collapse adjacent predictions until we
    # have the right number
    af = run_settings.get("augmentation_factor",1)
    if af > 1:
        y_collapsed = np.zeros((len(dataset.names),y.shape[1]))
        augmentation_factor = int(y.shape[0]/len(dataset.names))
        # collapse every <augmentation_factor> predictions by averaging
        for i,(low,high) in enumerate(zip(range(0,
                        y.shape[0], augmentation_factor),
                    range(augmentation_factor,y.shape[0]+augmentation_factor,
                                                    augmentation_factor))):
            # confused yet?
            # slice from low to high and take average down columns
            y_collapsed[i,:] = np.mean(y[low:high,:], axis=0)
        y = y_collapsed
    elif augment > 1:
        y_collapsed = np.zeros((N_examples,len(settings.classes)))
        # different kind of augmentation, has to be collapsed differently
        for row in range(N_examples):
            y_collapsed[row,:] = np.mean(np.vstack([r for r in 
                y[[i for i in range(row,N_examples*augment,N_examples)],:]]), 
                axis=0)
        y = y_collapsed            
        labels = dataset.y
    else:
        labels = dataset.y

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
    # add option for alternative data loading (for models with very large augmentation
    parser.add_argument('--altdata', nargs='?', help="Load alternative dataset"
            ", useful if your model has too much augmentation."
            " Should be path to alternative run settings json.", default=None)
    parser.add_argument('--augment', nargs='?', help='For online augmented '
                'models only. Will increase the number of times the script '
                'repeats predictions.', type=int, default=1)
    args = parser.parse_args()
    main(args.run_settings, verbose=args.v, altdata=args.altdata, 
            augment=args.augment)
