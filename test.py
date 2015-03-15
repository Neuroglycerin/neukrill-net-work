#!/usr/bin/env python

import neukrill_net.utils as utils
import neukrill_net.image_processing as image_processing
import neukrill_net.augment as augment
import neukrill_net.dense_dataset

import csv
import pickle
from sklearn.externals import joblib
import numpy as np
import glob
import os
import argparse
import six.moves

import pylearn2.utils
import pylearn2.config
import theano

def main(run_settings_path, verbose=False, altdata=None, augment=1, split=1):
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
                augment=augment, split=split)
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
        augment=1, altdata=None, split=1):
    # Based on the script found at:
    #   https://github.com/zygmuntz/pylearn2-practice/blob/master/predict.py
  


    # unpack settings
    settings = run_settings['settings']

    # first load the model
    if verbose:
        print("Loading model...")
    model = pylearn2.utils.serial.load(run_settings['pickle abspath'])

    # load the dataset, but split it if required.
    predictions = []
    for block in range(split):
        if verbose and split == 1:
            print("Loading data...")
        elif verbose:
            print("Loading data, split {0} of {1}...".format(block+1,split))
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
            proxdata.keywords['split'] = (block,split)
            # then instantiate the dataset
            dataset = pylearn2.config.yaml_parse._instantiate(proxdata)

        # make predictions
        predictions.append(make_predictions(model, dataset, settings, 
            run_settings, proxied, batch_size=100, verbose=verbose, 
            augment=augment, altdata=altdata))

    with open("/disk/scratch/neuroglycerin/dump/test.py.pkl", "wb") as f:
        import pickle
        pickle.dump(predictions, f)

    # stack predictions
    predictions = np.vstack(predictions)

    # get the names
    names = [os.path.basename(fpath) for fpath in settings.image_fnames['test']]

    # then write our results to csv 
    if verbose:
        print("Writing csv")
    utils.write_predictions(run_settings['submissions abspath'], predictions, 
            dataset.names, settings.classes)

def make_predictions(model, dataset, settings, run_settings, proxied,
                batch_size=100, verbose=False, augment=1, altdata=None):
    

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
    n_classes = len(settings.classes)
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
    y = np.zeros((N_examples*augment, n_classes))
    # didn't want to use xrange explicitly
    pcost = proxied.keywords['algorithm'].keywords['cost']
    cost = pylearn2.config.yaml_parse._instantiate(pcost)
    data_specs = cost.get_data_specs(model)
    data_specs = (data_specs[0].components[0],data_specs[1][0])
    i = 0 
    for _ in range(augment):
        # make sequential iterator
        iterator = dataset.iterator(batch_size=batch_size,num_batches=n_batches,
                            mode='even_sequential', data_specs=data_specs)
        for batch in iterator:
            if verbose:
                print("    Batch {0} of {1}".format(i+1,n_batches*augment))
            if type(X) == tuple:
                y[i*batch_size:(i+1)*batch_size,:] = f(batch[0],
                                                       batch[1])[:,:n_classes]
            else:
                y[i*batch_size:(i+1)*batch_size,:] = f(batch)[:,:n_classes]
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
        y_collapsed = np.zeros((N_examples,n_classes))
        # different kind of augmentation, has to be collapsed differently
        for row in range(N_examples):
            y_collapsed[row,:] = np.mean(np.vstack([r for r in 
                y[[i for i in range(row,N_examples*augment,N_examples)],:]]), 
                axis=0)
        y = y_collapsed            
    return y


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
    parser.add_argument('--split', nargs='?', help='Factor to split test set, '
            'to reduce problems with memory being filled when loading large '
            'augmented test sets.', type=int, default=1)
    args = parser.parse_args()
    main(args.run_settings, verbose=args.v, altdata=args.altdata, 
            augment=args.augment, split=args.split)
