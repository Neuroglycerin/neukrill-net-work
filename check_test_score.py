#!/usr/bin/env python
######################
## Script to check  ##
## the score of a   ##
## Pylearn2 pickle. ##
## Based on the     ##
## Holdout notebook.##
######################

import numpy as np
import pylearn2.utils
import pylearn2.config
import theano
import neukrill_net.dense_dataset
import neukrill_net.utils
import sklearn.metrics
import argparse
import os
import pylearn2.config.yaml_parse

def make_predictions(run_settings_path, verbose=False, augment=1):
    """
    Single function, calculates score, prints and
    returns it.
    """
    # load settings
    if verbose:
        print("Loading settings..")
    settings = neukrill_net.utils.Settings("settings.json")
    run_settings = neukrill_net.utils.load_run_settings(run_settings_path, 
            settings, force=True)

    # if augment not specified on cmdline may be specified in run_settings
    if augment == 1:
        augment = run_settings.get("augment",1)

    # load the model
    if verbose:
        print("Loading model...")
    model = pylearn2.utils.serial.load(run_settings['pickle abspath'])

    # load proxied YAML
    # format the YAML
    yaml_string = neukrill_net.utils.format_yaml(run_settings, settings)
    # load proxied objects
    proxied = pylearn2.config.yaml_parse.load(yaml_string, instantiate=False)

    # load the dataset
    dataset = neukrill_net.utils.dataset_from_yaml(proxied,
        training_set_mode='test', verbose=verbose)

    # find a good batch size 
    if verbose:
        print("Finding batch size...")
    if hasattr(dataset.X, 'shape'):
        N_examples = dataset.X.shape[0]
    else:
        N_examples = len(dataset.X)
    batch_size = 500
    while N_examples%batch_size != 0:
        batch_size += 1
    n_batches = int(N_examples/batch_size)
    n_classes = len(settings.classes)
    if verbose:
        print("    chosen batch size {0}"
                " for {1} batches per pas".format(batch_size,n_batches))

    # compiling theano forward propagation
    if verbose:
        print("Compiling forward prop...")
    model.set_batch_size(batch_size)
    X = model.get_input_space().make_batch_theano()
    Y = model.fprop(X)
    if type(X) == tuple:
        f = theano.function(X,Y)
    else:
        f = theano.function([X],Y)

    # compute probabilities
    if verbose:
        print("Making predictions...")
    y = np.zeros((N_examples*augment,n_classes))
    # get the data specs from the cost function using the model
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
                y[i*batch_size:(i+1)*batch_size,:] = f(batch[0],batch[1])[:,:n_classes]
            else:
                y[i*batch_size:(i+1)*batch_size,:] = f(batch[0])[:,:n_classes]
            i += 1

    # find augmentation factor
    af = run_settings.get("augmentation_factor",1)
    if af > 1:
        if verbose:
            print("Collapsing predictions...")
        y_collapsed = np.zeros((int(N_examples/af), n_classes)) 
        for i,(low,high) in enumerate(zip(range(0,dataset.y.shape[0],af),
                                    range(af,dataset.y.shape[0]+af,af))):
            y_collapsed[i,:] = np.mean(y[low:high,:], axis=0)
        y = y_collapsed
        # and collapse labels
        labels = dataset.y[range(0,dataset.y.shape[0],af)]
    elif augment > 1:
        y_collapsed = np.zeros((N_examples,n_classes))
        # different kind of augmentation, has to be collapsed differently
        for row in range(N_examples):
            y_collapsed[row,:] = np.mean(np.vstack([r for r in 
                y[[i for i in range(row,N_examples*augment,N_examples)],:]]), 
                axis=0)
        y = y_collapsed            
        labels = dataset.y
    else:
        labels = dataset.y
    # if these labels happen to have superclasses in them we better take them out
    labels = labels[:,:n_classes]

    return y,labels

def check_score(labels, run_settings, y_arrays, verbose=False):
    """
    Takes one or more arrays of predictions, and an array of the true
    labels and calculates the log loss.
    """
    if len(y_arrays) > 1:
        # if verbose, first find log loss of each model individually
        for y,fpath in zip(y_arrays,run_settings):
            logloss = sklearn.metrics.log_loss(labels,y)
            print("{1} log loss: {0}".format(logloss,fpath))

        # average the predictions from each model
        predictions = np.mean(np.array(y_arrays),axis=0)

        if verbose:
            print("Combined:")
        # calculate score
        logloss = sklearn.metrics.log_loss(labels,predictions)
        print("Log loss: {0}".format(logloss))
    else:
        # calculate score
        logloss = sklearn.metrics.log_loss(labels,y_arrays[0])
        print("Log loss: {0}".format(logloss))
    return logloss

if __name__=='__main__':
    parser = argparse.ArgumentParser(description='Check the log loss score'
                                                 'on the holdout test set.')
    parser.add_argument('run_settings', metavar='run_settings', type=str, 
        nargs='+', default=os.path.join("run_settings","alexnet_based.json"),
        help="Path to run settings json file.")
    # add verbose option
    parser.add_argument('-v', action="store_true", help="Run verbose.")
    parser.add_argument('--augment', nargs='?', help='For online augmented '
                'models only. Will increase the number of times the script '
                'repeats predictions. Do not use this if you are running this'
                ' with more than one json and one of the jsons does not'
                ' support online augmentation. In that case, specify an '
                '"augment" variable in the run_settings json for that '
                'specific model.', type=int, default=1)
    args = parser.parse_args()
    predictions = []
    for run_settings in args.run_settings:
        if args.v:
            print("Running {0} predictions.".format(run_settings))
        y,labels = make_predictions(run_settings, verbose=args.v, augment=args.augment)
        predictions.append(y)
    check_score(labels, args.run_settings, predictions, verbose=args.v)
