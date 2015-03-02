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

def check_score(run_settings_path, verbose=False):
    """
    Single function, calculates score, prints and
    returns it.
    """
    # load settings
    if verbose:
        print("Loading settings..")
    settings = neukrill_net.utils.Settings("settings.json")
    run_settings = neukrill_net.utils.load_run_settings(run_settings_path)
    
    # load the model
    if verbose:
        print("Loading model...")
    model = pylearn2.utils.serial.load(run_settings['pickle abspath'])

    # load the data
    if verbose:
        print("Loading data...")
    dataset = neukrill_net.dense_dataset.DensePNGDataset(
            settings_path=run_settings['settings_path'],
            run_settings=run_settings['run_settings_path'],
            train_or_predict='train',
            training_set_mode='test', force=True, verbose=verbose)

    # find a good batch size 
    if verbose:
        print("Finding batch size...")
    batch_size = 500
    while dataset.X.shape[0]%batch_size != 0:
        batch_size += 1
    n_batches = int(dataset.X.shape[0]/batch_size)
    if verbose:
        print("    chosen batch size {0}"
                " for {1} batches".format(batch_size,n_batches))

    # compiling theano forward propagation
    if verbose:
        print("Compiling forward prop...")
    model.set_batch_size(batch_size)
    X = model.get_input_space().make_batch_theano()
    Y = model.fprop(X)
    f = theano.function([X],Y)

    # compute probabilities
    if verbose:
        print("Making predictions...")
    y = np.zeros((dataset.X.shape[0],len(settings.classes)))
    for i in xrange(n_batches):
        if verbose:
            print("    Batch {0} of {1}".format(i+1,n_batches))
        x_arg = dataset.X[i*batch_size:(i+1)*batch_size,:]
        if X.ndim > 2:
            x_arg = dataset.get_topological_view(x_arg)
        y[i*batch_size:(i+1)*batch_size,:] = (f(x_arg.astype(X.dtype).T))

    # find augmentation factor
    af = run_settings.get("augmentation_factor",1)
    if af > 1:
        if verbose:
            print("Collapsing predictions...")
        y_collapsed = np.zeros((int(dataset.X.shape[0]/af), len(settings.classes))) 
        for i,(low,high) in enumerate(zip(range(0,dataset.y.shape[0],af),
                                    range(af,dataset.y.shape[0]+af,af))):
            y_collapsed[i,:] = np.mean(y[low:high,:], axis=0)
        y = y_collapsed
        # and collapse labels
        labels = dataset.y[range(0,dataset.y.shape[0],af)]
    else:
        labels = dataset.y

    # calculate score
    logloss = sklearn.metrics.log_loss(labels,y)
    print("Log loss: {0}".format(logloss))

    return logloss

if __name__=='__main__':
    parser = argparse.ArgumentParser(description='Check the log loss score'
                                                 'on the holdout test set.')
    parser.add_argument('run_settings', metavar='run_settings', type=str, 
        nargs='?', default=os.path.join("run_settings","alexnet_based.json"),
        help="Path to run settings json file.")
    # add verbose option
    parser.add_argument('-v', action="store_true", help="Run verbose.")
    args = parser.parse_args()
    check_score(args.run_settings, verbose=args.v)
