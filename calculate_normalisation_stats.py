#!/usr/bin/env python
##############################################
## Script to calculate the statistics (mean ##
## and variance) required to apply          ##
## normalisation to a dataset using online  ##
## augmentations.                           ##
##############################################

import numpy as np
import pylearn2.utils
import pylearn2.config
import theano
import neukrill_net.image_directory_dataset
import neukrill_net.utils as utils
import sklearn.metrics
import argparse
import os

def calculate_stats(run_settings_path, verbose=False):
    # load settings
    if verbose:
        print("Loading settings..")
    settings = neukrill_net.utils.Settings("settings.json")
    run_settings = neukrill_net.utils.load_run_settings(run_settings_path, 
            settings, force=True)

    # format the YAML file
    if verbose:
        print("Substituting YAML settings...")
    yaml_string = utils.format_yaml(run_settings, settings)

    # load the YAML
    if verbose:
        print("Loading YAML...")
    train = pylearn2.config.yaml_parse.load(yaml_string) 

    # pick a good batch size
    batch_size = 1000
    N = len(dataset.X)
    while N%batch_size != 0:
        batch_size += 1

    # get an iterator
    iterator = train.dataset.iterator(batch_size=batch_size)

    # iterate over an epoch, calculating the mean and variance
    mu = 0
    variances = []
    invN = 1./N
    for batch in iterator:
        mu += invN*np.sum(batch)
        variances.append(np.var(batch))
    variance = np.mean(variances)
    sigma = np.sqrt(variance)

    # write the results back into the settings file
    if 'normalise' in run_settings['preprocessing']:
        if run_settings['global_or_pixel'] == 'pixel':
            raise NotImplementedError("No significant gains seen from" 
                    " pixelwise so not implemented.")
        run_settings['preprocessing']['normalise']['mu'] = mu
        run_settings['preprocessing']['normalise']['sigma'] = sigma
    else:
        run_settings['preprocessing']['normalise'] = {}
        run_settings['preprocessing']['normalise']['global_or_pixel'] = 'global'
        run_settings['preprocessing']['normalise']['mu'] = mu
        run_settings['preprocessing']['normalise']['sigma'] = sigma

    # save to json
    utils.save_run_settings(run_settings)

if __name__=='__main__':
    parser = argparse.ArgumentParser(description='Check the log loss score'
                                                 'on the holdout test set.')
    parser.add_argument('run_settings', metavar='run_settings', type=str, 
        nargs='?', default=os.path.join("run_settings","alexnet_based.json"),
        help="Path to run settings json file.")
    # add verbose option
    parser.add_argument('-v', action="store_true", help="Run verbose.")
    args = parser.parse_args()
    calculate_stats(args.run_settings, verbose=args.v)
