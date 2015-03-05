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

    # set mu and sigma if they exist to non-values
    run_settings['preprocessing']['normalise']['mu'] = 0.0
    run_settings['preprocessing']['normalise']['sigma'] = 1.0

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
    N = len(train.dataset.X)
    while N%batch_size != 0:
        batch_size += 1
    if verbose:
        print("Chosen {0} batch size.".format(batch_size))

    # get an iterator
    iterator = train.dataset.iterator(batch_size=batch_size)

    # iterate over an epoch, calculating the mean and variance
    # (actually calculate a bunch of means and variances, but
    # it's _approximately_ correct)
    variances = []
    mus = []
    invNpixels = None
    if verbose:
        i = 1
        print("Processing batches:")
    for batch,y in iterator:
        mus.append(np.mean(batch)) 
        variances.append(np.var(batch))
        if verbose:
            print("    Batch {0} of {1}: mean {2}"
                " variance {3}".format(i,iterator.num_batches,mus[-1],variances[-1]))
            i += 1
    variance = np.mean(variances)
    # have to enforce floats
    sigma = float(np.sqrt(variance))
    mu = float(np.mean(mus))

    if verbose:
        print("Pixels have mu {0} and sigma {1}.".format(mu,sigma))
        print("Writing results to json.")
    # write the results back into the settings file
    if 'normalise' in run_settings['preprocessing']:
        if run_settings.get('global_or_pixel','global') == 'pixel':
            raise NotImplementedError("No significant gains seen from" 
                    " pixelwise so not implemented.")
        else:
            run_settings['preprocessing']['normalise']['global_or_pixel'] = 'global'
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
