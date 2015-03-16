#!/usr/bin/env python
"""
Script to combine predictions from different csvs to
slightly improve the log loss score. Will only work
if the models being combined all perform quite well.
"""

import os
import argparse
import neukrill_net.utils
import numpy as np
import json

def combine_csvs(csv_paths, verbose=False):
    """
    Takes two or more submission csv names and combines them by averaging the 
    predictions in each. Returns this array of combined predictions.
    """

    # load each csv as an array, filling empty 3d array
    predictions = np.zeros((len(csv_paths),130400,121))
    for i,cpath in enumerate(csv_paths):
        # unfortunately, have to check the csv name is correct
        if cpath.split(".")[-1] != "gz":
            cpath = cpath + ".gz"
        if verbose:
            print("Loading {0}...".format(cpath))
        # have to enforce str otherwise loadtxt breaks (doesn't like unicode)
        predictions[i,:] = np.loadtxt(str(cpath), skiprows=1, 
            usecols=range(1,122), delimiter=",")
    # average the arrays along the first axis
    if verbose:
        print("Averaging predictions...")
    predictions = np.mean(predictions, axis=0)
    return predictions

if __name__ == "__main__":
    # make a parser
    parser = argparse.ArgumentParser(description='Script to combine predictions'
            ' from different csvs to slightly improve the log loss score. Will'
            ' only work if the models being combined all perform quite well.')
    # run settings to combine
    parser.add_argument('run_settings', metavar='run_settings', type=str, 
            nargs='+', default=os.path.join("run_settings","default.json"),
            help="Paths to run settings json files.")
    # output file path
    parser.add_argument('-f', nargs='?', help='Output csv file name.', 
            type=str)
    # add verbose option
    parser.add_argument('-v', action="store_true", help="Run verbose.")
    args = parser.parse_args()
    # load the settings
    settings = neukrill_net.utils.Settings('settings.json')
    # get the csv paths
    csv_paths = []
    for rspath in args.run_settings:
        # load the run settings
        with open(rspath) as f:
            rs = json.load(f)
        csv_paths.append(rs["submissions abspath"])
    # pass the csv paths to the combiner to combine
    predictions = combine_csvs(csv_paths, verbose=args.v)
    # get the right filenames
    names = [os.path.basename(fpath) for fpath in settings.image_fnames['test']]
    if args.v:
        print("Writing new csv to {0}".format(args.f))
    # write the new csv to output file
    neukrill_net.utils.write_predictions(args.f, predictions, 
            names, settings.classes)
