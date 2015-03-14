#!/usr/bin/env python

import argparse
import csv
import gzip
import neukrill_net.utils
import os

def main(csv_path, settings, weight = 0.01):

    csv_path = os.path.abspath(csv_path)
    csv_in = gzip.open(csv_path, 'rb')
    reader = csv.reader(csv_in, delimiter = ',')
    
    with gzip.open(csv_path.split('.')[0] + '_prior_weighted.csv.gz', 'w') as csv_out:
        writer = csv.writer(csv_out, delimiter = ',')
        
        # Write a header row
        header = reader.next()
        writer.writerow(header)
        
        for j, row in enumerate(reader):
            row_out = []
            # Write image name
            row_out.append(row[0])
        
            # Iterate over predictions
            for i in range(1, len(row)):
                prediction = ((1 - weight) * float(row[i])) + (weight * settings.class_priors[i-1])
                row_out.append(prediction)
 
            if (j % 1000 == 0):
                print j
        
            # Write row
            writer.writerow(row_out)
    
    csv_in.close()
        

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description = 'Generate a submission file'
                                                   'from an existing one with'
                                                   'predictions weighted by'
                                                   'class priors.')

    parser.add_argument('csv_path', type = str, nargs = '?', 
                        help = 'Path to csv file.')

    parser.add_argument('--weight', nargs = '?', help = 'Weight to weight the'
                        'prediction by the class prior.', type = float,
                        default = 0.01)
    args = parser.parse_args()
    settings = neukrill_net.utils.Settings("settings.json")
    main(args.csv_path, settings, weight = args.weight)
