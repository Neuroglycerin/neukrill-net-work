#!/usr/bin/env python

import neukrill_net.utils as utils

import csv
import os
import gzip

def main():
    out_fname = 'submission_priorprobs.csv'
    settings = utils.Settings('settings.json')
    
    names = [os.path.basename(fpath) for fpath in settings.image_fnames['test']]
    
    with open(out_fname, 'w') as csv_out:
        out_writer = csv.writer(csv_out, delimiter=',')
        out_writer.writerow(['image'] + list(settings.classes))
        for index in range(len(names)):
            out_writer.writerow([names[index]] + list(settings.class_priors))
    
    with open(out_fname, 'rb') as f_in:
        f_out = gzip.open(out_fname + '.gz', 'wb')
        f_out.writelines(f_in)
        f_out.close()
    
if __name__ == '__main__':
    main()

