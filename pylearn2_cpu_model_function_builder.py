#!/usr/bin/env python
"""
Loads a pylearn2 model pickle, creates a new CPU compatible model and copies 
across original model parameters, compiles a theano function corresponding to a 
forward pass through the model and pickles this to a file. 
"""

import os
# force theano to use CPU
os.environ['THEANO_FLAGS']='device=cpu'
    
from pylearn2.utils import serial
import theano as th
import theano.tensor as tt
import cPickle as pickle
from argparse import ArgumentParser
from pylearn2.config import yaml_parse

def build_parser():
    parser = ArgumentParser(description=
        """Loads a pylearn2 model pickle, creates a new CPU compatible model
           and copies original model parameters, then compiles a theano function 
           corresponding to a forward pass through the model and pickles this
           to a file. 
        """)
    parser.add_argument('model_path', help='path to the model .pkl file')
    parser.add_argument('-s', '--yaml_spec_path', default=None,
                        help='path to the model spec YAML file')
    parser.add_argument('-o', '--out_dir', default=os.getcwd(),
                        help='path to output model prediction function pickle')            
    return parser

if __name__ == '__main__':
    parser = build_parser()
    args = parser.parse_args()
    print( 'Loading model at ' + args.model_path + '...' )
    model_gpu = serial.load( args.model_path )
    print( 'Model loaded. Creating new CPU runnable model from YAML...' )
    print('(current theano.config.device: {0})'.format(th.config.device))
    # hack to make model CPU compatible - 
    # construct new model from YAML spec and copy over parameters
    # https://groups.google.com/forum/#!topic/pylearn-users/7s_7lk4CyoI
    if args.yaml_spec_path is None:
        yaml_spec = model_gpu.yaml_src
    else:
        with open( args.yaml_spec_path, 'r' ) as f:
            yaml_spec = f.read() 
    model_cpu = yaml_parse.load( yaml_spec )
    model_cpu.set_param_values(model_gpu.get_param_values())
    print( 'Model converted.\nCompiling  function...' )
    X = model_cpu.get_input_space().make_theano_batch()
    Y = model_cpu.fprop( X )
    func = th.function( [X], Y)
    print('Function compiled. Pickling...')
    out_path = os.path.join( args.out_dir, 'model_function.pkl' )
    with open( out_path , 'w' ) as f:
        pickle.dump( func, f, protocol=pickle.HIGHEST_PROTOCOL )
    print( 'Pickled and saved to ' + out_path )
