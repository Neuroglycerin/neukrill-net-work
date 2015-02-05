#!/bin/python
#
# crude script to train on mnist
# should be folded into train 
# once the run settings parser is done

import numpy as np
import neukrill_net.nk_mlp
import neukrill_net.utils as utils
import os

def main():

    # yeah, that ought to parse it
    settings = utils.Settings('settings.json')
   
    # loading in mnist 
    train_path = os.path.join(settings.data_dir,"mnist_train.npz")
    test_path =  os.path.join(settings.data_dir,"mnist_test.npz")
    train_npz = np.load(train_path)
    test_npz = np.load(test_path)
    
    # sticking it all together
    X = np.vstack([train_npz['arr_0'],test_npz['arr_0']])
    y = np.hstack([train_npz['arr_1'],test_npz['arr_1']])

    # Testing out the mlp function
    mlp = neukrill_net.nk_mlp.MLP_sk_interface(verbose=True)

    mlp.fit(X,y)

if __name__ == "__main__":
    main()
