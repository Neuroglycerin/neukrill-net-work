#!/bin/bash
# execute it by typing 
# source start_script
# Courtesy of Krzysztof Geras
# Modified by Gavin Gray
# UI improved by Finlay Maguire

usage_1(){ echo "./start_script.sh GPU_CORES"; }

export gpu_number=$1

if [ -z "${gpu_number##*[!0-9]*}" ]
then
    usage_1
    return 1
fi

hostname=`hostname`

if [ $hostname = "stonesoup.inf.ed.ac.uk" ]; then
    echo "detected stonesoup, applying config"
	export LD_LIBRARY_PATH="/opt/cuda-5.0.35/lib:/opt/cuda-5.0.35/lib64"
	export CUDA_ROOT="/opt/cuda-5.0.35"
	export THEANO_FLAGS="device=gpu`echo -n $gpu_number`,floatX=float32,base_compiledir=~/.theano/stonesoup`echo -n $gpu_number`"
else
    echo "not stonesoup, applying default config"
	export THEANO_FLAGS="device=cpu,floatX=float32,base_compiledir=~/.theano/`echo -n $hostname`/`mktemp -u tmp.XXXXXXXXXXXXXXXXXXXXXXXXXXXXXX`"
fi

# undef gpu_number
unset gpu_number
