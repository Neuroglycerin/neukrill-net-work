#!/bin/bash
#execute it by typing 
#. ./start_script
# Courtesy of Krzysztof Geras
# Modified by Gavin Gray
# UI improved by Finlay Maguire

usage(){ echo "./start_script.sh -c GPU_CORES"; }
while getopts "hc:" OPTION; do
    case $OPTION in
        h) 
            usage
            exit 1
            ;;
        c) gpu_number=$OPTARG
            ;;
        *)
            usage
            exit 1
            ;;
    esac
done

if [ -z "${gpu_number##*[!0-9]*}" ]
then
    usage
    exit 1
fi

hostname=`hostname`

if [ $hostname = "stonesoup.inf.ed.ac.uk" ]; then
    echo "detected stonesoup, applying config"
	export LD_LIBRARY_PATH="/opt/cuda-5.0.35/lib;/opt/cuda-5.0.35/lib64"
	export CUDA_ROOT="/opt/cuda-5.0.35"
	export THEANO_FLAGS="device=gpu`echo -n $gpu_number`,floatX=float32,base_compiledir=~/.theano/stonesoup`echo -n $gpu_number`"
else
    echo "not stonesoup, applying default config"
	export THEANO_FLAGS="device=cpu,floatX=float32,base_compiledir=~/.theano/`echo -n $hostname`/`mktemp -u tmp.XXXXXXXXXXXXXXXXXXXXXXXXXXXXXX`"
fi
