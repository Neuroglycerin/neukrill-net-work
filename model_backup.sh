#!/bin/bash

usage="./model_backup.sh MODELFILE BACKUPDIR" 

if [ "$#" -ne 2 ]
then
    echo $usage;
    exit 1;
fi

original_fullpath=$1
original_fn=`basename $1`
ver=1
backup=${2}/${original_fn}_backup_${ver}

# if file exists increment version number
while [ -f "$backup" ]
do
    ver=$((ver+1))
    backup=${2}/${original_fn}_backup_${ver}
done
# stops when unique version number reached

# do backup
cp $original_fullpath $backup
