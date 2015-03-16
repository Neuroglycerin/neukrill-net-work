#!/bin/bash

usage="./model_backup.sh MODELFILE BACKUPDIR (run in dir of original file)"

if [ "$#" -ne 2 ]
then
    echo $usage;
    exit 1;
fi

original=$1
ver=1
backup=${2}/${1}_backup_${ver}

# if file exists increment version number
while [ -f "$backup" ]
do
    ver=$((ver+1))
    backup=${2}/${1}_backup_${ver}
done
# stops when unique version number reached

# do backup
cp $original $backup
