This repository is supposed to contain working code, scripts, etc related to the Neuroglycerin entry for the NDSB competition.

Installing tools
================

Ensure that `neukrill-net-tools` is installed by doing one of the following (editable/develop install so you can edit the code without needing to reload everything):

        pip install -e .
        python setup.py develop

See the tools repo for more details.

Startup Script for Theano
=========================

It's required to set some environment variables for Theano. We have a script
for doing this (could in future ensure they're set in the Python code).
First, make sure it's executable:

```bash
chmod +x start_script
```

Then source it ([don't run it](http://askubuntu.com/questions/53177/bash-script-to-set-environment-variables-not-working)):

```bash
source start_script
```

Main Scripts
============

`train.py` - fit, cross-validate and dump a classifier model

`test.py` - read a pickled classifier model and output a submission csv covering the test_data

Settings
========

The settings are used to find the data, among other things.
When starting to use the scripts, add your data path to the `settings.json`.
Specifically this should be the directory where you unzipped the files,
containing the `train` and `test` directories.

_Alternatively_, you could put the data in the `data` directory in your
repository as this is already in the `settings.json`.
