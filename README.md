This repository is supposed to contain working code, scripts, etc related to the Neuroglycerin entry for the NDSB competition.

Installing tools
================

Ensure that `neukrill-net-tools` is installed by doing one of the following (editable/develop install so you can edit the code without needing to reload everything):

        pip install -e .
        python setup.py develop

See the tools repo for more details.

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
