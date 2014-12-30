This repository is supposed to contain working code, scripts, etc related to the Neuroglycerin entry for the NDSB competition.

Ensure that `neukrill-net-tools` is installed by doing one of the following (editable/develop install so you can edit the code without needing to reload everything):

        pip install -e .
        python setup.py develop

`train.py` - fit, cross-validate and dump a classifier model

`test.py` - read a pickled classifier model and output a submission csv covering the test_data
