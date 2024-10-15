# How to install:
1) create new envroment
2) pip install -r requiremts.txt
3) python setup.py develop

# How to use:

Type `python -m hankelimputation --file=` [name of the file with missing data]

Extra optional parameters:
 - `--batch=` [if you want to process in batches state here, must be integer, 0 is all at once]
 - `--e=` [specify how close the model is fitting to the data, must be float, 0 < e < 10 , default 0.1]
 - `--maxtime=` [how long before the algorithm time out]


# Test code:
pip install pytest
python -m pytest

note: expect warning of "Objective contains too many subexpressions"
