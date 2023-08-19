#!/bin/bash -e
rm -fr pythonenv __pycache__
# create new venv
python -m venv pythonenv
. pythonenv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
