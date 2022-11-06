#!/bin/bash -e
rm -fr pythonenv __pycache__
# create new venv
python3 -m venv pythonenv
. pythonenv/bin/activate
python3 -m pip install --upgrade pip
pip install -r requirements.txt