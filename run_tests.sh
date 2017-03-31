#!/usr/bin/env bash

cd ${HOME}/workspace/
source venv/bin/activate
pip install -r requirements.txt --upgrade
py.test tests
