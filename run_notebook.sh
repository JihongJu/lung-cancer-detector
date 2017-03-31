#!/usr/bin/env bash

cd ${HOME}/workspace
source venv/bin/activate
pip install -r requirements.txt --upgrade
python -m ipykernel install --user --name=venv
jupyter notebook --ip 0.0.0.0 --port 8888 --no-browser
