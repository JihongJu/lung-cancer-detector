source venv/bin/activate
python -m ipykernel install --user --name=venv
jupyter notebook --ip 0.0.0.0 --port 8888 --no-browser
