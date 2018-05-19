# ml_course_project
DSGA-1003 Project
# Instructions for running the project on the Azure server
We expect the following files to be runnable directly on the server:

data_preparation.py
model_training_cleaned.ipynb
data_and_model_analysis.ipynb

### Data Preparation

First step is to source the conda environment:

```
conda env create -f /home/dustingodevais/asylum_project_runnable_code/appeals_env.yml 
source activate appeals_env
```

Then run the command
```
python /home/dustingodevais/ml_course_project/processing_scripts/data_preparation.py
```
Note that this will create all the data prep result files at /home/dustingodevais/asylum_project_runnable_code/data_for_model

You should see the following files:

```
appeals_data_final.csv
appeals_data_final.dta
appeals_data_final.pkl
non_appeals_data_final.csv
non_appeals_data_final.dta
non_appeals_data_final.pkl
```

### Running the analysis
```
# To be run on the remote server at location /home/dustingodevais/ml_course_project/notebooks
jupyter notebook --no-browser --port={port of choice e.g. 8883} --ip=0.0.0.0

# To be run on your local machine to open the port to the jupyter notebook running on the server
ssh -N -f -L localhost:{local port of choice e.g. 8886}:localhost:{port chose on the remote server e.g. 8883} {your username}@52.174.199.138
```
Next open your browser on your local machine at the local host port your chose e.g. http://localhost:8886/



### Troubleshooting

If you pass the token to the notebook and are still unable to access it, then you need to update your jupyter config file to accept incoming requests from all ip addresses.

Open your jupyter notebook config file, typically located at /home/{your username}/.jupyter/jupyter_notebook_config.py

If it's not there, then run `jupyter notebook --generate-config` to create it.

```
c.NotebookApp.allow_origin = '*'
c.NotebookApp.ip = '0.0.0.0'
```

