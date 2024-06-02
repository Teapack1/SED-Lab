# src/utils/prep.py

import os


# create directory tree for the project
def create_dir_tree():
    os.makedirs("DATA/DATASET", exist_ok=True)
    os.makedirs("DATA/TEST_DATA", exist_ok=True)
    os.makedirs("DATA/external", exist_ok=True)
    os.makedirs("DATA/raw", exist_ok=True)
    os.makedirs("DATA/processed", exist_ok=True)
    os.makedirs("experiments/features", exist_ok=True)
    os.makedirs("experiments/models", exist_ok=True)
    os.makedirs("MODEL/PLOTS", exist_ok=True)
    os.makedirs("config", exist_ok=True)


create_dir_tree()
