import os
from pathlib import Path
import logging

logging.basicConfig(level=logging.INFO, format='[%(asctime)s]: %(message)s:')


project_name = "Rank-YT-Tutorials"

list_of_files = [
    ".env",
    "requirements.txt",
    "src/data_fetch.py",
    "src/preprocess.py",
    "src/classify.py",
    "src/aggregate.py",
    "src/visualize.py",
    "notebooks/"
]


for filepath in list_of_files:
    filepath = Path(filepath)
    filedir, filename = os.path.split(filepath)

    if filedir != "":
        os.makedirs(filedir, exist_ok=True)
        logging.info(f"Creating directory:{filedir} for the file {filename}")

    
    if (not os.path.exists(filepath)) or (os.path.getsize(filepath) == 0):
        with open(filepath,'w') as f:
            pass
            logging.info(f"Creating empty file: {filepath}")


    
    else:
        logging.info(f"{filename} is already exists")

