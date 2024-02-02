import os
from typing import List

import torch
from clearml.automation.controller import PipelineDecorator
from clearml import TaskTypes
import numpy as np


def download_data(url:str, filename:str)->str:
    print("Downloading data from {}".format(url))
    return filename

def unpack(filename: str)->str:
    print("Extracting data from {}".format(filename))
    return filename

@PipelineDecorator.component(return_values=["parsed", "filename", "category"], cache=True, task_type=TaskTypes.data_processing, repo="https://github.com/FeU-aKlos/minimal-clearml-sample.git", repo_branch="pipeline", packages=["numpy", "torch", "clearml"])
def extract(url:str, filename:str)->np.ndarray:
    """
    Transform the data from ubyte format to tensor format.

    Args:
        filename (str): The path to the ubyte file.

    Returns:
        str: The filename of the stored tensor data.
    """
    import numpy as np
    from utils import some_function
    
    filename = some_function(filename)
    
    return np.random.rand(10, 10), filename, "images"

@PipelineDecorator.pipeline(name="custom pipeline logic v2", project="mnist-pipeline-project", version="0.0.1", repo="https://github.com/FeU-aKlos/minimal-clearml-sample.git", repo_branch="pipeline")
def create_workflow(train_images_url, train_labels_url,test_images_url, test_labels_url):
    urls = [test_labels_url]
    fns = ["test-labels-ubyte.gz"]
    
    parsed_data, filenames, categories = zip(*list(map(extract, urls, fns)))
    
    print(parsed_data, filenames, categories)

