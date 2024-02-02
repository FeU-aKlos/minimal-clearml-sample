import os
from typing import List

import torch
from clearml.automation.controller import PipelineDecorator
from clearml import TaskTypes
import numpy as np


def download_data(url:str, filename:str)->str:
    import urllib.request

    filename = os.path.basename(filename)
    urllib.request.urlretrieve(url, filename)
    print("Downloaded data to {}".format(filename))
    return filename

def unpack(filename: str)->str:
    """
    Extracts data from a gzipped file.

    Args:
        filename (str): The path to the gzipped file.

    Returns:
        str: The path to the extracted file.
    """
    import gzip
    import shutil

    print("Extracting data from {}".format(filename))
    output_filename = filename[:-3]

    if filename.endswith(".gz"):
        with gzip.open(filename, "rb") as f:
            with open(output_filename, "wb") as f_out:
                shutil.copyfileobj(f, f_out)
    print("Extracted data to {}".format(output_filename))
    return output_filename

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
    from utils import get_int
    
    filename = download_data(url, filename)
    filename = unpack(filename)

    parsed = None
    set = None
    category = None
    images_datatype = 2051
    labels_datatype = 2049
    nr_train_items = 60000
    nr_test_items = 10000

    if filename.endswith("ubyte"):  # FOR ALL "ubyte" FILES
        print("Reading ",filename)
        with open(filename, "rb") as f:
            data = f.read()
            dtype = get_int(data[:4])   # 0-3: THE MAGIC NUMBER TO WHETHER IMAGE OR LABEL
            length = get_int(data[4:8])  # 4-7: LENGTH OF THE ARRAY  (DIMENSION 0)

            if dtype == images_datatype:
                category = "images"
                num_rows = get_int(data[8:12])  # NUMBER OF ROWS  (DIMENSION 1)
                num_cols = get_int(data[12:16])  # NUMBER OF COLUMNS  (DIMENSION 2)
                buffer_offset = 16
                parsed = np.frombuffer(data, dtype=np.uint8, offset=buffer_offset)  # READ THE PIXEL VALUES AS INTEGERS
                parsed = parsed.reshape(length, num_rows, num_cols)  # RESHAPE THE ARRAY AS [NO_OF_SAMPLES x HEIGHT x WIDTH]
            elif dtype == labels_datatype:
                category = "labels"
                buffer_offset = 8
                parsed = np.frombuffer(data, dtype=np.uint8, offset=buffer_offset)  # READ THE LABEL VALUES AS INTEGERS
                parsed = parsed.reshape(length)  # RESHAPE THE ARRAY AS [NO_OF_SAMPLES]

            if length == nr_test_items:
                set = "test"
            elif length == nr_train_items:
                set = "train"
    else:
        raise TypeError("Not a valid file extension")

    if isinstance(parsed, type(None)) or isinstance(set, type(None)) or isinstance(category, type(None)):
        raise Exception("Something went wrong")    

    return parsed, filename, category

@PipelineDecorator.component(return_values=["data", "fn4storing"], parents=["extract"], cache=True, task_type=TaskTypes.data_processing, repo="https://github.com/FeU-aKlos/minimal-clearml-sample.git", repo_branch="pipeline", packages=["numpy", "torch", "clearml"])
def transform(data:np.ndarray, filename:str, category:str):
    import torch
    import numpy as np
    from utils import normalize

    fn4storing = filename + ".pt"
    if category == "images":
        data = normalize(data, axis=(1, 2))
        data = np.expand_dims(data, axis=1)
        data = torch.tensor(data, dtype=torch.float32)
    elif category == "labels":
        data = torch.tensor(data, dtype=torch.int64)
    print(data.shape)
    return data, fn4storing

@PipelineDecorator.component(return_values=["dataset.id"], parents=["transform"], cache=True, task_type=TaskTypes.data_processing, repo="https://github.com/FeU-aKlos/minimal-clearml-sample.git", repo_branch="pipeline", packages=["torch", "clearml"])
def store_in_dataset(datatensors:List[torch.Tensor], filenames:List[str], dataset_project:str, dataset_name:str="mnist"):
    from clearml import Dataset
    import os
    import torch
    
    dataset = Dataset.create(dataset_name=dataset_name, dataset_project=dataset_project)
    test_folder = os.path.join("/","tmp", "test")
    train_folder = os.path.join("/","tmp", "train")
    for data, filename in zip(datatensors, filenames):
        torch.save(data, os.path.join(test_folder if "test" in filename else train_folder,filename))

    
    dataset_paths_dict = {"train":train_folder, "test": test_folder}

    for key,path in dataset_paths_dict.items():
        dataset.add_files(path, dataset_path=key)
        dataset.upload()
        dataset.finalize()
    
    return dataset.id
    
@PipelineDecorator.component(return_values=["accuracy"], parents=["store_in_dataset"], cache=True, task_type=TaskTypes.training, repo="https://github.com/FeU-aKlos/minimal-clearml-sample.git", repo_branch="pipeline", packages=["torch"])
def train_model(dataset_id:str):
    import torch
    model = {}
    torch.save(model, "/tmp/model.pt")
    accuracy = 100
    return accuracy
    
@PipelineDecorator.component(return_values=["accuracy"], parents=["train_model"], cache=True, task_type=TaskTypes.testing, repo="https://github.com/FeU-aKlos/minimal-clearml-sample.git", repo_branch="pipeline")
def test_model(dataset_id:str):
    accuracy = 100
    return accuracy

@PipelineDecorator.pipeline(name="custom pipeline logic", project="mnist-pipeline-project", version="0.0.1", repo="https://github.com/FeU-aKlos/minimal-clearml-sample.git", repo_branch="pipeline")
def create_workflow(train_images_url, train_labels_url,test_images_url, test_labels_url):
    
    # urls = [train_images_url, train_labels_url,test_images_url, test_labels_url]
    # fns = ["/tmp/train-images-ubyte.gz", "/tmp/train-labels-ubyte.gz", "/tmp/test-images-ubyte.gz", "/tmp/test-labels-ubyte.gz"]
    urls = [test_labels_url]
    fns = ["test-labels-ubyte.gz"]
    
    parsed_data, filenames, categories = zip(*list(map(extract, urls, fns)))
    # #check if that works... i don't think so!
    # datasets, filenames = list(map(transform, parsed_data, filenames, categories, "mnist-pipeline-project"))
    
    # dataset_id = store_in_dataset(datasets, filenames, "mnist-pipeline-project")
    
    # train_model(dataset_id)
    
    # test_model(dataset_id)
