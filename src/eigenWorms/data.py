from sktime.utils.load_data import load_from_arff_to_dataframe
import torch
from torch import Tensor
import numpy as np
import os
import urllib.response
import zipfile
from sklearn.preprocessing import LabelEncoder
from sklearn import model_selection
from torch.utils.data import TensorDataset

def download(raw_data_dir):
        url = 'http://www.timeseriesclassification.com/Downloads/EigenWorms.zip'
        save_dir = raw_data_dir
        zipname = save_dir + '/eigenWorms.zip'
        ## download zipped data
        urllib.request.urlretrieve(url, zipname)
        ## unzip:
        with zipfile.ZipFile(zipname, 'r') as zip:
            zip.extractall(save_dir)

## taken from https://github.com/jambo6/neuralRDEs/blob/master/get_data/uea.py and changed slightly
def create_torch_data(train_file, test_file):
    """Creates torch tensors for test and training from the UCR arff format.

    Args:
        train_file (str): The location of the training data arff file.
        test_file (str): The location of the testing data arff file.

    Returns:
        data_train, data_test, labels_train, labels_test: All as torch tensors.
    """
    # Get arff format
    train_data, train_labels = load_from_arff_to_dataframe(train_file)
    test_data, test_labels = load_from_arff_to_dataframe(test_file)

    def convert_data(data):
        # Expand the series to numpy
        data_expand = data.applymap(lambda x: x.values).values
        # Single array, then to tensor
        data_numpy = np.stack([np.vstack(x).T for x in data_expand])
        return data_numpy

    train_data, test_data = convert_data(train_data), convert_data(test_data)

    # Encode labels as often given as strings
    encoder = LabelEncoder().fit(train_labels)
    train_labels, test_labels = encoder.transform(train_labels), encoder.transform(test_labels)
    train_labels, test_labels = torch.Tensor(train_labels), torch.Tensor(test_labels)

    return train_data, test_data, train_labels, test_labels

def process_data(raw_data_dir, processed_data_dir):
    rnd_state = 1234
    train_arff = raw_data_dir + '/EigenWorms_TRAIN.arff'
    test_arff = raw_data_dir + '/EigenWorms_TEST.arff'
    trainx, testx, trainy, testy = create_torch_data(train_arff,test_arff)
    datax = np.vstack((trainx, testx))
    datay = np.hstack((trainy, testy))

    train_data, test_data, train_labels, test_labels = model_selection.train_test_split(datax, datay, test_size=0.3, random_state=rnd_state)
    valid_data, test_data, valid_labels, test_labels = model_selection.train_test_split(test_data, test_labels,
                                                                                        test_size=0.5, random_state=rnd_state)

    train_dataset = TensorDataset(Tensor(train_data).float(), Tensor(train_labels).long())
    torch.save(train_dataset, processed_data_dir + '/training.pt')
    test_dataset = TensorDataset(Tensor(test_data).float(), Tensor(test_labels).long())
    torch.save(test_dataset, processed_data_dir + '/test.pt')
    valid_dataset = TensorDataset(Tensor(valid_data).float(), Tensor(valid_labels).long())
    torch.save(valid_dataset, processed_data_dir + '/validation.pt')

def EigenWorms():
    data_dir = os.getcwd()  + '/../../data/eigenworms'
    if not os.path.isdir(data_dir):
        os.mkdir(data_dir)

    raw_data_dir = data_dir + '/raw'
    if os.path.isdir(raw_data_dir):
        print("Data already downloaded")
    else:
        os.mkdir(raw_data_dir)
        print("Downloading data")
        download(raw_data_dir)
        print("Data download finished")

    processed_data_dir = data_dir + '/processed'
    if os.path.isdir(processed_data_dir):
        print("Data already processed")
    else:
        os.mkdir(processed_data_dir)
        process_data(raw_data_dir, processed_data_dir)
        print("Finished processing data")

    train_dataset = torch.load(processed_data_dir + '/training.pt')
    test_dataset = torch.load(processed_data_dir + '/test.pt')
    valid_dataset = torch.load(processed_data_dir + '/validation.pt')

    return train_dataset, test_dataset, valid_dataset
