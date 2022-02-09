import os
import urllib.request
import tarfile
import shutil

def download(raw_data_dir):
    url = 'http://download.tensorflow.org/data/speech_commands_v0.02.tar.gz'
    save_dir = raw_data_dir
    tarname = save_dir + '/google_speech_commands_v2.tar.gz'
    ## download zipped data
    urllib.request.urlretrieve(url, tarname)
    ## unzip:
    tar = tarfile.open(tarname, "r:gz")
    tar.extractall(save_dir)
    tar.close()

def move_files(original_fold, data_fold, data_filename):
    with open(data_filename) as f:
        for line in f.readlines():
            vals = line.split('/')
            dest_fold = os.path.join(data_fold, vals[0])
            if not os.path.exists(dest_fold):
                os.mkdir(dest_fold)
            shutil.move(os.path.join(original_fold, line[:-1]), os.path.join(data_fold, line[:-1]))

def create_train_fold(original_fold, data_fold, test_fold):
    # list dirs
    dir_names = list()
    for file in os.listdir(original_fold):
        if os.path.isdir(os.path.join(original_fold, file)):
            dir_names.append(file)

    # build train fold
    for file in os.listdir(original_fold):
        if os.path.isdir(os.path.join(original_fold, file)) and file in dir_names:
            shutil.move(os.path.join(original_fold, file), os.path.join(data_fold, file))

def make_dataset(gcommands_fold, out_path):
    validation_path = os.path.join(gcommands_fold, 'validation_list.txt')
    test_path = os.path.join(gcommands_fold, 'testing_list.txt')

    valid_fold = os.path.join(out_path, 'valid')
    test_fold = os.path.join(out_path, 'test')
    train_fold = os.path.join(out_path, 'train')

    if not os.path.exists(out_path):
        os.mkdir(out_path)
    if not os.path.exists(valid_fold):
        os.mkdir(valid_fold)
    if not os.path.exists(test_fold):
        os.mkdir(test_fold)
    if not os.path.exists(train_fold):
        os.mkdir(train_fold)

    move_files(gcommands_fold, test_fold, test_path)
    move_files(gcommands_fold, valid_fold, validation_path)
    create_train_fold(gcommands_fold, train_fold, test_fold)


def google12_v2():
    data_dir = os.getcwd()  + '/../../data/google_speech_command'
    if not os.path.isdir(data_dir):
        os.mkdir(data_dir)

    raw_data_dir = data_dir + '/raw'
    if os.path.isdir(raw_data_dir):
        print("Data already downloaded")
    else:
        print("Downloading data")
        os.mkdir(raw_data_dir)
        download(raw_data_dir)
        print("Data download finished")

    processed_data_dir = data_dir + '/processed'
    if os.path.isdir(processed_data_dir):
        print("Data already processed")
    else:
        os.mkdir(processed_data_dir)
        make_dataset(raw_data_dir, processed_data_dir)
        print("Finished processing data")
