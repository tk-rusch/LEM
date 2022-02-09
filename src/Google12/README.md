# Google 12 (Speech commands V2)
## Data preparation
The data gets automatically downloaded and split into train, test and validation 
sets through *data.py*.
Note that the download of the google speech command dataset v2 requires around 2.4 GB. 
So please make sure there is enough free space.

The correct grouping of the words into 12 targets 
gets done by **GCommandLoader** in *google12_data_loader.py*.
You are more than welcome to try also other groupings (i.e. google20, google35 and so on), 
by simply changing the **GSCmdV2Categs** dictionary in *google12_data_loader.py*

If you don't want to use our automated data pipeline, 
you can also download the data set with for instance *wget*:
```
wget http://download.tensorflow.org/data/speech_commands_v0.02.tar.gz
```
unpack it and split it according to the *testing_list.txt* and *validation_list.txt*, 
which can be found in the downloaded folder.

## Usage
To start the training with LEM, simply run:
```
python google12_task.py [args]
```

Options:
- nhid : hidden size
- epochs : max number of epochs
- device : computing device (GPU or CPU -- default: automatically chooses GPU if available)
- batch : batch size
- lr : learning rate
- drop : input dropout rate
- seed : random seed

The log of our best run can 
be found in the result directory.
