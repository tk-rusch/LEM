# EigenWorms
## Data preparation
The data gets automatically downloaded, processed and split into train, 
test and validation sets through *data.py*.

If you don't want to use our automated data pipeline, 
you can download the zip file here: http://www.timeseriesclassification.com/Downloads/EigenWorms.zip
and process the *.arff* files in the downloaded and unzipped folder.

## Usage
To start the training with LEM, simply run:
```
python eigenWorms_task.py [args]
```

Options:
- nhid : hidden size
- epochs : max number of epochs
- device : computing device (GPU or CPU -- default: automatically chooses GPU if available)
- batch : batch size
- lr : learning rate
- dt : parameter $Delta t$ of LEM
- seed : random seed

The log of our best run can 
be found in the result directory.
