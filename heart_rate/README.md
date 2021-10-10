# Heart-rate prediciton
## Data preparation
The data gets automatically downloaded and processed through *data.py*.
Note that all **Monash, UEA & UCR Time Series Regression** datasets have to be downloaded, although we only need the HR prediction data set, 
as only a full zip file (containing all data sets at once) is available online.

We process, however, only the heart-rate prediciton data set. But you are 
more than weclome to try also any of the other available data sets using LEM. 
To do so, simply change the **process_data** method in *data.py* accordingly.

If you don't want to use our automated data pipeline, 
you can download the zip file here: https://zenodo.org/record/3902651/files/Monash_UEA_UCR_Regression_Archive.zip?download=1
and process the *.ts* files in the **BIDMC32HR**
directory of the unzipped folder.

## Usage
To start the training with LEM, simply run:
```
python heart_rate_task.py [args]
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
