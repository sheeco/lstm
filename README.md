
>*Created by Eco Sun on 2016-11-23, last updated on 2017-11-22.*


An LSTM network for trajectory predicting.


### Dependency

- python-2.7.12
- [Theano-0.9.0](c697eeab84e5b8a74908da654b66ec9eca4f1291)
- [*Forked* Lasagne](4cd90af6f318caf2b883a26b72feb87383a0c695)

### Platform

- Windows 10, 64 bit
- PyCharm Community Edition 2016.3


### Files

File | Description
---- | -----------
CHANGE.md | Change log by each commit.
default.config | Default execution configurations stored as text file.
test.py | Contains the actual execution entry.
lstm/\_\_init__.py | Init file for the `lstm` module.
lstm/config.py | Methods involving global configuration access. (Since it's imported in each of the other files, one **MUST NOT** import any of them in this file.)
lstm/utils.py | Utility methods including file operations, assertions, logging, interactions...
lstm/sampler.py | Class `Sampler` designed for trace sample reading, loading, saving and updating.
lstm/model.py | Class `SocialLSTM` that implemented the network.
res/trace/* | Trajectory dataset files should be put under this directory as resources. The path could be configured with keyword 'path_trace'.
log/* | Log files would be stored under this folder, identified by timestamp of each execution.


### Trajectory Format

e.g. 

> 1.4970000000000000e+004	 -1.2499853354219340e+004	 -6.5815908801037222e+003	
> 1.5000000000000000e+004	 -1.2500406313549865e+004	 -6.5819316462464440e+003	
> 1.5030000000000000e+004	 -1.2501138629672572e+004	 -6.5828068052055951e+003	

- Each sample consists of a triple of `time`, `x` & `y`, separated by only whitespace strings.
- `time`: an integer in terms of seconds, `x` and `y`: euclidean coordinates converted from 3-dim GPS coordinates, in terms of meters.
- Each file contains all the trace samples from a single node, one sample per line, with no comment or description lines before or after.
- Values of `time` does not have to start from zero, but do have to maintain a constant interval from the start to the end, without any missing or additional values in between.


### Command Line Usage

`python test.py [-c | --config] [-t | --tag] [-i | --import`

Short Opt | Long Opt | Value | Example
--------- | -------- | ----- | -------
`-c` | `--config` | A wrapped `dict` string of configurations to update | `-c "{'nodes': ['1', '2'], 'ask': False}"`
`-t` | `--tag` | A string to attach to execution timestamp | `-t debug`
`-i` | `--import` | A wrapped path string of the parameter pickling file to import | `-i "log\[debug]2017-11-21-16-52-29\pickle\params-init.pkl"` 


### Execution Log Files

Plain text files functioned as execution reports.

File | Description
---- | -----------
args.log | Backup for the command line arguments.
console.log | Backup for console output during execution.
(train/test)-hitrate.log | Hit rate percents (for multiple hit ranges) recorded by each epoch during training/testing. (the briefest level)
(train/test)-epoch.log | Loss and deviations recorded by each epoch.
(train/test)-batch.log | Loss and deviations recorded by each batch.
(train/test)-sample.log | Prediction results (for multiple nodes) recorded by each sample. (the most detailed level)
pickle/ | Pickled (binary) files for network parameters saved after each epoch. Could be used for parameter import.
compare/ | Detailed prediction targets & results recorded by each epoch, formatted matlab-styled for matrix import.

 