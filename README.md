# lstm

An LSTM-RNN network for trajectory predicting.

## Dependency

- python-2.7.12
- [Theano-0.9.0](https://github.com/Theano/Theano/commit/c697eeab84e5b8a74908da654b66ec9eca4f1291)
- [*Forked* Lasagne](https://github.com/sheeco/Lasagne/commit/404861b44d7849c88509117ff92e4ff6c18af8ed)

## Platform

- Windows 10, 64 bit
- IntelliJ IDEA 2017.2.6


## Files

File                  | Description
----                  | -----------
CHANGE.md             | Change log by each commit.
default.config        | Default execution configurations as a `dict` string (plain text).
test.py               | A demo script running default test.
config.py             | Methods involving configuration processing.
utils.py              | Utility methods including file operations, assertions, logging, interactions...
dump.py               | Methods for prediction results dumping & panning rule dumping.
lstm/\_\_init__.py    | Init file for the `lstm` module.
lstm/sampler.py       | Class `Sampler` designed for trace sample reading, loading, saving and updating.
lstm/model.py         | Class `SocialLSTM` that implemented the network.
res/trace/*           | Trajectory dataset files should be put under this directory as resources. The path could be configured with keyword 'path_trace'.
log/*                 | Log files would be generated under this folder, identified by timestamp of each execution.


## Usage

### Trajectory Format

e.g. 

> 1.4970000000000000e+004	 -1.2499853354219340e+004	 -6.5815908801037222e+003	
> 1.5000000000000000e+004	 -1.2500406313549865e+004	 -6.5819316462464440e+003	
> 1.5030000000000000e+004	 -1.2501138629672572e+004	 -6.5828068052055951e+003	

- Each sample consists of a triple of `time`, `x` & `y`, separated by only whitespace strings.
- `time`: An integer in terms of seconds;`x` and `y`: Euclidean coordinates converted from 3-dim GPS coordinates, in terms of meters.
- Each file contains all the trace samples from a single node, one sample per line, with no comment or description lines before or after.
- Values of `time` does not have to start from zero, but DO have to maintain a constant interval from the start to the end, without any missing or additional values in between. (e.g. 120, 150, 180 ...)


### Command Line Arguments

For trace prediction: (test.py)

`python -O test.py [-c | --config] [-t | --tag] [-i | --import]`

Short Opt   | Long Opt   | Value                                                          | Example
---------   | --------   | -----                                                          | -------
`-c`        | `--config` | A wrapped `dict` string of configurations to update            | `-c "{'nodes': ['1', '2'], 'ask': False}"`
`-t`        | `--tag`    | A string to attach to execution timestamp                      | `-t demo`
`-i`        | `--import` | A wrapped path string of the parameter pickling file to import | `-i "log\[debug]2017-11-21-16-52-29\pickle\params-init.pkl"` 

For prediction dumping: (dump.py)

`python -O dump.py prediction [-p | --path <path-log-folders>] [-f | --from <folder-name-log|identifier>] 
                              [-e | --epoch <iepoch>] [-n | --name <filename-dump>] [-t | --to <folder-name-dump>]`

Short Opt   | Long Opt   | Value                                                                | Example
---------   | --------   | -----                                                                | -------
`-p`        | `--path`   | Where is the `log` folder?                                           | `-p ./res/log`
`-f`        | `--from`   | Dump from which log folder? Provide folder name or timestamp string. | `-f 2018-02-01-20-45-14`
`-e`        | `--epoch`  | Dump predictions of which epoch?                                     | `-e 108`
`-n`        | `--name`   | Name of the output file?                                             | `-n node1`
`-t`        | `--to`     | Dump output files to which folder?                                   | `-t ./res/dump`

For panning rule dumping: (dump.py)

`python -O dump.py pan [-f | --from <log-folder|identifier>] [-n | --name <dump-filename>]`

Short Opt   | Long Opt   | Value                                                                | Example
---------   | --------   | -----                                                                | -------
`-f`        | `--from`   | Dump from which log folder? Provide folder name or timestamp string. | `-f 2018-02-01-20-45-14`
`-n`        | `--name`   | Name of the output file?                                             | `-n node1`


### Configurations

Configurations are pre-defined in `default.config` & `lstm/default.config`, with descriptions provided in the comments. 

e.g.
	{
		# Group name
		'default':
			{
				# Path for trace files
				'path_trace': {
					'value': 'res/trace/NCSU/',
					'tags': ['path']
				},
			},
		'debug':
			# Nano-size net config for debugging
			{
				# Length of observed sequence
				'length_sequence_input': {
					'value': 4,
					'tags': ['build']
				},
			},
		'run':
			{
				# Length of observed sequence
				'length_sequence_input': {
					'value': 4,
					'tags': ['build']
				}
			}
	}

- Configurations are defined into 3 groups. The group 'default' is loaded at first, then group 'debug'/'run', 
at last the command line configurations. For the same configuration key, a newly loaded value would override the older one.
- Each configuration consists of a configuration key ('path_trace'), a default value ('res/trace/NCSU/') and a list of tags (['path', 'build']). 
The tags are only used for validation ('path' is used for format validation, 'build' is designed for parameter import validation but is not actually used at this moment).
- Only the pre-defined configuration keys are allowed. To enable a new configuration, one must add a key entry and provide with a default value in this file first.
- Which of the 'debug'/'run' groups actually gets loaded depends on the `__debug__` switch in python, i.e. the `-O` switch in command line. 
(`python -O test.py` means running; `python test.py` means debugging.)
- Some configurations (e.g. 'length_sequence_input' above) get assigned a different value to enable debugging with a nano-size network for speed-up purpose. 
The others (e.g. 'path_trace' above) are irrelevant to debug/run, hence are defined in group 'default'.


### Execution Log Files

Plain text files functioned as execution reports, stored under `log/`, identified by execution timestamp.

File                         | Description
----                         | -----------
config.log                   | Backup for the configurations used during execution.
args.log                     | Backup for the command line arguments.
console.log                  | Backup for console output during execution.
(train/test)-hitrate.log     | Hit rate percent (for multiple hit ranges) recorded by each epoch during training/testing. (the briefest level)
(train/test)-epoch.log       | Loss and deviations recorded by each epoch.
(train/test)-batch.log       | Loss and deviations recorded by each batch.
(train/test)-sample.log      | Prediction results (for multiple nodes) recorded by each sample. (the most detailed level)
pickle/                      | Pickled (binary) files for network parameters saved after each epoch. Could be used for parameter import.
compare/                     | Detailed prediction targets & results recorded by each epoch, formatted matlab-styled for the convenience of matrix import.

Also, most of the logs provide a set of column descriptions at its very first line.


### Export & Import of Network Parameters

Parameter values of the LSTM network could be exported or re-imported. 
They are extracted from (and could be re-assigned to) the network using interfaces provided by theano, 
and are serialized to disk files using the python serialization library `pickle`.

#### Export

Network parameters get automatically exported to '.pkl' files under `log/pickle/`. The pickling results are binary files that MUST NOT be temper with.

e.g. Files under `log/pickle/`

> params-init.pkl
> params-epoch1.pkl
> params-epoch2.pkl
> params-epoch3.pkl
> params-best-hitrange100-hitrate0.300000-epoch2.pkl
> params-best-hitrange50-hitrate0.250000-epoch2.pkl

- The initial (randomly generated) parameters as `params-init.pkl`.
- The parameters after the `n`th epoch of training as `params-epoch$n$.pkl`.
- Also, the parameters with the best hitrate result get an extra copy with a detailed filename for convenience. And would get updated during further training.

#### Import

1. Find the desired parameter file: e.g. `log/[demo]2017-07-20-22-27-03/pickle/params-best-hitrange50-hitrate0.250000-epoch2.pkl`
2. Run with parameter importing: `python -O test.py --import 'log/[demo]2017-07-20-22-27-03/pickle/params-best-hitrange50-hitrate0.250000-epoch2.pkl' --tag 'import-from-demo'`
Then the given parameters would be treated as initial parameters, instead of randomly generated ones.

Since the parameters are dependent on the network structure, one must make sure the network building configurations (tagged with 'build') are consistent with the execution to import from. 
(e.g. Importing parameters from a network with 10 as sequence length and assigning them to a network with 4 as sequence length is certainly not going to work.)
