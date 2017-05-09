#### Log Mark

*在 `CHANGE.md` 的更新日志中使用分类标记标签：*

　　 `ADD` 添加新功能、`REM` 删除现有功能、`MOD` 修改现有实现、`OPT` 功能优化、`TRY` 不确定尝试；

　　 `BUG` 待修复错误、`FIX` 修复错误、`TEST` 功能测试；

　　 `RFCT` 代码重构、`MNT` 版控维护、`NOTE` 附加说明、`TODO` 计划任务；


# Change Log

### 0.0.*


###### [2016-11-25](b35009fb246b93a56376a1122a12e5528d5cac19)

- ADD: 加载 batch 的函数 `load_batch_for_nodes`；
- RFCT: 重命名文件 `file_io.py` 为 `sample.py`，及一些函数重命名；

###### [2016-11-25](b9ab414351b1855919b3672874da2932c096d05c)

- MOD: 调整函数 `load_batch_for_nodes` 返回数组的维度次序为：节点->批->序列->采样维度；
- FIX: `sample.py` 中的输出错误；
- ADD: `config.py` 中的参数；

###### [2016-11-25](e06e548390787b7f073534d2f07affd7fdf18897)

- ADD: 修改自 [lstm_text_generation](https://github.com/Lasagne/Recipes/blob/master/examples/lstm_text_generation.py) 的 demo；
- [ ] BUG: 能够正常运行，但有时会得到 `NaN`；

###### [2016-12-01](e082892fc7e9d6d96adaa5f14cf0e6197d956e04)

- ADD: Allow `load_batch_for_nodes` to have either `int` or `list` passed in for param `filter_nodes`;
- ADD: Nano size configs for debugging; 
- ADD: Forwarding connections through the network are built & tested;
- [ ] TODO: While still having problems with the recurrent connection from previous LSTM hidden states to Social Pooling Layer;
- [ ] TRY: Whether `RecurrentContainerLayer` proposed in [#629](https://github.com/Lasagne/Lasagne/pull/629) would suffice;

###### [2016-12-02](09a53a25e789820e71f19145d92db3e9f15ec37f)

- ~~Fail to apply PR [stevenxxiu/Lasagne:recurrent](1d23b4022455ef8449b98c3805f2f0c919836f61) & 
  [skaae/Lasagne:merge_broadcast](8588929fba32b03e3df324b87fc451cc4b2ab225) via patch/git-format-patch approach.~~
  SHOULD use the fork-fetch-merge approach;
- OPT: Simplize module importing;

###### [2016-12-03](a3edd597c5fc475906d2f93a45f835e93c5bd23f)

- Merge PR [#629](https://github.com/Lasagne/Lasagne/issues/629) from [stevenxxiu/Lasagne:recurrent](1d23b4022455ef8449b98c3805f2f0c919836f61) & 
  [#633](https://github.com/Lasagne/Lasagne/issues/633) from [skaae/Lasagne:merge_broadcast](8588929fba32b03e3df324b87fc451cc4b2ab225) 
  into [my forked lasagne](https://github.com/sheeco/Lasagne) for dependency;
- MOD: Implement social pooling based on broadcastable `ElementwiseMergeLayer` (by skaae) instead of `ExpressionMergeLayer`;
- MOD: Have to alter the LSTM params sharing code for the new `LSTMCell` wrapper (by stevenxxiu);
- ADD: Try to use `RecurrentContainerLayer` to implement the recurrent connection from LSTM to S-Pooling;
- FIX: Theano compiling(?) problems in `social_mask` (probably undone);
- BUG: "Lack of input" reported by `lasagne.helper.get_output`. Haven't figured out the reason yet;

###### [2016-12-05](68aae71b8b1d7aacefee9bd6ec79882d960f5459)

- RFCT: Move helper funcitons into `utils.py`;
- ADD: `STRICT_BATCH` to indicate if only full-size batches are accepted, maybe mandatory to avoid indeterministic in `social_mask`;
- ADD: Try to define `SocialLSTMCell` herited from `CustomRecurrentCell` (untested);
- NOTE: Maybe should give up the `ReccurentContainerLayer` approach & overwrite `CustomRecurrentLayer` directly?

###### [2016-12-12](a7ac820e2778d45190204cdfd9349843859e5a1f)

- ADD: Instant sequence to the return values of `load_batch_for_nodes`;

###### [2016-12-12](2aa7ec08e92780cd1c7cb86c2fe549e0f217657b)

- TEST: Forwarding connections should be fine, only if passing in previous hidden states mannually;
- DEBUG: The overwriting in `SocialLSTMCell`, still "lack of input" for `layer_xy`. Likely has sth to do with line 291 & 551 in [recurrent.py](https://github.com/sheeco/Lasagne/blob/add565017d4636676028d16dcba1ec2b2870aa36/lasagne/layers/recurrent.py#L291);

###### [2016-12-22](b96a719c22d8ee8941e9c737b12a4597f031a30d)

- NOTE: Give up the previous cell&container approach from stevenxxiu;
- ADD: Try to define `RecurrentContainerLayer` by overwriting `CustomRecurrentLayer`;

##### Branch `shared_lstm`

###### [2017-03-13](4ff90b74528e4090fe2dd98f6460cbe3b7c24227)

- ADD: A simple model of shared LSTM, with dense decoder;
- TEST: Building & compiling should be fine;

###### [2017-03-14](8591dd9b5dcc7504806f030685bd0ec4b09f210d)

- OPT: for sample loading;

###### [2017-03-20](dd0b55bd7c2884eb8b0226890399ae9c7fe88f77)

- MOD: Dimension of network inputs modified to 3 to include the time column;
- ADD: NNL loss for training & Euclidean distance error for observation. 
- NOTE: Need new package [breze](https://github.com/breze-no-salt/breze) & consequently package [climin](https://github.com/BRML/climin), 
  for `pdf` (under `breze.arch.component.distributions.mvn`), the probability density function of multi-variate normal distribution;
- TEST: `compute_and_compile` should be fine for now;

###### [2017-03-21](8d138fdcc0785c27b2c97f8a91c36ea157ce2da6)

- MOD: Define probability density function of bivariate normal distribution directly. No longer need package breze;
- TRY: Different initialization metrics for weights & biases to avoid `NaN` or `inf` during computation;
- BUG: Values of sigma (standard deviations) are negative, or too small for further computation;

###### [2017-03-22](90b0cb1d2f1843343d5a0803cdc80260e1878dbe)

- FIX: Bugs with warnings in `sample.py`;

###### [2017-03-22](40295288a62c8f10da7f049cdab317db87e583c9)

- OPT: for `config.py` & warning function;

###### [2017-03-23](0643472e2a384f8eb94e12f94c092ffe2d7a7a94)

- ADD: Panning the entire coordinate system according to motion range, so as to keep coordinates positive; 

###### [2017-03-23](4211a71eee90da4fdb04876a5ef26ca71134c1e3)

- ADD: Apply different decoding layers to mu, sigma & rho to guarantee validity;
- TRY: Different initialization metrics for weights;

###### [2017-03-25](42a8f86e20844e01b90b430f81121ccb283b7cd1)

- RFCT: Extract file io into `file.py` & some other refactoring;
- ADD: Some utils;

###### [2017-03-27](b25784a81092e5c65d83965329575f925d1e19d7)

- MNT: Seperate `CHANGE.md` from `README.md`. & Rename `main.py` as `test.py`;

###### [2017-03-27](9bcd618823ad3177cf0c470f5353bc615a04140d)

- RFCT: Put initialization of weights & biases together for convenience;
- NOTE: With correlation==1.0, deviations=10000+, always get probs='inf'. Maybe should check value of correlation & `bivar_norm`;

###### [2017-03-28](a28aa72999fff3cd63476302885fc5ec3def8dde)

- TEST: `bivar_norm` is fine;
- NOTE: Correlation=+-1 has caused probs='inf'. Even correlation=+-0.1 would cause probs='inf';
- TEST: Keep correlation around +-`1.e-8` could prevent probs='inf', getting probs~=1.e-11 & loss ~= 900;
- ADD: Define `scaled_tanh` to prevent correlation from reaching 1 or -1. & Define `w_deviations` related to `N_NODES`;

###### [2017-03-30](cf217e3ac33c5f5515db19eaa4fe0aef785d7b19)

- RFCT: Wrap functions in `sample.py` into class `Sampler`. Tested;
- OPT: Add `utils.handle()`;

###### [2017-03-30](2cd5fe9d541c142f4abb0844a6847ee555e136d2)

- RFCT: Wrap functions in `model.py` into class `SharedLSTM`. & Some opt;

###### [2017-03-31](bc82a37756df3b1f44aca51a66bebf8d3f3e0794)

- ADD: Enable multiple hidden layers;
- OPT: for string formatting, warnings & configs;

###### [2017-03-31](ae4b7201986e93c5544ed95a37165643ca2c76bf)

- ADD: Class `Timer`. & Apply to building & compiling;

###### [2017-04-03](1404076eca67733b9df985831ea1285febc748ef)

- ADD: Enable scaled deviations, activated using sigmoid & then scaled within motion range;
- OPT: Enable checking for each single hidden layer during debugging. & Minor opt for network building;
- BUG: All the parameters (except for distribution layer) equal to 'nan' after 1st training;

###### [2017-04-04](ec6784434eab9b60e7875f397218b1cb96045cb4)

- RFCT: Wrap all the printing with `utils.xprint()` to control detail level;

###### [2017-04-05](65e5d9782ce5deeafa6784770751568f2aebc2eb)

- ADD: Enable recording & checking for all the parameters & layers during training;

###### [2017-04-06](88adfd9381202cd198a83bd2b44c275d9af5b7e1)

- ADD: Enable coordinates in sampler to be mapped to grid;

###### [2017-04-06](e0898e5d35bf5c0f5969a853edcbbd2573f015af)

- FIX: Bugs with printings. & Some opt;

###### [2017-04-06](a16e37ceedbd1f5c200ddd675f6bc21c81343c67)

- ADD: `Sampler.clip()` to enable devision of samples;

###### [2017-04-07](aec4d7f71507754770ec21fe3e369d65b9d7eb1d)

- ADD: Class `Logger` to manage log files;

###### [2017-04-08](ceae6c1f9778ea12b5ad5a60ba37cb1733a81c54)

- ADD: Log loss info to root logging directory (`/log`);

###### [2017-04-08](b062884540d50f24097f6bcc3bc2dbb0c31b5c24)

- ADD: Enable increasing of epoch number;

###### [2017-04-09](bb98a41deb0bdcb13aa064e6770be8dd45d022d6)

- ADD: Enable exporting & importing of model parameters;

###### [2017-04-10](af7bb86f56d7b80bc35d5a7cdf9db5d0d1a2b63e)

- RFCT: Redefine config as dictionary, to enable change & log of configs at any time;

###### [2017-04-11](f74c025ed4c8441ddd49845d47d4c67fad096e6a)

- FIX: Major bugs with sampler & some other minor bugs;

###### [2017-04-11](d63fe3e0c9ba1bbc5e351b5504059bd4e1a3caa8)

- RFCT: Seperate decoding, computing of loss & deviations from compiling function;

###### [2017-04-11](feb0fdd2abaf124e6a19277fe0d3ff762bf78adc)

- OPT: Remove all the references of `config.configuration` in default value definition of method arguments, 
  to enable config modification to work properly;
  
###### [2017-04-12](29f56437347cc92b33349e5f9e6ba5b2e265dfe6)

- RFCT: Some code style improvements according to [PEP8](https://www.python.org/dev/peps/pep-0008/#id36);

###### [2017-04-12](01271b55e4847273a4805746882dc4820584d5db)

- ADD: Enable command line configurations, e.g. `test.py -c|--config "{'num_node': 1, 'comment': 'something'}"`;

###### [2017-04-12](62765255fe862f2997b083e659817c03d12eb5ba)

- FIX: Major bugs with sampler. & Some other minor bugs & opts;

###### [2017-04-12](3b5f2155d341748f9b30174b223714ce478a37c8)

- ADD: Command `-t|--tag <some-tag>`, to tag the log folder e.g. "\[some-tag\]2017-04-12-16-00-00";

###### [2017-04-13](90213d5973a5d2deefc2c85949b77c6740285c7b)

- MOD: Choose node 31 & 32 from dataset NCSU for testing;

###### [2017-04-15](be6cb6e200abb3962809ec4ac665e0d8bd28d7d6)

- ADD: Enable to stop by `CTRL+C` between batches, and get logged properly;

###### [2017-04-15](8f7ae27b2275e8b221b2f82f9c489e5a53fc0591)

- MNT: Remove RNN packages written by YAN Yuan-Chi & jraiman package;

###### [2017-04-15](b5a5e18f2c351c14ae06bea3ac16a9b446788521)

- FIX: Bugs with pickling;

###### [2017-04-17](1e8b24675cb00e8737e3b95474c30c0920e41d46)

- ADD: Enable importing of configs & pickled params from command line, e.g. `-i|--import log/xx-xx-xx-xx-xx-xx/`;
- RFCT: Redefine structure of configs, config source & tags included;
- RFCT: Merge `filer.py` into `utils.py`. & Move mandatory initialzations into `__init__.py`;

###### [2017-04-17](84bb00b7b0103b18f6658517c8e0d4f53c0c1ed0)

- ADD: Log predictions to `prediction.log`;

###### [2017-04-17](170b26350e7159597916afe4e850bc7190ce60a3)

- ADD: Enable to change learning rate if training failure has occured. & Validate the finite of loss;
- RFCT: Remove useless config 'log_slot';
- RFCT: For asking & asserting methods;

###### [2017-04-17](5552ebbe8832b45e47191d6212ab979c59ea6764)

- ADD: Log command line args to file `args.log`;

###### [2017-04-18](b5cf6deff0e1fa045236548680bc888ef0a64e35)

- ADD: Enable timer to pause & resume;
- OPT: Pause the timer while waiting for answers from console, to improve the accuracy of timing;

###### [2017-04-18](bd93f761e1d2f6e22fee5897031670d2f0a5cdc0)

- OPT: Log deviations, predictions & targets together to `prediction.log`;
- FIX: Bug with root logger;

###### [2017-04-18](5112f7b3ce183e49e1138cc4caeef234bbd57497)

- ADD: Enable to choose training scheme among RMSProp, AdaGrad, Momentum SGD & Nesterov Momentum SGD;

###### [2017-04-18](450e0895b9d5a239714e919c98851afdc5b10681)

- RFCT: For assertion & file related methods in `utils`;

###### [2017-04-19](1e573fcf01948aea09ffabb29fd1bd107debf4f9)

- ADD: Enable to choose loss scheme from mean or sum;

###### [2017-04-19](5271c0ff1ac7161d1db4b1c173b82da0949742d0)

- OPT: For numpy array & error printing;

###### [2017-04-19](9ee8f48baa98da990466089c42fb6e1fabad6f2c)

- ADD: Enable to decay the learning rate by a constant ratio automatically after training failure;

###### [2017-04-19](87d88866d2527cb91fc0a3fa61d7ea02929623d8)

- OPT: Raise exception for unknown configuration key;

###### [2017-04-20](db5b84bb710ad0208b976efb31f25074b4cf15af)

- FIX: Bug with importing;

###### [2017-04-20](a4ed8875f6e0221925604d9b934000b9c554266d)

- FIX: Major bug with hidden layer structure;
- ADD: Enable to choose from parameter-shared / input-shared LSTM;

###### [2017-04-21](070705a70ba94db2936c8a84fa177fa176a79e97)

- FIX: Remove repeated computation in `SocialLSTM.train()` & `SocialLSTM.export_params()` to avoid unknown exception:

> Traceback (most recent call last):
>   File "test.py", line 14, in <module>
>     lstm.model.SocialLSTM.test()
>   File "D:\theano\lstm\lstm\model.py", line 1104, in test
>     model.export_params()
>   File "D:\theano\lstm\lstm\model.py", line 996, in export_params
>     params_all = self.check_params()
>   File "C:\Anaconda2\lib\site-packages\theano\compile\function_module.py", line 898, in __call__
>     storage_map=getattr(self.fn, 'storage_map', None))
>   File "C:\Anaconda2\lib\site-packages\theano\gof\link.py", line 325, in raise_with_op
>     reraise(exc_type, exc_value, exc_trace)
>   File "C:\Anaconda2\lib\site-packages\theano\compile\function_module.py", line 884, in __call__
>     self.fn() if output_subset is None else\
> ValueError: DeepCopyOp: the copy failed!
> Apply node that caused the error: DeepCopyOp(LSTM[10,1].W_hid_to_ingate)
> Toposort index: 191
> Inputs types: [CudaNdarrayType(float32, matrix)]
> Inputs shapes: [(32, 32)]
> Inputs strides: [(32, 1)]
> Inputs values: ['not shown']
> Outputs clients: [['output']]

###### [2017-04-21](2fd44ba7232051186d9f0b6dc5881818aa7719df)

- OPT: Set max length for network history to decrease RAM usage. & Log network history for future analysis;

###### [2017-04-21](ef98205288ef3d876f22d74bb7d09fab3e8d326c)

- FIX: Bug with command line importing argument;

###### [2017-04-21](eac766032385052be4c93a44efac604796d70faf)

- FIX: Bug with param pickling. No long compatible with old pickled files;

###### [2017-04-21](27cfb034b2da44de86d067ffa404e58dce55b7a7)

- OPT: Disable the recording of network history to speed up;

###### [2017-04-23](957e531f00f52ab19d3ecd8fcefd0d1866e1418d)

- OPT: For printing & logging;

###### [2017-04-24](4dab6013d9333d039393fdf40d9c12f542fb8e9a)

- ADD: Enable to choose from stopping, continuing & peeking during training;

###### [2017-04-24](360ae3d4044f4d353fad2ae012a692e767bdc006)

- ADD: Enable to choose decoding scheme between bivariant normal distribution / euclidean distance;

###### [2017-04-25](919aa2071f0cca1c330c2f490c17545e3cf19e57)

- FIX: Error during logging of loss by epoch;

###### [2017-04-26](6fbfac05519e77e67e5fd8e28f959b817f51fdbb)

- ADD: Enable to map coordinates to grid through config;

###### [2017-05-03](4832e5fdd7835321f2bc8b5a3b4e2658c957d625)

- RFCT: For sampler in SocialLSTM;

###### [2017-05-04](8eb12daac7910eac9165d58cecfcb9f2bde0d3f3)

- FIX: Bugs with instants & targets logging;

###### [2017-05-04](edffd7015000e9b4b9932c7007051eb814f6798e)

- RFCT: Seperate training control on different levels;

###### [2017-05-04](fc395728ec10576e4aae7dece6ae37d5b8f0b736)

- RFCT: Extract interpretation of menu asking into function;

###### [2017-05-08](20d9898f9bca5e2edf6d3cde0cb788e69a59e1a7)

- FIX: Wrong sampler arg passed into `SocialLSTM.train()`;
- RFCT: for `SocialLSTM.train()`;

###### [2017-05-08](3549db600e4ff735bc94f5b64622b9df41344bf2)

- ADD: Predicting function;

###### [2017-05-08](e6a45a0ad91b17b616922f3e49e7ec6a4dbca315)

- ADD: Save & export param values with the best training result;

###### 2017-05-09

- FIX: Tryout (predicting) should be actually training of one single epoch;
- NOTE: However parameter values would be restored after each tryout, thus to enable multiple tryouts during training without interference;
- FIX: Possible bug with parameter values;


- [ ] OPT: Change use of dict into OrderedDict or so;
- [ ] RFCT: Extract weight & bias initializers;
- [ ] ADD: Pre-training;
- [ ] ADD: Use `pprint` to improve printing;
