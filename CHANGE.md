### *Log Mark*

*在 `CHANGE.md` 的更新日志中使用分类标记标签：*

　　 `ADD` 添加新功能、`REM` 删除现有功能、`MOD` 修改现有实现、`OPT` 功能优化、`TRY` 不确定尝试；

　　 `BUG` 待修复错误、`FIX` 修复错误、`TEST` 功能测试；

　　 `RFCT` 代码重构、`MNT` 版控维护、`NOTE` 附加说明、`TODO` 计划任务；


# Change Log

## 0.0.*


#### [2016-11-25](b35009fb246b93a56376a1122a12e5528d5cac19)

- ADD: 加载 batch 的函数 `load_batch_for_nodes`；
- RFCT: 重命名文件 `file_io.py` 为 `sample.py`，及一些函数重命名；

#### [2016-11-25](b9ab414351b1855919b3672874da2932c096d05c)

- MOD: 调整函数 `load_batch_for_nodes` 返回数组的维度次序为：节点->批->序列->采样维度；
- FIX: `sample.py` 中的输出错误；
- ADD: `config.py` 中的参数；

#### [2016-11-25](e06e548390787b7f073534d2f07affd7fdf18897)

- ADD: 修改自 [lstm_text_generation](https://github.com/Lasagne/Recipes/blob/master/examples/lstm_text_generation.py) 的 demo；
- [ ] BUG: 能够正常运行，但有时会得到 `NaN`；

#### [2016-12-01](e082892fc7e9d6d96adaa5f14cf0e6197d956e04)

- ADD: Allow `load_batch_for_nodes` to have either `int` or `list` passed in for param `filter_nodes`;
- ADD: Nano size configs for debugging; 
- ADD: Forwarding connections through the network are built & tested;
- [ ] TODO: While still having problems with the recurrent connection from previous LSTM hidden states to Social Pooling Layer;
- [ ] TRY: Whether `RecurrentContainerLayer` proposed in [#629](https://github.com/Lasagne/Lasagne/pull/629) would suffice;

#### [2016-12-02](09a53a25e789820e71f19145d92db3e9f15ec37f)

- ~~Fail to apply PR [stevenxxiu/Lasagne:recurrent](1d23b4022455ef8449b98c3805f2f0c919836f61) & 
  [skaae/Lasagne:merge_broadcast](8588929fba32b03e3df324b87fc451cc4b2ab225) via patch/git-format-patch approach.~~
  SHOULD use the fork-fetch-merge approach;
- OPT: Simplize module importing;

#### [2016-12-03](a3edd597c5fc475906d2f93a45f835e93c5bd23f)

- Merge PR [#629](https://github.com/Lasagne/Lasagne/issues/629) from [stevenxxiu/Lasagne:recurrent](1d23b4022455ef8449b98c3805f2f0c919836f61) & 
  [#633](https://github.com/Lasagne/Lasagne/issues/633) from [skaae/Lasagne:merge_broadcast](8588929fba32b03e3df324b87fc451cc4b2ab225) 
  into [my forked lasagne](https://github.com/sheeco/Lasagne) for dependency;
- MOD: Implement social pooling based on broadcastable `ElementwiseMergeLayer` (by skaae) instead of `ExpressionMergeLayer`;
- MOD: Have to alter the LSTM params sharing code for the new `LSTMCell` wrapper (by stevenxxiu);
- ADD: Try to use `RecurrentContainerLayer` to implement the recurrent connection from LSTM to S-Pooling;
- FIX: Theano compiling(?) problems in `social_mask` (probably undone);
- BUG: "Lack of input" reported by `lasagne.helper.get_output`. Haven't figured out the reason yet;

#### [2016-12-05](68aae71b8b1d7aacefee9bd6ec79882d960f5459)

- RFCT: Move helper functions into `utils.py`;
- ADD: `STRICT_BATCH` to indicate if only full-size batches are accepted, maybe mandatory to avoid indetermination in `social_mask`;
- ADD: Try to define `SocialLSTMCell` inherited from `CustomRecurrentCell` (untested);
- NOTE: Maybe should give up the `ReccurentContainerLayer` approach & overwrite `CustomRecurrentLayer` directly?

#### [2016-12-12](a7ac820e2778d45190204cdfd9349843859e5a1f)

- ADD: Instant sequence to the return values of `load_batch_for_nodes`;

#### [2016-12-12](2aa7ec08e92780cd1c7cb86c2fe549e0f217657b)

- TEST: Forwarding connections should be fine, only if passing in previous hidden states manually;
- DEBUG: The overwriting in `SocialLSTMCell`, still "lack of input" for `layer_xy`. Likely has sth to do with line 291 & 551 in [recurrent.py](https://github.com/sheeco/Lasagne/blob/add565017d4636676028d16dcba1ec2b2870aa36/lasagne/layers/recurrent.py#L291);

#### [2016-12-22](b96a719c22d8ee8941e9c737b12a4597f031a30d)

- NOTE: Give up the previous cell&container approach from stevenxxiu;
- ADD: Try to define `RecurrentContainerLayer` by overwriting `CustomRecurrentLayer`;

##### Branch `shared_lstm`

#### [2017-03-13](4ff90b74528e4090fe2dd98f6460cbe3b7c24227)

- ADD: A simple model of shared LSTM, with dense decoder;
- TEST: Building & compiling should be fine;

#### [2017-03-14](8591dd9b5dcc7504806f030685bd0ec4b09f210d)

- OPT: for sample loading;

#### [2017-03-20](dd0b55bd7c2884eb8b0226890399ae9c7fe88f77)

- MOD: Dimension of network inputs modified to 3 to include the time column;
- ADD: NNL loss for training & Euclidean distance error for observation. 
- NOTE: Need new package [breze](https://github.com/breze-no-salt/breze) & consequently package [climin](https://github.com/BRML/climin), 
  for `pdf` (under `breze.arch.component.distributions.mvn`), the probability density function of multi-variate normal distribution;
- TEST: `compute_and_compile` should be fine for now;

#### [2017-03-21](8d138fdcc0785c27b2c97f8a91c36ea157ce2da6)

- MOD: Define probability density function of bivariate normal distribution directly. No longer need package breze;
- TRY: Different initialization metrics for weights & biases to avoid `NaN` or `inf` during computation;
- BUG: Values of sigma (standard deviations) are negative, or too small for further computation;

#### [2017-03-22](90b0cb1d2f1843343d5a0803cdc80260e1878dbe)

- FIX: Bugs with warnings in `sample.py`;

#### [2017-03-22](40295288a62c8f10da7f049cdab317db87e583c9)

- OPT: for `config.py` & warning function;

#### [2017-03-23](0643472e2a384f8eb94e12f94c092ffe2d7a7a94)

- ADD: Panning the entire coordinate system according to motion range, so as to keep coordinates positive; 

#### [2017-03-23](4211a71eee90da4fdb04876a5ef26ca71134c1e3)

- ADD: Apply different decoding layers to mu, sigma & rho to guarantee validity;
- TRY: Different initialization metrics for weights;

#### [2017-03-25](42a8f86e20844e01b90b430f81121ccb283b7cd1)

- RFCT: Extract file io into `file.py` & some other refactoring;
- ADD: Some utils;

#### [2017-03-27](b25784a81092e5c65d83965329575f925d1e19d7)

- MNT: Separate `CHANGE.md` from `README.md`. & Rename `main.py` as `test.py`;

#### [2017-03-27](9bcd618823ad3177cf0c470f5353bc615a04140d)

- RFCT: Put initialization of weights & biases together for convenience;
- NOTE: With correlation==1.0, deviations=10000+, always get probs='inf'. Maybe should check value of correlation & `bivar_norm`;

#### [2017-03-28](a28aa72999fff3cd63476302885fc5ec3def8dde)

- TEST: `bivar_norm` is fine;
- NOTE: Correlation=+-1 has caused probs='inf'. Even correlation=+-0.1 would cause probs='inf';
- TEST: Keep correlation around +-`1.e-8` could prevent probs='inf', getting probs~=1.e-11 & loss ~= 900;
- ADD: Define `scaled_tanh` to prevent correlation from reaching 1 or -1. & Define `w_deviations` related to `N_NODES`;

#### [2017-03-30](cf217e3ac33c5f5515db19eaa4fe0aef785d7b19)

- RFCT: Wrap functions in `sample.py` into class `Sampler`. Tested;
- OPT: Add `utils.handle()`;

#### [2017-03-30](2cd5fe9d541c142f4abb0844a6847ee555e136d2)

- RFCT: Wrap functions in `model.py` into class `SharedLSTM`. & Some opt;

#### [2017-03-31](bc82a37756df3b1f44aca51a66bebf8d3f3e0794)

- ADD: Enable multiple hidden layers;
- OPT: for string formatting, warnings & configs;

#### [2017-03-31](ae4b7201986e93c5544ed95a37165643ca2c76bf)

- ADD: Class `Timer`. & Apply to building & compiling;

#### [2017-04-03](1404076eca67733b9df985831ea1285febc748ef)

- ADD: Enable scaled deviations, activated using sigmoid & then scaled within motion range;
- OPT: Enable checking for each single hidden layer during debugging. & Minor opt for network building;
- BUG: All the parameters (except for distribution layer) equal to 'nan' after 1st training;

#### [2017-04-04](ec6784434eab9b60e7875f397218b1cb96045cb4)

- RFCT: Wrap all the printing with `utils.xprint()` to control detail level;

#### [2017-04-05](65e5d9782ce5deeafa6784770751568f2aebc2eb)

- ADD: Enable recording & checking for all the parameters & layers during training;

#### [2017-04-06](88adfd9381202cd198a83bd2b44c275d9af5b7e1)

- ADD: Enable coordinates in sampler to be mapped to grid;

#### [2017-04-06](e0898e5d35bf5c0f5969a853edcbbd2573f015af)

- FIX: Bugs with printings. & Some opt;

#### [2017-04-06](a16e37ceedbd1f5c200ddd675f6bc21c81343c67)

- ADD: `Sampler.clip()` to enable division of samples;

#### [2017-04-07](aec4d7f71507754770ec21fe3e369d65b9d7eb1d)

- ADD: Class `Logger` to manage log files;

#### [2017-04-08](ceae6c1f9778ea12b5ad5a60ba37cb1733a81c54)

- ADD: Log loss info to root logging directory (`/log`);

#### [2017-04-08](b062884540d50f24097f6bcc3bc2dbb0c31b5c24)

- ADD: Enable increasing of epoch number;

#### [2017-04-09](bb98a41deb0bdcb13aa064e6770be8dd45d022d6)

- ADD: Enable exporting & importing of model parameters;

#### [2017-04-10](af7bb86f56d7b80bc35d5a7cdf9db5d0d1a2b63e)

- RFCT: Redefine config as dictionary, to enable change & log of configs at any time;

#### [2017-04-11](f74c025ed4c8441ddd49845d47d4c67fad096e6a)

- FIX: Major bugs with sampler & some other minor bugs;

#### [2017-04-11](d63fe3e0c9ba1bbc5e351b5504059bd4e1a3caa8)

- RFCT: Separate decoding, computing of loss & deviations from compiling function;

#### [2017-04-11](feb0fdd2abaf124e6a19277fe0d3ff762bf78adc)

- OPT: Remove all the references of `config.configuration` in default value definition of method arguments, 
  to enable config modification to work properly;
  
#### [2017-04-12](29f56437347cc92b33349e5f9e6ba5b2e265dfe6)

- RFCT: Some code style improvements according to [PEP8](https://www.python.org/dev/peps/pep-0008/#id36);

#### [2017-04-12](01271b55e4847273a4805746882dc4820584d5db)

- ADD: Enable command line configurations, e.g. `test.py -c|--config "{'num_node': 1, 'comment': 'something'}"`;

#### [2017-04-12](62765255fe862f2997b083e659817c03d12eb5ba)

- FIX: Major bugs with sampler. & Some other minor bugs & opts;

#### [2017-04-12](3b5f2155d341748f9b30174b223714ce478a37c8)

- ADD: Command `-t|--tag <some-tag>`, to tag the log folder e.g. "\[some-tag\]2017-04-12-16-00-00";

#### [2017-04-13](90213d5973a5d2deefc2c85949b77c6740285c7b)

- MOD: Choose node 31 & 32 from dataset NCSU for testing;

#### [2017-04-15](be6cb6e200abb3962809ec4ac665e0d8bd28d7d6)

- ADD: Enable to stop by `CTRL+C` between batches, and get logged properly;

#### [2017-04-15](8f7ae27b2275e8b221b2f82f9c489e5a53fc0591)

- MNT: Remove RNN packages written by YAN Yuan-Chi & jraiman package;

#### [2017-04-15](b5a5e18f2c351c14ae06bea3ac16a9b446788521)

- FIX: Bugs with pickling;

#### [2017-04-17](1e8b24675cb00e8737e3b95474c30c0920e41d46)

- ADD: Enable importing of configs & pickled params from command line, e.g. `-i|--import log/xx-xx-xx-xx-xx-xx/`;
- RFCT: Redefine structure of configs, config source & tags included;
- RFCT: Merge `filer.py` into `utils.py`. & Move mandatory initializations into `__init__.py`;

#### [2017-04-17](84bb00b7b0103b18f6658517c8e0d4f53c0c1ed0)

- ADD: Log predictions to `prediction.log`;

#### [2017-04-17](170b26350e7159597916afe4e850bc7190ce60a3)

- ADD: Enable to change learning rate if training failure has occurred. & Validate the finite of loss;
- RFCT: Remove useless config 'log_slot';
- RFCT: For asking & asserting methods;

#### [2017-04-17](5552ebbe8832b45e47191d6212ab979c59ea6764)

- ADD: Log command line args to file `args.log`;

#### [2017-04-18](b5cf6deff0e1fa045236548680bc888ef0a64e35)

- ADD: Enable timer to pause & resume;
- OPT: Pause the timer while waiting for answers from console, to improve the accuracy of timing;

#### [2017-04-18](bd93f761e1d2f6e22fee5897031670d2f0a5cdc0)

- OPT: Log deviations, predictions & targets together to `prediction.log`;
- FIX: Bug with root logger;

#### [2017-04-18](5112f7b3ce183e49e1138cc4caeef234bbd57497)

- ADD: Enable to choose training scheme among RMSProp, AdaGrad, Momentum SGD & Nesterov Momentum SGD;

#### [2017-04-18](450e0895b9d5a239714e919c98851afdc5b10681)

- RFCT: For assertion & file related methods in `utils`;

#### [2017-04-19](1e573fcf01948aea09ffabb29fd1bd107debf4f9)

- ADD: Enable to choose loss scheme from mean or sum;

#### [2017-04-19](5271c0ff1ac7161d1db4b1c173b82da0949742d0)

- OPT: For numpy array & error printing;

#### [2017-04-19](9ee8f48baa98da990466089c42fb6e1fabad6f2c)

- ADD: Enable to decay the learning rate by a constant ratio automatically after training failure;

#### [2017-04-19](87d88866d2527cb91fc0a3fa61d7ea02929623d8)

- OPT: Raise exception for unknown configuration key;

#### [2017-04-20](db5b84bb710ad0208b976efb31f25074b4cf15af)

- FIX: Bug with importing;

#### [2017-04-20](a4ed8875f6e0221925604d9b934000b9c554266d)

- FIX: Major bug with hidden layer structure;
- ADD: Enable to choose from parameter-shared / input-shared LSTM;

#### [2017-04-21](070705a70ba94db2936c8a84fa177fa176a79e97)

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

#### [2017-04-21](2fd44ba7232051186d9f0b6dc5881818aa7719df)

- OPT: Set max length for network history to decrease RAM usage. & Log network history for future analysis;

#### [2017-04-21](ef98205288ef3d876f22d74bb7d09fab3e8d326c)

- FIX: Bug with command line importing argument;

#### [2017-04-21](eac766032385052be4c93a44efac604796d70faf)

- FIX: Bug with param pickling. No long compatible with old pickled files;

#### [2017-04-21](27cfb034b2da44de86d067ffa404e58dce55b7a7)

- OPT: Disable the recording of network history to speed up;

#### [2017-04-23](957e531f00f52ab19d3ecd8fcefd0d1866e1418d)

- OPT: For printing & logging;

#### [2017-04-24](4dab6013d9333d039393fdf40d9c12f542fb8e9a)

- ADD: Enable to choose from stopping, continuing & peeking during training;

#### [2017-04-24](360ae3d4044f4d353fad2ae012a692e767bdc006)

- ADD: Enable to choose decoding scheme between bivariate normal distribution / euclidean distance;

#### [2017-04-25](919aa2071f0cca1c330c2f490c17545e3cf19e57)

- FIX: Error during logging of loss by epoch;

#### [2017-04-26](6fbfac05519e77e67e5fd8e28f959b817f51fdbb)

- ADD: Enable to map coordinates to grid through config;

#### [2017-05-03](4832e5fdd7835321f2bc8b5a3b4e2658c957d625)

- RFCT: For sampler in SocialLSTM;

#### [2017-05-04](8eb12daac7910eac9165d58cecfcb9f2bde0d3f3)

- FIX: Bugs with instants & targets logging;

#### [2017-05-04](edffd7015000e9b4b9932c7007051eb814f6798e)

- RFCT: Separate training control on different levels;

#### [2017-05-04](fc395728ec10576e4aae7dece6ae37d5b8f0b736)

- RFCT: Extract interpretation of menu asking into function;

#### [2017-05-08](20d9898f9bca5e2edf6d3cde0cb788e69a59e1a7)

- FIX: Wrong sampler arg passed into `SocialLSTM.train()`;
- RFCT: for `SocialLSTM.train()`;

#### [2017-05-08](3549db600e4ff735bc94f5b64622b9df41344bf2)

- ADD: Predicting function;

#### [2017-05-08](e6a45a0ad91b17b616922f3e49e7ec6a4dbca315)

- ADD: Save & export param values with the best training result;

#### [2017-05-09](c7356e2c5f54098382f17113f25492e8b6e8e9d6)

- FIX: Tryout (predicting) should be actually training of one single epoch;
- NOTE: However parameter values would be restored after each tryout, thus to enable multiple tryouts during training without interference;
- FIX: Possible bug with parameter values;

#### [2017-05-09](5c7d9203e624d31b980d2bef19cd8505a066ccaf)

- OPT: Enable loose batch size;

#### [2017-05-10](b2bae08837905a811a5c10a0cddaf6c41566590a)

- FIX: Bug with best param value exporting;

#### [2017-05-10](eab208a32193f5f992be9e71f5a338d2cac14bf0)

- ADD: Enable to config trainset size;

#### [2017-05-10](4883b885628f35035426a5c17e51448fa12ac79b)

- FIX: & opt for printing & logging;

#### [2017-05-15](60e096513ea7b1521530a2ebac8955b71b36db01)

- OPT: for param pickling;

#### [2017-05-15](1862cc2f727c9d852e35d63ef40df1bc0d9ade9d)

- REM: Disable configuration importing. Only param importing is allowed now;

#### [2017-05-16](4bf4c893622d1c4088925c8e172e4379e73e76aa)

- OPT: for param pickling;

#### [2017-05-16](4a671bee91884ce5048fc25263c756f8cd55b31d)

- FIX: Bug with loss & deviation logging;

#### [2017-05-22](e31fb48b7935339aac1934fd46e779e8353722e3)

- FIX: Forget to reset entries after learning rate gets updated;
- FIX: Logging with multiple nodes;

#### [2017-05-22](b74812468ae3edd119c66c9d581c8a37eac68915)

- OPT: for param pickling;

#### [2017-05-23](ea4ef886fe29bb453fae7735e1b1fda30b1e4ebb)

- ADD: Social LSTM with Occupancy Map sharing. & Some OPT;

#### [2017-05-23](3e43fe9f57667afaa52062805905702866c89f8c)

- Merge branch [master](b74812468ae3edd119c66c9d581c8a37eac68915) & [social](ea4ef886fe29bb453fae7735e1b1fda30b1e4ebb);

#### [2017-05-24](9d1a1fcc2251d3832955a4e692ebb8990b84e1fb)

- ADD: Enable to config adaptive learning rate as decrement;

#### [2017-05-24](3f60184f77fb00fc8539ea76f5b0c31b3c335ead)

- ADD: Enable to adapt gradient clipping;

#### [2017-05-25](3edd2e2d913d94419d18f11744918a73fd3b1afb)

- ADD: Hitrate calculation & logging;
- FIX: Wrong calculation time of predictions & deviations;
- MOD: Change from deviation to hitrate as the metric to compete best param values;
- FIX: Bug with best param values;

#### [2017-05-29](575078143379eca398a7f8c12d71c4df34ab4975)

- FIX: Bug with stopping while training;

#### [2017-05-31](dddfb6fd5112ae8a7236394254c853c2928d41d5)

- FIX: Bug with hitrate printing;

#### [2017-05-31](441e262a39747445fabffa84cb61e5ec6a7a20e8)

- FIX: Bug with asking functions;

#### [2017-06-01](82ac680f65995e95eccb577cea486fe8ae8a24a0)

- OPT: Enable to tryout without training;

#### [2017-06-01](774b87107395167633a5a4281b3fc952686f3161)

- OPT: Backup some of the old logs instead of overwriting when learning rate gets updated;

#### [2017-06-04](a764f8f2eb778d3ad54dff7e8a23021af2b9016e)

- FIX: Bug with `InvalidTrainError` during testing;

#### [2017-06-05](4873171177773877eaade8e40b44677be9655f14)

- OPT: Accumulative hitrate;

#### [2017-06-07](b84d12ffa3a91f594a8bae109c381a13cbbe8f2f)

- OPT: Enable to apply all the samples as testset;
- TEST: Hitrate result of full testset is worse than those of separate trainset & testset (e.g. 27% out of 57% & 91%);
- TEST: Hitrate result of testset is repeatable under newly compiled train function with same initial learning rate;

#### [2017-06-07](d37dab258e5dccd969689f26359a48223df40505)

- FIX: Flip the sequence slicing to enable 'n-n' seq, & to maintain compatible with previous parameters for 'n-1' seq;
- TESTed: Hitrate result of 'n-1' seq is repeatable with flipped slicing;

#### [2017-06-07](a62ccd2c21c58c883081f2fe9853648e8719b2d9)

- OPT: Enable to exit without asking;
- OPT: Update some default configs;

#### [2017-06-09](816963178a4905ccc8a2da7ce6a27093f887e35f)

- RFCT: for compute-compile & adaptive learning;

#### [2017-06-09](cee61318f0883c386ce37036c697dfeb0da0ab3d)

- OPT: to avoid massive hitrate computation due to gigantic deviations;

#### [2017-06-09](63aaa31b8d3baafa96ed494086681896e2a029fc)

- FIX: Predict before actually do training;
- FIX: for return of tryout;
- TEST: Slightly worse than predicting after training, e.g. drop from 94% to 91.4%;

#### [2017-06-09](d40fee638d99f11643b7132c4d18a7950bcdb6ba)

- REM: Abandon network history;

#### [2017-06-14](758b3ea174e4bfab28add05133958e4ce3ac5d45)

- OPT: Enable to use predictions as training targets if real targets are missing;
- TEST: Use predictions as targets during tryout - get the same predictions all the way;

#### [2017-06-27](33da44b337d19e58e277e70a5c75ac786da41058)

- FIX: Fix hitrange comparison;
- MOD: Change to use tryout hitrate to decide the best record;

#### [2017-06-27](9753ba3c3710d629ee6f548acb74cf2c929eb0d2)

- OPT: for config logging;
- FIX: for exception handling in `sampler._filter_nodes_`;

#### [2017-07-10](019479f49b5fd86578c38e345febd0d72cc4a2b4)

- FIX: Crash when asking gets interrupted finally;
- FIX: Do not log by epoch if undone;

#### [2017-07-20](1f8f1ff2c25005c2aa006504b2bdf110d3678fc5)

- ADD: Save best records for hitrange 50m & 100m; 

#### [2017-11-14](b2c844a61baec1798ee03dc30cafa1c9ce5126d9)

- RFCT: Major refactoring for `Sampler`;

#### [2017-11-15](3a1cb50a647d292f058455077713a2b219422c1c)

- ADD: Enable to save samples(prediction results) to an (initially empty-like) sampler;
- ADD: Enable to apply unreliable input (use previous prediction results as the input samples during training);

#### [2017-11-15](b84212f62230f16173b4f9caa6174ed524e24f8d)

- OPT: for sampler;
- [x] RFCT: Change `node_identifiers` from list to dict;

#### [2017-11-21](f52833cb1f41a1568e6c52792141654023051264)

- RFCT: Fix some typos & code style problems;

#### [2017-11-21](696f67e9f5aa628e4f3b1ecf8a5e558ff20223c7)

- [x] RFCT: Move default configuration groups into data file `default.config`;
- RFCT: for config processing & echo messages;

#### [2017-11-22](eed4b08a15e450cfc66efc6b1dab5b4bfd1be265)

- MNT: Update readme;

#### [2017-11-28](2c8a32b79fcebed4f2010e4345314e0306036d33)

- MNT: Update readme & configuration descriptions;

#### [2017-11-28](2cf3bf5a2167eedac41cb17e999d65de733b6905)

- OPT: Enable list<int> as node filter, to improve robustness;

#### [2017-12-13](d6d3cbdebdd152f65136fc811925ee41d0ad2af7)

- MNT: Update readme & docstrings;

#### [2017-12-13](e2f4035b0161041a6e24a6b65dd24896e1439c22)

- FIX: Missing `config.log` for tryout-only execution;

#### [2017-12-26](4b174cac60fc6bde5007dd1f7a988e0d88349534)

- FIX: Wrong(real-time instead of 'original') parameters are saved to 'params-best-hitrange%d-hitrate%.1f-epoch%d.pkl' 
  (but parameters in 'params-epoch%d.pkl' are fine);

#### [2018-01-09](d03d9997eb51cb6dc5ba454f038615f9c1b92f64)

- MNT: Update `.gitignore` for IntelliJ project;

#### [2018-01-09](d41d8919a74e59873bcda45532cb0c7b2e139ff6)

- FIX: String format restriction in command line arguments;

#### [2018-01-18](91ca28f92162ba9048d4ee20c9bd999623c83d64)

- FIX: Export values of all the `TensorSharedVariable` involved with `func_train` (using `theano.compile.Function.get_shared()`
  instead of `Lasagne.layers.get_all_params()`) to fix the performance issue with re-imported models;

#### [2018-01-18](e5b933e80066a7a0f56cdb2b6880ee7a59f81d43)

- OPT: for timing display;

#### [2018-01-19](a9e9b6ee8465d26e8648841e08a0690500163127)

- [x] REM: Disable 'num_node' configuration;

#### [2018-01-19](3efb2bd0c818549cd24de5b87ec28873e8723f72)

- ADD: Enable to auto-exit once reaching expected hitrate configured with key 'expected_hitrate';

#### [2018-01-19](e01630a397e9f9adb973fd87c9abfd17f522f25b)

- RFCT: for configuration validation;

#### [2018-02-01](997d94124ffa306ff7c5dce14d73f4bc8cd7de37)

- RFCT: Major refactoring for `utils.py` & `config.py`;

#### [2018-02-04](6c8653c0e2065262cb96f9fec0db02b746c4fe40)

- RFCT: More refactoring for `utils.py` & `config.py` (`config.default_config_groups` is removed. Configs are loaded directly
  from files; Configs are separated into 2 files. Global configs & loggers are initialized in `utils.__init__()`);

#### [2018-02-04](2aae8ad8ff786b4f94f72d1e04ba0d50e716e414)

- ADD: Enable to dump predictions of desired epoch from previous execution into trace files;

#### [2018-02-27](548ee3f597a0914147579db973a6ce368886a7ba)

- OPT: for prediction dumping;

#### [2018-02-28](2ae41342e35220bc2d5882bb0281fe2dc9b65a86)

- ADD: Enable to dump panning rule for trace of desired node into pan files;
- RFCT: for `dump.py`;

#### [2018-03-29](037c91e6f10f528a74d359c07be38d561d7bda00)

- OPT: Update LSTM configs to smaller dimensions;

#### [2018-04-04](6cb689ea54fee37f990a232e605383c601dd8982)

- OPT: for prediction dumping;

#### 2018-05-17

- MNT: Update `README.md`;


## To-do List

- [ ] ADD: Enable to dump pickled parameters from previous execution;
- [ ] OPT: Keep sampler dividing compatibale with previous version?
- [ ] RFCT: Move `SocialLSTM.test()` into `test.py`?
- [ ] RFCT: Change to built-in `logger`;
- [ ] RFCT: Separate new class `OutputSampler` & `InputSampler` out from `Sampler`;
- [ ] RFCT: Extract weight & bias initializers;
- [ ] ADD: Pre-training;
- [ ] ADD: Use `pprint` to improve printing;
