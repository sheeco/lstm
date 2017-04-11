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

###### 2017-04-11

- FIX: A major bug in `SharedLSTM.test()` & some other minor bugs;


- [ ] FIX: Pass `config` into class initializers;
- [ ] RFCT: Extract weight & bias initializers;
- [ ] ADD: Pre-training;
- [ ] ADD: Use `pprint` to improve printing;
- [ ] TRY: Tuning `GRAD_CLIP`;
