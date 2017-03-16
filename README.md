
>***Created by Eco Sun on 2016-11-23***


### Dependency

- python-2.7.12
- numpy-1.11.2
- scipy-0.18.1
- mingw-4.7.0
- Theano-0.9.0.dev4
- **[Forked Lasagne](https://github.com/sheeco/Lasagne)**

### Platform

- Windows 10, 64 bit
- PyCharm Community Edition 2016.3


### Files

    README.md
    contineous/
    |-- ...
    disperse/
    |-- ...
    regression/
    |-- ...
    lstm/   
    |-- jraiman_lstm/
        |-- __init__.py
        |-- masked_loss.py   
    |-- __init__.py   
    |-- config.py   
    |-- file_io.py 
    |-- model.py    
    |-- train.py
 
 
---


#### Log Mark

*在 Git Commit Comment 中使用快速标签：*

　　 `+` 添加、`-` 删除、`~` 修改、`#` 优化、`?` 尝试、`!` 错误、`$` 修复、`%` 测试；

*在 `README.md` 的更新日志中使用分类标记标签：*

　　 `ADD` 添加新功能、`REM` 删除现有功能、`MOD` 修改现有实现、`OPT` 功能优化、`TRY` 不确定尝试；

　　 `BUG` 待修复错误、`FIX` 修复错误、`TEST` 功能测试；

　　 `RFCT` 代码重构、`MNT` 版控维护、`NOTE` 附加说明、`TODO` 计划任务；


# Update LOG

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

###### 2017-03-16

- 