
>***Created by Eco Sun on 2016-11-23***


### Dependency

- python-2.7.12
- numpy-1.11.2
- scipy-0.18.1
- mingw-4.7.0
- Theano-0.9.0.dev4
- [Lasagne latest version](639972e1496a3df331401de633f18be8b7ee9265)

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

###### [2016-11-27](ea01df35e2619edf0ab0690dd67825187a6160e6)

###### [2016-11-29](d8475aa25b4aa03a7c6ecd8c06ec48b9f1a5b9de)

###### [2016-11-30](c655f04af0fb13077deef1b958f9adb7873fe3c4)

###### [2016-12-01](e082892fc7e9d6d96adaa5f14cf0e6197d956e04)

- ADD: Allow `load_batch_for_nodes` to have either `int` or `list` passed in for param `filter_nodes`;
- ADD: Nano size configs for debugging; 
- ADD: Forwarding connections through the network are built & tested;
- [ ] TODO: While still having problems with the recurrent connection from previous LSTM hidden states to Social Pooling Layer;
- [x] TRY: Whether `RecurrentContainerLayer` proposed in [#629](https://github.com/Lasagne/Lasagne/pull/629) would suffice;

###### 2016-12-02

- Tried to apply PR [stevenxxiu/Lasagne:recurrent](1d23b4022455ef8449b98c3805f2f0c919836f61) & 
  [skaae/Lasagne:merge_broadcast](8588929fba32b03e3df324b87fc451cc4b2ab225) via patch/git-format-patch approach. FAIL;
- OPT: Import of lasagne;