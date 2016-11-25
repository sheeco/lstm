
>***Created by Eco Sun on 2016-11-23***


### Dependency

- python-2.7.12
- numpy-1.11.2
- scipy-0.18.1
- mingw-4.7.0
- Theano-0.9.0.dev4
- Lasagne-0.2.dev1

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


###### [ 2016-11-25 ]( b35009fb246b93a56376a1122a12e5528d5cac19 )

- ADD: 加载 batch 的函数 `load_batch_for_nodes`；
- RFCT: 重命名文件 `file_io.py` 为 `sample.py`，及一些函数重命名；