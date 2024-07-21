# Transformer

本项目包含non-streaming ASR和Streaming ASR，模型的pth文件和结果放在了result的文件夹中。

# Train & Test

训练函数和测试函数，都在solver.py文件里面

```
python solver.py
```

- solver.py：transformer，conformer的训练和识别
- streamingsolver.py： 基于transformer-Xl实现的一个流式结果的测试和训练
- T_transducer.py：transformer-transducer的训练和识别
- latencysolver.py：rnntransducer的训练和识别
- lmsolver.py：语言模型的训练