# Just-In-Time Checkpointing: Low Cost Error Recovery from Deep Learning Training Failures

!!! abstract
    ![](fig/author-list.png)

    * Paper: [Just-In-Time Checkpointing: Low Cost Error Recovery from Deep Learning Training Failures](https://dl.acm.org/doi/10.1145/3627703.3650085)

    * **EuroSys'24**

## 介绍

### 摘要

当前深度学习训练过程中需要使用大量 GPU 来处理大量数据，在运行的数周甚至数月时间中可能遇到硬件或软件故障。传统的周期性的检查点技术会在遇到故障时需要很长时间进行恢复，本文提出了一种 just-in-time（下称 JIT） 的检查点技术，支持故障时仅通过一个 minibatch 的迭代就恢复到故障前的训练阶段。

### 引言与背景

当前的故障恢复策略基本都基于周期性的写检查点，即在遇到故障时将训练状态恢复到之前的某个检查点状态。周期性的检查点需要在训练过程中消耗资源来保存状态，同时在恢复过程中也需要较长的 GPU 等待以及重新训练到原来状态的时间。作者在实践过程中观察到 GPU 集群中的故障很可能是**单个 GPU 或网络设备**的错误导致的，主机/CPU 以及同时多节点故障是极少数的情况。

<!-- 关注到用于训练的内存静默数据损坏 (Silent Data Corruption, SDC) 的比例很低，仅有 0.61%～1.76% 的 SDC 会造成故障，周期性的检查点仅在 -->

Just-In-Time checkpointing（下简称 JIT 或 JIT ckpt）希望仅在发生故障时写检查点，而不是周期性的进行检查点；同时对于常见的故障可以做到只重新进行一个 minibatch 的训练而恢复工作。要达到上述目的需要以下前提：训练过程是迭代式的，仅在每轮迭代的一小段时间里对模型状态进行修改；训练采取 DP 而 DP 会在不同的 GPU 上冗余保存模型状态，因此可以从其他 GPU 处获取丢失的状态。同时为了保障多机器同时出错导致无法使用 JIT 恢复的情况，可以设置一个低频率的周期性写检查点。

## 用户层 JIT ckpt

需要用户初始化 JIT 库并提供一个 `save_checkpoint` 函数供 JIT 库调用。

实现 JIT 最需要处理的两个问题是**监测故障**以及在故障后能够访问到**一致未损坏的 GPU 状态**。从检查故障到恢复的一般流程是：

1. 在聚合通讯过程中检查错误，一个 rank 有问题会导致其他几个正常 rank 遇到错误。
2. 监测到错误时，每个正常的 rank 会将其 CPU 状态拷贝到对应的检查点文件中。
3. scheduler 会被正常的 rank 通知，当至少一个检查点完成时，scheduler 会关闭当前任务并不在后边任务中使用故障的 GPU。
4. 在重启过程中，正常的 GPU 恢复到检查点状态并进行下一轮 minibatch 处理。

<!-- 有点太工程了，搁置 -->