# OpenTinyGPT

欢迎来到我们的OpenTinyGPT项目！这是我们的第二个项目，旨在通过实现一个简单的GPT范式: *next token prediction* 来帮助大家学习
Transformer模型，同时对于文本生成有一个基本的了解。

## 目录

- [项目简介](#项目简介)
- [任务要求](#任务要求)
- [快速开始](#快速开始)
- [技术背景](#技术背景)

## 项目简介

本项目的核心任务是，训练一个模型，当我们向模型输入一个指定的文本（如：`"The Month after July is"`）时，模型能够生成一个与输入文本相关的文本（如：`"The Month after July is August, this month is in summer ..."`）。大家需要在项目框架中实现GPT-style的模型，同时在训练完成后给出generate的实现。我们提供的语料，编码和目前的
加载框架都是非常naive的，大家可以按照自己的需求随意地修改目前的codebase，鼓励大家自己完全从0开始搭建框架。

### 项目基本架构
```
OpenTinyGPT/
├── config.py
├── model.py
├── train.py
├── generate.py 
└── README.md
```
## 任务要求

1.  **理解代码**: 你可以根据项目的codebase来修改 `model.py` 和 `train.py` 中的算法代码以实现训练需求，或者自己搭建框架。
2.  **模型训练**: 运行训练脚本 `train.py`，训练你自己的 GPT 模型。
3.  **条件生成**: 运行 `generate.py` 脚本，加载你训练好的模型，生成指定文本的相关文本，并观察生成效果。
4.  **以pr的方式提交你的项目到我们的主仓库**:


## 快速开始

1.  **创建项目**: (在此之前希望大家熟悉[git](https://liaoxuefeng.com/books/git/introduction/index.html))
    ```bash
    git clone https://github.com/MicrosoftClub-Xidian-Research/OpenTinyGPT.git
    cd OpenTinyGPT
    ```

2.  **创建环境**: 希望大家已经熟悉用[conda](https://zhuanlan.zhihu.com/p/94744929)创建环境了。

3.  **安装依赖**: 建议创建一个虚拟环境。
    ```bash
    conda create -n tinygpt python=3.10
    conda activate tinygpt
    pip install -r requirements.txt
    ```

4.  **训练模型**:
    ```bash
    # 这将开始训练，并会实时反馈当前的loss/val loss，最后存储vocab和model.
    python train.py
    ```

5.  **生成文本**:
    ```bash
    # 待训练完成后，运行此脚本查看生成效果，需要添加必要的参数.
    python generate.py
    ```

## 技术背景

### Transformer

**简介**:
伟大无需多言

**推荐阅读**:
*   **论文**: *Attention is All you need* - [https://arxiv.org/abs/1706.03762](https://arxiv.org/abs/1706.03762)  
*   **博客**: *Transformer 模型原理* - [https://zhuanlan.zhihu.com/p/338817680](https://zhuanlan.zhihu.com/p/338817680)
*   **博客**: *Next Token Prediction* - [https://www.cnblogs.com/wzzkaifa/p/19086539.html](https://www.cnblogs.com/wzzkaifa/p/19086539.html)
