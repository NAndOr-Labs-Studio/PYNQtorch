# PYNQtorch

[English](README.md)

## 概述

这是一套在 PYNQ 支持的设备上（特别是 ARM-v7l 架构）构建基于 HLS 加速的 PyTorch 的步骤。

该项目主要作为从事 AI 硬件工作的人员的学习和验证平台。

`hw/`和`backend/`文件夹提供了一个简陋的实现，展示了如何以简单的方式在 Python 中注册 torch 后端。这非常适合 HLS 算法的短期验证。

## 项目结构

```text
PYNQtorch/
├── notebook/
│   ├── backend_example/                # 后端示例
│   │   ├── example_EN.ipynb            # 英文版后端示例
│   │   └── example_CN.ipynb            # 中文版后端示例
│   └── hw_test/                        # 硬件测试
│       └── matrix_mult.ipynb           # Jupyter notebook
├── hw/
│   ├── kernel/                         # HLS 内核
│   │   └── <kernel_name>/
│   │       ├── <kernel_name>.cpp       # 内核源码
│   │       ├── <kernel_name>_tb.cpp    # 内核测试平台
│   │       ├── <kernel_name>.hpp       # 内核头文件
│   │       ├── description.json        # 内核描述
│   │       ├── hls_config.cfg          # HLS 配置
│   │       └── run_hls.tcl             # HLS Tcl 脚本
│   └── overlay/                        # Vivado 覆盖层
│       └── <overlay_name>/
│           ├── *.bit                   # FPGA 比特流
│           ├── *.hwh                   # FPGA 硬件交接文件
│           └── run_vivado.tcl          # 综合与实现 Tcl 脚本
├── sw/
│   ├── pytorch (submodule)             # PyTorch v2.8
│   ├── audio   (submodule)             # TorchAudio v2.8
│   └── vision  (submodule)             # TorchVision v0.23
├── backend/
│   └── pytorch_zynq/                   # 自定义后端示例
│       ├── __init__.py                 # 入口
│       ├── device.py                   # 属性定义
│       ├── hardware.py                 # PYNQ 覆盖层
│       ├── linear.py                   # 自定义线性绑定
│       └── ops.py                      # 适配软件算子到硬件
├── scripts/
│   └── environments.sh                 # 设置编译标志
├── LICENSE
└── README.md
```

## 硬件要求

- PYNQ Z2 (典型)

## 软件要求

- PYNQ v3.1.1
- ENVs (如果你想自己构建)

## 快速入门

### 快速入门指南

1. 将此仓库克隆到您的板卡上
2. 将 `backend/pytorch_zynq/` 复制到您的工作区
3. 导入后端并开始使用

## 从源码构建

从源码构建可能需要一些耐心才能成功。不过，硬件部分应该很简单，所以从硬件构建开始吧。

### HLS 部分

```cmd
cd hw/kernel/<kernel_name>
vitis-run --mode hls --tcl run_hls.tcl
```

### Vivado

```cmd
cd hw/overlay/<overlay_name>
vivado -mode batch -source run_vivado.tcl
```

### 软件构建

看起来构建 PyTorch 有点小菜一碟，但这肯定会花你一些时间！

确保你使用的是 Linux。编译在 Windows 上不起作用。

1. 更新你的源

   ```cmd
   sudo apt update
   sudo apt upgrade
   ```

2. 安装 qemu-user-static 以支持 binfmt 和其他依赖项

    ```cmd
    sudo apt install qemu-user-static debootstrap -y
    ```

3. 在 ARM-v7l 中制作 chroot

    使用 Trixie 是合理的，我不想解释这个。

   ```cmd
   sudo debootstrap --arch=armhf trixie /path/to/your/chroot
   ```

4. 运行 chroot

    ```cmd
    sudo mount -t sysfs sysfs /path/to/your/chroot/sys
    sudo mount -t proc proc /path/to/your/chroot/proc
    sudo mount --bind /dev /path/to/your/chroot/dev
    sudo chroot /path/to/your/chroot
    ```

5. 复制你的 sysroot

    `Trixie` 发行版中的 `GLIBC` 库太新了，而且我们需要板卡上的一些库以防万一。然后把它们都放到一个叫 "`sysroot`" 的文件夹里。

    你只需要下载（或复制）下面的文件夹：

    ```text
    /root/sysroot/ (推荐路径)
    ├── lib/                            # 板卡上的 /lib 路径
    └── usr/
        ├── include/                    # 板卡上的 /usr/include 路径
        └── lib/                        # 板卡上的 /usr/lib 路径
    ```

6. 修补你的 ARM-v7l 容器

    获取你需要的所有包并克隆此仓库。

    注意 PyTorch 推荐 **GCC 11**。较新的编译器可能会导致一系列 bug。

    将你的 apt 镜像发布版本替换为 `sid`，否则你会发现 gcc-11 缺失，哈哈。

    获取 `pyenv`，[点击这里查看如何安装](https://github.com/pyenv/pyenv)。

    通过 `pyenv` 安装 python 3.10.4

    ```cmd
    pyenv install 3.10.4
    pyenv global 3.10.4
    ```

    随意使用虚拟环境，用不用都行。

    然后，你需要初始化 torch 源代码，因为有很多子模块。

    ```cmd
    $(this repository) git submodule init --recursive
    ```

    这可能需要一段时间，喝杯茶吧。

    在你完全克隆源代码后，运行 `pip install -r requirements.txt` 来适配 torch 需要的所有依赖项，一些 pip 包将需要一些时间从源码编译。在那之前打发点时间吧。

    当所有提到的都搞定后，你现在正在运行 torch 编译前的最后一步。

    ```cmd
    $(this repository) source scripts/environments.sh
    ```

    这将创建一组典型的编译环境标志，不过你需要确保一些性能标志和依赖路径（例如 `sysroot`）适合你的系统。

7. 编译 torch

    ```cmd
    $(this repository) python sw/pytorch/setup.py bdist_wheel
    ```

    如果没有其他错误，编译后将创建 torch 的二进制 wheel 包。你可能需要自己修复一些牛魔的错误。

8. 编译 vision 和 audio 库（可选）

    过程与 torch 编译类似。使用 `environments.sh` 来帮助你。

## 笔记本

我们提供了一些 Jupyter notebook 来帮助您入门。您可以在 `notebook/` 文件夹中找到它们。

请注意，您必须创建自己的工作区来运行 notebook，包括将 `backend/pytorch_zynq/` 和硬件配置文件复制到正确的位置。这不是一个开箱即用的解决方案。

## 关于后端性能

我们知道 ZYNQ 系列的性能各不相同。在典型的 ZYNQ 7020（-1 速度等级）上，硬件矩阵乘法器平均提供 22 倍的加速率。随意探索不同设备上的性能。

## 许可证

AGPL-3.0 许可证
