# PYNQtorch

[简体中文](README_CN.md)

## Overview

A set of steps to build HLS-based-accelerated PyTorch on PYNQ-support devices, especially on ARM-v7l architecture.

The project is mainly used as a study and verification platform for those who are working on AI hardware.

`hw/` and `backend/` folder gives a shabby implementation for how to register a torch backend in python in an easy way. It's very suitable for short-term verification for HLS algorithm.

## Project Structure

```text
PYNQtorch/
├── notebook/
|   ├── backend_example/                # backend example
|   |   ├── example_EN.ipynb            # backend example in English
|   |   └── example_CN.ipynb            # backend example in Chinese
|   └── hw_test/                        # hardware test
│       └── matrix_mult.ipynb           # Jupyter notebook
├── hw/
│   ├── kernel/                         # HLS kernel
|   |   └── <kernel_name>/
│   │       ├── <kernel_name>.cpp       # kernel source
│   │       ├── <kernel_name>_tb.cpp    # kernel testbench
│   │       ├── <kernel_name>.hpp       # kernel header
│   │       ├── description.json        # kernel description
|   │       ├── hls_config.cfg          # HLS config
|   │       └── run_hls.tcl             # HLS Tcl
│   └── overlay/                        # Vivado overlay
|       └── <overlay_name>/
│           ├── *.bit                   # FPGA bitstream
│           ├── *.hwh                   # FPGA hardware handoff
│           └── run_vivado.tcl          # synth & impl Tcl
├── sw/
|   ├── pytorch (submodule)             # PyTorch v2.8
|   ├── audio   (submodule)             # TorchAudio v2.8
|   └── vision  (submodule)             # TorchVision v0.23
├── backend/
|   └── pytorch_zynq/                   # custom backend example
|       ├── __init__.py                 # entrance
|       ├── device.py                   # attribute definition
|       ├── hardware.py                 # PYNQ overlay layer
|       ├── linear.py                   # custom linear binding
|       └── ops.py                      # fitting sw operators to hw
├── scripts/
|   └── environments.sh                 # set compilation flags
├── LICENSE
└── README.md
```

## Hardware Requirements

- PYNQ Z2 (typical)

## Software Requirements

- PYNQ v3.1.1
- ENVs (if you want to build by yourself)

## Getting Started

### Quick start guide

1. clone this repository to your board
2. copy `backend/pytorch_zynq/` to your workspace
3. import backend & play

## Building from Source

Building from source may require some patience to succeed. However, hardware part should be easy so, start with hardware building.

### HLS part

```cmd
cd hw/kernel/<kernel_name>
vitis-run --mode hls --tcl run_hls.tcl
```

### Vivado

```cmd
cd hw/overlay/<overlay_name>
vivado -mode batch -source run_vivado.tcl
```

### Software building

It seems like building a PyTorch is somewhat a cake, but it will take you some time for sure!

Make sure you are using Linux. The compilation is not working on Windows.

1. Update your source

   ```cmd
   sudo apt update
   sudo apt upgrade
   ```

2. Install qemu-user-static for binfmt and other dependencies

    ```cmd
    sudo apt install qemu-user-static debootstrap -y
    ```

3. Make a chroot in ARM-v7l

    Using Trixie is reasonable, I don't want to explain that.

   ```cmd
   sudo debootstrap --arch=armhf trixie /path/to/your/chroot
   ```

4. Run chroot

    ```cmd
    sudo mount -t sysfs sysfs /path/to/your/chroot/sys
    sudo mount -t proc proc /path/to/your/chroot/proc
    sudo mount --bind /dev /path/to/your/chroot/dev
    sudo chroot /path/to/your/chroot
    ```

5. Copy your sysroot

    The `GLIBC` lib in `Trixie` release is too new, and we also need some libraries in your board in case of something unsuitable. Then put all of them into a folder called "`sysroot`".

    You just need to download (or copy) the folders below:

    ```text
    /root/sysroot/ (path recommended)
    ├── lib/                            # path /lib on your board
    └── usr/
        ├── include/                    # path /usr/include on your board
        └── lib/                        # path /usr/lib on your board
    ```

6. Patch your ARM-v7l container

    Get all package you need and clone this repository.

    Note that PyTorch recommends **GCC 11**. A newer compiler could lead to a series of bugs.

    Replace your apt mirror release to `sid`, or you will find gcc-11 is missing, lol.

    Get `pyenv`, [click here to find how to install](https://github.com/pyenv/pyenv).

    Install python 3.10.4 by `pyenv`

    ```cmd
    pyenv install 3.10.4
    pyenv global 3.10.4
    ```

    Feel free to use virtual environment.

    Then, you need to init torch source code as there are lots of submodules.

    ```cmd
    $(this repository) git submodule init
    $(this repository) git submodule update --init --recursive
    ```

    This may take a while, drink some tea then.

    After you totally clone the source code, run `pip install -r requirements.txt` to fit all dependencies the torch need, and some pip packages will take some time to compile from source. Kill time before that done.

    When all mentioned were taken, you are now running the last step before torch compilation.

    ```cmd
    $(this repository) source scripts/environments.sh
    ```

    This will create a set of typical compiling environment flags, you need to make sure that some performance flags and dependency paths (e.g. `sysroot`) suit your system, though.

7. Compile torch

    ```cmd
    $(this repository) python sw/pytorch/setup.py bdist_wheel
    ```

    This will create a binary wheel of torch after compilation, if there is no other errors. You may need to fix some cute errors by yourself.

8. Compile vision and audio library (optional)

    The procedure is similar to torch compilation. Use `environments.sh` to help you.

## Notebooks

We provide some Jupyter notebooks to help you get started. You can find them in the `notebook/` folder.

Note that you must need to create your own workspace to run the notebooks, including copy `backend/pytorch_zynq/` and hardware config files to the right place. It's not an open-box solution.

## About backend performance

We know ZYNQ series vary in performance. On the typical ZYNQ 7020 (-1 speed), the hardware matrix multiplier gives a 22x acceleration rate on average. Feel free to explore the performance on different devices.

## License

AGPL-3.0 license

