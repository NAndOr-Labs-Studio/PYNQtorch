# PYNQtorch

## Overview

A set of steps to build HLS-based-accelerated PyTorch on PYNQ-support devices, especially on ARM-v7l architecture.

The project is mainly used as a study and verification platform for those who is working on AI hardware.

hw folder gives a shabby implementation for how to regist a torch backend in python in a easy way. It's very suitable for short-term verification for HLS algorithm.

## Project Structure

```text
PYNQtorch/
├── notebook/
│   ├── *.ipynb                         # Jupyter notebook
│   ├── *.bit                           # FPGA bitstream
│   ├── *.hwh                           # FPGA hardware handoff
│   └── [others]                        # others
├── hw/
│   ├── kernel/                         # HLS kernel
|   |   └── <kernel_name>/
│   │       ├── <kernel_name>.cpp       # kernel source
│   │       ├── <kernel_name>_tb.cpp    # kernel testbench
│   │       ├── <kernel_name>.hpp       # kernel header
│   │       ├── description.json        # kernel description
│   │       ├── hls_config.cfg          # HLS congfig
│   │       └── run_hls.tcl             # HLS Tcl
│   └── overlay/                        # Vivado overlay
|       └── <overlay_name>/
│           ├── block_design.tcl        # Block Design Tcl
│           └── run_vivado.tcl          # synth & impl Tcl
├── sw/
|   ├── pytorch (submodule)             # PyTorch v2.8
|   ├── audio   (submodule)             # TorchAudio v2.8
|   └── vision  (submodule)             # TorchVision v0.23
├── backend/
|   └── pytorch_zynq                    # custom backend example
|       ├── __init__.py                 # entrance
|       ├── device.py                   # attribute define
|       ├── hardware.py                 # PYNQ overlay layer
|       ├── linear.py                   # custom linear binding
|       └── ops.py                      # fitting layer to hw
├── LICENSE
└── README.md
```

## Hardware Requirements

- PYNQ Z2 (typical)

## Software Requirements

- PYNQ v3.1.1
- ENVs (if you want to build by youself)

## Getting Started

### Quick start guide

1. clone this repository to your board
2. copy backend/pytorch_zynq to your workspace
3. import backend & play

## Building from Source

Builing from source may take some patience to success. However, hardware part should be easy so, grep hardware building on the start.

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

#### Environment preparation

Make sure you are using Linux. The compilation is not working on Windows.

1. update your source

   ```cmd
   sudo apt update & sudo apt upgrade
   ```

2. install qemu-user-static for binfmt and other dependencies

    ```cmd
    sudo apt install qemu-user-static debootstrap -y
    ```

3. make a chroot in ARM-v7l

   ```cmd
   sudo debootstrap --arch=armhf trxie /path/to/your/chroot
   ```

4. chroot

    ```cmd
    sudo mount -t sysfs sysfs /path/to/your/chroot/sys
    sudo mount -t proc proc /path/to/your/chroot/proc
    sudo mount --bind /dev /path/to/your/chroot/dev
    sudo chroot /path/to/your/chroot
    ```

5. patch your ARM-v7l container

    Get all package you need and clone this repository.

    Replace your apt mirror to `sid`, or you will find gcc-11 is missing, lol.

    Get `pyenv`, [click here to find how to install](https://github.com/pyenv/pyenv).

    Install python 3.10.4 by `pyenv`

    ```cmd
    pyenv install 3.10.4
    pyenv global 3.10.4
    ```

    init torch source code

    ```cmd
    $(this repository) git submodule init --recursive
    ```

    This may take a while, drink some tea then.

...

## Performance

...

## References

...

## License

AGPL-3.0 license
