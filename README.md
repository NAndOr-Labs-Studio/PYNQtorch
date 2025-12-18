# PYNQtorch

## Overview
A set of steps to build PyTorch on PYNQ-support devices.

## Project Structure
```
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
└── README.md
```

## Hardware Requirements
- TUL PYNQ-Z2
- MicroSD Card

## Software Requirements
- PYNQ v3.1.1
- ...

## Getting Started
### Quick start guide
1. upload files to PYNQ board
2. open Jupyter Notebook
3. ...

## Building from Source
### HLS
```cmd
cd hw/kernel/<kernel_name>
vitis-run --mode hls --tcl run_hls.tcl
```
### Vivado
```cmd
cd hw/overlay/<overlay_name>
vivado -mode batch -source run_vivado.tcl
```

## Performance
...

## References
...

## License
GPL-2.0 license