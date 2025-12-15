#Copyright (C) 2024, Advanced Micro Devices, Inc. All rights reserved.
#SPDX-License-Identifier: MIT

# you can revise the project name 
set project_name "matrix_mult"

# Create a project
open_project -reset ${project_name}

# Add design files
add_files ${project_name}.cpp
# Add test bench & files
add_files -tb ${project_name}_tb.cpp

# Set the top-level function
set_top ${project_name}

# ########################################################
# Create a solution
open_solution -reset ${project_name} -flow_target vivado

# Define technology and clock rate
set_part  {xc7z020-clg400-1}
create_clock -period 5

# Set variable to select which steps to execute
set hls_exec 3


csim_design
# Set any optimization directives
# End of directives

if {$hls_exec >= 1} {
	# Run Synthesis
   csynth_design
}
if {$hls_exec >= 2} {
	# Run Synthesis, RTL Simulation
   cosim_design
}
if {$hls_exec >= 3} { 
	# Run Synthesis, RTL Simulation, RTL implementation
   #export_design -format ip_catalog -version "1.00a" -library "hls" -vendor "xilinx.com" -description "A memory mapped IP created by Vitis HLS" -evaluate verilog
   export_design -format ip_catalog -ipname ${project_name} -library Vitis_DSP_Library -description "256*256 int8 dsp matrix multiplication" -evaluate verilog
}

exit