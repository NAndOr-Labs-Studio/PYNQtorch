# Set kernel variables
set PROJ "matrix_mult"
set SOLN "sol1"
set DESCRIPTION "256*256 int8 dsp matrix multiplication"

# Set FPGA part and clock period
set XPART xc7z020-clg400-1
set CLKP 5

# Set flags for various stages
set CSIM 1
set CSYNTH 1
set COSIM 1
set PACKAGE 1
set VIVADO_SYN 0
set VIVADO_IMPL 0

# Save current directory
set CUR_DIR [pwd]

# Create a project
open_project -reset ${PROJ}

# Add design files
add_files ${PROJ}.cpp
# Add test bench & files
add_files -tb ${PROJ}_tb.cpp

# Set the top-level function
set_top ${PROJ}

# ########################################################
# Create a solution
open_solution -reset ${SOLN} -flow_target vivado

# Define technology and clock rate
set_part $XPART
create_clock -period $CLKP

if {$CSIM == 1} {
  csim_design
}

if {$CSYNTH == 1} {
  csynth_design
}

if {$COSIM == 1} {
  cosim_design
}

if {$PACKAGE == 1} {
  export_design -rtl verilog -format ip_catalog -ipname ${PROJ} -display_name ${PROJ} -description ${DESCRIPTION}
}

if {$VIVADO_SYN == 1} {
  export_design -flow syn -rtl verilog -format ip_catalog -ipname ${PROJ} -display_name ${PROJ} -description ${DESCRIPTION}
}

if {$VIVADO_IMPL == 1} {
  export_design -flow impl -rtl verilog -format ip_catalog -ipname ${PROJ} -display_name ${PROJ} -description ${DESCRIPTION}
}

exit