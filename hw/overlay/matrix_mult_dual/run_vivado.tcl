# Set project variables
set PROJ "matrix_mult_dual"
set PROJ_DSP "matrix_mult"
set PROJ_FABRIC "matrix_mult_fabric"
set SOLN "sol1"
set DSGN "matrix_mult_dual"

# Set parallel jobs
set JOBS 20

# Create Vivado project
create_project ${PROJ} ./${PROJ} -part xc7z020clg400-1 -force

# Set IP repository path
set_property ip_repo_paths [list \
    "../../kernel/${PROJ_DSP}/${PROJ_DSP}/${SOLN}/impl/ip" \
    "../../kernel/${PROJ_FABRIC}/${PROJ_FABRIC}/${SOLN}/impl/ip" \
] [current_project]

update_ip_catalog

# Create block design
create_bd_design ${DSGN}

create_bd_cell -type ip -vlnv xilinx.com:ip:processing_system7:5.5 processing_system7_0

apply_bd_automation -rule xilinx.com:bd_rule:processing_system7 -config {make_external "FIXED_IO, DDR" Master "Disable" Slave "Disable" }  [get_bd_cells processing_system7_0]

set_property -dict [list CONFIG.PCW_USE_S_AXI_HP0 {1}] [get_bd_cells processing_system7_0]
set_property -dict [list CONFIG.PCW_USE_S_AXI_HP1 {1}] [get_bd_cells processing_system7_0]

create_bd_cell -type ip -vlnv xilinx.com:hls:${PROJ_DSP}:1.0 ${PROJ_DSP}_0
create_bd_cell -type ip -vlnv xilinx.com:hls:${PROJ_FABRIC}:1.0 ${PROJ_FABRIC}_0

apply_bd_automation -rule xilinx.com:bd_rule:axi4 \
  -config "Clk_master {Auto} Clk_slave {Auto} Clk_xbar {Auto} \
           Master {/processing_system7_0/M_AXI_GP0} \
           Slave {/${PROJ_DSP}_0/s_axi_CTRL} \
           ddr_seg {Auto} intc_ip {New AXI Interconnect} \
           master_apm {0}" \
  [get_bd_intf_pins ${PROJ_DSP}_0/s_axi_CTRL]

apply_bd_automation -rule xilinx.com:bd_rule:axi4 \
  -config "Clk_master {Auto} Clk_slave {Auto} Clk_xbar {Auto} \
           Master {/${PROJ_DSP}_0/m_axi_gmem} \
           Slave {/processing_system7_0/S_AXI_HP0} \
           ddr_seg {Auto} intc_ip {New AXI Interconnect} \
           master_apm {0}" \
  [get_bd_intf_pins processing_system7_0/S_AXI_HP0]

apply_bd_automation -rule xilinx.com:bd_rule:axi4 \
  -config "Clk_master {Auto} Clk_slave {Auto} Clk_xbar {Auto} \
           Master {/processing_system7_0/M_AXI_GP0} \
           Slave {/${PROJ_FABRIC}_0/s_axi_CTRL} \
           ddr_seg {Auto} intc_ip {New AXI Interconnect} \
           master_apm {0}" \
  [get_bd_intf_pins ${PROJ_FABRIC}_0/s_axi_CTRL]

apply_bd_automation -rule xilinx.com:bd_rule:axi4 \
  -config "Clk_master {Auto} Clk_slave {Auto} Clk_xbar {Auto} \
           Master {/${PROJ_FABRIC}_0/m_axi_gmem} \
           Slave {/processing_system7_0/S_AXI_HP1} \
           ddr_seg {Auto} intc_ip {New AXI Interconnect} \
           master_apm {0}" \
  [get_bd_intf_pins processing_system7_0/S_AXI_HP1]

validate_bd_design

# Generate output products
generate_target all [get_files  ./${PROJ}/${PROJ}.srcs/sources_1/bd/${DSGN}/${DSGN}.bd]
catch { config_ip_cache -export [get_ips -all ${DSGN}_processing_system7_0_0] }
catch { config_ip_cache -export [get_ips -all ${DSGN}_${PROJ_DSP}_0_0] }
catch { config_ip_cache -export [get_ips -all ${DSGN}_${PROJ_FABRIC}_0_0] }
catch { config_ip_cache -export [get_ips -all ${DSGN}_auto_pc_0] }
catch { config_ip_cache -export [get_ips -all ${DSGN}_rst_ps7_0_50M_0] }
catch { config_ip_cache -export [get_ips -all ${DSGN}_auto_pc_1] }
catch { config_ip_cache -export [get_ips -all ${DSGN}_auto_us_0] }
export_ip_user_files -of_objects [get_files ./${PROJ}/${PROJ}.srcs/sources_1/bd/${DSGN}/${DSGN}.bd] -no_script -sync -force -quiet
create_ip_run [get_files -of_objects [get_fileset sources_1] ./${PROJ}/${PROJ}.srcs/sources_1/bd/${DSGN}/${DSGN}.bd]
launch_runs ${DSGN}_processing_system7_0_0_synth_1 ${DSGN}_${PROJ_DSP}_0_0_synth_1 ${DSGN}_${PROJ_FABRIC}_0_0_synth_1 ${DSGN}_auto_pc_0_synth_1 ${DSGN}_rst_ps7_0_50M_0_synth_1 ${DSGN}_auto_pc_1_synth_1 ${DSGN}_auto_us_0_synth_1 -jobs ${JOBS}
export_simulation -of_objects [get_files ./${PROJ}/${PROJ}.srcs/sources_1/bd/${DSGN}/${DSGN}.bd] -directory ./${PROJ}/${PROJ}.ip_user_files/sim_scripts -ip_user_files_dir ./${PROJ}/${PROJ}.ip_user_files -ipstatic_source_dir ./${PROJ}/${PROJ}.ip_user_files/ipstatic -lib_map_path [list {modelsim=./${PROJ}/${PROJ}.cache/compile_simlib/modelsim} {questa=./${PROJ}/${PROJ}.cache/compile_simlib/questa} {riviera=./${PROJ}/${PROJ}.cache/compile_simlib/riviera} {activehdl=./${PROJ}/${PROJ}.cache/compile_simlib/activehdl}] -use_ip_compiled_libs -force -quiet

# Generate bd wrapper
make_wrapper -files [get_files ./${PROJ}/${PROJ}.srcs/sources_1/bd/${DSGN}/${DSGN}.bd] -top
add_files -norecurse ./${PROJ}/${PROJ}.gen/sources_1/bd/${DSGN}/hdl/${DSGN}_wrapper.v

# Launch synthesis and implementation
launch_runs synth_1 -jobs ${JOBS}
wait_on_run synth_1

launch_runs impl_1 -jobs ${JOBS}
wait_on_run impl_1

launch_runs impl_1 -to_step write_bitstream -jobs ${JOBS}
wait_on_run impl_1

#move and rename bitstream to final location
file copy -force ./${PROJ}/${PROJ}.runs/impl_1/${DSGN}_wrapper.bit ${PROJ}.bit
file copy -force ./${PROJ}/${PROJ}.gen/sources_1/bd/${DSGN}/hw_handoff/${DSGN}.hwh ${PROJ}.hwh