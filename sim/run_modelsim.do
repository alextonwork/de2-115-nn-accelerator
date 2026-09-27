# ModelSim (Intel FPGA Starter/Lite Edition) script.
# In the ModelSim transcript:  cd <repo>/sim   then   do run_modelsim.do
if {[file exists work]} { vdel -lib work -all }
vlib work
vlog -work work ../rtl/mac.v
vlog -work work +incdir+../tb/vectors ../tb/tb_mac.v
vsim -voptargs=+acc work.tb_mac
add wave -divider inputs
add wave sim:/tb_mac/clk sim:/tb_mac/in_valid sim:/tb_mac/in_start sim:/tb_mac/in_last
add wave -radix decimal sim:/tb_mac/a sim:/tb_mac/b
add wave -divider pipeline
add wave -radix decimal sim:/tb_mac/dut/prod_r
add wave -divider outputs
add wave sim:/tb_mac/out_done
add wave -radix decimal sim:/tb_mac/acc sim:/tb_mac/result
run -all
wave zoom full
