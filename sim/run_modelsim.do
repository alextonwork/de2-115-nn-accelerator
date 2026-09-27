# ModelSim (Intel FPGA Starter/Lite Edition) script.
# In the ModelSim transcript:
#   cd <repo>/sim
#   do run_modelsim.do              ;# MAC unit testbench (default)
#   do run_modelsim.do tb_nn_core   ;# full forward pass
#   do run_modelsim.do tb_top       ;# board top level
if {$argc > 0} { set TB $1 } else { set TB tb_mac }

if {[file exists work]} { vdel -lib work -all }
vlib work
vlog -work work ../rtl/mac.v ../rtl/weight_rom.v ../rtl/nn_core.v ../rtl/hex7seg.v ../rtl/de2_115_top.v
vlog -work work +incdir+../tb/vectors ../tb/$TB.v
vsim -voptargs=+acc work.$TB

if {$TB == "tb_mac"} {
    add wave -divider inputs
    add wave sim:/tb_mac/clk sim:/tb_mac/in_valid sim:/tb_mac/in_start sim:/tb_mac/in_last
    add wave -radix decimal sim:/tb_mac/a sim:/tb_mac/b
    add wave -divider pipeline
    add wave -radix decimal sim:/tb_mac/dut/prod_r
    add wave -divider outputs
    add wave sim:/tb_mac/out_done
    add wave -radix decimal sim:/tb_mac/acc sim:/tb_mac/result
} elseif {$TB == "tb_nn_core"} {
    add wave sim:/tb_nn_core/clk sim:/tb_nn_core/start sim:/tb_nn_core/busy sim:/tb_nn_core/done
    add wave -divider fsm
    add wave sim:/tb_nn_core/dut/state sim:/tb_nn_core/dut/layer
    add wave -radix unsigned sim:/tb_nn_core/dut/neuron sim:/tb_nn_core/dut/k sim:/tb_nn_core/dut/rom_addr
    add wave -divider mac
    add wave -radix decimal sim:/tb_nn_core/dut/weight sim:/tb_nn_core/dut/s_act
    add wave sim:/tb_nn_core/dut/mac_done
    add wave -radix decimal sim:/tb_nn_core/dut/mac_result
    add wave -divider results
    add wave -radix decimal sim:/tb_nn_core/dut/hidden sim:/tb_nn_core/dut/logit
    add wave sim:/tb_nn_core/pred
} else {
    add wave sim:/tb_top/KEY sim:/tb_top/SW sim:/tb_top/LEDG sim:/tb_top/HEX0 sim:/tb_top/HEX6
    add wave -radix hex sim:/tb_top/dut/logit
}
run -all
wave zoom full
