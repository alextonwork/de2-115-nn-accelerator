#!/bin/sh
# Open-source alternative to ModelSim. Run from anywhere: sh sim/run_iverilog.sh
set -e
cd "$(dirname "$0")"
iverilog -g2005 -Wall -I ../tb/vectors -o tb_mac.vvp ../rtl/mac.v ../tb/tb_mac.v
vvp -n tb_mac.vvp
