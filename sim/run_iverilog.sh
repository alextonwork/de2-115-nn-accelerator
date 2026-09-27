#!/bin/sh
# Open-source alternative to ModelSim. Runs every testbench and fails if any fails.
# Run from anywhere: sh sim/run_iverilog.sh
set -e
cd "$(dirname "$0")"
RTL="../rtl/mac.v ../rtl/weight_rom.v ../rtl/nn_core.v ../rtl/hex7seg.v ../rtl/de2_115_top.v"
status=0
for tb in tb_mac tb_nn_core tb_top; do
    echo "=== $tb"
    iverilog -g2005 -I ../tb/vectors -o $tb.vvp $RTL ../tb/$tb.v
    vvp -n $tb.vvp | tee $tb.log | grep -E "TEST|FAIL|x=|SW1" || true
    grep -q "TEST PASSED" $tb.log || status=1
done
exit $status
