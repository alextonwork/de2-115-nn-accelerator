#!/bin/sh
# Open-source alternative to ModelSim. Runs every testbench and fails if any fails.
# Run from anywhere: sh sim/run_iverilog.sh
set -e
cd "$(dirname "$0")"
RTL="../rtl/mac.v ../rtl/weight_rom.v ../rtl/nn_core.v ../rtl/nn_core_par.v ../rtl/hex7seg.v ../rtl/de2_115_top.v ../rtl/de2_115_mnist_top.v ../rtl/conv_pool.v ../rtl/cnn_core.v"
status=0

# run <name> <testbench> [iverilog -P overrides...]
run() {
    name=$1; tb=$2; shift 2
    echo "=== $name"
    iverilog -g2005 -I ../tb/vectors -s $tb "$@" -o $name.vvp $RTL ../tb/$tb.v
    vvp -n $name.vvp | tee $name.log | grep -E "TEST|FAIL|x=|SW1|SW=|SW17|accuracy|latency" || true
    grep -q "TEST PASSED" $name.log || status=1
}

for tb in tb_mac tb_nn_core tb_top tb_mnist tb_conv_pool; do
    run $tb $tb
done

# parallel core: every lane count, 200 images each (the single-MAC run above
# already covers all 1000), then the board top at both ends of the sweep
for n in 1 2 4 8 16 32; do
    run tb_mnist_par$n tb_mnist_par -P tb_mnist_par.N_MAC=$n -P tb_mnist_par.N_CASES=200
done
for n in 1 32; do
    run tb_mnist_top$n tb_mnist_top -P tb_mnist_top.N_MAC=$n
done
# CNN: 200 images keeps CI quick (all 1000 pass: run -P tb_cnn.N_CASES=1000)
run tb_cnn tb_cnn -P tb_cnn.N_CASES=200
exit $status
