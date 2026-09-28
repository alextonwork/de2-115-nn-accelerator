# ---------------------------------------------------------------------------
# sweep_n_mac.tcl  -  compile the MNIST board design once per MAC count and
#                     collect area, memory, multipliers and Fmax into a CSV
#
# From the quartus/ directory (Quartus command prompt or any shell with
# quartus_sh on the PATH):
#
#     quartus_sh -t sweep_n_mac.tcl              (N_MAC = 1 2 4 8 16 32)
#     quartus_sh -t sweep_n_mac.tcl 8 32         (just these)
#
# Writes ../results/n_mac_sweep.csv and keeps each run's summary reports in
# ../results/n_mac_sweep/N<n>/. Then run python/plot_sweep.py for the plot
# and the README table. Each compile takes a few minutes.
#
# The N_MAC override is removed from the .qsf at the end, so a normal GUI
# compile goes back to the default in rtl/de2_115_mnist_top.v.
# ---------------------------------------------------------------------------
load_package flow

set project de2_115_mnist
set ns      {1 2 4 8 16 32}
if {$argc > 0} { set ns $argv }

set out_dir ../results
set csv     $out_dir/n_mac_sweep.csv

# MNIST 196-32-10 latency, same formula as rtl/nn_core_par.v (checked in
# simulation by tb/tb_mnist_par.v)
proc cycles {n} {
    set g1 [expr {32 / $n}]
    set g2 [expr {(10 + $n - 1) / $n}]
    return [expr {$g1*197 + $g2*33 + 9 + (10 - ($g2 - 1)*$n)}]
}

proc read_file {path} {
    set f [open $path r]
    set text [read $f]
    close $f
    return $text
}

# first number after a label in a Quartus report, commas removed ("" if absent)
proc grab {text pattern} {
    if {[regexp -- $pattern $text -> value]} {
        return [string map {, ""} $value]
    }
    return ""
}

file mkdir $out_dir
set rows {}

project_open $project -revision $project

foreach n $ns {
    puts "=================== N_MAC = $n ==================="
    set_parameter -name N_MAC $n
    export_assignments

    if {[catch {execute_flow -compile} err]} {
        puts "Compile failed for N_MAC = $n: $err"
        lappend rows "$n,[cycles $n],,,,,,,,compile failed"
        continue
    }

    set fit [read_file output_files/$project.fit.rpt]
    set sta [read_file output_files/$project.sta.rpt]

    set les   [grab $fit {Total logic elements\s*;\s*([\d,]+)}]
    set regs  [grab $fit {Total registers\s*;\s*([\d,]+)}]
    set m9k   [grab $fit {M9Ks\s*;\s*([\d,]+)}]
    set mbits [grab $fit {Total memory bits\s*;\s*([\d,]+)}]
    set mul9  [grab $fit {Embedded Multiplier 9-bit elements\s*;\s*([\d,]+)}]

    # Fmax from the slow (worst-case) 85 C corner, the number Quartus signs off on
    set fmax ""
    set rfmax ""
    set i [string first "Slow 1200mV 85C Model Fmax Summary" $sta]
    if {$i >= 0} {
        regexp -- {;\s*([\d.]+) MHz\s*;\s*([\d.]+) MHz\s*;\s*CLOCK_50} \
            [string range $sta $i end] -> fmax rfmax
    }
    set slack ""
    set i [string first "Slow 1200mV 85C Model Setup Summary" $sta]
    if {$i >= 0} {
        regexp -- {;\s*CLOCK_50\s*;\s*(-?[\d.]+)} [string range $sta $i end] -> slack
    }

    puts "N_MAC=$n  LEs=$les  regs=$regs  M9K=$m9k  mem_bits=$mbits  mult9=$mul9  Fmax=$fmax MHz  slack=$slack ns"
    lappend rows "$n,[cycles $n],$les,$regs,$m9k,$mbits,$mul9,$fmax,$rfmax,$slack"

    set keep $out_dir/n_mac_sweep/N$n
    file mkdir $keep
    foreach r {fit.summary sta.summary map.summary} {
        if {[file exists output_files/$project.$r]} {
            file copy -force output_files/$project.$r $keep/
        }
    }
}

set_parameter -name N_MAC -remove
export_assignments
project_close

set f [open $csv w]
puts $f "n_mac,cycles,logic_elements,registers,m9k,memory_bits,mult_9bit,fmax_mhz,restricted_fmax_mhz,setup_slack_ns_at_50mhz"
foreach r $rows { puts $f $r }
close $f
puts "wrote $csv"
