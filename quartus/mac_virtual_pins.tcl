# Quartus Tcl console (View > Utility Windows > Tcl Console) with the project open:
#   source ../quartus/mac_virtual_pins.tcl
# Makes the MAC's data ports "virtual" so a standalone fit doesn't need 90+ real
# pins and doesn't let I/O delay dominate timing. Only clk stays a real pin.
foreach p {rst_n in_valid in_start in_last a[*] b[*] out_done acc[*] result[*]} {
    set_instance_assignment -name VIRTUAL_PIN ON -to $p
}
export_assignments
