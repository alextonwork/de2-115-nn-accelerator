create_clock -name CLOCK_50 -period 20.000 [get_ports CLOCK_50]
derive_clock_uncertainty
# Switches, keys, LEDs and 7-segment are human-speed: no timing requirement.
set_false_path -from [get_ports {SW[*] KEY[*]}]
set_false_path -to   [get_ports {LEDG[*] LEDR[*] HEX*}]
