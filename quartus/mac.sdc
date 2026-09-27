# Timing constraint for synthesizing mac.v standalone (DE2-115 CLOCK_50 = 50 MHz)
create_clock -name clk -period 20.000 [get_ports clk]
derive_clock_uncertainty
