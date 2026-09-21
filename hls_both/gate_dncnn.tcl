# quick csim-only gate for the DnCNN -- run from the dncnn/ folder
open_project -reset gate_dn
set_top denoiser_top
add_files denoiser_hls.cpp -cflags "-DPREC=32 -I."
add_files -tb denoiser_tb.cpp -cflags "-DPREC=32 -I."
open_solution -reset "sol"
set_part {xczu7ev-ffvc1156-2-e}
create_clock -period 10 -name default
csim_design
exit
