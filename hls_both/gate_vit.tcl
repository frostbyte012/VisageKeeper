# quick csim-only gate for the ViT -- run from the vit/ folder
open_project -reset gate_vit
set_top vit_denoiser_top
add_files vit_denoiser_hls.cpp -cflags "-DPREC=32 -I."
add_files -tb vit_denoiser_tb.cpp -cflags "-DPREC=32 -I."
open_solution -reset "sol"
set_part {xczu7ev-ffvc1156-2-e}
create_clock -period 10 -name default
csim_design
exit
