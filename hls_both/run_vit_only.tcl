# run_vit_only.tcl -- synthesize ONLY the ViT denoiser, all three precisions.
# Run from the hls_both/vit folder:
#     cd C:\vit_hls\hls_both\vit
#     vitis-run --mode hls --tcl ../run_vit_only.tcl
#
# The DnCNN results are already done -- no need to rebuild them.
#
# csim is skipped entirely here: correctness was already gated
# (8.3697e-04 PASS at Float32 on this machine), and ViT csim costs ~22 min
# per precision in debug mode. Synthesis is what Table III needs.

foreach P {32 16 8} {
    puts ""
    puts "=================================================="
    puts "  ViT denoiser, PREC=$P"
    puts "=================================================="

    open_project -reset vit_$P
    set_top vit_denoiser_top
    add_files vit_denoiser_hls.cpp -cflags "-DPREC=$P -I."
    open_solution -reset "sol"
    set_part {xczu7ev-ffvc1156-2-e}
    create_clock -period 10 -name default

    csynth_design

    if { [catch { export_design -format ip_catalog -rtl verilog } msg] } {
        puts "WARNING: export_design PREC=$P: $msg"
    }
    close_project
}

puts ""
puts "=================================================="
puts "  DONE -- reports at:"
puts "    vit_32/sol/syn/report/vit_denoiser_top_csynth.rpt"
puts "    vit_16/sol/syn/report/vit_denoiser_top_csynth.rpt"
puts "    vit_8/sol/syn/report/vit_denoiser_top_csynth.rpt"
puts "=================================================="
exit
