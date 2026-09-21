# run_dncnn_only.tcl -- synthesize ONLY the DnCNN denoiser, all precisions.
# Run from the hls_both/dncnn folder:
#     cd C:\vit_hls\hls_both\dncnn
#     vitis-run --mode hls --tcl ../run_dncnn_only.tcl
#
# csim is skipped: correctness already gated at 7.7486e-07 PASS on this
# machine. Synthesis is what Table III needs. Same part and clock as the
# ViT run, so the two are directly comparable.

foreach P {32 16 8} {
    puts ""
    puts "=================================================="
    puts "  DnCNN denoiser, PREC=$P"
    puts "=================================================="

    open_project -reset dncnn_$P
    set_top denoiser_top
    add_files denoiser_hls.cpp -cflags "-DPREC=$P -I."
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
puts "    dncnn_32/sol/syn/report/denoiser_top_csynth.rpt"
puts "    dncnn_16/sol/syn/report/denoiser_top_csynth.rpt"
puts "    dncnn_8/sol/syn/report/denoiser_top_csynth.rpt"
puts "=================================================="
exit
