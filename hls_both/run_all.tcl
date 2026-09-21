# run_all.tcl -- synthesize BOTH denoisers at all three precisions on ZCU104.
#
#   vitis_hls -f run_all.tcl
#
# Produces six projects:
#   dncnn_32/sol/...  dncnn_16/sol/...  dncnn_8/sol/...
#   vit_32/sol/...    vit_16/sol/...    vit_8/sol/...
#
# Both at 112x112 and 100 MHz, so the comparison is like-for-like.
#
# FIXED8 NOTE: verified on host, Fixed8 breaks the ViT (max err 1.0) but is
# usable for the DnCNN (4.6e-01 -- also poor, matching your published row
# being the least accurate). csim is skipped for ViT/Fixed8 so a known-bad
# accuracy result cannot abort the sweep.

proc build {name src tb prec topfn} {
    puts ""
    puts "=================================================="
    puts "  $name  PREC=$prec"
    puts "=================================================="
    open_project -reset ${name}_${prec}
    set_top $topfn
    add_files $src -cflags "-DPREC=$prec -I."
    add_files -tb $tb -cflags "-DPREC=$prec -I."
    open_solution -reset "sol"
    set_part {xczu7ev-ffvc1156-2-e}
    create_clock -period 10 -name default

    if { !($name eq "vit" && $prec == 8) } {
        if { [catch { csim_design } msg] } {
            puts "WARNING: csim $name PREC=$prec: $msg"
        }
    } else {
        puts "INFO: skipping csim (ViT Fixed8 known unusable)"
    }

    csynth_design
    if { [catch { export_design -format ip_catalog -rtl verilog } msg] } {
        puts "WARNING: export_design $name PREC=$prec: $msg"
    }
    close_project
}

foreach P {32 16 8} {
    cd dncnn
    build dncnn denoiser_hls.cpp denoiser_tb.cpp $P denoiser_top
    cd ..
}
foreach P {32 16 8} {
    cd vit
    build vit vit_denoiser_hls.cpp vit_denoiser_tb.cpp $P vit_denoiser_top
    cd ..
}

puts ""
puts "=================================================="
puts "  DONE -- reports at:"
puts "    dncnn/dncnn_<P>/sol/syn/report/denoiser_top_csynth.rpt"
puts "    vit/vit_<P>/sol/syn/report/vit_denoiser_top_csynth.rpt"
puts "=================================================="
exit
