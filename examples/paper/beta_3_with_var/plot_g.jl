import OpiForm

include("defs.jl")

function create_g_plots()
  for μ in μs
    prefix_mfl = "$(dir_label)/N_micro=$(N_micro)/N_mfl=$(N_mfl)/σ²=$(β_σ²)/μ=$μ"
    mfl_mono_dir = "results/$(prefix_mfl)/meanfield-mono"

    @info "[plot_g] directory $(mfl_mono_dir)"

    OpiForm.plot_g_init_multi(mfl_mono_dir; g_max=25.0)
  end
end

create_g_plots()
