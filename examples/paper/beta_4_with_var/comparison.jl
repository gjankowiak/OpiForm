import OpiForm as OF

include("defs.jl")

function perform_comparison()
  micro_dirs = String[]
  mfl_dirs = String[]
  mfl_mono_dirs = String[]

  for μ in μs
    prefix_micro = "$(dir_label)/N_micro=$(N_micro)/σ²=$(β_σ²)/μ=$μ"
    prefix_mfl = "$(dir_label)/N_micro=$(N_micro)/N_mfl=$(N_mfl)/σ²=$(β_σ²)/μ=$μ"

    push!(micro_dirs, "results/$(prefix_micro)/micro")
    push!(mfl_dirs, "results/$(prefix_mfl)/meanfield-multi")
    push!(mfl_mono_dirs, "results/$(prefix_mfl)/meanfield-mono")
  end

  OF.compare_variance_lfr_multi(micro_dirs, mfl_dirs, mfl_mono_dirs;
    t_max=8.0, cutoff_factor=0.1, backend=:cairo, stddev_min=1e-3, show_rates=true, t_min_rate=2.5, t_max_rate=8)
end

perform_comparison()

