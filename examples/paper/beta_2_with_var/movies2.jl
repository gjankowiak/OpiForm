import OpiForm as OF

include("defs.jl")

function create_movies()
  for μ in μs
    reference_dir = get_reference_dir(μ)
    prefix_micro = "$(dir_label)/N_micro=$(N_micro)/σ²=$(β_σ²)/μ=$μ"
    prefix_mfl = "$(dir_label)/N_micro=$(N_micro)/N_mfl=$(N_mfl)/σ²=$(β_σ²)/μ=$μ"

    c_ids, c_expectations = OF.load_lfr_community_data(reference_dir)
    try
      OF.plot_ω_f_with_single(
        "results/$(prefix_micro)/micro",
        "results/$(prefix_mfl)/meanfield-multi",
        "results/$(prefix_mfl)/meanfield-mono";
        community_ids=c_ids, scale_x=false, xlims=[-0.5, 0.5], ylims=[-0.5, 7.0])
    catch ex
      @error "could not generate movie for μ=$μ"
      println(ex)
    end
  end
end

create_movies()
