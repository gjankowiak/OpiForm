import OpiForm
import DelimitedFiles

###################
# STATUS ##########
# #################
# Tue 10 Dec 14:42:37 CET 2024
# Modified to re-run µ = 0.1

include("defs.jl")
include("./movies.jl")

function generate_reference(μ)
  params = get_micro_params(μ, 1e-2)
  params = merge(params, (
    init_method_omega=:from_lfr,
    init_method_adj_matrix=:from_lfr
  ))
  reference_dir = get_reference_dir(μ)
  OpiForm.prepare_directory(reference_dir, params, :micro, force=true)
end

function get_reference_dir(μ)
  return "results/$(dir_label)/N_micro=$(N_micro)/reference/μ=$μ"
end

function get_micro_params(μ, β_σ²)

  c_mean_padding = 0.5
  expectation_bounds = (-1 + c_mean_padding, 1 - c_mean_padding)

  base_params = OpiForm.Params.get_default_params()

  base_params = merge(base_params, (
    N_micro=N_micro,
    N_mfl=N_mfl,
  ))

  params = merge(base_params, (
    max_iter=max_iter,
    δt=1e-4,
    store_every_iter=100,
    store_g=false,
    mfl_single_group=false,
    debug_multigroup=false,
    CFL_violation=:abort,
    init_method_omega=:from_lfr_with_ref,
    init_method_adj_matrix=:from_file,
    init_method_f=:from_kde_omega,
    init_method_g=:from_kde_adj_matrix,
    init_micro_filename=joinpath(get_reference_dir(μ), "data.hdf5"),
    init_lfr_communities_dir=get_reference_dir(μ),
    init_lfr_max_tries=3,
    init_lfr_args=(k_mean, k_max),
    init_lfr_kwargs=(mixing_parameter=μ,
      nmin=nmin,
      nmax=nmax,
      μ_community_bounds=expectation_bounds,
      μ_community_distrib=:equidistributed,
      β_σ²=β_σ²,
    ),
    init_lfr_target_n_communities=target_communities
  ))

  return params
end

function main()

  function f(tup; skip_micro::Bool=false, skip_mono::Bool=false)

    μ, β_σ² = tup

    params = get_micro_params(μ, β_σ²)

    params = merge(params, (mfl_single_group=false,))
    mono_prefix = "multi"

    prefix_micro = "$(dir_label)/N_micro=$(N_micro)/σ²=$(β_σ²)/μ=$μ"
    prefix_mfl = "$(dir_label)/N_micro=$(N_micro)/N_mfl=$(N_mfl)/σ²=$(β_σ²)/μ=$μ"

    store_dir_micro = "results/$(prefix_micro)/micro"

    # #
    # Run the micro model
    # #
    params_micro = params
    if !skip_micro
      OpiForm.Micro.launch(store_dir_micro, params_micro; force=true)
    end

    reference_dir = params.init_lfr_communities_dir
    c_ids, c_expectations = OpiForm.load_lfr_community_data(reference_dir)

    # #
    # Run the meanfield model using the initial data and graph of the micro model (with KDE)
    # #
    errs = []
    for single_group in [true, false]
      if skip_mono && single_group
        continue
      end
      params = merge(params, (mfl_single_group=single_group,))
      mono_prefix = single_group ? "mono" : "multi"
      store_dir_mfl = "results/$(prefix_mfl)/meanfield-$(mono_prefix)"

      params_lLF = merge(params, (
        flux=:lLF, f_dependent_g=false,
        f_init_func=OpiForm.Params.build_f_init_func_beta_weighted(; communities=c_ids, community_expectations=c_expectations, σ²=β_σ²),
        init_method_omega=:from_file,
        init_method_adj_matrix=:from_file,
        init_micro_filename=joinpath(store_dir_micro, "data.hdf5"),
      ))
      err = OpiForm.MeanField.launch(store_dir_mfl, params_lLF; force=false)
      push!(errs, merge(err, (mono=single_group,)))
    end
    return errs
  end

  skip_micro = false
  skip_mono = false
  skip_ref = false
  skip_computations = false
  skip_movies = false

  if !skip_computations
    for μ in μs
      if !skip_ref
        generate_reference(μ)
      end

      errs = f((μ, β_σ²); skip_micro=skip_micro, skip_mono=skip_mono)

      for err in errs
        if !err.success
          @error "Computation failed for μ=$(μ) [mono=$(err.mono)]"
        end
      end
    end

    @info "Computations done"
  end

  if !skip_movies
    for μ in μs
      reference_dir = get_reference_dir(μ)
      prefix_micro = "$(dir_label)/N_micro=$(N_micro)/σ²=$(β_σ²)/μ=$μ"
      prefix_mfl = "$(dir_label)/N_micro=$(N_micro)/N_mfl=$(N_mfl)/σ²=$(β_σ²)/μ=$μ"

      c_ids, c_expectations = OpiForm.load_lfr_community_data(reference_dir)
      try
        OpiForm.plot_ω_f_with_single(
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

  @info "All done!"

end

main()
