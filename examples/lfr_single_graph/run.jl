using Distributed
import REPL.TerminalMenus as TM

@everywhere import OpiForm
@everywhere import DelimitedFiles
# import OpiForm

function main()
  @everywhere N_micro = 1000
  @everywhere N_mfl = 301

  @everywhere function generate_reference(μ)
    params = get_micro_params(μ, 1e-2)
    params = merge(params, (
      init_method_omega=:from_lfr,
      init_method_adj_matrix=:from_lfr
    ))
    reference_dir = get_reference_dir(μ)
    OpiForm.prepare_directory(reference_dir, params, :micro, force=true)
  end

  @everywhere function get_reference_dir(μ)
    return "results/LFR_single_graph/N_micro=$(N_micro),N_mfl=$(N_mfl)/reference/μ=$μ"
  end

  @everywhere function get_micro_params(μ, β_σ²)
    k_mean = 15
    k_max = 40

    nmin = div(N_micro, 5)
    nmax = div(N_micro, 3)

    c_mean_padding = 0.25
    expectation_bounds = (-1 + c_mean_padding, 1 - c_mean_padding)

    base_params = OpiForm.Params.get_default_params()

    base_params = merge(base_params, (
      N_micro=N_micro,
      N_mfl=N_mfl,
    ))

    params = merge(base_params, (
      max_iter=25000,
      δt=1e-3,
      store_every_iter=100,
      store_g=false,
      init_method_omega=:from_lfr_with_ref,
      init_method_adj_matrix=:from_file,
      init_method_f=:from_f_init,
      init_method_g=:from_kde_adj_matrix,
      init_micro_filename=joinpath(get_reference_dir(μ), "data.hdf5"),
      init_lfr_communities_dir=get_reference_dir(μ),
      init_lfr_args=(k_mean, k_max),
      init_lfr_kwargs=(mixing_parameter=μ,
        nmin=nmin,
        nmax=nmax,
        μ_community_bounds=expectation_bounds,
        μ_community_distrib=:equidistributed,
        β_σ²=β_σ²,
      ),
      init_lfr_target_n_communities=3
    ))

    return params
  end

  @everywhere function f(tuple)

    run_id, μ, β_σ² = tuple

    prefix = "LFR_single_graph/N_micro=$(N_micro),N_mfl=$(N_mfl)/σ²=$(β_σ²)/μ=$μ"
    store_dir_micro = "results/$(prefix)/micro-$(run_id)"

    params = get_micro_params(μ, β_σ²)

    # Run the micro model
    params_micro = params
    OpiForm.Micro.launch(store_dir_micro, params_micro; force=true)
    # try
    #   OpiForm.Micro.launch(store_dir_micro, params_micro; force=true)
    # catch e
    #   @error "Micro run failed"
    #   if e isa String
    #     @error e
    #   else
    #     @error e.msg
    #   end
    #   return
    # end

    reference_dir = params.init_lfr_communities_dir
    c_ids, c_expectations = load_lfr_community_data(reference_dir)

    # Run the meanfield model using the initial data and graph of the micro model (with KDE)
    store_dir_mfl = "results/$(prefix)/meanfield-$(run_id)"
    params_lLF = merge(params, (
      flux=:lLF, f_dependent_g=false,
      f_init_func=OpiForm.Params.build_f_init_func_beta_weighted(; communities=c_ids, community_expectations=c_expectations, σ²=β_σ²),
      init_method_omega=:from_file,
      init_method_adj_matrix=:from_file,
      init_micro_filename=joinpath(store_dir_micro, "data.hdf5"),
    ))
    try
      OpiForm.MeanField.launch(store_dir_mfl, params_lLF; force=true)
    catch e
      @error "MFL run failed"
      if e isa String
        @error e
      else
        @error e.msg
      end
      return
    end
  end

  #β_σ²s = [1e-3, 2e-3, 4e-3, 1.2e-2]

  # μs = [1e-3; 5e-3; 1e-2; range(5e-2, 5e-1, 7)]
  # β_σ²s = [1e-3, 4e-3, 1.2e-2]

  μs = [1e-3; 5e-3; 1e-2; 5e-2; 5e-1]
  β_σ²s = [1e-3, 4e-3, 1.2e-2]

  n_runs = 5

  pmap(generate_reference, μs)
  pmap(f, Iterators.product(1:n_runs, μs, β_σ²s))

  # Postprocessing

  include("./comparison.jl")
  include("./plot_g.jl")

  perform_comparison()
  create_g_plots()


  options = ["no", "yes"]
  menu = TM.RadioMenu(options)
  choice = TM.request("Create movies (will take a long time)?", menu)

  if choice == 2
    include("./movies.jl")
    @info "creating movies"
    create_movies()
  end

  @info "All done!"

end

main()
