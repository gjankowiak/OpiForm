using Distributed
@everywhere import OpiForm
# import OpiForm

function main()
  @everywhere N_micro = 1000
  @everywhere N_mfl = 1201

  @everywhere function f(tuple)

    run_id, μ = tuple

    k_mean = 15
    k_max = 40

    nmin = div(N_micro, 5)
    nmax = div(N_micro, 3)

    base_params = OpiForm.Params.get_default_params()

    base_params = merge(base_params, (
      N_micro=N_micro,
      N_mfl=N_mfl,
    ))

    c_mean_padding = 0.25

    params = merge(base_params, (
      max_iter=16000,
      δt=5e-4,
      store_every_iter=1000,
      store_g=false,
      init_method_omega=:from_lfr,
      init_method_adj_matrix=:from_lfr,
      init_method_f=:from_kde_omega,
      init_method_g=:from_kde_adj_matrix,
      init_lfr_args=(k_mean, k_max),
      init_lfr_kwargs=(mixing_parameter=µ,
        nmin=nmin,
        nmax=nmax,
        µ_community_bounds=(-1 + c_mean_padding, 1 - c_mean_padding),
        µ_community_distrib=:equidistributed,
        β_σ²=1e-2,
      ),
      init_lfr_target_n_communities=3
    ))

    prefix_old = "LFR/N=$(N_micro)/μ=$μ"
    prefix_new = "LFR/N_micro=$(N_micro),N_mfl=$(N_mfl)/μ=$μ"

    # Run the micro model
    store_dir_micro = "results/$(prefix_old)/micro-$(run_id)"
    # params_micro = params
    # try
    #   OpiForm.Micro.launch(store_dir_micro, params_micro; force=true)
    # catch e
    #   @error "Micro run failed"
    #   @error e
    #   #rethrow()
    #   #catch_backtrace()
    #   return
    # end

    # Run the meanfield model using the initial data and graph of the micro model (with KDE)
    store_dir_mfl = "results/$(prefix_new)/meanfield-$(run_id)"
    params_lLF = merge(params, (
      flux=:lLF, f_dependent_g=false,
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

    # OpiForm.compare_variance([
    #     store_dir_mfl
    #   ], [
    #     store_dir_micro
    #   ])
  end

  μs = 10 .^ (range(-3, 0, 10))

  n_runs = 5

  pmap(f, Iterators.product(1:n_runs, µs))

end

main()
