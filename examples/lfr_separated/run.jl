using Distributed
@everywhere import OpiForm
@everywhere import DelimitedFiles
# import OpiForm

function main()
  @everywhere N_micro = 1000
  @everywhere N_mfl = 301

  @everywhere function f(tuple)

    run_id, μ, β_σ² = tuple

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
    expectation_bounds = (-1 + c_mean_padding, 1 - c_mean_padding)

    params = merge(base_params, (
      max_iter=25000,
      δt=1e-3,
      store_every_iter=100,
      store_g=false,
      init_method_omega=:from_lfr,
      init_method_adj_matrix=:from_lfr,
      init_method_f=:from_f_init,
      init_method_g=:from_kde_adj_matrix,
      init_lfr_args=(k_mean, k_max),
      init_lfr_kwargs=(mixing_parameter=μ,
        nmin=nmin,
        nmax=nmax,
        µ_community_bounds=(-1 + c_mean_padding, 1 - c_mean_padding),
        µ_community_distrib=:equidistributed,
        β_σ²=β_σ²,
      ),
      init_lfr_target_n_communities=3
    ))

    prefix = "LFR_separated_2/N_micro=$(N_micro),N_mfl=$(N_mfl)/σ²=$(β_σ²)/μ=$μ"

    # Run the micro model
    store_dir_micro = "results/$(prefix)/micro-$(run_id)"
    params_micro = params
    try
      OpiForm.Micro.launch(store_dir_micro, params_micro; force=true)
    catch e
      @error "Micro run failed"
      if e isa String
        @error e
      else
        @error e.msg
      end
      return
    end
    c_ids = vec(map(Int, DelimitedFiles.readdlm(joinpath(store_dir_micro, "c_ids.csv"))))
    c_expectations = vec(DelimitedFiles.readdlm(joinpath(store_dir_micro, "c_expectations.csv")))

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

  µs = [1e-3; 5e-3; 1e-2; 5e-1; range(0.05, 1.0, 7)]
  β_σ²s = [1e-3, 4e-3, 1.2e-2]

  n_runs = 5

  pmap(f, Iterators.product(1:n_runs, µs, β_σ²s))

end

main()
