using Distributed
@everywhere import OpiForm
# import OpiForm

function main()
  @everywhere N_micro = 1000


  @everywhere function f(tuple)

    run_id, p = tuple

    base_params = OpiForm.Params.get_default_params()

    base_params = merge(base_params, (
      N_micro=N_micro,
    ))

    base_params = merge(base_params, (
      init_method_omega=:from_sampling_f_init,
      init_method_adj_matrix=:from_graph,
      init_micro_graph_type=:erdos_renyi,
      init_method_f=:from_f_init,
      init_method_g=:from_kde_adj_matrix,
    ))

    prefix = "Erdos-Renyi_2/N=$(N_micro)/p=$p"

    params = merge(base_params, (
      init_micro_graph_args=(N_micro, p),
    ))

    # Run the micro model
    store_dir_micro = "results/$(prefix)/micro-$(run_id)"
    params_micro = params
    try
      OpiForm.Micro.launch(store_dir_micro, params_micro; force=true)
    catch e
      @error "Micro run failed"
      @error e
      return
    end

    # Run the meanfield model using the initial data and graph of the micro model (with KDE)
    store_dir_mfl = "results/$(prefix)/meanfield-$(run_id)"
    params_lLF = merge(params, (
      flux=:lLF, f_dependent_g=false,
      init_method_omega=:from_file,
      init_method_adj_matrix=:from_file,
      init_micro_filename=joinpath(store_dir_micro, "data.hdf5"),
    ))
    try
      OpiForm.MeanField.launch(store_dir_mfl, params_lLF; force=true)
    catch
      @error "MFL run failed"
      return
    end
  end

  p_crit = log(N_micro) / N_micro

  ps = (10 .^ range(log10(p_crit) * 1.01, -0.05; length=30))

  n_runs = 5

  pmap(f, Iterators.product(1:n_runs, ps))

end

main()
