using Distributed
@everywhere import OpiForm
# import OpiForm

function main()
  ps = [1e-3; 1e-2; 0.1:0.1:0.9...]
  ps = (10 .^ range(-3, -1; length=12))[1:end-1]

  ps = [0.001, 0.0015199110829529332, 0.0023101297000831605, 0.003511191734215131,
    0.005336699231206307, 0.008111308307896872, 0.01, 0.012328467394420659,
    0.01873817422860384, 0.02848035868435802, 0.04328761281083059, 0.0657933224657568,
    0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]

  @everywhere function f(p)
    # function f(p)

    base_params = OpiForm.Params.get_default_params()

    N_micro = 1000

    base_params = merge(base_params, (
      N_micro=N_micro,
    ))

    f_init = (x) -> 1.0

    base_params = merge(base_params, (
      f_init_func=f_init,
      init_method_omega=:from_sampling_f_init,
      init_method_adj_matrix=:from_graph,
      init_micro_graph_type=:erdos_renyi,
      init_method_f=:from_f_init,
      init_method_g=:from_kde_adj_matrix,
    ))

    prefix = "Erdos-Renyi/uniform/N=$(N_micro)/p=$p"

    params = merge(base_params, (
      init_micro_graph_args=(N_micro, p),
    ))

    # Run the micro model
    store_dir_micro = "results/$(prefix)/micro"
    params_micro = params
    try
      OpiForm.Micro.launch(store_dir_micro, params_micro; force=true)
    catch
      @error "Micro run failed"
      return
    end

    # Run the meanfield model using the initial data and graph of the micro model (with KDE)
    store_dir_mfl = "results/$(prefix)/meanfield"
    params_lLF = merge(params, (
      flux=:lLF, f_dependent_g=false,
      init_method_omega=:from_file,
      init_method_adj_matrix=:from_file,
      init_micro_filename=joinpath(store_dir_micro, "data.hdf5")
    ))
    try
      OpiForm.MeanField.launch(store_dir_mfl, params_lLF; force=true)
    catch
      @error "MFL run failed"
      return
    end

    OpiForm.compare_variance([
        store_dir_mfl
      ], [
        store_dir_micro
      ])
  end

  pmap(f, ps)
  # f(ps[1])

end

main()
