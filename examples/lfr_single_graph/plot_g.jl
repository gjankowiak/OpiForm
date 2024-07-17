import OpiForm

function create_g_plots()
  base_dir = get(ARGS, 1, "results/LFR_single_graph/N_micro=1000,N_mfl=301/")

  @info "Using base directory $(base_dir)"

  mfl_dirs = String[]

  dirs = walkdir(base_dir)

  regex_meanfield = r"meanfield-[0-9]+$"

  for d_tuple in dirs
    d = d_tuple[1]
    if endswith(d, regex_meanfield) && isfile(joinpath(d, "data.hdf5"))
      push!(mfl_dirs, d)
    end
  end

  for d in mfl_dirs
    OpiForm.plot_g_init(d; g_max=5.0)
  end
end
