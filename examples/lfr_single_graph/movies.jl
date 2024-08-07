import OpiForm

function create_movies()

  μs = [1e-3; 5e-3; 1e-2; 5e-2; 5e-1]
  β_σ²s = [0.001; 0.004; 0.012]

  for β_σ² in β_σ²s
    for µ in µs
      base_dir = "results/LFR_single_graph/N_micro=1000,N_mfl=301/σ²=$β_σ²/μ=$µ/"

      @info "Using base directory $(base_dir)"

      mfl_dirs = String[]
      micro_dirs = String[]

      dirs = walkdir(base_dir; follow_symlinks=true)

      regex_micro = r"micro-[0-9]+$"
      regex_meanfield = r"meanfield-[0-9]+$"

      # @show dirs

      limit = 10

      for d_tuple in dirs
        d = d_tuple[1]
        if endswith(d, regex_micro) && isfile(joinpath(d, "data.hdf5")) && length(micro_dirs) < limit
          push!(micro_dirs, d)
        elseif endswith(d, regex_meanfield) && isfile(joinpath(d, "data.hdf5")) && length(mfl_dirs) < limit
          push!(mfl_dirs, d)
        else
        end
      end

      OpiForm.plot_results_no_g(;
        meanfield_dirs=mfl_dirs,
        micro_dirs=micro_dirs,
        stride=1,
      )

      OpiForm.plot_results_no_g(;
        meanfield_dirs=mfl_dirs,
        micro_dirs=micro_dirs,
        stride=1,
        center_micro=true
      )
    end
  end

end
