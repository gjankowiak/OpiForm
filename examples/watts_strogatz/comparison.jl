import OpiForm

function main()

  # for β in [0.2]
  for β in 0:0.1:1

    base_dir = get(ARGS, 1, "results/Watts-Strogatz/N=1000/β=$β")

    @info "Using base directory $(base_dir)"

    mfl_dirs = String[]
    micro_dirs = String[]

    dirs = walkdir(base_dir)

    regex_micro = r"micro-[0-9]+$"
    regex_meanfield = r"meanfield-[0-9]+$"

    # @show dirs

    for d_tuple in dirs
      d = d_tuple[1]
      if endswith(d, regex_micro) && isfile(joinpath(d, "data.hdf5"))
        push!(micro_dirs, d)
      elseif endswith(d, regex_meanfield) && isfile(joinpath(d, "data.hdf5"))
        push!(mfl_dirs, d)
      end
    end

    a = 1

    #OpiForm.compare_variance_ws_all(mfl_dirs, micro_dirs)
    OpiForm.compare_variance_ws_EA(mfl_dirs, micro_dirs; cutoff_factor=0.4)
  end

end

main()
