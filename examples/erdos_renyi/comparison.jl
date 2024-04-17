import OpiForm

function main()
  base_dir = get(ARGS, 1, "results/Erdos-Renyi_2/N=1000")

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

  #OpiForm.compare_variance_er(mfl_dirs, micro_dirs)
  OpiForm.compare_variance_er_EA(mfl_dirs, micro_dirs; cutoff_factor=0.2)
end

main()
