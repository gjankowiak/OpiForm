import OpiForm

function main()

  base_dir = get(ARGS, 1, "results/Erdos-Renyi/uniform/N=1000")
  base_dir = get(ARGS, 1, "results/Erdos-Renyi/N=1000")
  base_dir = get(ARGS, 1, "results/Erdos-Renyi_2/N=1000")

  @info "Using base directory $(base_dir)"

  mfl_dirs = String[]
  micro_dirs = String[]

  dirs = readdir(base_dir)

  for d in dirs
    micro_dir = joinpath(base_dir, d, "micro")
    mfl_dir = joinpath(base_dir, d, "meanfield")
    if isfile(joinpath(micro_dir, "data.hdf5")) && isfile(joinpath(mfl_dir, "data.hdf5"))
      push!(micro_dirs, micro_dir)
      push!(mfl_dirs, mfl_dir)
    end
  end

  mfl_dir = mfl_dirs[6]
  micro_dir = micro_dirs[6]

  @show mfl_dir

  OpiForm.compare_variance(mfl_dirs, micro_dirs)
  OpiForm.plot_result(;
    meanfield_dir=mfl_dir,
    micro_dir=micro_dir,
    half_connection_matrix=true,
    center_histogram=false,
    stride=10,
  )

end

main()
