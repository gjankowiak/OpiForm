import OpiForm

base_dir = get(ARGS, 1, "results/Erdos-Renyi_2/N=1000")

@info "Using base directory $(base_dir)"

mfl_dirs = String[]
micro_dirs = String[]

dirs = readdir(base_dir)

@show dirs

for d in dirs
  micro_dir = joinpath(base_dir, d, "micro")
  mfl_dir = joinpath(base_dir, d, "meanfield")
  if isfile(joinpath(micro_dir, "data.hdf5")) && isfile(joinpath(mfl_dir, "data.hdf5"))
    push!(micro_dirs, micro_dir)
    push!(mfl_dirs, mfl_dir)
  end
end

OpiForm.compare_variance_er(mfl_dirs, micro_dirs)
OpiForm.compare_variance_er_EA(mfl_dirs, micro_dirs)
