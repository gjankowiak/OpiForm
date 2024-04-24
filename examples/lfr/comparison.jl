import OpiForm
import HDF5

function has_h5_key(fn, key)
  h5 = HDF5.h5open(fn, "r")
  return haskey(h5, key)
end

function main()

  #base_dir = get(ARGS, 1, "results/LFR/N=1000")
  base_dir = get(ARGS, 1, "results/LFR/N_micro=1000,N_mfl=601/")
  base_dir = get(ARGS, 1, "results/LFR/N_micro=1000,N_mfl=1201/")
  base_dir = get(ARGS, 1, "results/LFR/N_micro=1000,N_mfl=301/")

  @info "Using base directory $(base_dir)"

  mfl_dirs = String[]
  micro_dirs = String[]

  dirs = walkdir(base_dir; follow_symlinks=true)

  regex_micro = r"micro-[0-9]+$"
  regex_meanfield = r"meanfield-[0-9]+$"

  # @show dirs

  for d_tuple in dirs
    d = d_tuple[1]
    data_path = joinpath(d, "data.hdf5")
    if endswith(d, regex_micro) && isfile(data_path)
      push!(micro_dirs, d)
    elseif endswith(d, regex_meanfield) && isfile(data_path) && has_h5_key(data_path, "f")
      push!(mfl_dirs, d)
    else
    end
  end

  OpiForm.compare_variance_lfr_EA(mfl_dirs, micro_dirs; cutoff_factor=0.2, t_max=10, stddev_min=10^(-2.5))

end

main()
