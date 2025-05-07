import OpiForm

include("defs.jl")

function create_movies()

  base_dir = "results/$(dir_label)/"

  @info "Using base directory $(base_dir)"

  micro_dirs = [joinpath(base_dir, "micro")]
  mfl_dirs = [joinpath(base_dir, "meanfield-$(suffix)") for suffix in ["multi", "mono"]]

  center_micro = false

  OpiForm.plot_results_no_g(;
    output_filename="multi-group.mp4",
    meanfield_dirs=mfl_dirs[1:1],
    micro_dirs=micro_dirs,
    stride=1,
    center_micro=center_micro
  )

  OpiForm.plot_results_no_g(;
    output_filename="mono-group.mp4",
    meanfield_dirs=mfl_dirs[2:2],
    micro_dirs=micro_dirs,
    stride=1,
    center_micro=center_micro
  )
end
