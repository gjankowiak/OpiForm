import OpiForm

#########################
#          RUN          #
#########################

# Get sensible defaults. You can also define you own.
# You can get a description of a parameter by calling
# OpiForm.Params.describe
# Example: OpiForm.Params.describe(:N_micro)
# To get a list of all parameters, their types and default values,
# call OpiForm.Params.describe()
params = OpiForm.Params.get_default_params()

# Tweak some values from the defaults
# params = merge(params, (
#   init_micro_graph_type = :barabasi_albert,
#   init_micro_graph_args = (params.N_micro, 10)
# ))

OpiForm.display_params(params)

prefix = "test"

# Run the micro model
store_dir_micro = "results/$(prefix)_micro"
params_micro = params
OpiForm.Micro.launch(store_dir_micro, params_micro; force=true)

# Run the meanfield model using the initial data and graph of the micro model (with KDE)
store_dir_mfl = "results/$(prefix)_meanfield"
params_lLF = merge(params, (
  flux=:lLF, f_dependent_g=false,
  init_method_omega=:from_file,
  init_method_adj_matrix=:from_file,
  init_method_f=:from_kde_omega,
  init_method_g=:from_kde_adj_matrix,
  init_micro_filename = joinpath(store_dir_micro, "data.hdf5")
))
OpiForm.MeanField.launch(store_dir_mfl, params_lLF; force=true)

#########################
#         PLOT          #
#########################

# Movie

# OpiForm.plot_result("$(prefix).mp4";
#   meanfield_dir="results/$(prefix)_meanfield",
#   micro_dir="results/$(prefix)_micro",
#   half_connection_matrix=true,
#   center_histogram=false
# )

# Movie with centered histogram

OpiForm.plot_result("$(prefix)_centered.mp4";
  meanfield_dir="results/$(prefix)_meanfield",
  micro_dir="results/$(prefix)_micro",
  half_connection_matrix=true,
  center_histogram=false
)

# Comvergence plots

OpiForm.compare_variance(
  [
    "results/$(prefix)_meanfield"
  ], [
    "results/$(prefix)_micro"
  ]
)
