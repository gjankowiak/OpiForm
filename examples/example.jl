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

# Tweak some values from the defaults:

# by changing the connection density,
params = merge(params, (
  connection_density=0.1,
))
prefix = "$(params.init_method_adj_matrix)/connection_density=$(params.connection_density)"

# or by generating the adjacency matrix from a graph
# params = merge(params, (
#   init_method_adj_matrix=:from_graph,
#   # choose the type of graph
#   init_micro_graph_type=:barabasi_albert,
#   # tweak the generator's parameters
#   init_micro_graph_args=(params.N_micro, 20)
# ))
# prefix = "$(params.init_method_adj_matrix)/$(params.init_micro_graph_type)_$(params.init_micro_graph_args)"

OpiForm.display_params(params)


# Run the micro model
store_dir_micro = "results/$(prefix)/micro"
params_micro = params
OpiForm.Micro.launch(store_dir_micro, params_micro; force=true)

# Run the meanfield model using the initial data and graph of the micro model (with KDE)
store_dir_mfl = "results/$(prefix)/meanfield"
params_lLF = merge(params, (
  flux=:lLF, f_dependent_g=false,
  init_method_omega=:from_file,
  init_method_adj_matrix=:from_file,
  init_method_f=:from_kde_omega,
  init_method_g=:from_kde_adj_matrix,
  init_micro_filename=joinpath(store_dir_micro, "data.hdf5")
))
OpiForm.MeanField.launch(store_dir_mfl, params_lLF; force=true)

#########################
#         PLOT          #
#########################

# Movie

OpiForm.plot_result(;
  meanfield_dir="results/$(prefix)/meanfield",
  micro_dir="results/$(prefix)/micro",
  half_connection_matrix=true,
  center_histogram=false
)

# Comvergence plots

OpiForm.compare_variance(
  [
    "results/$(prefix)/meanfield"
  ], [
    "results/$(prefix)/micro"
  ]
)
