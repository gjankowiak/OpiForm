# OpiForm, multimodel network-based opinion formation simulation
#
# Copyright (C) 2024  Gaspard Jankowiak
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
#
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <https://www.gnu.org/licenses/>.


module Params

export get_default_params, DEFAULTS

import OrderedCollections
import Markdown: @md_str

import Polynomials

function f_init_func(ω::Float64)
  return exp(-10 * (ω + 0.5)^2) + 0.5 * exp(-15 * (ω - 0.5)^2) + 0.2
end

function α_init_func(ω::Float64, m::Float64)
  stddev = 0.3
  r = exp(-(ω - m)^2 / stddev^2) + 0.4
  return r
end

# Debate kernel (for epistemic bubbles)
function D_func(distance)
  return -distance
end

# Radicalization kernel (ie attractive term for echo chambers)
function R_func(distance)
  return -distance
end

# Polarization kernel (ie repulsive term for echo chambers)
function P_func(distance)
  return distance
end

DEFAULTS = OrderedCollections.OrderedDict(
  :N_mfl => (
    type=Int64,
    default=301,
    desc=md"""
    size of the discretization for the meanfield model.
    """),
  :N_micro => (
    type=Int64,
    default=300,
    desc=md"""
    number of agents for the micro model.
    """),
  :N_sampling => (
    type=Int64,
    default=1000,
    desc=md"""
    size of the discretization used for the sampling of `f_init_func` and `α_init_func`.
    Beware that a full `N_sampling` x `N_sampling` matrix will be allocated.
    """),
  :δt => (
    type=Float64,
    default=1e-3,
    desc=md"""
    time step.
    """
  ),
  :max_iter => (
    type=Int64,
    default=20_000,
    desc=md"""
    maximum number of time iterations.
    """
  ),
  :init_method_omega => (
    type=Symbol,
    values=[:from_file, :from_sampling_f_init],
    default=:from_sampling_f_init,
    desc=md"""
    initialization method for ω (micro model).
    * :from\_file, a HDF5 file should be provided with the `init_micro_filename` parameter. ω is read from the "omega/\$init\_micro\_iter" key. The size of the vector should match `N_micro`.
    * :from\_sampling\_f\_init, ω will be initialized by drawing `N_micro` samples from the function defined in `f_init_func`.
    """
  ),
  :init_method_adj_matrix => (
    type=Symbol,
    values=[:from_file, :from_sampling_α_init, :from_graph],
    default=:from_sampling_α_init,
    desc=md"""
    initialization method for the adjacency matrix.
    * :from\_file, a HDF5 file should be provided with the `init_micro_filename` parameter. The adjacency matrix is read from the "adj\_matrix" key. The size of the matrix should be `N_micro` x `N_micro`.
    * :from\_sampling\_α\_init, sample entries from `α_init_func`. The number of non-zero entries will be determined from `N_micro` and `connection_density`. 
    * :from\_graph, a graph will be generated according to the `init_micro_graph_type` option and its adjacency matrix will be used. 
    """
  ),
  :init_method_f => (
    type=Symbol,
    values=[:from_file, :from_f_init, :from_kde_omega],
    default=:from_kde_omega,
    desc=md"""
    initialization method for f.

    Possible values:
    * :from\_file, a HDF5 file should be provided with the `init_mfl_filename` parameter. The initial value is read from the "f/\$init\_mfl\_iter" key. The size of the vector should be `N_mfl`.
    * :from\_f\_init, the function in the `f_init_func` parameter is evaluated on a regular grid of size `N_mfl`.
    * :from\_kde\_omega, perform a Kernel Density Estimation from `ω` using the KernelDensity.jl package.
    """
  ),
  :init_method_g => (
    type=Symbol,
    values=[:from_file, :from_α_init, :from_kde_adj_matrix],
    default=:from_kde_omega,
    desc=md"""
    initialization method for f.

    Possible values:
    * :from\_file, a HDF5 file should be provided with the `init_mfl_filename` parameter. The initial value is read from the "g/\$init\_mfl\_iter" key. The size of the matrix should be `N_mfl` x `N_mfl`.
    * :from\_α\_init, the function in the `α_init_func` parameter is evaluated on a regular grid of size `N_mfl` x `N_mfl`. The initial value for g is then f\_init * α\_init * f\_init' / connection\_density.
    * :from\_kde\_adj\_matrix, perform a Kernel Density Estimation based on ω and the adjacency matrix using the KernelDensity.jl package.
    """
  ),
  :init_micro_filename => (
    type=Union{Symbol,Nothing},
    default=nothing,
    desc=md"""
    filename pointing to the initial data for the micro system. Only used if `init_method_omega` or `init_method_adj_matrix` is set to :from\_file.
    """
  ),
  :init_mfl_filename => (
    type=Union{Symbol,Nothing},
    default=nothing,
    desc=md"""
    filename pointing to the initial data for the meanfield system. Only used if `init_method_f` or `init_method_g` is set to :from\_file.
    """
  ),
  :init_micro_iter => (
    type=Int64,
    default=0,
    desc=md"""
    iteration in the file for which the data for `ω` and `adj_matrix` should be loaded. 0 corresponds to the initial data.
    """
  ),
  :init_mfl_iter => (
    type=Int64,
    default=0,
    desc=md"""
    iteration in the file for which the data for `f` and `g` should be loaded. 0 corresponds to the initial data.
    """
  ),
  :init_micro_graph_type => (
    type=Symbol,
    default=:barabasi_albert,
    desc=md"""
    symbol for the Graphs.jl constructor. See the Graphs.jl documentation [^1] for the possible values.

    [^1]: https://juliagraphs.org/Graphs.jl/dev/core\_functions/simplegraphs\_generators/
    """
  ),
  :init_micro_graph_args => (
    type=Tuple,
    default=(301, 10),
    desc=md"""
    arguments to pass to the Graphs.jl constructor. See the Graphs.jl documentation [^1] for the possible values.

    [^1]: https://juliagraphs.org/Graphs.jl/dev/core\_functions/simplegraphs\_generators/
    """
  ),
  :init_micro_graph_kwargs => (
    type=Tuple,
    default=Dict{Symbol,Any}(),
    desc=md"""
    keyword arguments to pass to the Graphs.jl constructor. See the Graphs.jl documentation [^1] for the possible values.

    [^1]: https://juliagraphs.org/Graphs.jl/dev/core\_functions/simplegraphs\_generators/
    """
  ),
  :f_init_func => (
    type=Function,
    default=f_init_func,
    desc=md"""
    function $[-1, 1] \to \mathbb{R}_+$ to initialize `f`. Will be normalized for you.
    """
  ),
  :α_init_func => (
    type=Function,
    default=α_init_func,
    desc=md"""
    function $[-1, 1] \times [-1, 1] \to \mathbb{R}_+$ to initialize `g`. Will be normalized for you. See also the `connection_density` parameter.
    """
  ),
  :constant_α => (
    type=Bool,
    default=false,
    desc=md"""
    use a constant `α` for the initialization, it will be scaled for you. This is more efficient that setting `α_init_func` to a constant function.
    """
  ),
  :constant_g => (
    type=Bool,
    default=false,
    desc=md"""
    set `g` to be constant in time and space. This should *not* be used unless you are debugging.
    """
  ),
  :full_adj_matrix => (
    type=Bool,
    default=false,
    desc=md"""
    use a full adjacency matrix. This is more efficient that passing a full graph to `init_micro_graph_type`, for example.
    """
  ),
  :f_dependent_g => (
    type=Bool,
    default=false,
    desc=md"""
    do not use the evolution equation for `g` but set `g` to `f .* α .* f' / connection_density`.
    """
  ),
  :connection_density => (
    type=Float64,
    default=0.1,
    desc=md"""
    used for the normalization of the adjacency matrix and `g`. The number of non-zero entries in the adjacency matrix will be roughly `N_micro^2*connection_density`, so the resulting density will
    be `connection_density`, i.e. `connection_density` *should be less than 1*!
    """
  ),
  :plot_scale => (
    type=Function,
    default=identity,
    desc=md"unused."
  ),
  :plot_every => (
    type=Int64,
    default=10,
    desc=md"unused."
  ),
  :plot_backend => (
    type=Symbol,
    default=:makie,
    desc=md"unused."
  ),
  :store_every_iter => (
    type=Int64,
    default=10,
    desc=md"""
    one every `store_every_iter` iteration will be saved to disk.
    """
  ),
  :σ => (
    type=Float64,
    default=1.0,
    desc=md"""
    parameter which balances the effects of echo chambers (EC) and epystemic bubbles (EB). Can be between 0 (EC only) and 1 (EB only).
    """
  ),
  :EC_type => (
    type=Symbol,
    values=[:characteristic, :super_gaussian],
    default=:super_gaussian,
    desc=md"""
    window function for the computation of echo chambers.

    * :characteristic, original model, uses $1_{|ω-m| < EC_ρ}(m)$
    * :super\_gaussian, uses $clip(exp(- (|ω-m|²/EC_ρ)^EC_power), EC_clip_value)$
                     where `clip(x,v) = x if x > v, 0 otherwise`.

    Note that :super\_gaussian approximates :characteristic as `EC_power` → ∞ and `EC_clip_value` → 0.
    """
  ),
  :EC_ρ => (
    type=Float64,
    default=2e-1,
    desc=md"""
    echo chamber radius.
    """
  ),
  :EC_power => (
    type=Float64,
    default=3.0,
    desc=md"""
    echo chamber super-gaussian power. See `EC_type`.
    """
  ),
  :EC_clip_value => (
    type=Float64,
    default=2e-1,
    desc=md"""
    echo chamber radius.
    echo chamber super-gaussian clip value. See `EC_type`.
    """
  ),
  :normalize_chambers => (
    type=Bool,
    default=false,
    desc=md"""
    whether to normalize echo chambers. TODO.
    """
  ),
  :D_func => (
    type=Function,
    default=D_func,
    desc=md"""
    Debate kernel (for epistemic bubbles), function of the distance $|ω - m|$. Default: distance → -distance.
    """
  ),
  :R_func => (
    type=Function,
    default=R_func,
    desc=md"""
    Radicalization kernel (ie attractive term for echo chambers), function of the distance $|ω - m|$. Default: distance → -distance.
    """
  ),
  :P_func => (
    type=Function,
    default=P_func,
    desc=md"""
    Polarization kernel (ie repulsive term for echo chambers), function of the distance $|ω - m|$. Default: distance → distance.
    """
  ),

  # Choice of flux:
  # :LF (Lax-Friedrich)
  # :lLF (local Lax-Friedrich)
  # :godunov
  # :constant_godunov
  # :upwind
  # :KT
  :flux => (
    type=Symbol,
    values=[:LF, :lLF],
    default=:lLF,
    desc=md"""
    choice of the flux for the finite volume scheme (meanfield model).
    """
  ),
  :approx_a_prime => (
    type=Bool,
    default=false,
    desc=md"""
    try to approximate `a'` when computing the fluxes in the finite volume scheme.
    """
  ),
  :time_stepping => (
    type=Symbol,
    values=[:simple, :RK4],
    default=:simple,
    desc=md"""
    the time stepping method.
    """
  ),
  :LF_relaxation => (
    type=Float64,
    default=1.0,
    desc=md"""
    relaxation parameter for the LF/lLF fluxes. This _should_ be set to 1.0.
    """
  ),
  :godunov_entropy_fix => (
    type=Bool,
    default=false,
    desc=md"""
    The use entropy fix for the Godunov scheme
    See Leveque 12.3 (FVM for Hyperbolic Problems)
    """
  ),
  :KT_flux_limiter => (
    type=Symbol,
    values=[:minmod],
    default=:midmod,
    desc=md"""
    type of flux limiter for the KT scheme.
    """
  ),
  :log_debug => (
    type=Bool,
    default=false,
    desc=md"whether to log debug information."
  ),
  :constant_a => (
    type=Bool,
    default=false,
    desc=md"whether to set and keep `a` constant in the finite volume scheme (for debugging only)."
  ),
  :CFL_violation => (
    type=Symbol,
    values=[:ignore, :warn, :throw],
    default=:throw,
    desc=md"""how to handle CFL violations."""
  )
)

function get_default_params()
  return NamedTuple([k => v.default for (k, v) in DEFAULTS])
end

function display_parameter_description(key, value)
  vtype = value.type
  vdefault = value.default
  display(md"**$key** ($vtype) [ $vdefault ]")
  display(value.desc)
  println()
end

function describe(key)
  display_parameter_description(key, DEFAULTS[key])
end

function describe()
  for (k, v) in DEFAULTS
    display_parameter_description(k, v)
  end
end

end # module Params
