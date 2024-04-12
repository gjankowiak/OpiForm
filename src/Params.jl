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

import TOML

function f_init_func_exp(ω::Float64)
  return exp(-10 * (ω + 0.5)^2) + 0.5 * exp(-15 * (ω - 0.5)^2) + 0.2
end

function f_init_func_cosine(ω::Float64)
  h = (x) -> abs(x) <= 1 ? 0.5 * (1 - cos(π * (x - 1))) : 0.0
  return 0.5 * h(4 * (ω - 0.5)) + h(3(ω + 0.50))
end

function α_init_func(ω::Float64, m::Float64)
  stddev = 0.3
  r = exp(-(ω - m)^2 / stddev^2) + 0.4
  return r
end

function g_init_func(ω::Float64, m::Float64)
  return 1.0
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
    * :from\_LFR, for each community `c` in the LFR graph, draw the opinion from the beta distribution with expectation `µ_c`, where the community expectation `µ_c` is drawn from [-1, 1]. See `init_lfr_args` for details.
    """
  ),
  :init_method_adj_matrix => (
    type=Symbol,
    values=[:from_file, :from_sampling_α_init, :from_graph, :from_lfr],
    default=:from_sampling_α_init,
    desc=md"""
    initialization method for the adjacency matrix.
    * :from\_file, a HDF5 file should be provided with the `init_micro_filename` parameter. The adjacency matrix is read from the "adj\_matrix" key. The size of the matrix should be `N_micro` x `N_micro`.
    * :from\_sampling\_α\_init, sample entries from `α_init_func`. The number of non-zero entries will be determined from `N_micro` and `connection_density`. 
    * :from\_graph, a graph will be generated according to the `init_micro_graph_type` option and its adjacency matrix will be used. 
    * :from\_LFR, generate a  Lancichinetti-Fortunato-Radicchi model benchmarks graph (using LFRBenchmarkGraphs.jl [^1])

    [^1] https://fcdimitr.github.io/LFRBenchmarkGraphs.jl/stable/
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
    values=[:from_file, :from_α_init, :from_g_init, :from_kde_adj_matrix],
    default=:from_kde_omega,
    desc=md"""
    initialization method for f.

    Possible values:
    * :from\_file, a HDF5 file should be provided with the `init_mfl_filename` parameter. The initial value is read from the "g/\$init\_mfl\_iter" key. The size of the matrix should be `N_mfl` x `N_mfl`.
    * :from\_α\_init, the function in the `α_init_func` parameter is evaluated on a regular grid of size `N_mfl` x `N_mfl`. The initial value for g is then f\_init * α\_init * f\_init' / connection\_density.
    * :from\_g\_init, the function in the `g_init_func` parameter is evaluated on a regular grid of size `N_mfl` x `N_mfl`.
    * :from\_kde\_adj\_matrix, perform a Kernel Density Estimation based on ω and the adjacency matrix using the KernelDensity.jl package.
    """
  ),
  :init_micro_filename => (
    type=String,
    default="",
    desc=md"""
    filename pointing to the initial data for the micro system. Only used if `init_method_omega` or `init_method_adj_matrix` is set to :from\_file.
    """
  ),
  :init_mfl_filename => (
    type=String,
    default="",
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
    symbol for the Graphs.jl constructor. Good choices are

    * `:dorogovtsev_mendes`, low degree for most nost nodes, a few nodes of high degree. Takes the number nodes as argument.
    * `:barabasi_albert`, most nodes with degree k, but with fat tail (towards high degree). Takes `n` and `k` as arguments, where `n` is the number of nodes.
    * `:erdos_renyi`, degree mostly the same for all nodes, given by `n`*`d`. Takes `n` and `d` as argument. 
    * `:barbell_graph`, two complete subgraphs (cliques) of size `m` and `n`, connected by 1 edge. Takes `m` and `n` as arguments.

    See the Graphs.jl documentation [^1] for the all possible values.

    [^1]: https://juliagraphs.org/Graphs.jl/dev/core\_functions/simplegraphs\_generators/
    """
  ),
  :init_micro_graph_max_tries => (
    type=Int64,
    default=10,
    desc=md"""
    graph generation will be attempted this many times until the graph is connected
    """
  ),
  :init_micro_graph_args => (
    type=Tuple,
    default=(301, 10),
    desc=md"""
    arguments to pass to the Graphs.jl constructor. This is always a `Tuple`, so for a single argument `n`, use `(n,)`. See the Graphs.jl documentation [^1] for the possible values.

    [^1]: https://juliagraphs.org/Graphs.jl/dev/core\_functions/simplegraphs\_generators/
    """
  ),
  :init_micro_graph_kwargs => (
    type=Tuple,
    default=Dict{Symbol,Any}(),
    desc=md"""
    keyword arguments to pass to the Graphs.jl constructor. This is always a `Dict{Symbol,Any}`, use `Dict(:key => value)`. See the Graphs.jl documentation [^1] for the possible values.

    [^1]: https://juliagraphs.org/Graphs.jl/dev/core\_functions/simplegraphs\_generators/
    """
  ),
  :init_lfr_args => (
    type=Tuple,
    default=(div(301, 10), div(301, 5)),
    desc=md"""
    (k_avg::Integer, k_max::Integer)
    2nd and 3rd arguments to pass to the LFRBenchmarkGraphs.jl constructor, see the documentation [^1] for the possible values.

    [^1]: https://fcdimitr.github.io/LFRBenchmarkGraphs.jl/stable/
    """
  ),
  :init_lfr_kwargs => (
    type=NamedTuple,
    default=(
      μ_community_distrib=:equidistributed,
      μ_community_bounds=(-1 + 5e-2, 1 - 5e-2),
      β_σ²=1e-2
    ),
    desc=md"""
    keyword arguments to pass to the LFRBenchmarkGraphs.jl constructor, see the documentation [^1] for the possible values.
    Some other arguments can be passed:

    * `µ_community_distrib`: how `μ_community` is sampled. Can be:
        * `:equidistributed` (default), the samples are computed as `range(µ_community_bounds...; length=n_communities)`
        * `:uniform`, the samples are draw uniformely from the interval given by `μ_community_bounds`.
    * `µ_community_bounds`: the bounds for the expectation `μ_community`
    * `β_σ²`: the variance for the β distribution use to samples `ω_i`

    [^1]: https://fcdimitr.github.io/LFRBenchmarkGraphs.jl/stable/
    """
  ),
  :init_lfr_max_tries => (
    type=Int64,
    default=10,
    desc=md"""
    LFR graph generation will be attempted this many times until the graph has has many communities as specified by `init_lfr_target_n_communities`.
    """
  ),
  :init_lfr_target_n_communities => (
    type=Int64,
    default=0,
    desc=md"""
    If positive, require the LFR graph to have this number of communities.
    If no admissible graph has been generated after `init_lfr_max_tries`, throws ErrorException.
    """
  ),
  :f_init_func => (
    type=Function,
    default=f_init_func_cosine,
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
  :g_init_func => (
    type=Function,
    default=g_init_func,
    desc=md"""
    function $[-1, 1] \times [-1, 1] \to \mathbb{R}_+$ to initialize `g`. Will NOT be normalized.
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
  :mfl_connectivity_factor => (
    type=Float64,
    default=1.0,
    desc=md"""
    scaling factor for the time derivative of f and g
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
  :store_g => (
    type=Bool,
    default=true,
    desc=md"""
    whether to save `g` (the main contribution to disk usage)
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

function to_toml(store_dir::String, params::NamedTuple)
  function conv(value)
    if value isa Tuple
      array = Vector{Any}()
      for item in value
        push!(array, item)
      end
      return array
    elseif value isa NamedTuple
      return Dict(pairs(value))
    else
      return string(value)
    end
  end
  open(joinpath(store_dir, "params.toml"), "w") do f
    TOML.print(conv, f, pairs(params))
  end
end

function from_toml(store_dir::String)
  p = open(joinpath(store_dir, "params.toml"), "r") do f
    TOML.parse(f)
  end
  params = Dict{Symbol,Any}()
  for (k, v) in p
    key = Symbol(k)
    key_type = DEFAULTS[key].type
    if key_type == Symbol
      params[key] = Symbol(v)
    elseif key_type in [Int64, Float64, String, Bool]
      params[key] = v
    elseif key_type == Tuple
      params[key] = Tuple(v)
    elseif key_type == NamedTuple
      try
        params[key] = NamedTuple(v)
      catch
        try
          k = tuple(map(Symbol, collect(keys(v)))...)
          vals = collect(values(v))
          params[key] = NamedTuple{k}(vals)
        catch
          params[key] = v
        end
      end
    else
      params[key] = v
      continue
    end
  end
  return NamedTuple(params)
end

end # module Params
