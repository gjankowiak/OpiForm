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


function get_ωx_ωy!(ωx::Vector{Float64}, ωy::Vector{Float64}, adj_matrix::SpA.SparseMatrixCSC{Int64,Int64}, ω::Vector{Float64})
  N = length(ω)
  rows = SpA.rowvals(adj_matrix)
  got = 0
  for j in 1:N
    nzr = SpA.nzrange(adj_matrix, j)
    nnz_in_col = length(nzr)

    ωx[got+1:got+nnz_in_col] .= ω[rows[nzr]]
    ωy[got+1:got+nnz_in_col] .= ω[j]

    got += nnz_in_col
  end
end

"""
    get_ωx_ωy(adj_matrix::SpA.SparseMatrixCSC{Int64,Int64}, ω::Vector{Float64})

Returns two vectors `ωx` and `ωy` s.t. ∀ k in 1:K, ∃ (i,j) such that A[i,j] ≠ 0 and ω[i] = ωx[k], ω[j] = ωy[k].
"""
function get_ωx_ωy(adj_matrix::SpA.SparseMatrixCSC{Int64,Int64}, ω::Vector{Float64})
  nnz = SpA.nnz(adj_matrix)
  ωx, ωy = zeros(nnz), zeros(nnz)
  get_ωx_ωy!(ωx, ωy, adj_matrix, ω)
  return ωx, ωy
end

function to_graph(adj_matrix::SpA.SparseMatrixCSC)
  return Graphs.SimpleGraphs.SimpleGraph(adj_matrix)
end

function to_adj_matrix(g::Graphs.SimpleGraphs.SimpleGraph)
  return SpA.sparse(g)
end

function compute_kde(x, ops::Vector{Float64})
  h_SJ = KernelDensitySJ.bwsj(ops)
  kde_ops = KernelDensity.kde(ops, boundary=(-1, 1), bandwidth=h_SJ)
  interp_kde_ops = KernelDensity.InterpKDE(kde_ops)
  return KernelDensity.pdf(interp_kde_ops, x)
end

function compute_kde(x, adj_matrix::SpA.SparseMatrixCSC, ops::Vector{Float64})
  h_SJ = KernelDensitySJ.bwsj(ops)
  ωx, ωy = get_ωx_ωy(adj_matrix, ops)

  kde_a = KernelDensity.kde([ωx ωy], boundary=((-1, 1), (-1, 1)), bandwidth=(h_SJ, h_SJ))
  interp_kde_a = KernelDensity.InterpKDE(kde_a)

  return [KernelDensity.pdf(interp_kde_a, _x, _y) for _x in x, _y in x]
end

function scale_poly(poly::Polynomials.AbstractPolynomial)
  I = Polynomials.integrate(poly)
  return poly / (I(1) - I(-1))
end

function sample_poly_dist(poly::Polynomials.AbstractPolynomial, n::Int64)
  # Samples a polynomial distribution
  I = Polynomials.integrate(poly)
  scaled_I = I - I(-1)
  scaled_I = scaled_I / scaled_I(1)

  # approximate extrema of the density
  x = range(-1, 1, length=n)
  fmin, fmax = extrema(poly.(x))
  @info "min(f), max(f) ~ $fmin, $fmax"

  img_samples = rand(n)

  samples = sort(collect(map(x -> Roots.find_zero(scaled_I - x, (-1, 1), Roots.Bisection()), img_samples)))

  @info "Histogram of the samples of f_init:\n" * string(UnicodePlots.histogram(samples; nbins=50, vertical=true))

  return samples
end

"""
    scale_to_pdf(f::Function, x::AbstractRange)

Return the function x → f(x) / ∫f(x) dx, where the integration is done with a midpoint rule and params.N_sampling points.
"""
function scale_to_pdf(f::Function, x::AbstractRange)
  δx = x[2] - x[1]
  int_f = sum(f, x) / δx
  f_scaled(x) = f(x) / int_f
  return f_scaled
end

function scale_g_init(g_init_func_unscaled)
  return 1 / Cubature.hcubature(g_init_func_unscaled, [-1; -1], [1; 1])[1]
end

function compute_ω_inf_mfl(g_init_func_scaled, x)
  if isnothing(g_init_func_scaled)
    return 0.0
  else
    @info "Computing g's first moment"
    # hack because Cubature sometimes stalls
    # return (Cubature.pcubature(x -> x[1] * g_init_func_scaled(x[1], x[2]), [-1; -1], [1; 1])[1]) / g_init_integral
    δx = x[2] - x[1]
    return δx^2 * sum(x -> x[1] * g_init_func_scaled(x[1], x[2]), Iterators.product(x, x))
  end
end

"""
    fast_sampling(f::Function, n::Int64, N_sampling::Int64)

Return `n` samples from the probability distribution f/∫f discretized on a grid of size `N_sampling`
"""
function fast_sampling(f::Function, n::Int64, N_sampling::Int64)
  x = build_x(N_sampling)
  @show extrema(x)

  f_x = f.(x)

  accumulator = Vector{Float64}(undef, N_sampling)
  samples = Vector{Float64}(undef, n)

  # Compute the cumulative distribution function by summing and normalizing
  cumsum!(accumulator, f_x)
  accumulator ./= accumulator[end]

  # number of samples already found
  k = 0

  while k < n

    trial = rand()
    idx_range = searchsorted(accumulator, trial)
    if length(idx_range) == 0
      if idx_range.stop == 0
        # trial < accumulator[1]
        rand_left = -1
        rand_right = x[1]
        samples[k+1] = rand_left + (rand_right - rand_left) * rand()
        k += 1
      elseif idx_range.start == N_sampling
        # trial > accumulator[end], this should never happen
        samples[k+1] = 1
        k += 1
        continue
      else
        # accumulator[idx_range.stop] < trial < accumulator[idx_range.start]
        # this should be the most common case.
        # We pick a sample uniformely between the two bounds
        rand_left = x[idx_range.stop]
        rand_right = x[idx_range.start]
        samples[k+1] = rand_left + (rand_right - rand_left) * rand()
        k += 1
        continue
      end
    else
      # accumulator[idx_range.start] = trial = accumulator[idx_range.stop]
      # this should never happen, and corresponds to zeros of f
      continue
    end
  end

  return sort(samples)
end

"""
    fast_sampling(params::NamedTuple, α::Matrix{Float64}, n::Int64; symmetric::Bool=true)

Return a matrix `adj_matrix` with exactly `n` ones, where the entries are sampled according to the distribution `α`.

The returned matrix is symmetric if the `symmetric` flag is set.
"""
function fast_sampling(params::NamedTuple, α::Matrix{Float64}, n::Int64; symmetric::Bool=true)
  adj_matrix = SpA.spzeros(Int64, params.N_micro, params.N_micro)

  if !params.constant_α
    accumulator = Vector{Float64}(undef, params.N_micro^2)

    cumsum!(accumulator, vec(α))
    accumulator ./= accumulator[end]
  end

  accepted = 0
  tried = 0

  n_digits = Int(ceil(log10(n)))

  while accepted < n
    tried += 1
    if !params.constant_α
      t = rand()
      r = searchsorted(accumulator, t)
      if length(r) == 0
        idx = r.start
      else
        idx = rand(r)
      end

      p, q = divrem(idx - 1, params.N_micro)
      i, j = p + 1, q + 1
    else
      i, j = rand(1:params.N_micro), rand(1:params.N_micro)
    end

    if adj_matrix[i, j] == 0
      adj_matrix[i, j] = 1
      accepted += 1
      if symmetric
        adj_matrix[j, i] = 1
        accepted += 1
        tried += 1
      end

      if accepted % 100 == 0
        print("  $(lpad(accepted, n_digits, " "))/$(n) ($(round(accepted/tried*100; digits=5))% accepted)", "\b"^100)
      end
    end
  end
  return adj_matrix
end

"""
    sample_g_init(params::NamedTuple, scaled_funcs::NamedTuple, ops::Vector{Float64}; symmetric::Bool=true, zero_diagonal::Bool=false)

Returns an adjacency matrix, the number of entries and their distribution is computed according to the passed parameters.

Returns nothing if the flag params.full_adj_matrix is set.
"""
function sample_g_init(params::NamedTuple, scaled_funcs::NamedTuple, ops::Vector{Float64}; symmetric::Bool=true, zero_diagonal::Bool=false)
  params.full_adj_matrix && return nothing

  g_init_integral = params.connection_density * params.N_micro

  @assert length(ops) > 0 "No samples provided!"
  @assert params.connection_density < 1 "params.connection_density >= 1!"

  if !params.constant_α
    # Compute all the couples g_init(ω_i, ω_j)
    if zero_diagonal
      α_init_samples = [i == j ? 0.0 : scaled_funcs.α_init_func_scaled(ops[i], ops[j]) for i = 1:params.N_micro, j = 1:params.N_micro]
    else
      α_init_samples = [scaled_funcs.α_init_func_scaled(ops[i], ops[j]) for i = 1:params.N_micro, j = 1:params.N_micro]
    end
  else
    if zero_diagonal
      α_init_samples = [i == j ? 0.0 : g_init_integral / 4 for i = 1:params.N_micro, j = 1:params.N_micro]
    else
      α_init_samples = [g_init_integral / 4 for i = 1:params.N_micro, j = 1:params.N_micro]
    end
  end

  # DEFINITION G
  f_ops_samples = scaled_funcs.f_init_func_scaled.(ops)
  g_init_samples = f_ops_samples .* α_init_samples .* f_ops_samples' / params.connection_density
  sum_g_samples = sum(g_init_samples)

  # integrate over the samples to get the corresponding connectivity
  s = (2 / params.N_micro)^2
  discrete_connectivity = sum_g_samples * s

  # check that it matches the connectivity parameter
  # this should only be expected to match if f_init is a constant distribution
  # i.e. the samples in the opinion spaces are uniformly distributed
  @info "Connectivities ∫g (match expected if f_init is constant):"
  @info "from parameter               : $(params.connection_density * params.N_micro)"
  @info "from discretization of g_init: $discrete_connectivity"

  # Sampling the adjacency matrix A

  # Number of non-zero entries in the adjea
  N_entries = round(Int, params.N_micro^2 * params.connection_density)
  # check that we are not trying to find to many (more than the size of the matrix)
  @assert (N_entries <= params.N_micro * (params.N_micro - 1)) "Trying to get $(N_entries) entries, only $(params.N_micro*(params.N_micro-1)) available!"
  @info "Sampling $(N_entries) entries from g_init (max: $(params.N_micro*(params.N_micro-1)))"

  adj_matrix = fast_sampling(params, α_init_samples, N_entries; symmetric=symmetric)

  # compute the discrepency between A and the #I_i
  sharp_I = params.N_micro * sum(α_init_samples; dims=2) * s
  discrepency = sum(adj_matrix; dims=2) - sharp_I

  @info "Theoretical sparsity: $(params.connection_density)"
  @info "     Actual sparsity: $(SpA.nnz(adj_matrix)/params.N_micro/params.N_micro)"

  @info "Discrepency Σ_j A_ij - #I_i (params.connection_density=$(params.connection_density)):"
  @info "\n" * string(UnicodePlots.histogram(discrepency; nbins=50, vertical=true))

  @info "Heat map, α_init(ω_i, ω_j):"
  println(UnicodePlots.heatmap(α_init_samples; width=80, height=80))

  @info "Adjacency matrix:\n" * string(UnicodePlots.spy(reverse(adj_matrix; dims=1); width=80, height=30))

  @info "Heat map, g_init(ω_i, ω_j):"
  println(UnicodePlots.heatmap(g_init_samples; width=80, height=80))

  if !Graphs.is_connected(Graphs.SimpleGraph(adj_matrix))
    @warn "The adjacency matrix is not connected!"
  end

  return adj_matrix
end

"""
    scale_f_α(params::NamedTuple)

Scaled the functions f and α, returns them in a NamedTuple along with g (scaled)
"""
function scale_f_α(params::NamedTuple)
  if !params.constant_α
    _α = params.α_init_func
  else
    _α = (_) -> params.connection_density
  end

  x = build_x(params.N_sampling)
  f_init_func_scaled = scale_to_pdf(params.f_init_func, x)

  # DEFINITION G
  g_init_unscaled_uni = (x -> _α(x[1], x[2]) * params.f_init_func(x[1]) * params.f_init_func(x[2]) / params.connection_density)
  g_prefactor = OpiForm.scale_g_init(g_init_unscaled_uni)
  α_init_func_scaled = (ω, m) -> g_prefactor * _α(ω, m)
  # DEFINITION_G
  g_init_func_scaled = (ω, m) -> α_init_func_scaled(ω, m) * params.f_init_func(ω) * params.f_init_func(m) / params.connection_density

  return (
    f_init_func_scaled=f_init_func_scaled,
    g_init_func_scaled=α_init_func_scaled,
    α_init_func_scaled=g_init_func_scaled
  )
end

# filter for NamedTuples only from v1.11
filter_nt_fields = if VERSION < v"1.11"
  (f, nt) -> begin
    NamedTuple{filter(f, keys(nt))}(nt)
  end
else
  filter
end


function initialize_LFR(params::NamedTuple, lfr_args...; lfr_kwargs...)

  lfr_kwargs_nt = (; lfr_kwargs...)

  # The β distribution has support on [0, 1], these are helper functions to scale to and back from [0, 1]
  scale_to_01 = x -> 0.5 * (x + 1)
  scale_from_01 = x -> 2x - 1

  GEN_KEYS = [:is_directed, :nmin, :nmax, :tau, :tau2, :fixed_range, :mixing_parameter,
    :overlapping_nodes, :overlap_membership, :excess, :defect, :seed, :clustering_coeff]

  lfr_gen_kwargs = filter_nt_fields(in(GEN_KEYS), lfr_kwargs_nt)

  g, c_ids = LFR.lancichinetti_fortunato_radicchi(params.N_micro, lfr_args...; seed=rand(Int32), lfr_gen_kwargs...)

  idc_sort = sortperm(c_ids)
  inv_idc_sort = invperm(idc_sort)
  c_ids_sorted = c_ids[idc_sort]

  n_communities = length(unique(c_ids_sorted))
  @info "[LFR] generated graph with $(n_communities) communities"
  #
  # rescale bounds to [0, 1]
  bounds = collect(lfr_kwargs_nt.µ_community_bounds)
  bounds_01 = scale_to_01.(bounds)

  if lfr_kwargs_nt.µ_community_distrib == :equidistributed
    community_means = collect(range(bounds_01..., length=n_communities))[Random.randperm(n_communities)]
  elseif lfr_kwargs_nt.μ_community_distrib == :uniform
    community_means = rand(n_communities) * (bounds_01[2] - bounds_01[1]) .+ bounds_01[1]
  else
    @error "unkown value '$(lfr_kwargs_nt.μ_community_distrib)' for parameter µ_community_distrib"
  end

  ω_0 = zeros(params.N_micro)

  got = 0

  for i in 1:n_communities
    µ = community_means[i]
    σ² = lfr_kwargs_nt.β_σ²
    ν = µ * (1 - µ) / σ² - 1
    local a = µ * ν
    local b = (1 - µ) * ν
    beta_dist = Distributions.Beta(a, b)

    idc = searchsorted(c_ids_sorted, i)

    community_size = length(idc)

    got += length(idc)

    samples = Distributions.rand(beta_dist, community_size)
    ω_0[idc] .= scale_from_01.(samples)
  end

  ω_0 = ω_0[inv_idc_sort]

  return SpA.sparse(g), ω_0, c_ids
end

function prepare_initial_data(store_dir::String, params::NamedTuple, mode::Symbol)
  if mode == :micro || (mode == :mfl && (params.init_method_f == :from_kde_omega || params.init_method_g == :from_kde_adj_matrix))
    if params.init_method_omega == :from_file
      # Load ops from HDF5 file
      iter = get(params, :init_micro_iter, 0)
      if iter == 0
        ω_0 = load_hdf5_data(params.init_micro_filename, "omega_init")
      else
        ω_0 = load_hdf5_data(params.init_micro_filename, "omega/$iter")
      end
      @assert size(ω_0) == (params.N_micro,) "The size of ω_0 from the file is different from N_micro ($(size(ω_0)) vs $(params.N_micro))"
    elseif params.init_method_omega == :from_sampling_f_init
      # Scale and sample f_init
      @info "Sampling f_init to get ω_0"
      ω_0 = fast_sampling(params.f_init_func, params.N_micro, params.N_sampling)
    elseif params.init_method_omega == :from_lfr
      @assert params.init_method_adj_matrix == :from_lfr "init_method_omega set to :from_ldr but init_method_adj_matrix is not!"
    else
      throw("Unknown value $(params.init_method_omega) for parameter init_method_omega")
    end

    if params.init_method_adj_matrix == :from_file
      # Load adj_matrix from HDF5 file
      adj_matrix = load_hdf5_sparse(params.init_micro_filename, "adj_matrix")
      @assert size(adj_matrix) == (params.N_micro, params.N_micro)
    elseif params.init_method_adj_matrix == :from_sampling_α_init
      scaled_funcs = scale_f_α(params)
      adj_matrix = sample_g_init(params, scaled_funcs, ω_0)
    elseif params.init_method_adj_matrix == :from_graph
      tries = 0
      graph = Graphs.Graph()
      while tries < params.init_micro_graph_max_tries
        tries += 1
        graph = getfield(Graphs.SimpleGraphs, params.init_micro_graph_type)(params.init_micro_graph_args...; params.init_micro_graph_kwargs...)
        if Graphs.is_connected(graph)
          break
        end
      end
      if tries == params.init_micro_graph_max_tries
        throw("Could not generate a connected graph after $(tries) tries")
      end
      adj_matrix = SpA.sparse(graph)
      @assert !isnothing(adj_matrix)
    elseif params.init_method_adj_matrix == :from_lfr
      tries = 0
      adj_matrix, ω_0, c_ids = nothing, nothing, nothing
      while tries < params.init_lfr_max_tries
        tries += 1
        adj_matrix, ω_0, c_ids = initialize_LFR(params, params.init_lfr_args...; params.init_lfr_kwargs...)
        if params.init_lfr_target_n_communities > 0 && (length(unique(c_ids)) == params.init_lfr_target_n_communities)
          break
        end
      end
      if isnothing(adj_matrix)
        throw(ErrorException("Could not generate a LFR graph with $(params.init_lfr_target_n_communities) communities with the given parameters"))
      end
    else
      throw("Unknown value $(params.init_method_adj_matrix) for parameter init_method_adj_matrix")
    end

    # store data
    store_hdf5_data(joinpath(store_dir, "data.hdf5"), "omega_init", ω_0)
    store_hdf5_sparse(joinpath(store_dir, "data.hdf5"), "adj_matrix", adj_matrix)

    ##################
    ### Plot graph ###
    ##################

    # Switch to CairoMakie
    # CairoMakie.activate!()

    # Define the colormap
    cmap(x) = x < 0 ? CairoMakie.Makie.RGB{Float64}(1.0, 1 + x, 1 + x) : CairoMakie.Makie.RGB{Float64}(1 - x, 1 - x, 1.0)
    node_colors = map(cmap, ω_0)
    graph = Graphs.SimpleGraphs.SimpleGraph(adj_matrix)
    n_edges = Graphs.ne(graph)

    # Plot the graph using 3 different layouts
    for layout in [:Stress, :Spring, :Shell]
      layout_name = lowercase(string(layout))
      if layout == :Stress && !Graphs.is_connected(graph)
        @warn "The graph is not connected, skipping layout '$layout_name'"
        continue
      end
      layout_fn = joinpath(store_dir, "graph_$(layout_name).svg")
      try
        fig = CairoMakie.Figure(size=(2000, 2000))
        ax = CairoMakie.Axis(fig[1, 1])
        CairoMakie.hidedecorations!(ax)
        GraphMakie.graphplot!(ax, graph; layout=getfield(GraphMakie, layout)(), node_color=node_colors, alpha=0.1, edge_width=0.1)
        CairoMakie.save(layout_fn, fig)
        @info "Graph view saved at $(layout_fn)"
      catch
        @warn "Graph view failed with layout '$layout_name' (disconnected graph?)"
      end
    end

    if params.init_method_adj_matrix == :from_lfr
      fig = M.Figure(size=(2048, 1152))
      c = M.Makie.ColorSchemes.Paired_12.colors

      n_colors = length(c)

      cs = [c[1+(i%n_colors)] for i in c_ids]

      n_communities = length(unique(c_ids))

      ax1 = M.Axis(fig[1, 1], title="LFR graph (colored by community)", xticklabelsvisible=false, yticklabelsvisible=false)
      GraphMakie.graphplot!(ax1, graph; edge_width=0.1, node_color=cs, node_size=12)

      ax2 = M.Axis(fig[1, 3], title="LFR graph (colored by opinion)", xticklabelsvisible=false, yticklabelsvisible=false)
      g_ops = GraphMakie.graphplot!(ax2, graph; edge_width=0.1, node_color=ω_0, node_size=12,
        node_attr=(colorrange=(-1, 1),))
      fig[2, 3] = cb = M.Colorbar(fig, g_ops, label="Opinions", vertical=false)

      ax3 = M.Axis(fig[1, 2], title="Opinions histograms (aggregated by community)", xticklabelsvisible=true, yticklabelsvisible=false, xlabel=M.L"\omega_i")
      idc_sort = sortperm(c_ids)
      c_ids_sorted = c_ids[idc_sort]
      ω_0_sorted = ω_0[idc_sort]
      for i in 1:n_communities
        idc = searchsorted(c_ids_sorted, i)

        M.hist!(ax3, ω_0_sorted[idc], scale_to=1.0, offset=i, direction=:y, bins=range(-1, 1, 200), color=c[1+(i%n_colors)])
      end

      lfr_plot_fn = joinpath(store_dir, "graph_LFR.svg")
      M.save(lfr_plot_fn, fig)
      @info "LFR Graph view saved at $(lfr_plot_fn)"

      open(joinpath(store_dir, "c_ids.csv"), "w") do c_ids_csv
        writedlm(c_ids_csv, c_ids)
      end
    end

    #GLMakie.activate!()
  end
  if mode == :mfl
    x = build_x(params.N_mfl)
    δx = x[2] - x[1]
    if params.init_method_f == :from_file
      # Load data from HDF5 file
      iter = get(params, :init_mfl_iter, 0)
      f_0 = load_hdf5_data(params.init_mfl_filename, "f/$iter")
      @assert size(f_0) == (params.N_mfl, 1)
    elseif params.init_method_f == :from_f_init
      # Evaluate and scale f_init
      f_0 = params.f_init_func.(x)
      f_0_integral = δx * sum(f_0)
      f_0 ./= f_0_integral
    elseif params.init_method_f == :from_kde_omega
      # Compute KDE
      f_0 = compute_kde(x, ω_0)
    else
      throw("Unknown value $(params.init_method_f) for parameter init_method_f")
    end
    if params.init_method_g == :from_file
      # Load data from HDF5 file
      iter = get(params, :init_mfl_iter, 0)
      g_0 = load_hdf5_data(params.init_mfl_filename, "g/$iter")
      @assert size(g_0) == (params.N_mfl, params.N_mfl)
      α = nothing
    elseif params.init_method_g == :from_α_init
      # Evaluate and scale f_init * α_init * f_init' / connection_density
      α = [params.α_init_func(x[i], x[j]) for i in eachindex(x), j in eachindex(x)]
      g_0 = f_0 .* α .* f_0' / params.connection_density
      g_0_integral = δx^2 * sum(g_0)
      g_0 ./= g_0_integral
    elseif params.init_method_g == :from_g_init
      g_0 = [params.g_init_func(x[i], x[j]) for i in eachindex(x), j in eachindex(x)]
    elseif params.init_method_g == :from_kde_adj_matrix
      @assert !isnothing(adj_matrix) "The adjacency matrix is Nothing. I don't know how to perform a KDE if full_adj_matrix is set."
      # Compute KDE
      g_0 = compute_kde(x, adj_matrix, ω_0)
      α = nothing
    else
      throw("Unknown value $(params.init_method_g) for parameter init_method_g")
    end
    success = store_hdf5_data(joinpath(store_dir, "data.hdf5"), [
      "f_init" => f_0, "g_init" => g_0, "alpha" => α,
    ])
    if !success
      @warn "Failed to store MFL initial data correctly!"
    end
  end

end

function prepare_directory(store_dir::String, params::NamedTuple, mode::Symbol; force::Bool=false)

  if isdir(store_dir)
    if force
      @warn "Removing test dir $(store_dir)"
      rm(store_dir; force=true, recursive=true)
    else
      throw("Directory $(store_dir) already exists!")
    end
  end

  mkpath(store_dir)
  @info "Output set to $(store_dir)/"

  fmt_logger = LoggingExtras.FormatLogger(joinpath(store_dir, "output_" * string(mode) * ".log"); append=true) do io, args
    if args.level == LoggingExtras.Info
      println(io, "[", args.level, "] ", args.message)
    else
      println(io, "[", args.level, "] ", args.message, "\n", "@ ", args._module, " ", args.file, ":", args.line)
    end
  end

  if params.log_debug
    console_log_level = Logging.Debug
  else
    console_log_level = Logging.Info
  end

  console_logger = Logging.ConsoleLogger(stdout, console_log_level)

  tee_logger = LoggingExtras.TeeLogger(
    console_logger,
    LoggingExtras.MinLevelLogger(fmt_logger, LoggingExtras.Info)
  )
  Logging.global_logger(tee_logger)

  @info Dates.now()

  try
    dirty = length(read(`git status -s -uno --porcelain`, String)) > 0
    if dirty
      @warn "Working tree is dirty!"
      @warn read(`git status -uno`, String)
    end
  catch
    @warn "git status failed"
  end

  # TODO: remove this and fix Plotting.jl accordingly
  meta = Dict(
    "julia_version" => string(VERSION),
    "connection_density" => params.connection_density,
    "delta_t" => params.δt,
    "flux" => string(params.flux),
    "N_mfl" => params.N_mfl,
    "N_micro" => params.N_micro
  )
  try
    meta["commit"] = read(`git show -s --oneline`, String)
  catch
    meta["commit"] = "failed"
    @warn "git show failed"
  end

  open(joinpath(store_dir, "metadata.toml"), "w") do metafile
    TOML.print(metafile, meta)
  end

  prepare_initial_data(store_dir, params, mode)

  serialize_params(store_dir, params)

  Params.to_toml(store_dir, params)
end
