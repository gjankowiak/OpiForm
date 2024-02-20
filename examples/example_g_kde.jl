import Graphs
import KernelDensity
import UnicodePlots
# import CairoMakie as M
import GLMakie as M
import GraphMakie as GM
import SparseArrays as SpA

function gauss(x; σ::Float64=1.0, μ::Float64=0.0)
  return 1 / (sqrt(2π) * σ) * exp(-0.5 * ((x - μ) / σ)^2)
end

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

function sample_normal_interval(n::Int64; σ::Real=1, bounds::Tuple{T1,T2}) where {T1,T2<:Real,Real}
  ω = Float64[]
  while length(ω) < n
    need = n - length(ω)
    new_samples = filter(x -> bounds[1] .< x .< bounds[2], randn(need) * σ)
    append!(ω, new_samples)
  end
  return ω
end

function get_ωx_ωy(adj_matrix::SpA.SparseMatrixCSC{Int64,Int64}, ω::Vector{Float64})
  nnz = SpA.nnz(adj_matrix)
  ωx, ωy = zeros(nnz), zeros(nnz)
  get_ωx_ωy!(ωx, ωy, adj_matrix, ω)
  return ωx, ωy
end

function compute_g_kde(n, graph_type, graph_args)
  σ = 1 / 3

  ω = sample_normal_interval(n; σ=σ, bounds=(-1, 1))

  println(UnicodePlots.histogram(ω, nbins=50, vertical=true))

  g = getfield(Graphs.SimpleGraphs, graph_type)(graph_args...)
  A = SpA.sparse(g)

  ωx, ωy = get_ωx_ωy(A, ω)

  kde_ω = KernelDensity.kde(ω)
  interp_kde_ω = KernelDensity.InterpKDE(kde_ω)

  kde_adj_matrix = KernelDensity.kde((ωx, ωy))
  interp_kde_adj_matrix = KernelDensity.InterpKDE(kde_adj_matrix)

  n_x = 1001
  x = range(-1, 1; length=n_x)
  δx = 2 / n_x

  eval_kde_adj_matrix = [KernelDensity.pdf(interp_kde_adj_matrix, _x, _y) for _x in x, _y in x]

  fig = M.Figure()

  ax_topleft = M.Axis(fig[1, 1], aspect=1, title="Histogram of ω  and corresponding KDE (N=$n)")
  ax_topright = M.Axis(fig[1, 3], aspect=1, title="Kernel Density Estimation (KDE) of gᴺ")
  ax_topmid = M.Axis(fig[1, 2], aspect=1, backgroundcolor=:black, title="2D histogram for gᴺ")
  ax_bottommid = M.Axis(fig[2, 2], aspect=1, title="Histogram for the nodes' degree")
  ax_bottomleft = M.Axis(fig[2, 1], aspect=1, title="Structure of the graph ($graph_type$graph_args)")
  ax_bottomright = M.Axis(fig[2, 3], aspect=1, title="Histogram for ωᵢ · #Iᵢ")

  M.lines!(ax_topleft, x, KernelDensity.pdf(interp_kde_ω, x))
  #M.lines!(ax_topleft, x, gauss.(x; σ=σ))
  M.hist!(ax_topleft, ω, bins=50, normalization=:pdf)

  M.hist!(ax_bottommid, vec(sum(A; dims=1)))

  M.hist!(ax_bottomright, ω .* vec(sum(A; dims=1)))

  n_edges = SpA.nnz(A)
  if graph_type == :barabasi_albert
    edge_width = 10 / sqrt(n_edges)
  else
    edge_width = 500 / n_edges
  end
  GM.graphplot!(ax_bottomleft, g, alpha=0.1, edge_width=edge_width, node_size=0.5)

  M.heatmap!(ax_topright, x, x, eval_kde_adj_matrix, colormap=:ice)
  M.hexbin!(ax_topmid, ωx, ωy, cellsize=2 / 100, colormap=:ice, threshold=0)

  display(fig)
end

n = 500

# see https://juliagraphs.org/Graphs.jl/dev/core_functions/simplegraphs_generators

# Low degree for most nost, a few nodes of high degree
# compute_g_kde(n, :dorogovtsev_mendes, (n,))
#
# Most nodes with degree k, but with fat tail (towards high degree)
k = n ÷ 5
compute_g_kde(n, :barabasi_albert, (n, k))

# Degree mostly the same for all node, given by n*d 
# d = 2e-2
# compute_g_kde(n, :erdos_renyi, (n, d))

# Two complete subgrapgs (cliques) connected by 1 edge
# compute_g_kde(n, :barbell_graph, (n - n ÷ 2, n ÷ 2))
