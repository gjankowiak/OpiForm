module Discrete

import HDF5
import UnicodePlots
import ..OpiForm: prepare, @fmt

# Move to parent?
function build_adj_matrix(mode, Ns; density=1.0)
  if mode == :clusters
    N = sum(Ns)
    M = BitMatrix(undef, N, N)
    fill!(M, 0)
    local s = 1
    for n in Ns
      fill!(view(M, s:s+n-1, s:s+n-1), 1)
      s += n
    end
    return M
  elseif mode == :density
    if density == 0
      return zeros(N, N)
    end
    M = 0.5rand(N, N)
    return (M + M') .< density
  else
    throw(ArgumentError("unknown mode"))
  end
end

function compute_diffs!(dst::Matrix{Float64}, params::NamedTuple, ops::Vector{Float64})
  # ops_diff[i,j] = ops[j] - ops[i]
  for j in 1:params.N_discrete
    dst[:, j] .= ops[j] .- ops
  end

  # slower:
  #dst .= -repeat(ops, 1, params.N_discrete) + repeat(ops, 1, params.N_discrete)'
end

function compute_diffs(params::NamedTuple, ops::Vector{Float64})
  return -repeat(ops, 1, params.N_discrete) + repeat(ops', params.N_discrete, 1)
end

function delta_EC!(inc::Matrix{Float64}, params::NamedTuple, ops::Vector{Float64}, ops_diff::Matrix{Float64})
  repeat_ops = repeat(ops', params.N_discrete, 1)

  B_ρ = -params.EC_ρ .<= ops_diff .<= params.EC_ρ
  B_lower = -params.EC_ρ .> ops_diff
  B_higher = ops_diff .> params.EC_ρ

  size_B_ρ = vec(sum(B_ρ; dims=2))
  size_B_lower = vec(sum(B_lower; dims=2))
  size_B_higher = vec(sum(B_higher; dims=2))

  μ_ρ = vec(sum(B_ρ .* repeat_ops; dims=2)) ./ max.(1, size_B_ρ)
  μ_lower = vec(sum(B_lower .* repeat_ops; dims=2)) ./ max.(1, size_B_lower)
  μ_higher = vec(sum(B_higher .* repeat_ops; dims=2)) ./ max.(1, size_B_higher)

  delta_group = vec(sum(B_ρ .* ops_diff; dims=2))

  # Attracting term
  inc[:, 2] = delta_group / params.N_discrete

  # Repulsing term
  inc[:, 3] = (-size_B_lower .* (μ_lower - µ_ρ) - size_B_higher .* (μ_higher - µ_ρ)) / params.N_discrete

end

function delta_EC_potential!(inc::Matrix{Float64}, params::NamedTuple, ops::Vector{Float64}, ops_diff::Matrix{Float64}, ∇u)
  ops_diff .= ∇u.(abs.(ops_diff)) .* sign.(ops_diff)
  inc[:, 2] = vec(sum(ops_diff; dims=2)) / (params.N_discrete - 1)
end

function delta_EB!(inc::Matrix{Float64}, tmp::Matrix{Float64}, params::NamedTuple, ops::Vector{Float64}, ops_diff::Matrix{Float64}, adj_mat, neighbors::Vector{Int64})
  if params.full_adj_matrix
    tmp .= ops_diff
  else
    tmp .= ops_diff .* adj_mat
  end
  inc[:, 1] = vec(sum(params.D_func, tmp; dims=2)) ./ neighbors
end

function affinity!(M, x; diag_factor=1e-1)
  throw("Not implemented")

  # Set all columns of M to -x
  M .= -x

  # Add x to all rows of M
  M .+= x'

  # Square all elements and flip the sign, so that Mij = -|xi - xj|²
  M .^= 2
  M .*= -1

  # The value of the diagonal elements of M controls (in some way)
  # the number of clusters resulting from affinity propagation.
  # In our case, if Mii is close to 0, there are many clusters,
  # and fewer as Mii goes to min Mij.
  m = minimum(M)

  #M[diagind(M)] = diag_factor * m

  return M
end

function launch(params_in, store_path)
  params = merge(params_in, (
    store_path=store_path,
  ))

  prepare(params, :discrete)

  #M = zeros(params.N_discrete, params.N_discrete)

  ops = copy(params.ops_init)

  i = 1

  ops_diff = Matrix{Float64}(undef, params.N_discrete, params.N_discrete)
  tmp = Matrix{Float64}(undef, params.N_discrete, params.N_discrete)

  if params.full_adj_matrix
    neighbors = fill(params.N_discrete - 1, (params.N_discrete,))
  else
    neighbors = Vector{Int64}(vec(max.(1, sum(params.adj_matrix; dims=2) .- 1)))
  end
  @info neighbors[1]

  # increment
  # 1st column: EB                   | (Debate)
  # 2nd column: EC (attractive term) | (Radicalization)
  # 3rd column: EC (repulsive term)  | (Polarization)
  inc = zeros(params.N_discrete, 3)

  if params.store
    store_i = [0]
    store_ops = [copy(ops)]
  end

  factors = [params.σ; 1 - params.σ; 1 - params.σ]

  while i <= params.max_iter
    i += 1
    if i % 10 == 0
      print(" [i=$(lpad(i, 5, " "))]" * "\b"^100)
    end

    compute_diffs!(ops_diff, params, ops)

    if params.σ < 1
      throw("Not implemented")
      delta_EC_potential!(inc, params, ops, ops_diff, ∇u)
    end

    if params.σ > 0
      delta_EB!(inc, tmp, params, ops, ops_diff, params.adj_matrix, neighbors)
    end

    ops .= clamp.(ops .- params.δt * vec(inc * factors), -1, 1)

    if params.store && i % params.store_every_iter == 0
      push!(store_i, i)
      push!(store_ops, copy(ops))
    end

  end

  println()

  if params.store
    @info "Saving data to disk @ $(params.store_path)"
    HDF5.h5open(joinpath(params.store_path, "data_discrete.h5"), "w") do fid
      fid["i"] = store_i
      fid["ops"] = hcat(store_ops...)
      if !params.full_adj_matrix
        fid["adj_matrix"] = Matrix(params.adj_matrix)
      end
    end
  end

  @info "Done"

end

function run_and_plot(Ns, ∇u, ops_zero, params; old_plot=Missing)
  M = build_adj_matrix(:clusters, Ns)

  hist = run(ops_zero, M, ∇u, params)'
  return plot_hist(Ns, hist, params; old_plot=old_plot)
end
end

