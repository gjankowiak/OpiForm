module Micro

import HDF5
import UnicodePlots
import ..OpiForm: prepare_directory, load_hdf5_data, store_hdf5_data, load_hdf5_sparse, @fmt

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

function compute_diffs!(dst::Matrix{Float64}, params::NamedTuple, ω::Vector{Float64})
  # ω_diff[i,j] = ω[j] - ω[i]
  for j in 1:params.N_micro
    dst[:, j] .= ω[j] .- ω
  end

  # slower:
  #dst .= -repeat(ω, 1, params.N_micro) + repeat(ω, 1, params.N_micro)'
end

function compute_diffs(params::NamedTuple, ω::Vector{Float64})
  return -repeat(ω, 1, params.N_micro) + repeat(ω', params.N_micro, 1)
end

function delta_EC!(inc::Matrix{Float64}, params::NamedTuple, ω::Vector{Float64}, ω_diff::Matrix{Float64})
  repeat_ω = repeat(ω', params.N_micro, 1)

  B_ρ = -params.EC_ρ .<= ω_diff .<= params.EC_ρ
  B_lower = -params.EC_ρ .> ω_diff
  B_higher = ω_diff .> params.EC_ρ

  size_B_ρ = vec(sum(B_ρ; dims=2))
  size_B_lower = vec(sum(B_lower; dims=2))
  size_B_higher = vec(sum(B_higher; dims=2))

  μ_ρ = vec(sum(B_ρ .* repeat_ω; dims=2)) ./ max.(1, size_B_ρ)
  μ_lower = vec(sum(B_lower .* repeat_ω; dims=2)) ./ max.(1, size_B_lower)
  μ_higher = vec(sum(B_higher .* repeat_ω; dims=2)) ./ max.(1, size_B_higher)

  delta_group = vec(sum(B_ρ .* ω_diff; dims=2))

  # Attracting term
  inc[:, 2] = delta_group / params.N_micro

  # Repulsing term
  inc[:, 3] = (-size_B_lower .* (μ_lower - µ_ρ) - size_B_higher .* (μ_higher - µ_ρ)) / params.N_micro

end

function delta_EC_potential!(inc::Matrix{Float64}, params::NamedTuple, ω::Vector{Float64}, ω_diff::Matrix{Float64}, ∇u)
  ω_diff .= ∇u.(abs.(ω_diff)) .* sign.(ω_diff)
  inc[:, 2] = vec(sum(ω_diff; dims=2)) / (params.N_micro - 1)
end

function delta_EB!(inc::Matrix{Float64}, tmp::Matrix{Float64}, params::NamedTuple, ω::Vector{Float64}, ω_diff::Matrix{Float64}, adj_mat, neighbors::Vector{Int64})
  if params.full_adj_matrix
    tmp .= ω_diff
  else
    tmp .= ω_diff .* adj_mat
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

function launch(store_dir::String, params::NamedTuple; force::Bool=false)

  prepare_directory(store_dir, params, :micro; force=force)

  ω = load_hdf5_data(joinpath(store_dir, "data.hdf5"), "omega_init")
  adj_matrix = load_hdf5_sparse(joinpath(store_dir, "data.hdf5"), "adj_matrix")

  i = 1

  ω_diff = Matrix{Float64}(undef, params.N_micro, params.N_micro)
  tmp = Matrix{Float64}(undef, params.N_micro, params.N_micro)

  if params.full_adj_matrix
    neighbors = fill(params.N_micro - 1, (params.N_micro,))
  else
    neighbors = Vector{Int64}(vec(max.(1, sum(adj_matrix; dims=2) .- 1)))
  end
  @info neighbors[1]

  # increment
  # 1st column: EB                   | (Debate)
  # 2nd column: EC (attractive term) | (Radicalization)
  # 3rd column: EC (repulsive term)  | (Polarization)
  inc = zeros(params.N_micro, 3)

  store_i = [0]
  store_ω = [copy(ω)]

  factors = [params.σ; 1 - params.σ; 1 - params.σ]

  while i <= params.max_iter
    i += 1
    if i % 10 == 0
      print(" [i=$(lpad(i, 5, " "))]" * "\b"^100)
    end

    compute_diffs!(ω_diff, params, ω)

    if params.σ < 1
      throw("Not implemented")
      delta_EC_potential!(inc, params, ω, ω_diff, ∇u)
    end

    if params.σ > 0
      delta_EB!(inc, tmp, params, ω, ω_diff, adj_matrix, neighbors)
    end

    ω .= clamp.(ω .- params.δt * vec(inc * factors), -1, 1)

    if i % params.store_every_iter == 0
      push!(store_i, i)
      push!(store_ω, copy(ω))
    end

  end

  println()

  @info "Saving data to disk @ $(store_dir)"
  store_hdf5_data(joinpath(store_dir, "data.hdf5"), [
    "i" => store_i, "omega" => hcat(store_ω...)
  ])

  @info "Done"

end

end
