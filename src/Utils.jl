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


"""
    find_free_suffix(prefix::String; suffix_length::Int64=3)

Return an inexisting filename of the form `prefix`_`suffix`, where `suffix` represents an integer, left padded with zero to a length of `suffix_length`.
"""
function find_free_suffix(prefix::String; suffix_length::Int64=3)
  for k in 0:(10^suffix_length-1)
    fn = string(prefix, "_", lpad(k, suffix_length, '0'))
    if !isfile(fn)
      return fn
    end
  end
  throw("Could not find a free suffix of length $suffix_length for prefix $prefix")
end

function symmetry_defect(A::Matrix{T}) where {T<:Real}
  if size(A, 1) != size(A, 2)
    return Inf
  end
  N = size(A, 1)
  m = 0
  i_defect, j_defect = 0, 0
  @inbounds for i in 1:N
    for j in i:N
      d = abs(A[i, j] - A[j, i])
      if d > m
        m = d
        i_defect, j_defect = i, j
      end
    end
  end
  return m, (i_defect, j_defect)
end

function issymmetric(A::Matrix{T}; tol::Float64=1e-8) where {T<:Real}
  if size(A, 1) != size(A, 2)
    return false
  end
  N = size(A, 1)
  @inbounds for i in 1:N
    for j in i:N
      if abs(A[i, j] - A[j, i]) > tol
        return false
      end
    end
  end
  return true
end

function find_support_bounds(f::Array{Float64,2}, x; tol::Float64=1e-5)
  # If no element is larger than tol, we return 0, so that the choice of the value in
  # the ternary operator (here, x[1]) is arbitrary
  left_bound = col -> (idx = findfirst(x -> x > tol, col); return isnothing(idx) ? x[1] : x[idx])
  right_bound = col -> (idx = findlast(x -> x > tol, col); return isnothing(idx) ? x[1] : x[idx])
  return [(left_bound(f[:, i]), right_bound(f[:, i])) for i in axes(f, 2)]
end

function find_support_bounds(f_a::Vector{Array{Float64,2}}, x_a::Vector; tol::Float64=1e-5)
  return map(i -> find_support_bounds(f_a[i], x_a[i]; tol=tol), eachindex(f_a))
end

function peak2peak(v; dims)
  return vec(map(x -> x[2] - x[1], extrema(v; dims=dims)))
end

function get_memory_usage(type)
  mem = parse(Int, chomp(read(`ps -p $(getpid()) -h -o size`, String)))
  if type == String
    gbs = Int(floor(mem / 1_000_000))
    mbs = Int(ceil((mem - gbs * 1_000_000) / 1_000))
    if gbs > 0
      return "$(gbs).$(mbs) GB"
    else
      return "$(mbs) MB"
    end
  elseif type in [Int, UInt]
    return mem
  else
    throw("not implemented for type $(type)")
  end
end

function clip(x, v)
  return (x > v) ? x : 0.0
end

function rand_symmetric(N, δ)
  v = rand(N, N)
  # symmetrize
  v = 0.5 * (v + v')

  # rescale
  m, M = extrema(v)
  v = (v .- m) / (M - m)

  # map
  f = x -> x <= 0.5 ? 2x : (2x - 1)
  v = f.(v)

  return v .< δ
end

function speyes(N, n_groups)
  # compute group sizes
  d, r = divrem(N, n_groups)
  sizes = [k <= r ? d + 1 : d for k in 1:n_groups]

  # build sparse matrix
  M_sparse = SpA.blockdiag([
    SpA.sparse(ones(s, s)) for s in sizes
  ]...)

  nnz = sum(sizes .^ 2)

  # Check for memory efficiency
  # https://en.wikipedia.org/wiki/Sparse_matrix
  if nnz < 0.5 * (N * (N - 1) - 1)
    # we can keep the sparse matrix
    return M_sparse
  else
    return Array(M_sparse)
  end
end

function build_x(N)
  δx = 2 / N
  x_l, x_r = -1 + 0.5δx, 1 - 0.5δx

  return range(x_l, x_r, length=N)
end

function load_hdf5_data(filename::String, key::String)
  h5 = HDF5.h5open(filename, "r")
  try
    r = read(h5[key])
    close(h5)
    return r
  catch
    close(h5)
    @warn "Failed to load value for key '$key' from file $filename"
    return nothing
  end
end

"""
    store_hdf5_data(filename::String, key_data_pairs::Vector{Pair{String,T}}) where {T<:Any}

Store open the HDF5 storage at `filename`, write the non-nothing values to it and close it.
"""
function store_hdf5_data(filename::String, key_data_pairs::Vector{Pair{String,T}}) where {T<:Any}
  h5 = HDF5.h5open(filename, "cw")
  for (key, data) in key_data_pairs
    if isnothing(data)
      continue
    end
    try
      h5[key] = data
    catch
      close(h5)
      return false
    end
  end
  close(h5)
  return true
end

function store_hdf5_data(filename::String, key::String, data::Any)
  return store_hdf5_data(filename, [key => data])
end

function store_hdf5_sparse(filename::String, key::String, A::SpA.SparseMatrixCSC)
  i, j, v = get_ijv(A)
  (m, n) = size(A)
  HDF5.h5open(filename, "cw") do h5
    h5["$key/i"] = i
    h5["$key/j"] = j
    h5["$key/v"] = v
    h5["$key/m"] = m
    h5["$key/n"] = n
  end
end


function load_hdf5_sparse(filename::String, key::String; cids_fn::String="")::SpA.SparseMatrixCSC

  function get_ids(fn)
    f = open(fn, "r")
    return map(Int, vec(readdlm(f)))
  end

  h5 = HDF5.h5open(filename, "r")
  try
    m = read(h5["$key/m"])
    n = read(h5["$key/n"])
    i = read(h5["$key/i"])
    j = read(h5["$key/j"])
    v = read(h5["$key/v"])
    close(h5)

    if !isempty(cids_fn)
      c_ids = get_ids(cids_fn)
      idx_a = sortperm(c_ids)
      inv_idx_a = invperm(idx_a)

      rank = (i) -> inv_idx_a[i]

      i_sorted = collect(map(rank, i))
      j_sorted = collect(map(rank, j))
      v_sorted = collect(map(rank, v))

      return SpA.sparse(i_sorted, j_sorted, v_sorted, m, n)
    else
      return SpA.sparse(i, j, v, m, n)
    end

  catch
    close(h5)
    return nothing
  end
end

"""
    get_ijv(A::SA.SparseMatrixCSC)

Return the I, J and V vectors needed to build a SparseMatrixCSC
"""
function get_ijv(A::SpA.SparseMatrixCSC)
  v = SpA.nonzeros(A)
  i = SpA.rowvals(A)
  j = Array(SpA.ColumnIndices(A))
  return i, j, v
end

function load_metadata(dn::String)
  return TOML.parsefile(joinpath(dn, "metadata.toml"))
end


macro left(v, fill=0.0)
  return esc(:(SA.shiftedarray($v, 1, $fill)))
end

macro right(v, fill=0.0)
  return esc(:(SA.shiftedarray($v, -1, $fill)))
end

macro up_mat(v, fill=0.0)
  return esc(:(SA.shiftedarray($v, (1, 0), $fill)))
end

macro down_mat(v, fill=0.0)
  return esc(:(SA.shiftedarray($v, (-1, 0), $fill)))
end

macro left_mat(v, fill=0.0)
  return esc(:(SA.shiftedarray($v, (0, 1), $fill)))
end

macro right_mat(v, fill=0.0)
  return esc(:(SA.shiftedarray($v, (0, -1), $fill)))
end

function display_params(params::NamedTuple)
  r = "Parameters:\n"
  for k in keys(params)
    r *= rpad(k, 30, ' ') * " : $(params[k])\n"
  end
  @info r
end

function serialize_params(store_dir::String, params::NamedTuple)
  Serialization.serialize(joinpath(store_dir, "params.dat"), params)
end

function deserialize_params(store_dir::String)
  return Serialization.deserialize(joinpath(store_dir, "params.dat"))
end

function longest_prefix(str_a::Array{String}; existing_dir::Bool=false)
  n = length(str_a)

  if n == 0
    return ""
  elseif n == 1
    return str_a[1]
  end

  # sort the array
  sorted_str_a = sort(str_a)

  # find the longest common prefix between the first and last element
  prefix = ""
  i = 1
  first_str = sorted_str_a[1]
  last_str = sorted_str_a[end]
  for i in eachindex(first_str)
    if !(i in eachindex(last_str))
      break
    end
    if first_str[i] == last_str[i]
      prefix *= first_str[i]
      i += 1
    else
      break
    end
  end
  if existing_dir
    if !isdir(prefix)
      prefix = splitdir(prefix)[1]
    end
  end
  return prefix
end

function split_run_path(fns::Vector{String})
  return map(x -> (y = split(x, "-"); (join(y[1:end-1], "-"), last(y))), fns)
end
#
# The β distribution has support on [0, 1], these are helper functions to scale to and back from [0, 1]
scale_to_01 = x -> 0.5 * (x + 1)
scale_from_01 = x -> 2x - 1

function load_lfr_community_data(dir::String)
  c_ids = vec(map(Int, readdlm(joinpath(dir, "c_ids.csv"))))
  c_expectations = vec(readdlm(joinpath(dir, "c_expectations.csv")))
  return c_ids, c_expectations
end

function beta_μσ²_to_ab(μ, σ²)
  ν = µ * (1 - µ) / σ² - 1
  a = µ * ν
  b = (1 - µ) * ν
  return a, b
end
