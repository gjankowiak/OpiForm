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
  HDF5.h5open(filename, "cw") do h5
    h5["$key/i"] = i
    h5["$key/j"] = j
    h5["$key/v"] = v
  end
end

function load_hdf5_sparse(filename::String, key::String)::SpA.SparseMatrixCSC
  h5 = HDF5.h5open(filename, "r")
  try
    i = read(h5["$key/i"])
    j = read(h5["$key/j"])
    v = read(h5["$key/v"])
    close(h5)
    return SpA.sparse(i, j, v)
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
