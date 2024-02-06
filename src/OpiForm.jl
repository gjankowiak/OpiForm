module OpiForm

import UnicodePlots

import Logging
import LoggingExtras

import BenchmarkTools
import StatsBase

import Graphs

import Polynomials
import Roots

import Cubature

import Dates

import ShiftedArrays as SA

import SparseArrays as SpA

import Serialization

import GLMakie
#import WGLMakie
import CairoMakie

import HDF5

import Statistics

macro fmt(v)
  r = string(v)
  return :(string($r, " = ", $(esc(v))))
end

GLMakie.activate!()
M = GLMakie

function set_makie_backend(backend)
  global M
  if backend == :gl
    GLMakie.activate!()
    M = GLMakie
    # elseif backend == :wgl
    #     WGLMakie.activate!()
    #     M = WGLMakie
  elseif backend == :cairo
    CairoMakie.activate!()
    M = CairoMakie
  else
    throw("Unkown backend '%(backend)'")
  end
end

function get_makie_backend()
  return M
end

function sample_poly_dist(poly, n)
  # Samples a polynomial distribution
  I = Polynomials.integrate(poly)
  scaled_I = I - I(-1)
  scaled_I = scaled_I / scaled_I(1)

  img_samples = rand(n)

  samples = sort(collect(map(x -> Roots.find_zero(scaled_I - x, (-1, 1), Roots.Bisection()), img_samples)))

  @info "Histogram of the samples of f_init:\n" * string(UnicodePlots.histogram(samples; nbins=50, vertical=true))

  return samples
end

function scale_g_init(g_init_func, connectivity)
  # Start by normalizing the input distribution
  # so that the connectivity parameter is matched
  g_init_integral = Cubature.hcubature(g_init_func, [-1; -1], [1; 1])[1]

  function g_init_func_scaled(ω, m)
    return g_init_func([ω; m]) / g_init_integral * connectivity
  end

  return g_init_func_scaled
end

function compute_ω_inf_mf(g_init_func)
  g_init_integral = Cubature.hcubature(g_init_func, [-1; -1], [1; 1])[1]
  return Cubature.hcubature(x -> x[1] * g_init_func(x), [-1; -1], [1; 1])[1]
end

function sample_g_init(f_samples, g_init_func_scaled, connectivity; symmetric::Bool=true)
  N_discrete = length(f_samples)

  @assert N_discrete > 0 "No samples provided!"
  @assert connectivity < N_discrete "Connectivity >= N_discrete!"

  # Compute all the couples g_init(ω_i, ω_j)
  # FIXME: use the symmetry to speed up the computation
  g_init_samples = [i == j ? 0.0 : g_init_func_scaled(f_samples[i], f_samples[j]) for i = 1:N_discrete, j = 1:N_discrete]
  sum_g_samples = sum(g_init_samples)

  # integrate over the samples to get the corresponding connectivity
  s = (2 / N_discrete)^2
  discrete_connectivity = sum_g_samples * s

  # check that it matches the connectivity parameter
  # this should only be expected to match if f_init is a constant distribution
  # i.e. the samples in the opinion spaces are uniformly distributed
  @info "Connectivities ∫g (match expected if f_init is constant):"
  @info "from parameter               : $connectivity"
  @info "from discretization of g_init: $discrete_connectivity"

  # Sampling the adjacency matrix A

  # allocate it
  A = SpA.spzeros(Int, (N_discrete, N_discrete))

  # Number of non-zero entries in the adjea
  N_entries = round(Int, N_discrete * connectivity)
  # check that we are not trying to find to many (more than the size of the matrix)
  @assert (N_entries <= N_discrete * (N_discrete - 1)) "Trying to get $(N_entries) entries, only $(N_discrete*(N_discrete-1)) available!"
  @info "Sampling $(N_entries) entries from g_init (max: $(N_discrete*(N_discrete-1)))"

  # initialize some data useful to display the progress
  N_digits = Int(ceil(log10(N_entries)))
  accepted = 0
  tried = 0


  # do the actually sampling
  # this is very slow if the matrix is large and sparse
  @time begin
    while accepted < N_entries
      # sample a couple of indices (uniformely)
      i, j = rand(1:N_discrete), rand(1:N_discrete)
      tried += 1

      # we do not allow self-connection
      # we allow at most one connection
      if i == j || A[i, j] > 0
        continue
      end

      if rand() < g_init_func_scaled(f_samples[i], f_samples[j]) / sum_g_samples
        A[i, j] = 1
        accepted += 1
        if symmetric
          A[j, i] = 1
          tried += 1
          accepted += 1
        end

        print("  $(lpad(accepted, N_digits, " "))/$(N_entries) ($(round(accepted/tried*100; digits=5))% accepted)", "\b"^100)
      end

    end
    println()

  end

  # compute the discrepency between A and the #I_i
  sharp_I = N_discrete * sum(g_init_samples; dims=2) * s
  discrepency = sum(A; dims=2) - sharp_I

  @info "Theoretical sparsity: $(connectivity/N_discrete)"
  @info "     Actual sparsity: $(SpA.nnz(A)/N_discrete/N_discrete)"

  @info "Discrepency Σ_j A_ij - #I_i (connectivity=$connectivity):"
  @info "\n" * string(UnicodePlots.histogram(discrepency; nbins=50, vertical=true))

  @info "Heat map, g_init(ω_i, ω_j):"
  println(UnicodePlots.heatmap(g_init_samples; width=80, height=80))

  @info "Adjacency matrix:\n" * string(UnicodePlots.spy(reverse(A; dims=1); width=80, height=30))

  if !Graphs.is_connected(Graphs.SimpleGraph(A))
    @warn "The adjacency matrix is not connected!"
  end

  return (A=A, discrepency=discrepency)
end

function check_sampling(samples, P; use_makie=false)
  # normalize to distribution to plot
  n_factor = Polynomials.integrate(P)(1) - Polynomials.integrate(P)(-1)

  # Plot the histogram (scaled if using Makie) and the distribution function
  # to check that both match
  t = range(-1, 1, 500)

  # UnicodePlots can only show two separate plots,
  # use Makie for a stacked plot which is better for checking that
  # the sampling matches the distribution
  println(UnicodePlots.lineplot(t, P.(t)))
  println(UnicodePlots.histogram(samples; nbins=60, vertical=true))

  if use_makie
    fig = M.Figure()
    ax = M.Axis(fig[1, 1])
    M.lines!(ax, t, P.(t) / n_factor)
    M.hist!(ax, samples; bins=100, normalization=:pdf)
    display(fig)
  end
end

function prepare(params, mode)
  if params.store
    if isdir(params.store_path)
      if startswith(params.store_path, "results/test_")
        @warn "Removing test dir $(params.store_path)"
        rm(params.store_path; force=true, recursive=true)
      else
        throw("Directory $(params.store_path) already exists!")
      end
    end
    mkpath(params.store_path)
    @info "Output set to $(params.store_path)/"

    fmt_logger = LoggingExtras.FormatLogger(joinpath(params.store_path, "output_" * string(mode) * ".log"); append=true) do io, args
      if args.level == LoggingExtras.Info
        println(io, "[", args.level, "] ", args.message)
      else
        println(io, "[", args.level, "] ", args.message, "\n", "@ ", args._module, " ", args.file, ":", args.line)
      end
    end
  end

  if params.log_debug
    console_log_level = Logging.Debug
  else
    console_log_level = Logging.Info
  end

  console_logger = Logging.ConsoleLogger(stdout, console_log_level)

  if params.store
    tee_logger = LoggingExtras.TeeLogger(
      console_logger,
      LoggingExtras.MinLevelLogger(fmt_logger, LoggingExtras.Info)
    )
    Logging.global_logger(tee_logger)
  else
    Logging.global_logger(console_logger)
  end

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

  if params.store
    open(joinpath(params.store_path, "metadata.txt"), "w") do meta
      write(meta, "Julia version: $VERSION\n")
      try
        write(meta, "Commit:")
        write(meta, read(`git show -s --oneline`, String))
      catch
        @warn "git show failed"
      end
    end
    Serialization.serialize(joinpath(params.store_path, "params.dat"), params)
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

function load_hdf5_data(filename::String, key::String)
  data = HDF5.h5open(filename)
  try
    return read(data[key])
  catch
    return nothing
  end
end

function plot_result(output_filename::String; meanfield_dir::Union{String,Nothing}=nothing, discrete_dir::Union{String,Nothing}=nothing)
  mf_dirs = isnothing(meanfield_dir) ? String[] : [meanfield_dir]
  d_dirs = isnothing(discrete_dir) ? String[] : [discrete_dir]
  return plot_results(output_filename; meanfield_dirs=mf_dirs, discrete_dirs=d_dirs)
end

#mutable struct GMatricesBuffer
#  start_idx::Int64
#  end_idx::Int64
#  size::Int64
#  g_a::Vector{Matrix{Float64}}
#  hdf5_handle::HDF5.File
#end
#
#function load(v::Array{String}, size::Int64)::GMatricesBuffer
#
#end

function plot_results(output_filename::String;
  meanfield_dirs::Vector{String}=String[],
  discrete_dirs::Vector{String}=String[])

  # check that the dirs actually exist
  @assert all(isdir, meanfield_dirs)
  @assert all(isdir, discrete_dirs)

  has_mf, has_d = length(meanfield_dirs) > 0, length(discrete_dirs) > 0
  @assert length(discrete_dirs) <= 1 "currently only a single discrete result is supported"

  K_mf = length(meanfield_dirs)

  if !(has_mf || has_d)
    @error "No dir provided"
    return
  end

  obs_i = M.Observable(1)

  if has_mf
    labels = map(dn -> endswith("/", dn) ? basename(dirname(dn)) : basename(dn), meanfield_dirs)
    f_a = map(dn -> load_hdf5_data(joinpath(dn, "data_meanfield.h5"), "f"), meanfield_dirs)

    N_a = map(f -> size(f, 1), f_a)
    max_iter = minimum(map(f -> size(f, 2), f_a))

    obs_g_k = nothing
    function get_g_k(dn, i)
      try
        return load_hdf5_data(joinpath(dn, "data_meanfield.h5"), "g/$(i-1)")
      catch
        return nothing
      end
    end

    obs_g_k = map(dn -> (M.@lift get_g_k(dn, $obs_i)), meanfield_dirs)

    full_g = isnothing(obs_g_k[1][])

    function build_x(N)
      δx = 2 / N
      x_l, x_r = -1 + 0.5δx, 1 - 0.5δx

      return range(x_l, x_r, length=N)
    end

    x_a = map(build_x, N_a)
  end

  if has_d
    discrete_dir = discrete_dirs[1]
    ops = load_hdf5_data(joinpath(discrete_dir, "data_discrete.h5"), "ops")
    mean_ops = Statistics.mean(ops)
    adj_matrix_full = load_hdf5_data(joinpath(discrete_dir, "data_discrete.h5"), "adj_matrix")
    adj_matrix = isnothing(adj_matrix_full) ? nothing : SpA.sparse(adj_matrix_full)
    N_discrete = size(ops, 1)
    if !isnothing(adj_matrix)
      adj_matrix_nnz = SpA.nnz(adj_matrix)

      sharp_I = vec(sum(adj_matrix; dims=2))
      ω_inf_d = sum(ops[:, 1] .* sharp_I) ./ sharp_I
      xs = Vector{Float64}(undef, adj_matrix_nnz)
      ys = Vector{Float64}(undef, adj_matrix_nnz)

      function get_xy(opis)
        rows = SpA.rowvals(adj_matrix)
        got = 0
        for j in 1:N_discrete
          nzr = SpA.nzrange(adj_matrix, j)
          nnz_in_col = length(nzr)

          xs[got+1:got+nnz_in_col] .= opis[rows[nzr]]
          ys[got+1:got+nnz_in_col] .= opis[j]

          got += nnz_in_col
        end
        @assert got == adj_matrix_nnz
      end

      get_xy(ops[:, 1])
      obs_xs = M.Observable(xs)
      obs_ys = M.Observable(xs)
    else
      ω_inf_d = sum(ops[:, 1]) / N_discrete
    end
  end


  if has_mf && has_d
    max_iter = min(max_iter, size(ops, 2))
  end

  set_makie_backend(:gl)

  fig = M.Figure(size=(1024, 720))
  ax1 = M.Axis(fig[1:4, 1])
  ax1.title = "f / ω_i"

  ax2 = M.Axis(fig[1:4, 2])
  ax2.title = "f / ω_i"

  if has_mf && !full_g
    ax3 = M.Axis(fig[6:9, 1])
    ax3.title = "f(ω) g(ω,m) f(m)"
  end

  if has_d && !isnothing(adj_matrix)
    ax4 = M.Axis(fig[6:9, 2])
    ax4.title = "connections"
  end

  g_bottom = fig[5, 1:2] = M.GridLayout()

  if has_d
    obs_ops = M.@lift ops[:, $obs_i]
    obs_extrema_ops = M.@lift extrema($obs_ops)

    M.hist!(ax1, obs_ops; bins=50, normalization=:pdf)
    M.vlines!(ax1, ω_inf_d, color=:grey, ls=0.5)
    M.hist!(ax2, obs_ops; bins=50, normalization=:pdf)
    M.vlines!(ax2, ω_inf_d, color=:grey, ls=0.5)

    if !isnothing(adj_matrix)
      M.scatter!(ax4, obs_xs, obs_ys, alpha=0.2, markersize=4)
      M.limits!(ax4, (-1, 1), (-1, 1))
    end
  end

  if has_mf
    obs_f_a = [M.@lift f_a[k][:, $obs_i] for k in 1:K_mf]

    function find_support(f_a)
      left_idc = map(f -> (idx = findfirst(x -> x > 1e-5, f[]); return (isnothing(idx) ? 1 : idx)), f_a)
      right_idc = map(f -> (idx = findlast(x -> x > 1e-5, f[]); return (isnothing(idx) ? length(f[]) : idx)), f_a)

      left_x = [x_a[i][left_idc[i]] for i in 1:K_mf]
      right_x = [x_a[i][right_idc[i]] for i in 1:K_mf]

      if has_d
        return (left=min(minimum(obs_ops[]), minimum(left_x)), right=max(maximum(obs_ops[]), maximum(right_x)))
      else
        return (left=minimum(left_x), right=maximum(right_x))
      end
    end

    function find_max(f_a)
      return maximum(map(f -> maximum(f[]), f_a))
    end

    if !full_g
      obs_gff_a = [M.Observable(obs_f_a[k][]' .* obs_g_k[k][] .* obs_f_a[k][]) for k in 1:K_mf]
    end

    for k in 1:K_mf
      M.lines!(ax1, x_a[k], obs_f_a[k], label=labels[k])

      #M.vspan!(ax2, -δx / 2, δx / 2, color=:grey, alpha=0.3)
      M.barplot!(ax2, x_a[k], obs_f_a[k], gap=0)

      if !full_g
        M.heatmap!(ax3, x_a[k], x_a[k], obs_gff_a[k])
        #M.heatmap!(ax3, x_a[k], x_a[k], M.@lift log10.(abs.($o)))
      end
    end

    legend = M.Legend(g_bottom[1, 1], ax1)

  end

  i_range = 1:10:max_iter

  function step_i(i)
    if i % 10 == 0 || i == max_iter
      print("  ", lpad(i, 5, " "), "/", max_iter, "\b"^50)
    end

    if has_mf && any(f -> any(isnan, f[:, i]), f_a)
      return
    end

    obs_i[] = i
    if has_mf
      first_mass = 2 / N_a[1] * sum(f_a[1][:, i])
      ax1.title = "$i, M[1] = $(round(first_mass; digits=6))"
    else
      ax1.title = string(i)
    end

    if has_d && !isnothing(adj_matrix)
      get_xy(obs_ops[])
      obs_xs[] = xs
      obs_ys[] = ys
    end

    if !full_g
      for k in 1:K_mf
        obs_gff_a[k][] = obs_f_a[k][]' .* obs_g_k[k][] .* obs_f_a[k][]
      end
    end

    if has_mf
      support = find_support(obs_f_a)
      max_f = find_max(obs_f_a)
      M.ylims!(ax1, low=-1, high=1.3 * max_f)
      M.xlims!(ax1, low=support.left, high=support.right)
      M.xlims!(ax2, low=support.left, high=support.right)
      M.ylims!(ax2, low=-1, high=1.3 * max_f)
    else
      M.autolimits!(ax1)
      M.autolimits!(ax2)
      M.xlims!(ax2, low=obs_extrema_ops[][:left], high=obs_extrema_ops[][:right])
    end
  end

  M.record(step_i, fig, output_filename, i_range)
  @info ("movie saved at $output_filename")

  println()

end

include("Meanfield.jl")
include("Discrete.jl")

end
