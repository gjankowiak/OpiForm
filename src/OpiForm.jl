module OpiForm

import UnicodePlots

import Logging
import LoggingExtras

import BenchmarkTools
import StatsBase

import TOML

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

function scale_poly(poly)
  I = Polynomials.integrate(poly)
  return poly / (I(1) - I(-1))
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

function scale_g_init(g_init_func_unscaled, target_g_integral)
  g_init_unscaled_integral = Cubature.hcubature(g_init_func_unscaled, [-1; -1], [1; 1])[1]
  return target_g_integral / g_init_unscaled_integral
end

function compute_ω_inf_mf(g_init_func_scaled, g_init_integral)
  if isnothing(g_init_func_scaled)
    return 0.0
  else
    return (Cubature.hcubature(x -> x[1] * g_init_func_scaled(x[1], x[2]), [-1; -1], [1; 1])[1]) / g_init_integral
  end
end

function fast_sampling!(A, α, n; symmetric::Bool=true, constant_α::Bool=false)
  N = size(A, 1)

  if !constant_α

    # this can overflow
    acc = cumsum(vec(α))
    acc ./= acc[end]
  end

  accepted = 0
  tried = 0

  n_digits = Int(ceil(log10(n)))

  while accepted < n
    tried += 1
    if !constant_α
      t = rand()
      r = searchsorted(acc, t)
      if length(r) == 0
        idx = r.start
      else
        idx = rand(r)
      end

      p, q = divrem(idx - 1, N)
      i, j = p + 1, q + 1
    else
      i, j = rand(1:N), rand(1:N)
    end

    if A[i, j] == 0
      A[i, j] = 1
      accepted += 1
      if symmetric
        A[j, i] = 1
        accepted += 1
        tried += 1
      end

      if accepted % 100 == 0
        print("  $(lpad(accepted, n_digits, " "))/$(n) ($(round(accepted/tried*100; digits=5))% accepted)", "\b"^100)
      end
    end
  end
end

function sample_g_init(f_samples, f_init_func_scaled, α_init_func_scaled, connection_density, constant_α::Bool;
  symmetric::Bool=true, fast::Bool=true)
  N_discrete = length(f_samples)
  g_init_integral = connection_density * N_discrete

  @assert N_discrete > 0 "No samples provided!"
  @assert connection_density < 1 "connection_density >= 1!"

  if !constant_α
    # Compute all the couples g_init(ω_i, ω_j)
    # FIXME: use the symmetry to speed up the computation
    α_init_samples = [i == j ? 0.0 : α_init_func_scaled(f_samples[i], f_samples[j]) for i = 1:N_discrete, j = 1:N_discrete]
  else
    α_init_samples = [i == j ? 0.0 : g_init_integral / 4 for i = 1:N_discrete, j = 1:N_discrete]
  end

  # DEFINITION G
  f_ops_samples = f_init_func_scaled.(f_samples)
  g_init_samples = 0.5 * N_discrete * f_ops_samples .* α_init_samples .* f_ops_samples'

  sum_α_samples = sum(α_init_samples)
  sum_g_samples = sum(g_init_samples)

  # integrate over the samples to get the corresponding connectivity
  s = (2 / N_discrete)^2
  discrete_connectivity = sum_g_samples * s

  # check that it matches the connectivity parameter
  # this should only be expected to match if f_init is a constant distribution
  # i.e. the samples in the opinion spaces are uniformly distributed
  @info "Connectivities ∫g (match expected if f_init is constant):"
  @info "from parameter               : $(connection_density * N_discrete)"
  @info "from discretization of g_init: $discrete_connectivity"

  # Sampling the adjacency matrix A

  # allocate it
  A = SpA.spzeros(Int, (N_discrete, N_discrete))

  # Number of non-zero entries in the adjea
  N_entries = round(Int, N_discrete^2 * connection_density)
  # check that we are not trying to find to many (more than the size of the matrix)
  @assert (N_entries <= N_discrete * (N_discrete - 1)) "Trying to get $(N_entries) entries, only $(N_discrete*(N_discrete-1)) available!"
  @info "Sampling $(N_entries) entries from g_init (max: $(N_discrete*(N_discrete-1)))"

  # initialize some data useful to display the progress
  N_digits = Int(ceil(log10(N_entries)))
  accepted = 0
  tried = 0

  # do the actually sampling
  # this is very slow if the matrix is large and sparse
  if fast
    fast_sampling!(A, α_init_samples, N_entries; symmetric=symmetric, constant_α=constant_α)
  else
    while accepted < N_entries
      # sample a couple of indices (uniformely)
      i, j = rand(1:N_discrete), rand(1:N_discrete)
      tried += 1

      # we do not allow self-connection
      # we allow at most one connection
      if i == j || A[i, j] > 0
        continue
      end

      if constant_α || (rand() < α_init_samples(i, j) / sum_α_samples)
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
  sharp_I = N_discrete * sum(α_init_samples; dims=2) * s
  discrepency = sum(A; dims=2) - sharp_I

  @info "Theoretical sparsity: $(connection_density)"
  @info "     Actual sparsity: $(SpA.nnz(A)/N_discrete/N_discrete)"

  @info "Discrepency Σ_j A_ij - #I_i (connection_density=$connection_density):"
  @info "\n" * string(UnicodePlots.histogram(discrepency; nbins=50, vertical=true))

  @info "Heat map, α_init(ω_i, ω_j):"
  println(UnicodePlots.heatmap(α_init_samples; width=80, height=80))

  @info "Adjacency matrix:\n" * string(UnicodePlots.spy(reverse(A; dims=1); width=80, height=30))

  @info "Heat map, g_init(ω_i, ω_j):"
  @show extrema(g_init_samples)
  println(UnicodePlots.heatmap(g_init_samples; width=80, height=80))

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
    meta = Dict(
      "julia_version" => string(VERSION),
      "omega_inf_mf" => params.ω_inf_mf_init,
      "connection_density" => params.connection_density,
      "N_discrete" => params.N_discrete
    )
    try
      meta["commit"] = read(`git show -s --oneline`, String)
    catch
      meta["commit"] = "failed"
      @warn "git show failed"
    end

    open(joinpath(params.store_path, "metadata.toml"), "w") do metafile
      TOML.print(metafile, meta)
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

function plot_result(output_filename::String; meanfield_dir::Union{String,Nothing}=nothing, discrete_dir::Union{String,Nothing}=nothing, kwargs...)
  mf_dirs = isnothing(meanfield_dir) ? String[] : [meanfield_dir]
  d_dirs = isnothing(discrete_dir) ? String[] : [discrete_dir]
  return plot_results(output_filename; meanfield_dirs=mf_dirs, discrete_dirs=d_dirs, kwargs...)
end

function plot_results(output_filename::String;
  meanfield_dirs::Vector{String}=String[],
  discrete_dirs::Vector{String}=String[],
  kwargs...)

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


  half_connection_matrix = get(kwargs, :half_connection_matrix, false)

  center_histogram = get(kwargs, :center_histogram, false)
  if center_histogram
    @warn "Centering histograms, plots are biased!"
  end

  obs_i = M.Observable(1)
  obs_iter = M.Observable(0)

  if has_mf
    labels = map(dn -> endswith("/", dn) ? basename(dirname(dn)) : basename(dn), meanfield_dirs)
    i_mf_a = map(dn -> load_hdf5_data(joinpath(dn, "data_meanfield.h5"), "i"), meanfield_dirs)
    f_a = map(dn -> load_hdf5_data(joinpath(dn, "data_meanfield.h5"), "f"), meanfield_dirs)

    α_init_a = map(dn -> load_hdf5_data(joinpath(dn, "data_meanfield.h5"), "α_init"), meanfield_dirs)

    N_discrete_mf_a = map(dn -> (
        meta = TOML.parsefile(joinpath(dn, "metadata.toml"));
        return meta["N_discrete"]
      ), meanfield_dirs)

    N_a = map(f -> size(f, 1), f_a)

    obs_g_k = nothing
    function get_g_iter(dn, iter)
      try
        return load_hdf5_data(joinpath(dn, "data_meanfield.h5"), "g/$iter")
      catch
        return nothing
      end
    end

    obs_g_k = map(dn -> (M.@lift get_g_iter(dn, $obs_iter)), meanfield_dirs)

    constant_g = isnothing(obs_g_k[1][])

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
    i_d_a = map(dn -> load_hdf5_data(joinpath(dn, "data_discrete.h5"), "i"), discrete_dirs)
    adj_matrix_full = load_hdf5_data(joinpath(discrete_dir, "data_discrete.h5"), "adj_matrix")
    adj_matrix = isnothing(adj_matrix_full) ? nothing : SpA.sparse(adj_matrix_full)
    N_discrete = size(ops, 1)
    if !isnothing(adj_matrix)
      @assert SpA.is_hermsym(adj_matrix, identity)

      adj_matrix_nnz = SpA.nnz(adj_matrix)

      sharp_I = vec(sum(adj_matrix; dims=2))
      ω_inf_d = sum(ops[:, 1] .* sharp_I) ./ sum(sharp_I)
      xs = Vector{Float64}(undef, adj_matrix_nnz)
      ys = Vector{Float64}(undef, adj_matrix_nnz)

      obs_xs = M.Observable(view(xs, 1:adj_matrix_nnz))
      obs_ys = M.Observable(view(ys, 1:adj_matrix_nnz))

      function get_xy(opis)
        rows = SpA.rowvals(adj_matrix)
        got = 0
        for j in 1:N_discrete
          nzr = SpA.nzrange(adj_matrix, j)
          nnz_in_col = length(nzr)

          if half_connection_matrix
            opis_x = view(opis, rows[nzr])

            idc = opis_x .> opis[j]
            nnz_in_col = length(filter(isone, idc))

            xs[got+1:got+nnz_in_col] .= opis_x[idc]
            ys[got+1:got+nnz_in_col] .= opis[j]

            got += nnz_in_col
          else
            xs[got+1:got+nnz_in_col] .= opis[rows[nzr]]
            ys[got+1:got+nnz_in_col] .= opis[j]

            got += nnz_in_col
          end
        end

        if center_histogram
          xs .-= ω_inf_d
          ys .-= ω_inf_d
        end

        if half_connection_matrix
          obs_xs[] = view(xs, 1:got)
          obs_ys[] = view(ys, 1:got)
        end

      end

      get_xy(ops[:, 1])
    else
      ω_inf_d = sum(ops[:, 1]) / N_discrete
    end
  end


  set_makie_backend(:gl)

  fig = M.Figure(size=(1024, 720))
  ax1 = M.Axis(fig[1:4, 1])
  ax1.title = "f / ω_i"

  if has_mf
    ax2 = M.Axis(fig[1:4, 2])
    ax2.title = "g(ω,m)"

    ax3 = M.Axis(fig[1:4, 3])
    ax3.title = "f α f"

    g_bottom = fig[5, 1:3] = M.GridLayout()
  else
    g_bottom = fig[5, 1:2] = M.GridLayout()
  end



  if has_mf
    obs_f_a = [M.@lift f_a[k][:, $obs_i] for k in 1:K_mf]
    obs_fαf_a = [M.@lift 0.5 * N_discrete_mf_a[k] * f_a[k][:, $obs_i] .* α_init_a[k] .* f_a[k][:, $obs_i]' for k in 1:K_mf]

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

    if !constant_g
      obs_gff_a = [M.Observable(obs_g_k[k][]) for k in 1:K_mf]
    end

    for k in 1:K_mf
      M.lines!(ax1, x_a[k], obs_f_a[k], label=labels[k])

      #M.barplot!(ax3, x_a[k], obs_f_a[k], gap=0)

      if !constant_g
        M.heatmap!(ax2, x_a[k], x_a[k], obs_gff_a[k])
      end
      M.heatmap!(ax3, x_a[k], x_a[k], obs_fαf_a[k])
    end

    legend = M.Legend(g_bottom[1, 1], ax1)

  end

  if has_d
    if center_histogram
      obs_ops = M.@lift (ops[:, $obs_i] .- ω_inf_d)
    else
      obs_ops = M.@lift ops[:, $obs_i]
    end
    obs_extrema_ops = M.@lift extrema($obs_ops)

    M.hist!(ax1, obs_ops; bins=50, normalization=:pdf)
    M.vlines!(ax1, ω_inf_d, color=:grey, ls=0.5)
    # M.hist!(ax3, obs_ops; bins=50, normalization=:pdf)
    # M.vlines!(ax3, ω_inf_d, color=:grey, ls=0.5)

    if !isnothing(adj_matrix)
      M.scatter!(ax2, obs_xs, obs_ys, alpha=0.0005, markersize=4, color=:black, strokewidth=0.05, transparency=true)
      M.limits!(ax2, (-1, 1), (-1, 1))
      M.limits!(ax3, (-1, 1), (-1, 1))
    end
  end

  function step_i(tuple)
    i, iter = tuple

    if has_mf && any(f -> any(isnan, f[:, i]), f_a)
      return
    end

    obs_i[] = i
    obs_iter[] = iter
    if has_mf
      first_mass = 2 / N_a[1] * sum(f_a[1][:, i])
      ax1.title = "$iter, M[1] = $(round(first_mass; digits=6))"
    else
      ax1.title = string(iter)
    end

    if has_d && !isnothing(adj_matrix)
      get_xy(obs_ops[])
    end

    if !constant_g
      for k in 1:K_mf
        obs_gff_a[k][] = obs_g_k[k][]
      end
    end

    if has_mf
      support = find_support(obs_f_a)
      max_f = find_max(obs_f_a)
      M.ylims!(ax1, low=-1, high=1.3 * max_f)
      M.xlims!(ax1, low=support.left, high=support.right)
      # M.xlims!(ax3, low=support.left, high=support.right)
      # M.ylims!(ax3, low=-1, high=1.3 * max_f)

      M.xlims!(ax2, low=support.left, high=support.right)
      M.ylims!(ax2, low=support.left, high=support.right)

    else
      M.autolimits!(ax1)
      # M.autolimits!(ax3)
      # M.xlims!(ax3, low=obs_extrema_ops[][:left], high=obs_extrema_ops[][:right])
    end

  end

  i_range = enumerate(i_mf_a[1])
  M.record(step_i, fig, output_filename, i_range)
  @info ("movie saved at $output_filename")

  println()

end

function get_ω_inf_mf(dir::String)
  p = joinpath(dir, "metadata.toml")
  r = TOML.parsefile(p)["omega_inf_mf"]
  return r
end

function compare_peak2peak(
  meanfield_dir::String,
  discrete_dir::String,
)
  return compare_peak2peak([meanfield_dir], [discrete_dir])
end

function compare_peak2peak(
  meanfield_dirs::Vector{String},
  discrete_dirs::Vector{String},
)

  # check that the dirs actually exist
  @assert all(isdir, meanfield_dirs)
  @assert all(isdir, discrete_dirs)

  K_mf, K_d = length(meanfield_dirs), length(discrete_dirs)

  if K_mf == 0 && K_d == 0
    @error "No dir provided"
    return
  end

  if K_mf > 0
    labels_mf = map(dn -> endswith("/", dn) ? basename(dirname(dn)) : basename(dn), meanfield_dirs)
    i_mf_a = map(dn -> load_hdf5_data(joinpath(dn, "data_meanfield.h5"), "i"), meanfield_dirs)
    f_a = map(dn -> load_hdf5_data(joinpath(dn, "data_meanfield.h5"), "f"), meanfield_dirs)

    N_a = map(f -> size(f, 1), f_a)

    function build_x(N)
      δx = 2 / N
      x_l, x_r = -1 + 0.5δx, 1 - 0.5δx

      return range(x_l, x_r, length=N)
    end

    x_a = map(build_x, N_a)

    ω_inf_mf_init_a = map(dn -> get_ω_inf_mf(dn), meanfield_dirs)

    support_bounds_mf_a = find_support_bounds(f_a, x_a)
    support_width_mf_a = [map(x -> x[2] - x[1], s) for s in support_bounds_mf_a]

    # function get_g_iter(dn, iter)
    #   try
    #     return load_hdf5_data(joinpath(dn, "data_meanfield.h5"), "g/$iter")
    #   catch
    #     return nothing
    #   end
    # end

    # function compute_weighted_avgs(k)
    #   dn = meanfield_dirs[k]
    #   ω_inf_mf = zeros(length(i_mf_a[k]))
    #   for (i, iter) in enumerate(i_mf_a[k])
    #     g = get_g_iter(dn, iter)
    #     @assert !isnothing(g)
    #     ω_inf_mf[i] = sum(g .* x_a[k]) / sum(g)
    #   end
    #   return ω_inf_mf
    # end

    g_M1_a = map(dn -> load_hdf5_data(joinpath(dn, "data_meanfield.h5"), "g_M1"), meanfield_dirs)

  end

  if K_d > 0
    labels_d = map(dn -> endswith("/", dn) ? basename(dirname(dn)) : basename(dn), discrete_dirs)
    i_d_a = map(dn -> load_hdf5_data(joinpath(dn, "data_discrete.h5"), "i"), discrete_dirs)
    ops_a = map(dn -> load_hdf5_data(joinpath(dn, "data_discrete.h5"), "ops"), discrete_dirs)

    adj_matrix_full_a = map(dn -> load_hdf5_data(joinpath(dn, "data_discrete.h5"), "adj_matrix"), discrete_dirs)
    adj_matrix_a = map(adj_matrix_full -> isnothing(adj_matrix_full) ? nothing : SpA.sparse(adj_matrix_full), adj_matrix_full_a)
    N_discrete_a = map(ops -> size(ops, 1), ops_a)

    function compute_weighted_avg(k)
      adj_matrix = adj_matrix_a[k]
      ops = ops_a[k]
      N_discrete = N_discrete_a[k]

      if !isnothing(adj_matrix)
        sharp_I = vec(sum(adj_matrix; dims=2))
        n_connections = sum(sharp_I)

        return [sum(ops[:, k] .* sharp_I) ./ n_connections for k in axes(ops, 2)]
      else
        return [sum(ops[:, k]) / (N_discrete - 1) for k in axes(ops, 2)]
      end
    end

    ω_inf_d_a = [compute_weighted_avg(k) for k in 1:K_d]
    p2p_d_a = [peak2peak(ops; dims=1) for ops in ops_a]
    extrema_d_a = [extrema(ops; dims=1) for ops in ops_a]
  end

  set_makie_backend(:gl)

  fig = M.Figure(size=(1024, 720))
  ax1 = M.Axis(fig[1, 1], yscale=log10, xlabel="iterations", title=M.L"\max_i\;\omega_i - \min_i\;\omega_i")
  ax2 = M.Axis(fig[1, 2], xlabel="iterations", title=M.L"\omega_\inf \quad \min_i\; \omega_i \quad \max_i\; \omega_i")

  for k in 1:K_d
    r = i_d_a[k]
    p2p = p2p_d_a[k]
    M.lines!(ax1, r, p2p, label=labels_d[k])

    M.lines!(ax2, r, ω_inf_d_a[k], label=labels_d[k])
    left_bounds = vec(map(v -> v[1], extrema_d_a[k]))
    right_bounds = vec(map(v -> v[2], extrema_d_a[k]))
    M.band!(ax2, r, left_bounds, right_bounds, alpha=0.2)
  end

  for k in 1:K_mf
    p2p = support_width_mf_a[k]
    r = i_mf_a[k]
    M.lines!(ax1, r, p2p, label=labels_mf[k])

    M.hlines!(ax2, ω_inf_mf_init_a[k])
    M.lines!(ax2, r, g_M1_a[k])
    M.band!(ax2, r, map(v -> v[1], support_bounds_mf_a[k]), map(v -> v[2], support_bounds_mf_a[k]), alpha=0.2)
  end

  #M.axislegend()

  display(fig)

end

include("Meanfield.jl")
include("Discrete.jl")

end
