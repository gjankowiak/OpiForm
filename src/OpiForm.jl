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

import LinearAlgebra as LA

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

function scale_poly(poly)
  I = Polynomials.integrate(poly)
  return poly / (I(1) - I(-1))
end

function sample_poly_dist(poly, n)
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

function scale_g_init(g_init_func_unscaled, target_g_integral)
  g_init_unscaled_integral = Cubature.hcubature(g_init_func_unscaled, [-1; -1], [1; 1])[1]
  return target_g_integral / g_init_unscaled_integral
end

function compute_ω_inf_mf(g_init_func_scaled, g_init_integral)
  if isnothing(g_init_func_scaled)
    return 0.0
  else
    @info "Computing g's first moment"
    # hack because Cubature sometimes stalls
    # return (Cubature.pcubature(x -> x[1] * g_init_func_scaled(x[1], x[2]), [-1; -1], [1; 1])[1]) / g_init_integral
    n = 1000
    δx = 2 / n
    x = range(-1 + 0.5δx, 1 - δx, length=n)
    return (δx^2 * sum(x -> x[1] * g_init_func_scaled(x[1], x[2]), Iterators.product(x, x))) / g_init_integral
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
  symmetric::Bool=true, fast::Bool=true, zero_diagonal::Bool=false)
  N_discrete = length(f_samples)
  g_init_integral = connection_density * N_discrete

  @assert N_discrete > 0 "No samples provided!"
  @assert connection_density < 1 "connection_density >= 1!"

  if !constant_α
    # Compute all the couples g_init(ω_i, ω_j)
    if zero_diagonal
      α_init_samples = [i == j ? 0.0 : α_init_func_scaled(f_samples[i], f_samples[j]) for i = 1:N_discrete, j = 1:N_discrete]
    else
      α_init_samples = [α_init_func_scaled(f_samples[i], f_samples[j]) for i = 1:N_discrete, j = 1:N_discrete]
    end
  else
    if zero_diagonal
      α_init_samples = [i == j ? 0.0 : g_init_integral / 4 for i = 1:N_discrete, j = 1:N_discrete]
    else
      α_init_samples = [g_init_integral / 4 for i = 1:N_discrete, j = 1:N_discrete]
    end
  end

  # DEFINITION G
  f_ops_samples = f_init_func_scaled.(f_samples)
  g_init_samples = f_ops_samples .* α_init_samples .* f_ops_samples' / connection_density
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
      if zero_diagonal && i == j
        continue
      end

      # we allow at most one connection
      if A[i, j] > 0
        continue
      end

      if constant_α || (rand() < α_init_samples(i, j) / sum_α_samples)
        A[i, j] = 1
        accepted += 1
        if symmetric && i != j
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

function scale_and_sample(α_init_func, f_init_poly_unscaled, connection_density, N_discrete::Int, constant_α::Bool, full_adj_matrix::Bool)
  if !constant_α
    _α = α_init_func
  else
    _α = (_) -> connection_density
  end

  f_init_poly = OpiForm.scale_poly(f_init_poly_unscaled)

  f_init_func(x) = f_init_poly(x)

  # DEFINITION G
  g_init_unscaled_uni = (x -> _α(x) * f_init_func(x[1]) * f_init_func(x[2]) / connection_density)
  g_prefactor = OpiForm.scale_g_init(g_init_unscaled_uni, 1.0)
  α_init_func_scaled = (ω, m) -> g_prefactor * _α([ω, m])
  # DEFINITION_G
  g_init_func_scaled = (ω, m) -> α_init_func_scaled(ω, m) * f_init_func(ω) * f_init_func(m) / connection_density

  ω_inf_mf_init = OpiForm.compute_ω_inf_mf(g_init_func_scaled, 1.0)

  # if g_init is nothing, it will be evaluated from g_init_func_scaled
  g_init = nothing

  ### ops_init ###
  ops_init = OpiForm.sample_poly_dist(f_init_poly, N_discrete)

  ### adj_matrix ###
  if !full_adj_matrix
    g_sampling_result = OpiForm.sample_g_init(ops_init, f_init_func, α_init_func_scaled, connection_density, constant_α)
    adj_matrix = g_sampling_result.A
  else
    adj_matrix = nothing
  end

  return (
    f_init_poly=f_init_poly,
    f_init_func=f_init_func,
    α_init_func_scaled=α_init_func_scaled,
    g_init_func_scaled=g_init_func_scaled,
    g_init=g_init,
    ops_init=ops_init,
    adj_matrix=adj_matrix,
    ω_inf_mf_init=ω_inf_mf_init
  )
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
      "delta_t" => params.δt,
      "flux" => string(params.flux),
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


include("Utils.jl")
include("Plotting.jl")
include("Meanfield.jl")
include("Discrete.jl")

end
