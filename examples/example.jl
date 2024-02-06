import OpiForm
import Polynomials
import SparseArrays as SpA

function main(suffix)

  #########################
  #      PARAMETERS       #
  #########################

  ### Discretization ###
  N = 301
  N_discrete = 200
  δt = 1e-3
  max_iter = 10000

  ### Visualization ###
  plot_scale = identity
  plot_every = 10
  plot_backend = :makie

  # Storage
  store = true
  store_overwrite = false

  ### Model ###

  # Average number of connections per individual,
  # g_init will be normalized so that ∫g = connectivity.
  # The number of non-zero entries in the adjacency matrix
  # will be roughly N_discrete*connectivity, so the resulting density will
  # be connectivity/N_discrete, i.e. connectivity should be less than N_discrete!
  connectivity = N_discrete / 20

  # σ = 0.0 => Echo chambers only
  # σ = 1.0 => Epistemic bubbles only
  σ = 1.0

  # Echo chamber type
  # :characteristic, original model, uses 1_{|ω-m| < EC_ρ}(m)
  # :super_gaussian, uses clip(exp(- (|ω-m|²/EC_ρ)^EC_power), EC_clip_value)
  #                  where clip(x,v) = x if x > v, 0 otherwise
  # Note that :super_gaussian approximates :characteristic as EC_power → ∞ and EC_clip_value → 0
  EC_type = :super_gaussian

  # Echo chamber radius
  EC_ρ = 2e-1

  # Echo chamber super-gaussian parameters
  EC_power = 3.0
  EC_clip_value = 0.0

  normalize_chambers = false

  ## Debate kernel (for epistemic bubbles)
  D_kern_factor = 1.0
  function D_func(distance)
    return -D_kern_factor * distance
  end
  function D_kern(ω, m)
    return D_func(ω - m)
  end

  # Radicalization kernel (ie attractive term for echo chambers)
  R_kern_factor = 1.0
  function R_func(distance)
    return -R_kern_factor * distance
  end
  function R_kern(ω, m)
    return R_func(ω - m)
  end

  # Polarization kernel (ie repulsive term for echo chambers)
  P_kern_factor = 1.0
  function P_func(distance)
    return P_kern_factor * distance
  end
  function P_kern(ω, m)
    return P_func(ω - m)
  end

  ### Initial conditions ###

  #### f_init will be normalized ####

  PP = Polynomials.Polynomial

  # centered bump
  # f_init_poly = PP([1.0, -1.0]) * PP([1, 1])

  # symmetric bumps (chebyshev)
  n_bumps = 2
  chebyshev_coeffs = zeros(n_bumps * 2 + 1)
  chebyshev_coeffs[end] = 1
  f_init_poly = 1.0 - Polynomials.ChebyshevT{Float64}(chebyshev_coeffs)

  # asymmetric bumps (chebyshev)
  # n_bumps = 3
  # chebyshev_coeffs = zeros(n_bumps * 2)
  # chebyshev_coeffs[end] = 1
  # f_init_func = x -> 1.0 - Polynomials.ChebyshevT{Float64}(chebyshev_coeffs)(x)

  function f_init_func(x)
    return f_init_poly(x)
  end

  #### g (connectivity discribution) ####

  function g_init_func(x::Vector{Float64})
    ω, m = x
    stddev = 0.3
    r = exp(-(ω - m)^2 / stddev^2) + 0.1
    return r
  end

  g_init_func_scaled = OpiForm.scale_g_init(g_init_func, connectivity)

  # if g_init is nothing, it will be evaluated from g_init_func_scaled
  g_init = nothing

  # g_init = OpiForm.speyes(N, 3)
  # g_init = ones(N,N)

  # g_init = rand_symmetric(N, 0.1)

  full_g = false
  full_adj_matrix = false

  ### ops_init ###
  ops_init = OpiForm.sample_poly_dist(f_init_poly, N_discrete)

  ### adj_matrix ###
  g_sampling_result = OpiForm.sample_g_init(ops_init, g_init_func_scaled, connectivity)
  adj_matrix = g_sampling_result.A

  ### Solver ###
  # Choice of flux:
  # :LF (Lax-Friedrich)
  # :lLF (local Lax-Friedrich)
  # :godunov
  # :full_godunov
  # :upwind
  # :KT
  flux = :lLF

  # Try to approximate a'
  approx_a_prime = false

  # Time stepping method
  # :simple
  # :RK4 (not convervative?)
  time_stepping = :simple

  LF_relaxation = 1.0

  # The use entropy fix for the Godunov scheme
  # See Leveque 12.3 (FVM for Hyperbolic Problems)
  godunov_entropy_fix = false

  # KT flux limiter
  # minmod oder superbee
  KT_flux_limiter = :minmod

  ### DEBUG ###
  log_debug = false
  constant_a = false

  # How to deal with CFL violations
  # :ignore
  # :warn
  # :throw
  # :lower_step (not implemented)
  CFL_violation = :throw

  ##############################################################@

  params = (
    # domain
    N=N,
    N_discrete=N_discrete,
    δt=δt,
    max_iter=max_iter,

    # model
    # "Debate" term, attractive, interactions from connectivity matrix
    D_kern_factor=D_kern_factor,
    D_func=D_func,
    D_kern=D_kern,
    #
    # "Radicalization" term, attractive, interaction with mean from EC
    R_kern_factor=R_kern_factor,
    R_func=R_func,
    R_kern=R_kern,
    #
    # "Polarization" term, attractive, interaction between mean from EC and mean from EC^c
    P_kern_factor=P_kern_factor,
    P_func=P_func,
    P_kern=P_kern,

    # bubble-chamber balance parameter
    connectivity=connectivity,
    σ=σ,
    EC_ρ=EC_ρ,
    EC_type=EC_type,
    EC_power=EC_power,
    EC_clip_value=EC_clip_value,
    normalize_chambers=normalize_chambers,

    # initial conditions

    ### Continuous model
    ### f
    f_init_poly=f_init_poly,
    f_init_func=f_init_func,
    ### g
    g_init_func_scaled=g_init_func_scaled,
    g_init=g_init,
    full_g=full_g, # if true, then g = 1 (modulo normalization). Overrides g_init.

    ### Discrete model
    full_adj_matrix=full_adj_matrix,
    ### u
    ops_init=ops_init,
    adj_matrix=adj_matrix,

    # visualization
    plot_scale=plot_scale, # can be :log10
    plot_every=plot_every,
    plot_backend=plot_backend,

    # storage
    store=store,
    store_overwrite=store_overwrite,

    # solver
    flux=flux,
    approx_a_prime=approx_a_prime,
    time_stepping=time_stepping,
    LF_relaxation=LF_relaxation,
    godunov_entropy_fix=godunov_entropy_fix,

    # debug
    log_debug=log_debug,
    constant_a=constant_a,
    CFL_violation=CFL_violation
  )

  OpiForm.set_makie_backend(:gl)

  params_lLF = merge(params, (flux=:lLF,))
  store_path = "results/test_meanfield_lLF" * suffix
  OpiForm.MeanField.launch(params_lLF, store_path)

  store_path = "results/test_discrete" * suffix
  OpiForm.Discrete.launch(params, store_path)

end

suffix = "_sampling_g0"
main(suffix)

OpiForm.plot_result("test_sampling_g0.mp4", meanfield_dir="results/test_meanfield_lLF" * suffix, discrete_dir="results/test_discrete" * suffix)
