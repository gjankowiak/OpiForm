import OpiForm
import Polynomials
import SparseArrays as SpA

function main(suffix)

  #########################
  #      PARAMETERS       #
  #########################

  ### Discretization ###
  N = 301
  N_discrete = 1000
  δt = 1e-3
  max_iter = 20_000

  ### Visualization ###
  plot_scale = identity
  plot_every = 10
  plot_backend = :makie

  # Storage
  store = true
  store_overwrite = false
  store_every_iter = 10

  ### Model ###
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
  # D_kern = (ω, m) -> -D_kern_factor * exp(-abs(ω - m)) * sign(ω - m)
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

  #### f_init_func NEEDS TO BE NORMALIZED ####

  # centered bump
  #f_init_poly_unscaled = PP([1.0, -1.0]) * PP([1, 1])
  # @show f_init_poly

  # side bump
  #f_init_poly = 0
  #function f_init_func(x)
  #  return x < 0.0 ? 1 - 4 * (x + 0.5)^2 : 0.0
  #end

  # symmetric bumps (chebyshev)
  # n_bumps = 2
  # chebyshev_coeffs = zeros(n_bumps * 2 + 1)
  # chebyshev_coeffs[end] = 1
  # f_init_poly_unscaled = 1.0 - Polynomials.ChebyshevT{Float64}(chebyshev_coeffs)
  #
  # flatter symmetric bumps (chebyshev)
  n_bumps = 2
  chebyshev_coeffs = zeros(n_bumps * 2 + 1)
  chebyshev_coeffs[end] = 1
  f_init_poly_unscaled = 2.0 - Polynomials.ChebyshevT{Float64}(chebyshev_coeffs)

  # asymmetric bumps (chebyshev)
  # n_bumps = 3
  # chebyshev_coeffs = zeros(n_bumps * 2)
  # chebyshev_coeffs[end] = 1
  # f_init_poly_unscaled = 1.0 - Polynomials.ChebyshevT{Float64}(chebyshev_coeffs)

  # flatter asymmetric bumps (chebyshev)
  # n_bumps = 3
  # chebyshev_coeffs = zeros(n_bumps * 2)
  # chebyshev_coeffs[end] = 1
  # f_init_poly_unscaled = 2.0 - Polynomials.ChebyshevT{Float64}(chebyshev_coeffs)

  # more asymmetric bumps (chebyshev)
  # n_bumps = 3
  # chebyshev_coeffs = zeros(n_bumps * 2)
  # chebyshev_coeffs[1] = -1
  # chebyshev_coeffs[2] = 1
  # chebyshev_coeffs[end] = 1

  #f_init_poly_unscaled = 1.0 - Polynomials.ChebyshevT{Float64}(chebyshev_coeffs)

  # Cosine
  # f_init_func = x -> 0.3 * (1e-3 + cos(2 * π * x))
  # f_init_func = x -> 0.3 * (1e-3 + cos(π * x / 2))
  # f_init_func = x -> 0.3 * (1 + 1e-3 - cos(π * x / 2))

  # Constant
  #f_init_poly = Polynomials.Polynomial([0.5])
  #f_init_func = x -> 0.5

  # Linear
  # f_init_func = x -> 1.0 + 0.3 * x

  # Step
  # f_init_func = x -> (2 + round(2 * x)) * (x < 0)

  #### α (connectivity discribution) ####

  # use a constant α for the initialization, it will be scaled for you
  constant_α = false

  # set g as constant in space and time, this should NOT be set to true unless debugging
  constant_g = false

  # use a full adjacency matrix
  full_adj_matrix = false

  # set g = c f(ω) α(ω,m) f(m), i.e. do not consider g as an independent unknown
  f_dependent_g = false

  # Density of the connections
  # g_init will be normalized so that
  # ∫g = λ (see the overleaf file),
  #    = N_discete * connection_density.
  # The number of non-zero entries in the adjacency matrix
  # will be roughly N_discrete^2*connection_density, so the resulting density will
  # be connection_density, i.e. connection_density should be less than 1!

  connection_density = 0.1
  @assert connection_density < 1

  function α_init_func(x::Vector{Float64})
    ω, m = x
    stddev = 0.3
    r = exp(-(ω - m)^2 / stddev^2) + 0.4
    return r
  end

  sas_result = OpiForm.scale_and_sample(α_init_func, f_init_poly_unscaled, connection_density, N_discrete, constant_α, full_adj_matrix)

  ### Solver ###
  # Choice of flux:
  # :LF (Lax-Friedrich)
  # :lLF (local Lax-Friedrich)
  # :godunov
  # :constant_godunov
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
    connection_density=connection_density,
    σ=σ,
    EC_ρ=EC_ρ,
    EC_type=EC_type,
    EC_power=EC_power,
    EC_clip_value=EC_clip_value,
    normalize_chambers=normalize_chambers,

    # initial conditions

    ### Continuous model
    ### f
    f_init_poly=sas_result.f_init_poly,
    f_init_func=sas_result.f_init_func,
    ### g
    g_init_func_scaled=sas_result.g_init_func_scaled,
    α_init_func_scaled=sas_result.α_init_func_scaled,
    g_init=sas_result.g_init,
    f_dependent_g=f_dependent_g,
    constant_α=constant_α, # if true, then g = 1 (modulo normalization). Overrides g_init.
    constant_g=constant_g, # legacy parameter, unused, should be false
    ω_inf_mf_init=sas_result.ω_inf_mf_init,

    ### Discrete model
    full_adj_matrix=full_adj_matrix,
    ### u
    ops_init=sas_result.ops_init,
    adj_matrix=sas_result.adj_matrix,

    # visualization
    plot_scale=plot_scale, # can be :log10
    plot_every=plot_every,
    plot_backend=plot_backend,

    # storage
    store=store,
    store_overwrite=store_overwrite,
    store_every_iter=store_every_iter,

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

  params_lLF = merge(params, (flux=:lLF, f_dependent_g=false))
  store_path = "results/test_meanfield_" * suffix
  OpiForm.MeanField.launch(params_lLF, store_path)

  params_lLF_fdg = merge(params, (flux=:lLF, f_dependent_g=true))
  store_path = "results/test_meanfield_fdg_" * suffix
  OpiForm.MeanField.launch(params_lLF_fdg, store_path)

  store_path = "results/test_discrete_" * suffix
  OpiForm.Discrete.launch(params, store_path)

end

suffix = "2_bumps_flatter"

#########################
#          RUN          #
#########################

# main(suffix)

#########################
#         PLOT          #
#########################

# Movie

# OpiForm.plot_result("test_$(suffix).mp4";
#   meanfield_dir="results/test_meanfield_" * suffix,
#   discrete_dir="results/test_discrete_" * suffix,
#   half_connection_matrix=true,
#   center_histogram=false
# )

# Movie with centered histogram

OpiForm.plot_result("test_$(suffix)_centered_histogram.mp4";
  meanfield_dir="results/test_meanfield_" * suffix,
  discrete_dir="results/test_discrete_" * suffix,
  half_connection_matrix=true,
  center_histogram=true
)

# Comvergence plots

OpiForm.compare_peak2peak(
  [
    "results/test_meanfield_" * suffix,
    "results/test_meanfield_fdg_" * suffix,
  ], [
    "results/test_discrete_" * suffix,
  ]
)
