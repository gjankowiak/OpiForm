import OpiForm
import Polynomials
import UnicodePlots

function sample_ω_inf(N_micro)

  ### Initial conditions ###

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
  # n_bumps = 2
  # chebyshev_coeffs = zeros(n_bumps * 2 + 1)
  # chebyshev_coeffs[end] = 1
  # f_init_poly_unscaled = 2.0 - Polynomials.ChebyshevT{Float64}(chebyshev_coeffs)

  # asymmetric bumps (chebyshev)
  # n_bumps = 3
  # chebyshev_coeffs = zeros(n_bumps * 2)
  # chebyshev_coeffs[end] = 1
  # f_init_poly_unscaled = 1.0 - Polynomials.ChebyshevT{Float64}(chebyshev_coeffs)

  # flatter asymmetric bumps (chebyshev)
  n_bumps = 3
  chebyshev_coeffs = zeros(n_bumps * 2)
  chebyshev_coeffs[end] = 1
  f_init_poly_unscaled = 2.0 - Polynomials.ChebyshevT{Float64}(chebyshev_coeffs)

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
  # will be roughly N_micro^2*connection_density, so the resulting density will
  # be connection_density, i.e. connection_density should be less than 1!

  connection_density = 0.1
  @assert connection_density < 1

  function α_init_func(x::Vector{Float64})
    ω, m = x
    stddev = 0.3
    r = exp(-(ω - m)^2 / stddev^2) + 0.4
    return r
  end

  sas_result = OpiForm.scale_and_sample(α_init_func, f_init_poly_unscaled, connection_density, N_micro, constant_α, full_adj_matrix)

  r = [sas_result.ω_inf_d_init; sas_result.ω_inf_mfl_init]
  @show r
  return r
end

N_r = [20; 50; 100; 250; 500; 1000; 2000; 3000]
ω_inf = hcat(map(sample_ω_inf, N_r)...)

p = UnicodePlots.lineplot(N_r, ω_inf[1, :]; width=80, xlabel="N_micro", name="ω_∞_micro")
UnicodePlots.lineplot!(p, N_r, ω_inf[2, :], name="ω_∞_mfl")
