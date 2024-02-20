module MeanField

import UnicodePlots
import HDF5

import Graphs: SimpleGraph, is_connected, connected_components

import ..OpiForm: SA, SpA, M, clip, rand_symmetric, speyes, prepare_directory, issymmetric, symmetry_defect,
  build_x, load_hdf5_data, store_hdf5_data, @fmt, @left, @right

function check_connected(M::Array{Int64,2})
  g = SimpleGraph(M)
  return is_connected(g)
end

function minmod(a, b)
  # FIXME: 
  @assert !(a .|> isinf |> any) "\n minmod:inf input (a)"
  @assert !(b .|> isinf |> any) "\n minmod:inf input (b)"
  @assert !(a .|> isnan |> any) "\n minmod:NaN input (a)"
  @assert !(b .|> isnan |> any) "\n minmod:NaN input (b)"
  r1 = @. sign(a) + sign(b)
  r2 = @. min(abs(a), abs(b))
  @assert !(r1 .|> isnan |> any) "\n minmod:NaN r1!"
  @assert !(r2 .|> isnan |> any) "\n minmod:NaN r2!"
  r = @. 0.5 * (sign(a) + sign(b)) * min(abs(a), abs(b))
  @assert !(r .|> isnan |> any) "\n minmod:NaN result!"
  return r
end

function compute_df!(dst, params::NamedTuple, f, a, a_prime)
  f_l = @left f
  f_r = @right f
  a_l = @left a
  a_r = @right a

  if params.approx_a_prime
    a_prime_l = @left a_prime
    a_prime_r = @right a_prime
  end

  if params.flux == :lLF
    # local Lax-Fridiredrich
    if params.approx_a_prime
      max_C_l = maximum(abs, [a + a_prime .* f a_l + a_prime_l .* f_l], dims=2)
      max_C_r = maximum(abs, [a + a_prime .* f a_r + a_prime_r .* f_r], dims=2)
    else
      max_C_l = maximum(abs, [a a_l], dims=2)
      max_C_r = maximum(abs, [a a_r], dims=2)
    end

    if params.CFL_violation != :ignore
      CFL_failed_at_idx = findfirst(isone, max.(max_C_l, max_C_r) .> 0.5params.δx / params.δt)
      if !isnothing(CFL_failed_at_idx)
        if params.CFL_violation == :warn
          @warn("CFL not met at x=$(params.x[CFL_failed_at_idx]) (idx=$(CFL_failed_at_idx[1]))")
        elseif params.CFL_violation == :throw
          throw("CFL not met at x=$(params.x[CFL_failed_at_idx]) (idx=$(CFL_failed_at_idx[1]))")
        else
          throw("Unkown CFL_violation setting '$(params.CFL_violation)'")
        end
      end
    end

    flux_l = 0.5 * (f_l .* a_l .+ f .* a .- params.LF_relaxation * max_C_l .* (f .- f_l))
    flux_r = 0.5 * (f_r .* a_r .+ f .* a .- params.LF_relaxation * max_C_r .* (f_r .- f))
  elseif params.flux == :LF
    # Lax-Friedrich
    max_C = maximum(abs, a + a_prime .* f)
    flux_l = 0.5 * (f_l .* a_l .+ f .* a .- params.LF_relaxation * max_C * (f .- f_l))
    flux_r = 0.5 * (f_r .* a_r .+ f .* a .- params.LF_relaxation * max_C * (f_r .- f))
  elseif params.flux == :upwind
    # upwind
    flux_l = ((f_l .< f) .* min.(a .* f, a_l .* f_l) .+
              (f_l .>= f) .* max.(a .* f, a_l .* f_l))
    flux_r = ((f .< f_r) .* min.(a_r .* f_r, a .* f) .+
              (f .>= f_r) .* max.(a_r .* f_r, a .* f))
  elseif params.flux == :constant_godunov
    # wave speed
    s_l = (f .* a .- f_l .* a_l) ./ (f .- f_l)
    s_r = (f_r .* a_r .- f .* a) ./ (f_r .- f)

    transonic_l = f_l .< 0 .< f
    transonic_r = f .< 0 .< f_r

    flux_l = ((f_l .< f) .* min.(a .* f, a_l .* f_l) .+
              (f_l .>= f) .* max.(a .* f, a_l .* f_l))
    flux_r = ((f .< f_r) .* min.(a_r .* f_r, a .* f) .+
              (f .>= f_r) .* max.(a_r .* f_r, a .* f))

  elseif params.flux == :godunov
    # godunov

    if params.godunov_entropy_fix
      throw("not implemented")
      transonic_l = f_l .< 0 .< f
      transonic_r = f .< 0 .< f_r

      flux_l = ((f_l .< f) .* min.(a .* f, a_l .* f_l) .+
                (f_l .>= f) .* max.(a .* f, a_l .* f_l))
      flux_r = ((f .< f_r) .* min.(a_r .* f_r, a .* f) .+
                (f .>= f_r) .* max.(a_r .* f_r, a .* f))
    else
      # Similar to upwind
      flux_l = ((f_l .< f) .* min.(a .* f, a_l .* f_l) .+
                (f_l .>= f) .* max.(a .* f, a_l .* f_l))
      flux_r = ((f .< f_r) .* min.(a_r .* f_r, a .* f) .+
                (f .>= f_r) .* max.(a_r .* f_r, a .* f))
    end
  elseif params.flux == :KT # (Kurganov-Tadmor)
    λ = params.δt / params.δx
    (α_l, α_r) = max.(abs.(a), abs.(a_l)), max.(abs.(a), abs.(a_r))
    α_lll = @left α_l
    α_rrr = @right α_r

    f_x = @. minmod((f - f_l) / params.δx, (f_r - f) / params.δx)
    #@assert !(f_x .|> isnan |> any) "\n NaN detected in f_x"

    #f_x_r = @. 0.5 / params.δx * minmod((w_rr - w_r) / (1 + λ * (α_r - α_rrr)), (w_r - w) / (1 + λ * (α_r - α_l)))
    #f_x_l = @. 0.5 / params.δx * minmod((w - w_l) / (1 + λ * (α_l - α_r)), (w_l - w_ll) / (1 + λ * (α_l - α_lll)))

    f_x_ll = @left f_x
    f_x_rr = @right f_x

    f_l_L = @. f_l + params.δx * f_x_ll * (0.5 - λ * α_l)
    f_r_L = @. f + params.δx * f_x * (0.5 - λ * α_r)
    f_l_R = @. f - params.δx * f_x * (0.5 - λ * α_l)
    f_r_R = @. f_r - params.δx * f_x_rr * (0.5 - λ * α_r)

    f_p_l_L = @. f_l_L - 0.5 * params.δt * a_l
    f_p_r_L = @. f_r_L - 0.5 * params.δt * a
    f_p_l_R = @. f_l_R - 0.5 * params.δt * a
    f_p_r_R = @. f_r_R - 0.5 * params.δt * a_r

    w_p_l = @. (0.5 * (f_l + f)
                +
                0.25 * (params.δx - α_l * params.δt) * (f_x_ll - f_x)
                -
                0.5 / α_l * (a * f_p_l_R - a_l * f_p_l_L)
    )
    w_p_r = @. (0.5 * (f + f_r)
                +
                0.25 * (params.δx - α_r * params.δt) * (f_x - f_x_rr)
                -
                0.5 / α_r * (a_r * f_p_r_R - a * f_p_r_L)
    )

    #@assert !(w_p_l .|> isinf |> any) "\n inf detected in w_p_l"
    #@assert !(w_p_r .|> isinf |> any) "\n inf detected in w_p_r"
    #@assert !(w_p_l .|> isnan |> any) "\n NaN detected in w_p_l"
    #@assert !(w_p_r .|> isnan |> any) "\n NaN detected in w_p_r"

    _w_p = @. 0.5 * params.δt * (α_l - α_r) * f_x - λ / (1 - λ * (a_l + a_r)) * (a * (f_p_r_L - f_p_l_R))
    w_p = @. f + _w_p

    #@assert !(w_p .|> isinf |> any) "\n inf detected in w_p"
    #@assert !(w_p .|> isnan |> any) "\n NaN detected in w_p"

    w_p_ll = @left w_p
    w_p_rr = @right w_p

    #@show extrema(@. 1 + λ * (α_r - α_rrr))
    #@show extrema(@. 1 + λ * (α_r - α_l))

    #@show extrema(@. (w_p_rr - w_p_r) / (1 + λ * (α_r - α_rrr)))
    #@show extrema(@. (w_p_r - w_p) / (1 + λ * (α_r - α_l)))

    f_x_p_r = @. 0.5 * params.δx * minmod((w_p_rr - w_p_r) / (1 + λ * (α_r - α_rrr)), (w_p_r - w_p) / (1 + λ * (α_r - α_l)))
    f_x_p_l = @. 0.5 * params.δx * minmod((w_p - w_p_l) / (1 + λ * (α_l - α_r)), (w_p_l - w_p_ll) / (1 + λ * (α_l - α_lll)))

    # @assert !(f_x_p_r .|> isnan |> any) "\n NaN detected in f_x_p_r"
    # @assert !(f_x_p_l .|> isnan |> any) "\n NaN detected in f_x_p_l"

    # TODO: check factor
    @. dst = (
      λ * α_l * w_p_l + (1 - λ * (α_l + α_r)) * w_p
      +
      λ * α_r * w_p_r + 0.5 * params.δx * ((λ * α_l)^2 * f_x_p_l - (λ * α_r)^2 * f_x_p_r)
    )
    return

  else
    throw("unknown flux '%(params.flux)'")
  end

  # Neumann boundary conditions
  flux_l[1] = 0
  flux_r[end] = 0

  dst .= flux_r - flux_l
end

function compute_dg!(dst, params::NamedTuple, g, a, a_prime)
  # Exact same computation as for f but along both dimensions.
  # There is probably a way to take advantage of the symmetry of g.
  g_lω, g_rω = SA.shiftedarray(g, (1, 0), 0.0), SA.shiftedarray(g, (-1, 0), 0.0)
  g_lm, g_rm = SA.shiftedarray(g, (0, 1), 0.0), SA.shiftedarray(g, (0, -1), 0.0)

  a_l = @left a
  a_r = @right a

  if params.approx_a_prime
    a_prime_l = @left a_prime
    a_prime_r = @right a_prime
  end

  # Lax-Friedrich flux
  if params.flux == :lLF
    if params.approx_a_prime
      throw("not implemented")

      # for f
      # max_C_l = maximum(abs, [a + a_prime .* f a_l + a_prime_l .* f_l], dims=2)
      # max_C_r = maximum(abs, [a + a_prime .* f a_r + a_prime_r .* f_r], dims=2)

      max_C_l = maximum(abs, [
          a+a_prime.*g a_l+a_prime_l.*g_lω a_l+a_prime_l.*g_lm
        ], dims=2)

      max_C_r = maximum(abs, [
          a+a_prime.*g a_r+a_prime_r.*g_rω a_r+a_prime_r.*g_rm
        ], dims=2)

    else
      max_C_l = maximum(abs, [a a_l], dims=2)
      max_C_r = maximum(abs, [a a_r], dims=2)
    end

    # Lax-Friedrich flux
    flux_lω = 0.5 * (g_lω .* a_l .+ g .* a .- params.LF_relaxation * max_C_l .* (g .- g_lω))
    flux_rω = 0.5 * (g_rω .* a_r .+ g .* a .- params.LF_relaxation * max_C_r .* (g_rω .- g))

    flux_lm = 0.5 * (g_lm .* a_l' .+ g .* a' .- params.LF_relaxation * max_C_l' .* (g .- g_lm))
    flux_rm = 0.5 * (g_rm .* a_r' .+ g .* a' .- params.LF_relaxation * max_C_r' .* (g_rm .- g))
  elseif params.flux == :LF
    max_C = maximum(abs, a)

    # Lax-Friedrich flux
    flux_lω = 0.5 * (g_lω .* a_l .+ g .* a .- params.LF_relaxation * max_C .* (g .- g_lω))
    flux_rω = 0.5 * (g_rω .* a_r .+ g .* a .- params.LF_relaxation * max_C .* (g_rω .- g))

    flux_lm = 0.5 * (g_lm .* a_l' .+ g .* a' .- params.LF_relaxation * max_C .* (g .- g_lm))
    flux_rm = 0.5 * (g_rm .* a_r' .+ g .* a' .- params.LF_relaxation * max_C .* (g_rm .- g))
  elseif params.flux == :KT # (Kurganov-Tadmor)
    throw("unknown flux '%(params.flux)'")
  end


  # Neumann boundary conditions
  flux_lω[1] = 0
  flux_rω[end] = 0

  flux_lm[1] = 0
  flux_rm[end] = 0

  @. dst = flux_rω - flux_lω + flux_rm - flux_lm


end

function compute_a!(a_dst, a_prime_dst, µ_dst, µC_dst, params::NamedTuple, f, g)
  # compute EB
  #
  # g is normalized so that ∫∫g = connection_density*N_micro
  # η(ω,m) = g(ω,m) / ∫ g(ω,m') dm'

  if params.constant_g
    # if g is constant,  g = connection_density*N_micro / (params.Ω_width)²
    # η(ω,m) = 1 / params.Ω_width
    η = 1 / (params.Ω_width)
  else
    g_mass = params.δx * sum(g; dims=2)
    g_mass_inv = 1 ./ g_mass
    g_mass_inv[g_mass_inv.>1/params.int_threshold] .= 0
    η = g .* g_mass_inv
  end

  EB = params.δx * sum(η .* params.D_matrix; dims=2)

  # Check the normalization.
  # Original model: chamber_size = 1
  # Alernative 1: params.δx * sum(params.EC_mask_matrix; dims=2) -- unstable with lLF

  # compute EC
  if params.normalize_chambers
    chamber_size = params.δx * sum(params.EC_mask_matrix; dims=2)
  else
    chamber_size = 1.0
  end
  chamber_mass = params.δx * sum(f' .* params.EC_mask_matrix; dims=2) ./ chamber_size
  chamber_mass_inv = 1 ./ chamber_mass
  chamber_mass_inv[chamber_mass_inv.>1/params.int_threshold] .= 0.0
  # p = chamber_mass_inv' .* f .* params.EC_mask_matrix
  p = chamber_mass_inv .* f .* params.EC_mask_matrix
  µ = params.δx * sum(params.x' .* p; dims=2) ./ chamber_size
  µ_dst .= µ

  if params.normalize_chambers
    chamberC_size = params.δx * sum((1 .- params.EC_mask_matrix); dims=2)
  else
    chamberC_size = 1.0
  end
  chamberC_mass = params.δx * sum(f' .* (1 .- params.EC_mask_matrix); dims=2) ./ chamberC_size
  chamberC_mass_inv = 1 ./ chamber_mass
  chamberC_mass_inv[chamberC_mass_inv.>1/params.int_threshold] .= 0.0
  pC = chamberC_mass_inv .* f .* (1 .- params.EC_mask_matrix)
  µC = params.δx * sum(params.x' .* pC; dims=2) ./ chamberC_size
  µC_dst .= µC

  EC_in = params.R_kern.(params.x, µ)
  EC_out = params.P_kern.(µ, μC)

  a_dst .= params.σ .* EB .+ (1 - params.σ) * (
    chamber_mass .* EC_in .+ chamberC_mass .* EC_out
  )
  if params.approx_a_prime
    a_prime_dst .= (1 - params.σ) * params.δx .* (EC_in ./ chamber_size .+ EC_out ./ chamberC_size)
  end
end

function display_params(params::NamedTuple)
  r = "Parameters:\n"
  for (i, k) in enumerate(keys(params))
    if k in [:g_init, :D_matrix, :x, :EC_mask_matrix, :ops_init, :adj_matrix]
      continue
    else
      r *= "$k = $(params[k]), "
      if i % 3 == 2
        r *= "\n"
      end
    end
  end
  @info r
end

function launch(store_dir::String, params_in::NamedTuple; force::Bool=false)
  if params_in.constant_g
    @warn "Parameter constant_g is deprecated!"
  end

  prepare_directory(store_dir, params_in, :mfl, force=force)

  # Domain parameters
  # FIXME: this is not used consistently
  Ω_left, Ω_right = -1, 1
  Ω_width = Ω_right - Ω_left

  N_mfl = params_in.N_mfl

  x = build_x(N_mfl)
  δx = x[2] - x[1]

  # Model parameters
  if params_in.constant_a
    @warn "a is set constant!"
  end

  for k in [:D_kern_factor, :R_kern_factor, :P_kern_factor]
    if !isapprox(params_in[k], 1)
      @warn("$k is not 1!")
    end
  end

  # Echo chamber mask matrix
  if params_in.EC_type == :characteristic
    EC_idx_span = Int(floor(min(N_mfl - 1, params_in.EC_ρ / δx)))
    EC_mask_matrix = SpA.spdiagm([i => ones(N_mfl - abs(i)) for i in -EC_idx_span:EC_idx_span]...)
  elseif params_in.EC_type == :super_gaussian
    if N_mfl % 2 != 1
      @warn("Using super gaussian echo chamber with even N_mfl, this breaks symmetry")
    end
    # define the mask matrix
    EC_mask_matrix_full = zeros(N_mfl, N_mfl)

    # compute the full mask centered at 0
    EC_mask = clip.(exp.(-(x .^ 2 / params_in.EC_ρ^2) .^ params_in.EC_power), params_in.EC_clip_value)

    # fill the matrix rowwise by shifting EC_mask
    EC_mask_shift = div(N_mfl, 2) + 1
    for i in 1:N_mfl
      EC_mask_matrix_full[i, max(1, i + 1 - EC_mask_shift):min(N_mfl, i + N_mfl - EC_mask_shift)] =
        EC_mask[max(1, EC_mask_shift - (i - 1)):min(N_mfl, N_mfl + EC_mask_shift - i)]
    end

    use_sparse = sum(iszero, EC_mask) < 0.5 * (N_mfl - 1)
    if use_sparse
      EC_mask_matrix = SpA.sparse(EC_mask_matrix_full)
    else
      @warn "Echo chamber matrix (EC_mask_matrix) is not sparse"
      EC_mask_matrix = EC_mask_matrix_full
    end
  else
    throw("Unknown EC_type '$(params_in.EC_type)'")
  end

  # Debate matrix
  D_matrix = params_in.D_kern.(x, x')

  # Threshold for small integrals, integrals smaller than this
  # will have their inverse treated as zero.
  int_threshold = 1e-10

  params = merge(params_in, (
    # domain
    Ω_left=Ω_left,
    Ω_right=Ω_right,
    Ω_width=Ω_width,
    δx=δx,
    x=x,

    # model
    D_matrix=D_matrix,
    EC_mask_matrix=EC_mask_matrix,
    int_threshold=int_threshold,
  ))

  f_init = load_hdf5_data(joinpath(store_dir, "data.hdf5"), "f_init")
  f = copy(f_init)
  if !params.constant_g
    g = load_hdf5_data(joinpath(store_dir, "data.hdf5"), "g_init")
  else
    g = nothing
  end

  i = 0

  α = load_hdf5_data(joinpath(store_dir, "data.hdf5"), "alpha")
  if params.f_dependent_g
    @assert !isnothing(α) "α is Nothing, but f_dependent_g is set!"
  end

  if !isdir(store_dir)
    throw("Directory $(store_dir) does not exist!")
  end
  store_i = [i]
  store_f = [copy(f)]
  if !params.constant_g
    store_g = [(0, copy(g))]
    store_g_M1 = [sum(x .* g) / sum(g)]
  end

  # Initial mass
  mass_init = δx * sum(f)

  a = zeros(params.N_mfl)
  a_prime = zeros(params.N_mfl)
  µ, µC = zeros(params.N_mfl), zeros(params.N_mfl)

  df = zeros(params.N_mfl)
  if !params.constant_g
    dg = zeros(params.N_mfl, params.N_mfl)
  end

  compute_a!(a, a_prime, µ, µC, params, f, g)
  a_init = copy(a)

  if params.time_stepping == :RK4
    RK4_f = zeros(params.N_mfl, 4)
    RK4_df = zeros(params.N_mfl, 4)
    RK4_g = zeros(params.N_mfl, params.N_mfl, 4)
    RK4_dg = zeros(params.N_mfl, params.N_mfl, 4)
  end

  ## Print parameters and plot initial conditions before starting

  display_params(params)

  if abs(params.LF_relaxation - 1) > 1e-10
    @warn "LF_relaxation != 1 ($(params.LF_relaxation))"
  end


  while i < params.max_iter

    # Check that g is roughly symmetric
    symmetry_tol = 1e-8
    if !issymmetric(g, tol=symmetry_tol)
      throw("g is not symmetric after iteration $i (with tolerance $symmetry_tol)")
    end

    i += 1
    if i % 10 == 0
      print("\b"^200 * "[i=$(lpad(i, 5, " "))]")
    end

    if params.time_stepping == :simple
      if params.constant_a
        fill!(a, 1.0)
        fill!(a_prime, 0.0)
      else
        compute_a!(a, a_prime, µ, µC, params, f, g)
      end
      compute_df!(df, params, f, a, a_prime)
      if !params.constant_g && !params.f_dependent_g
        compute_dg!(dg, params, g, a, a_prime)
      end
      @assert !(df .|> isnan |> any) "\n NaN detected in df at iteration $i"

      ###################################
      #          TIME STEPPING          #
      # #################################

      if params.flux == :KT
        f .= df
      else
        f .-= params.δt / params.δx * df
        if !params.constant_g
          if params.f_dependent_g
            # DEFINITION G
            g .= (f .* α .* f') ./ params.connection_density
          else
            g .-= params.δt / params.δx * dg
          end
        end
      end
    elseif params.time_stepping == :RK4
      if params.f_dependent_f
        throw("not implemented")
      end
      # Stage 1
      stage = 1

      if params.constant_a
        fill!(a, 1.0)
        fill!(a_prime, 0.0)
      else
        compute_a!(a, a_prime, µ, µC, params, f, g)
      end

      compute_df!(view(RK4_df, :, stage), params, f, a, a_prime)
      if !params.constant_g
        compute_dg!(view(RK4_dg, :, stage), params, g, a, a_prime)
      end

      RK4_f[:, stage] .= f .- 0.5 .* params.δt ./ params.δx .* RK4_df[:, stage]
      if !params.constant_g
        RK4_g[:, :, stage] .= g .- 0.5 .* params.δt ./ params.δx .* RK4_dg[:, :, stage]
      end

      # Stage 2
      stage = 2

      if !params.constant_a
        compute_a!(a, a_prime, µ, µC, params, RK4_f[:, stage-1], RK4_g[:, :, stage-1])
      end

      compute_df!(view(RK4_df, :, stage), params, RK4_f[:, stage-1], a, a_prime)
      if !params.constant_g
        compute_dg!(view(RK4_dg, :, stage), params, RK4_g[:, :, stage-1], a, a_prime)
      end

      RK4_f[:, stage] .= f .- 0.5 .* params.δt ./ params.δx .* RK4_df[:, stage]
      if !params.constant_g
        RK4_g[:, :, stage] .= g .- 0.5 .* params.δt ./ params.δx .* RK4_dg[:, :, stage]
      end

      # Stage 3
      stage = 3

      if !params.constant_a
        compute_a!(a, a_prime, µ, µC, params, RK4_f[:, stage-1], RK4_g[:, :, stage-1])
      end

      compute_df!(view(RK4_df, :, stage), params, RK4_f[:, stage-1], a, a_prime)
      if !params.constant_g
        compute_dg!(view(RK4_dg, :, stage), params, RK4_g[:, :, stage-1], a, a_prime)
      end

      RK4_f[:, stage] .= f .- params.δt ./ params.δx .* RK4_df[:, stage]
      if !params.constant_g
        RK4_g[:, :, stage] .= g .- params.δt ./ params.δx .* RK4_dg[:, :, stage]
      end

      # Stage 4
      stage = 4

      if !params.constant_a
        compute_a!(a, a_prime, µ, µC, params, RK4_f[:, stage-1], RK4_g[:, :, stage-1])
      end

      compute_df!(view(RK4_df, :, stage), params, RK4_f[:, stage-1], a, a_prime)
      if !params.constant_g
        compute_dg!(view(RK4_dg, :, stage), params, RK4_g[:, :, stage-1], a, a_prime)
      end

      df .= (RK4_df[:, 1] .+ 2RK4_df[:, 2] + 2RK4_df[:, 3] + RK4_df[:, 4]) ./ 6
      if !params.constant_g
        dg .= (RK4_dg[:, :, 1] .+ 2RK4_dg[:, :, 2] + 2RK4_dg[:, :, 3] + RK4_dg[:, :, 4]) ./ 6
      end

      f -= params.δt / params.δx * df
      if !params.constant_g
        g -= params.δt / params.δx * dg
      end
    else
      throw("Unkown time-stepping method '$(params.time_stepping)'")
    end

    if i % params.store_every_iter == 0
      push!(store_i, i)
      push!(store_f, copy(f))
      if !params.constant_g
        push!(store_g_M1, sum(x .* g) / sum(g))
        push!(store_g, (i, copy(g)))
        if length(store_g) > 100
          store_hdf5_data(joinpath(store_dir, "data.hdf5"), ["g/$i" => g for (i, g) in store_g])
          empty!(store_g)
        end
      end
    end
  end

  @info "Saving data to disk @ $(store_dir)"
  store_pairs = vcat(
    ["i" => store_i, "f" => hcat(store_f...)],
  )
  if !params.constant_g
    append!(store_pairs,
      ["g/$i" => g for (i, g) in store_g],
      ["g_M1" => store_g_M1]
    )
  end

  store_hdf5_data(joinpath(store_dir, "data.hdf5"), store_pairs)

  p = UnicodePlots.lineplot(f, width=100, height=30, yscale=params.plot_scale, name="f")
  UnicodePlots.lineplot!(p, f_init, name="f_init")
  println(p)

  p = UnicodePlots.lineplot(a, width=100, height=30, name="a")
  UnicodePlots.lineplot!(p, a_init, name="a_init")
  println(p)

  if !params.constant_g
    p = UnicodePlots.lineplot(store_g_M1, width=100, height=30, name="∫∫ ω g(ω,m) dω dm")
    println(p)
  end

  @info @fmt extrema(f)

  @info @fmt mass_init
  mass = δx * sum(f)
  @info @fmt mass

  @info "Done"
end

end

