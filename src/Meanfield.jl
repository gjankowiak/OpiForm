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


module MeanField

import UnicodePlots
import HDF5

import Graphs: SimpleGraph, is_connected, connected_components

import ..OpiForm: SA, SpA, M, clip, rand_symmetric, speyes, prepare_directory, issymmetric, symmetry_defect,
  build_x, load_hdf5_data, store_hdf5_data, @fmt, @left, @right, @up_mat, @down_mat, @left_mat, @right_mat

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
  n_groups = size(f, 2)

  f_l = SA.shiftedarray(f, (1, 0), 0.0)
  f_r = SA.shiftedarray(f, (-1, 0), 0.0)
  a_l = SA.shiftedarray(a, (1, 0), 0.0)
  a_r = SA.shiftedarray(a, (-1, 0), 0.0)

  if params.approx_a_prime
    a_prime_l = SA.shiftedarray(a_prime, (1, 0), 0.0)
    a_prime_r = SA.shiftedarray(a_prime, (-1, 0), 0.0)
  end

  if params.flux == :lLF
    # local Lax-Fridriedrich
    if params.approx_a_prime
      max_C_l = maximum(abs, reshape([a + a_prime .* f a_l + a_prime_l .* f_l], (params.N_mfl, n_groups, 2)), dims=3)
      max_C_r = maximum(abs, reshape([a + a_prime .* f a_r + a_prime_r .* f_r], (params.N_mfl, n_groups, 2)), dims=3)
    else
      max_C_l = reshape(maximum(abs, reshape([a a_l], (params.N_mfl, n_groups, 2)), dims=3), (params.N_mfl, n_groups))
      max_C_r = reshape(maximum(abs, reshape([a a_r], (params.N_mfl, n_groups, 2)), dims=3), (params.N_mfl, n_groups))
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
    throw("Not implemented")
    max_C = maximum(abs, a + a_prime .* f)
    flux_l = 0.5 * (f_l .* a_l .+ f .* a .- params.LF_relaxation * max_C * (f .- f_l))
    flux_r = 0.5 * (f_r .* a_r .+ f .* a .- params.LF_relaxation * max_C * (f_r .- f))
  elseif params.flux == :LW_Richtmyer
    # Richtmyer
    throw("Not implemented")
    flux_l = 0.5 * a .* ((f .+ f_l) .- params.δt / params.δx * (f .* a .- f_l .* a_l))
    flux_r = 0.5 * a .* ((f_r .+ f) .- params.δt / params.δx * (f_r .* a_r .- f .* a))
  elseif params.flux == :upwind
    # upwind
    throw("Not implemented")
    flux_l = ((f_l .< f) .* min.(a .* f, a_l .* f_l) .+
              (f_l .>= f) .* max.(a .* f, a_l .* f_l))
    flux_r = ((f .< f_r) .* min.(a_r .* f_r, a .* f) .+
              (f .>= f_r) .* max.(a_r .* f_r, a .* f))
  elseif params.flux == :constant_godunov
    # wave speed
    throw("Not implemented")
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
    throw("Not implemented")

    if params.godunov_entropy_fix
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
    throw("not implemented")
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
    throw("unknown flux '$(params.flux)'")
  end

  # Neumann boundary conditions
  flux_l[1,:] .= 0
  flux_r[end,:] .= 0

  dst .= flux_r - flux_l
end

function compute_dg!(dst, params::NamedTuple, g, a, a_prime)
  # Exact same computation as for f but along both dimensions.
  # There is probably a way to take advantage of the symmetry of g.
  n_groups = size(g, 3)

  g_lω, g_rω = SA.shiftedarray(g, (1, 0, 0, 0), 0.0), SA.shiftedarray(g, (-1, 0, 0, 0), 0.0)
  g_lm, g_rm = SA.shiftedarray(g, (0, 1, 0, 0), 0.0), SA.shiftedarray(g, (0, -1, 0, 0), 0.0)

  a_l = SA.shiftedarray(a, (1, 0), 0.0)
  a_r = SA.shiftedarray(a, (-1, 0), 0.0)

  if params.approx_a_prime
    a_prime_l = SA.shiftedarray(a_prime, (1, 0), 0.0)
    a_prime_r = SA.shiftedarray(a_prime, (-1, 0), 0.0)
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
      max_C_l = reshape(maximum(abs, reshape([a a_l], (params.N_mfl, n_groups, 2)), dims=3), (params.N_mfl, n_groups))
      max_C_r = reshape(maximum(abs, reshape([a a_r], (params.N_mfl, n_groups, 2)), dims=3), (params.N_mfl, n_groups))
    end

    flux_lω = zeros(params.N_mfl, params.N_mfl, n_groups, n_groups)
    flux_rω = zeros(params.N_mfl, params.N_mfl, n_groups, n_groups)
    flux_lm = zeros(params.N_mfl, params.N_mfl, n_groups, n_groups)
    flux_rm = zeros(params.N_mfl, params.N_mfl, n_groups, n_groups)

    for p = 1:n_groups
      for q = 1:n_groups
        # Lax-Friedrich flux
        flux_lω[:,:,p,q] = 0.5 * (g_lω[:,:,p,q] .* a_l[:,p] .+ g[:,:,p,q] .* a[:,p] .- params.LF_relaxation * max_C_l[:,p] .* (g[:,:,p,q] .- g_lω[:,:,p,q]))
        flux_rω[:,:,p,q] = 0.5 * (g_rω[:,:,p,q] .* a_r[:,p] .+ g[:,:,p,q] .* a[:,p] .- params.LF_relaxation * max_C_r[:,p] .* (g_rω[:,:,p,q] .- g[:,:,p,q]))

        flux_lm[:,:,p,q] = 0.5 * (g_lm[:,:,p,q] .* a_l[:,q]' .+ g[:,:,p,q] .* a[:,q]' .- params.LF_relaxation * max_C_l[:,q] .* (g[:,:,p,q] .- g_lm[:,:,p,q]))
        flux_rm[:,:,p,q] = 0.5 * (g_rm[:,:,p,q] .* a_r[:,q]' .+ g[:,:,p,q] .* a[:,q]' .- params.LF_relaxation * max_C_r[:,q] .* (g_rm[:,:,p,q] .- g[:,:,p,q]))
      end
    end

  elseif params.flux == :LF
    throw("not implemented")
    max_C = maximum(abs, a)

    # Lax-Friedrich flux
    flux_lω = 0.5 * (g_lω .* a_l .+ g .* a .- params.LF_relaxation * max_C .* (g .- g_lω))
    flux_rω = 0.5 * (g_rω .* a_r .+ g .* a .- params.LF_relaxation * max_C .* (g_rω .- g))

    flux_lm = 0.5 * (g_lm .* a_l' .+ g .* a' .- params.LF_relaxation * max_C .* (g .- g_lm))
    flux_rm = 0.5 * (g_rm .* a_r' .+ g .* a' .- params.LF_relaxation * max_C .* (g_rm .- g))
  elseif params.flux == :LW_Richtmyer
    throw("not implemented")
    # Richtmyer
    flux_lω = 0.5 * g .* ((g .+ g_lω) .- params.δt / params.δx * (g_lω .* a_l .- g .* a))
    flux_rω = 0.5 * g .* ((g_rω .+ g) .- params.δt / params.δx * (g_rω .* a_r .- g .* a))

    flux_lm = 0.5 * g .* ((g .+ g_lm) .- params.δt / params.δx * (g_lm .* a_l' .- g .* a'))
    flux_rm = 0.5 * g .* ((g_rm .+ g) .- params.δt / params.δx * (g_rm .* a_r' .- g .* a'))
  elseif params.flux == :KT
    throw("not implemented")
    λ = params.δt / params.δx

    (α_l, α_r) = max.(abs.(a), abs.(a_l)), max.(abs.(a), abs.(a_r))
    α_lll = @left α_l
    α_rrr = @right α_r

    #
    # g_ω
    #

    f_x = @. minmod((g - g_lω) / params.δx, (g_rω - g) / params.δx)

    f_x_ll = @up_mat f_x
    f_x_rr = @down_mat f_x

    f_l_L = @. g_lω + params.δx * f_x_ll * (0.5 - λ * α_l)
    f_r_L = @. g + params.δx * f_x * (0.5 - λ * α_r)
    f_l_R = @. g - params.δx * f_x * (0.5 - λ * α_l)
    f_r_R = @. g_rω - params.δx * f_x_rr * (0.5 - λ * α_r)

    f_p_l_L = @. f_l_L - 0.5 * params.δt * a_l
    f_p_r_L = @. f_r_L - 0.5 * params.δt * a
    f_p_l_R = @. f_l_R - 0.5 * params.δt * a
    f_p_r_R = @. f_r_R - 0.5 * params.δt * a_r

    w_p_l = @. (0.5 * (g_lω + g)
                +
                0.25 * (params.δx - α_l * params.δt) * (f_x_ll - f_x)
                -
                0.5 / α_l * (a * f_p_l_R - a_l * f_p_l_L)
    )
    w_p_r = @. (0.5 * (g + g_rω)
                +
                0.25 * (params.δx - α_r * params.δt) * (f_x - f_x_rr)
                -
                0.5 / α_r * (a_r * f_p_r_R - a * f_p_r_L)
    )

    _w_p = @. 0.5 * params.δt * (α_l - α_r) * f_x - λ / (1 - λ * (a_l + a_r)) * (a * (f_p_r_L - f_p_l_R))
    w_p = @. g + _w_p

    w_p_ll = @up_mat w_p
    w_p_rr = @down_mat w_p

    f_x_p_r = @. 0.5 * params.δx * minmod((w_p_rr - w_p_r) / (1 + λ * (α_r - α_rrr)), (w_p_r - w_p) / (1 + λ * (α_r - α_l)))
    f_x_p_l = @. 0.5 * params.δx * minmod((w_p - w_p_l) / (1 + λ * (α_l - α_r)), (w_p_l - w_p_ll) / (1 + λ * (α_l - α_lll)))

    # TODO: check factor
    d_g_ω = @. (
      λ * α_l * w_p_l + (1 - λ * (α_l + α_r)) * w_p
      +
      λ * α_r * w_p_r + 0.5 * params.δx * ((λ * α_l)^2 * f_x_p_l - (λ * α_r)^2 * f_x_p_r)
    )

    #
    # g_m
    #

    f_x = @. minmod((g - g_lm) / params.δx, (g_rm - g) / params.δx)

    f_x_ll = @left_mat f_x
    f_x_rr = @right_mat f_x

    f_l_L = @. g_lm + params.δx * f_x_ll * (0.5 - λ * α_l')
    f_r_L = @. g + params.δx * f_x * (0.5 - λ * α_r')
    f_l_R = @. g - params.δx * f_x * (0.5 - λ * α_l')
    f_r_R = @. g_rm - params.δx * f_x_rr * (0.5 - λ * α_r')

    f_p_l_L = @. f_l_L - 0.5 * params.δt * a_l'
    f_p_r_L = @. f_r_L - 0.5 * params.δt * a'
    f_p_l_R = @. f_l_R - 0.5 * params.δt * a'
    f_p_r_R = @. f_r_R - 0.5 * params.δt * a_r'

    w_p_l = @. (0.5 * (g_lm + g)
                +
                0.25 * (params.δx - α_l' * params.δt) * (f_x_ll - f_x)
                -
                0.5 / α_l' * (a' * f_p_l_R - a_l' * f_p_l_L)
    )
    w_p_r = @. (0.5 * (g + g_rm)
                +
                0.25 * (params.δx - α_r' * params.δt) * (f_x - f_x_rr)
                -
                0.5 / α_r' * (a_r' * f_p_r_R - a' * f_p_r_L)
    )

    _w_p = @. 0.5 * params.δt * (α_l - α_r)' * f_x - λ / (1 - λ * (a_l + a_r)') * (a' * (f_p_r_L - f_p_l_R))
    w_p = @. g + _w_p

    w_p_ll = @left_mat w_p
    w_p_rr = @right_mat w_p

    f_x_p_r = @. 0.5 * params.δx * minmod((w_p_rr - w_p_r) / (1 + λ * (α_r - α_rrr)'), (w_p_r - w_p) / (1 + λ * (α_r - α_l)'))
    f_x_p_l = @. 0.5 * params.δx * minmod((w_p - w_p_l) / (1 + λ * (α_l - α_r)'), (w_p_l - w_p_ll) / (1 + λ * (α_l - α_lll)'))

    # TODO: check factor
    d_g_m = @. (
      λ * α_l' * w_p_l + (1 - λ * (α_l + α_r)') * w_p
      +
      λ * α_r' * w_p_r + 0.5 * params.δx * ((λ * α_l')^2 * f_x_p_l - (λ * α_r')^2 * f_x_p_r)
    )

    dst .= 0.5 .* (d_g_ω .+ d_g_m)


    return
  else
    throw("unknown flux '$(params.flux)'")
  end


  # Neumann boundary conditions
  # FIXME INDICES
  flux_lω[1] = 0
  flux_rω[end] = 0

  flux_lm[1] = 0
  flux_rm[end] = 0

  @. dst = flux_rω - flux_lω + flux_rm - flux_lm


end

function compute_a!(a_dst, a_prime_dst, µ_dst, µC_dst, params::NamedTuple, f, g)
  n_groups = size(f, 2)
  # compute EB
  #
  # g is normalized so that ∫∫g = connection_density*N_micro
  # η(ω,m) = g(ω,m) / ∫ g(ω,m') dm'

  if params.constant_g
    # if g is constant,  g = connection_density*N_micro / (params.Ω_width)²
    # η(ω,m) = 1 / params.Ω_width
    η = 1 / (params.Ω_width)
  else
    g_mass = params.δx * sum(g; dims=(2,4))
    g_mass_inv = 1 ./ g_mass
    g_mass_inv[g_mass_inv.>1/params.int_threshold] .= 0
    η = g .* g_mass_inv
  end

  EB = reshape(params.δx * sum(η .* params.D_matrix; dims=(2,4)), (params.N_mfl, n_groups))

  # Check the normalization.
  # Original model: chamber_size = 1
  # Alernative 1: params.δx * sum(params.EC_mask_matrix; dims=2) -- unstable with lLF

  # compute EC
  if params.σ < 1
    throw("not implemented")
  else
    a_dst .= EB
    if params.approx_a_prime
      a_prime_dst .= 0.0
    end
  end
end

function compute_f_stats(f::Vector{Float64}, g::Matrix{Float64}, x::AbstractVector)
  g_M1_n = compute_g_M1_normalized(g, x)
  f_var = compute_f_var(f, x, g_M1_n)
  return (g_M1_n=g_M1_n, f_var=f_var)
end

function compute_g_M1_normalized(g::Matrix{Float64}, x::AbstractVector)
  return sum(x .* g) / sum(g)
end

function compute_f_var(f::Vector{Float64}, x::AbstractVector, center::Float64)
  δx = x[2] - x[1]
  M2 = sum(f .* (x .- center) .^ 2) * δx
  return M2
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

  # Echo chamber mask matrix
  if params_in.σ == 1
    EC_mask_matrix = nothing
  else
    throw("Unknown EC_type '$(params_in.EC_type)'")
  end

  # Debate matrix
  D_matrix = params_in.D_func.(x .- x')

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

  # FIXME:
  f_init = load_hdf5_data(joinpath(store_dir, "data.hdf5"), "f_init")
  n_groups = size(f_init, 2)
  f = copy(f_init)
  if !params.constant_g
    g = load_hdf5_data(joinpath(store_dir, "data.hdf5"), "g_init")
  else
    g = nothing
  end

  i = 0

  # FIXME: remove
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
    if params.store_g
      store_g = [(0, copy(g))]
    end
    # FIXME:
    # f_stats = compute_f_stats(f, g, x)
    # store_g_M1_n = [f_stats.g_M1_n]
    # store_f_var = [f_stats.f_var]
  end

  # Initial mass
  mass_init = δx * sum(f)

  a = zeros(params.N_mfl, n_groups)
  a_prime = zeros(params.N_mfl, n_groups)
  µ, µC = zeros(params.N_mfl), zeros(params.N_mfl)

  df = zeros(params.N_mfl, n_groups)
  if !params.constant_g
    dg = zeros(params.N_mfl, params.N_mfl, n_groups, n_groups)
  end

  compute_a!(a, a_prime, µ, µC, params, f, g)
  a_init = copy(a)

  ## Print parameters and plot initial conditions before starting

  if abs(params.LF_relaxation - 1) > 1e-10
    @warn "LF_relaxation != 1 ($(params.LF_relaxation))"
  end

  mfl_λ = params.mfl_connectivity_factor

  while i < params.max_iter

    # Check that g is roughly symmetric
    symmetry_tol = 1e-8
    if params.perform_checks && !issymmetric(g, tol=symmetry_tol)
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
        # FIXME: connectivity factor missing
        f .= df
        if !params.constant_g
          g .= dg
        end
      else
        f .-= params.δt / params.δx * mfl_λ * df
        if !params.constant_g
          if params.f_dependent_g
            # DEFINITION G
            g .= (f .* α .* f') ./ params.connection_density
          else
            g .-= params.δt / params.δx * mfl_λ * dg
          end
        end
      end
    else
      throw("Unkown time-stepping method '$(params.time_stepping)'")
    end

    if i % params.store_every_iter == 0
      push!(store_i, i)
      push!(store_f, copy(f))
      if !params.constant_g
        # f_stats = compute_f_stats(f, g, x)
        # push!(store_g_M1_n, f_stats.g_M1_n)
        # push!(store_f_var, f_stats.f_var)
        if params.store_g
          push!(store_g, (i, copy(g)))
          if length(store_g) > 100
            store_hdf5_data(joinpath(store_dir, "data.hdf5"), ["g/$i" => g for (i, g) in store_g])
            empty!(store_g)
          end
        end
      end
    end
  end

  @info "Saving data to disk @ $(store_dir)"
  store_pairs = vcat(
    ["i" => store_i, "f" => cat(store_f...; dims=3)],
  )
  if !params.constant_g
    # append!(store_pairs,
    #   ["f_var" => store_f_var],
    #   ["g_M1_n" => store_g_M1_n]
    # )
    if params.store_g
      append!(store_pairs,
        ["g/$i" => g for (i, g) in store_g],
      )
    end
  end

  store_hdf5_data(joinpath(store_dir, "data.hdf5"), store_pairs)

  @info @fmt mass_init
  mass = δx * sum(f)
  @info @fmt mass

  @info "Done"
end

end

