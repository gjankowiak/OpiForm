
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


module MeanfieldMono

import ..OpiForm: SA, SpA, M, clip, rand_symmetric, speyes, prepare_directory, issymmetric, symmetry_defect,
  build_x, load_hdf5_data, store_hdf5_data, @fmt, @left, @right, @up_mat, @down_mat, @left_mat, @right_mat

function compute_df_mono!(dst, params::NamedTuple, f, a, a_prime)
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
  else
    throw("not implemented")
  end

  # Neumann boundary conditions
  flux_l[1] = 0
  flux_r[end] = 0

  dst .= flux_r - flux_l
end

function compute_dg_mono!(dst, params::NamedTuple, g, a, a_prime)
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
  else
    throw("not implemented")
  end
  #
  # Neumann boundary conditions
  flux_lω[1,:] .= 0
  flux_rω[end,:] .= 0

  flux_lm[:,1] .= 0
  flux_rm[:,end] .= 0

  @. dst = flux_rω - flux_lω + flux_rm - flux_lm

end

function compute_a_mono!(a_dst, a_prime_dst, µ_dst, µC_dst, params::NamedTuple, f, g)
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
  if params.σ < 1
    throw("not implemented")
  else
    a_dst .= EB
    if params.approx_a_prime
      a_prime_dst .= 0.0
    end
  end
end

end

