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


module OpiForm

export find_free_suffix

import UnicodePlots

import Logging
import LoggingExtras

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

import DelimitedFiles: writedlm, readdlm

import LFRBenchmarkGraphs as LFR
import Random
import Distributions
# 
# global M
# 
# try
#   import GLMakie as M
# catch
#   import CairoMakie as M
# end

import CairoMakie as M
import CairoMakie

import KernelDensity
import KernelDensitySJ
import Graphs
import GraphMakie

import HDF5

macro fmt(v)
  r = string(v)
  return :(string($r, " = ", $(esc(v))))
end

function set_makie_backend(backend)
  return
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


include("Utils.jl")

include("Params.jl")
import .Params

include("Initialization.jl")
include("Plotting.jl")
include("Meanfield.jl")
include("Micro.jl")

end
