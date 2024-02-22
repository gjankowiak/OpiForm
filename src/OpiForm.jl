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

import KernelDensity
import Graphs
import GraphMakie

import GLMakie
import CairoMakie
import GraphMakie

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


include("Utils.jl")

include("Params.jl")
import .Params

include("Initialization.jl")
include("Plotting.jl")
include("Meanfield.jl")
include("Micro.jl")

end
