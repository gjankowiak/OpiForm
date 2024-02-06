# OpiForm

## Usage

This package is still in development. To use, start from a fresh julia environment:

    $ julia --project=.

Fetch the package as a development package:

    ] dev --local https://github.com/gjankowiak/OpiForm_alpha

Copy the example file to the current directory:

    ; cp dev/OpiForm_alpha/example.jl .

Fetch the Polynomials package:

    ] add Polynomials

You should be ready to run the example:

    julia> include("example.jl")
