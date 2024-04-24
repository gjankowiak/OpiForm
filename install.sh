#!/bin/bash

set -x
set -e

mkdir 'OpiForm'
cd 'OpiForm'
JULIA_PKG_DEVDIR="$PWD/dev" julia --project=. -e 'import Pkg; Pkg.develop(url="https://github.com/gjankowiak/OpiForm"); Pkg.resolve(); Pkg.instantiate()'
cp dev/OpiForm/examples/example.jl .

echo 'You should be ready to go! Try:'
echo 'cd OpiFrom'
echo 'julia --project=. example.jl'
