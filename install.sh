#!/bin/bash

set -x
set -e

mkdir 'OpiForm'
cd 'OpiForm'
JULIA_PKG_DEVDIR="$PWD/dev" julia --project=. -e 'import Pkg; Pkg.develop(url="https://github.com/gjankowiak/OpiForm"); Pkg.resolve(); Pkg.instantiate()'

echo 'You should be ready to go! Try:'
echo 'cd OpiFrom'
echo "julia --project=. -e 'include(\"examples/paper/beta_2_with_var/run_monoproc.jl\")'"
