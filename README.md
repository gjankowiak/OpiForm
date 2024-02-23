# OpiForm

## Installation

This one liner will setup OpiForm for you:

    curl https://raw.githubusercontent.com/gjankowiak/OpiForm_alpha/master/install.sh | bash

## Usage

Switch to the OpiForm directory, you should then be ready to run the example:

    $ julia --project=. include("example.jl")

## Parameters

You can get a list of available parameters, along with the required types and default values by calling `OpiForm.Params.describe()`.
