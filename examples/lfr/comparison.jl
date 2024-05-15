import OpiForm
import HDF5
import DelimitedFiles: writedlm

include("../check_degree.jl")

function has_h5_key(fn, key)
  h5 = HDF5.h5open(fn, "r")
  return haskey(h5, key)
end

function perform_comparison()

  β_σ²s = [0.001; 0.002; 0.004; 0.012]

  for β_σ² in β_σ²s
    base_dir = "results/LFR/N_micro=1000,N_mfl=301/σ²=$β_σ²/"

    @info "Using base directory $(base_dir)"

    check_degree(base_dir, :LFR; max_degree=100)

    mfl_dirs = String[]
    micro_dirs = String[]

    dirs = walkdir(base_dir; follow_symlinks=true)

    regex_micro = r"micro-[0-9]+$"
    regex_meanfield = r"meanfield-[0-9]+$"

    # @show dirs

    for d_tuple in dirs
      d = d_tuple[1]
      data_path = joinpath(d, "data.hdf5")
      if endswith(d, regex_micro) && isfile(data_path)
        push!(micro_dirs, d)
      elseif endswith(d, regex_meanfield) && isfile(data_path) && has_h5_key(data_path, "f")
        push!(mfl_dirs, d)
      else
      end
    end

    (aggregates_micro, aggregates_mfl, mfl_cutoff) = OpiForm.compare_variance_lfr_EA(mfl_dirs, micro_dirs; cutoff_factor=0.2, t_max=10, stddev_min=10^(-2.5))

    agg_dir = joinpath(base_dir, "aggregates_convergence")
    mkpath(agg_dir)

    open(joinpath(agg_dir, "mfl_cutoff.tsv"), "w") do f
      write(f, "$mfl_cutoff\n")
    end

    micro_rates = vcat([[aggregates_micro[i][:param] aggregates_micro[i][:rates]] for i in eachindex(aggregates_micro)]...)
    mfl_rates = vcat([[aggregates_mfl[i][:param] aggregates_mfl[i][:rates]] for i in eachindex(aggregates_mfl)]...)

    @show micro_rates
    open(joinpath(agg_dir, "micro.tsv"), "w") do f
      writedlm(f, micro_rates, '\t')
    end
    open(joinpath(agg_dir, "mfl.tsv"), "w") do f
      writedlm(f, mfl_rates, '\t')
    end

    for a in aggregates_micro
      dn = joinpath(agg_dir, "mu=$(a[:param])")
      mkpath(dn)
      open(joinpath(dn, "micro.tsv"), "w") do f
        writedlm(f, [a[:time] a[:stddev]])
      end
    end

    for a in aggregates_mfl
      dn = joinpath(agg_dir, "mu=$(a[:param])")
      mkpath(dn)
      open(joinpath(dn, "mfl.tsv"), "w") do f
        writedlm(f, [a[:time] a[:stddev]])
      end
    end

  end

end
