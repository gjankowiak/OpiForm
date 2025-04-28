import OpiForm as OF
# import GLMakie
import CairoMakie
import DelimitedFiles as DF
CairoMakie.activate!()
import TOML

function compare_variance_lfr_multi(
  micro_dirs::Vector{String},
  meanfield_dirs::Vector{String},
  meanfield_mono_dirs::Vector{String};
  cutoff_factor::Float64=1.0,
  t_max::Real=0,
  t_min_rate::Real=0,
  t_max_rate::Real=Inf,
  stddev_min::Real=0,
  backend::Symbol=:cairo,
  show_rates::Bool=true,
  show_errors::Bool=false
)

  if backend == :cairo
    M = CairoMakie
    CairoMakie.activate!()
  else
    M = GLMakie
    GLMakie.activate!()
  end

  dirs = [micro_dirs..., meanfield_dirs..., meanfield_mono_dirs...]

  # check that the dirs actually exist
  @assert all(isdir, micro_dirs)
  @assert all(isdir, meanfield_dirs)
  @assert all(isdir, meanfield_mono_dirs)

  K_mfl, K_mfl_mono, K_d = length(meanfield_dirs), length(meanfield_mono_dirs), length(micro_dirs)

  if all(isempty, dirs)
    @error "No dir provided"
    return
  end

  prefix = OF.longest_prefix(dirs, existing_dir=true)

  if K_mfl > 0
    # labels_mfl = map(dn -> endswith("/", dn) ? basename(dirname(dn)) : basename(dn), meanfield_dirs)
    i_mfl_a = map(dn -> OF.load_hdf5_data(joinpath(dn, "data.hdf5"), "i"), meanfield_dirs)
    f_a = map(dn -> OF.load_hdf5_data(joinpath(dn, "data.hdf5"), "f"), meanfield_dirs)
    params_mfl_a = map(OF.Params.from_toml, meanfield_dirs)
    mu_mfl_a = map(x -> x.init_lfr_kwargs.mixing_parameter, params_mfl_a)

    times_mfl_a = [params_mfl_a[k].δt * i_mfl_a[k] for k in 1:K_mfl]

    N_a = map(f -> size(f, 1), f_a)

    f_stddev_a = map(dn -> sqrt.(OF.load_hdf5_data(joinpath(dn, "data.hdf5"), "f_var")), meanfield_dirs)
    rates_mfl_a = OF.unzip([OF.compute_rate_regression(i_mfl_a[k], f_stddev_a[k], params_mfl_a[k].δt; t_min=t_min_rate, t_max=t_max_rate) for k in 1:K_mfl])
  end


  if K_mfl_mono > 0
    # labels_mfl = map(dn -> endswith("/", dn) ? basename(dirname(dn)) : basename(dn), meanfield_mono_dirs)
    i_mfl_mono_a = map(dn -> OF.load_hdf5_data(joinpath(dn, "data.hdf5"), "i"), meanfield_mono_dirs)
    f_mono_a = map(dn -> OF.load_hdf5_data(joinpath(dn, "data.hdf5"), "f"), meanfield_mono_dirs)
    params_mfl_mono_a = map(OF.Params.from_toml, meanfield_mono_dirs)
    mu_mfl_mono_a = map(x -> x.init_lfr_kwargs.mixing_parameter, params_mfl_mono_a)

    times_mfl_mono_a = [params_mfl_mono_a[k].δt * i_mfl_mono_a[k] for k in 1:K_mfl_mono]

    N_mono_a = map(f -> size(f, 1), f_mono_a)

    f_stddev_mono_a = map(dn -> sqrt.(OF.load_hdf5_data(joinpath(dn, "data.hdf5"), "f_var")), meanfield_mono_dirs)
    rates_mfl_mono_a = OF.unzip([OF.compute_rate_regression(i_mfl_mono_a[k], f_stddev_mono_a[k], params_mfl_mono_a[k].δt; t_min=t_min_rate, t_max=t_max_rate) for k in 1:K_mfl_mono])
  end


  if K_d > 0
    # labels_d = map(dn -> endswith("/", dn) ? basename(dirname(dn)) : basename(dn), micro_dirs)
    i_d_a = map(dn -> OF.load_hdf5_data(joinpath(dn, "data.hdf5"), "i"), micro_dirs)
    ω_a = map(dn -> OF.load_hdf5_data(joinpath(dn, "data.hdf5"), "omega"), micro_dirs)
    params_d_a = map(OF.Params.from_toml, micro_dirs)
    mu_d_a = map(x -> x.init_lfr_kwargs.mixing_parameter, params_d_a)

    adj_matrix_a = map(dn -> OF.load_hdf5_sparse(joinpath(dn, "data.hdf5"), "adj_matrix"), micro_dirs)
    N_micro_a = map(ω -> size(ω, 1), ω_a)

    times_d_a = [params_d_a[k].δt * i_d_a[k] for k in 1:K_d]

    function compute_weighted_avg(k)
      adj_matrix = adj_matrix_a[k]
      ω = ω_a[k]
      N_micro = N_micro_a[k]

      if !isnothing(adj_matrix)
        sharp_I = vec(sum(adj_matrix; dims=2))
        n_connections = sum(sharp_I)

        return [sum(ω[:, k] .* sharp_I) ./ n_connections for k in axes(ω, 2)]
      else
        return [sum(ω[:, k]) / (N_micro - 1) for k in axes(ω, 2)]
      end
    end

    ω_inf_d_a = [compute_weighted_avg(k) for k in 1:K_d]
    stddevs_d_a = [compute_stddev(ω_a[k], ω_inf_d_a[k]) for k in 1:K_d]
    rates_d_a = OF.unzip([OF.compute_rate_regression(i_d_a[k], stddevs_d_a[k], params_d_a[k].δt; t_min=t_min_rate, t_max=t_max_rate) for k in 1:K_d])

  end

  if show_rates
    fig = M.Figure(size=(1920, 1080))
  else
    fig = M.Figure(size=(720, 1080))
  end
  ax1 = M.Axis(fig[1, 1], yscale=log10, xlabel="time", title="Standard deviations")
  if show_rates
    ax2 = M.Axis(fig[1, 2], xlabel="μ", title="Convergence rates", xscale=log10)
    if show_errors
      ax3 = M.Axis(fig[1, 3], xlabel="μ", title="Average relative regression error", xscale=log10, yscale=identity)
    end
  end

  mfl_cutoff = cutoff_factor * params_mfl_a[1].δt * i_mfl_a[1][end]

  M.vspan!(ax1, t_min_rate, t_max_rate; color=(:blue, 0.1))

  # Microscopic stddev
  for k in 1:K_d
    label = k == 1 ? "Micro" : nothing
    M.lines!(ax1, times_d_a[k], stddevs_d_a[k], label=label, linestyle=:dot, linewidth=2,
      color=OF.p_to_color(mu_d_a[k]; mapping=log10, pmin=minimum(mu_d_a), pmax=maximum(mu_d_a)))
  end

  # Continuous stddev
  for k in 1:K_mfl
    label = k == 1 ? "Continuous (multi groups)" : nothing
    M.lines!(ax1, times_mfl_a[k], f_stddev_a[k], label=label, linewidth=2,
      color=OF.p_to_color(mu_mfl_a[k]; mapping=log10, pmin=minimum(mu_mfl_a), pmax=maximum(mu_mfl_a)))
  end
  #
  # Continuous stddev (single group)
  for k in 1:K_mfl_mono
    label = k == 1 ? "Continuous (single group)" : nothing
    M.lines!(ax1, times_mfl_mono_a[k], f_stddev_mono_a[k], label=label, linestyle=:dash, linewidth=2, color=:black)
  end

  M.axislegend(ax1)

  if t_max > 0
    M.xlims!(ax1, low=0, high=t_max)
  end

  if stddev_min > 0
    M.ylims!(ax1, low=stddev_min)
  end

  if show_rates
    M.scatterlines!(ax2, mu_d_a, -rates_d_a[:, 1], label="Micro", linestyle=:dot, color=:black)
    M.scatterlines!(ax2, mu_mfl_a, -rates_mfl_a[:, 1], label="Continuous (multi groups)", color=:black)
    M.scatterlines!(ax2, mu_mfl_mono_a, -rates_mfl_mono_a[:, 1], label="Continuous (single group)", color=:black, linestyle=:dash)
    # M.scatter!(ax2, mu_d_a, -rates_d_a[:, 1], label="Micro")
    # M.scatter!(ax2, mu_mfl_a, -rates_mfl_a[:, 1], label="MFL (multi group)")
    # M.scatter!(ax2, mu_mfl_mono_a, -rates_mfl_mono_a[:, 1], label="MFL (single group)")

    if show_errors
      M.lines!(ax3, mu_d_a, rates_d_a[:, 4], label="Micro", linestyle=:dot, color=:black)
      M.lines!(ax3, mu_mfl_a, rates_mfl_a[:, 4], label="Continuous (multi groups)", color=:black)
      M.lines!(ax3, mu_mfl_mono_a, rates_mfl_mono_a[:, 4], label="Continuous (single group)", color=:black, linestyle=:dash)
    end

    M.axislegend(ax2, position=:lt)
  end

  tuples = [
    ("var_micro", times_d_a, mu_d_a, stddevs_d_a, rates_d_a),
    ("var_mfl_multi", times_mfl_a, mu_mfl_a, f_stddev_a, rates_mfl_a),
    ("var_mfl_mono", times_mfl_mono_a, mu_mfl_mono_a, f_stddev_mono_a, rates_mfl_mono_a)
  ]

  min_time_samples = minimum([minimum(map(length, t[2])) for t in tuples])

  for t in tuples
    DF.writedlm("$prefix/$(t[1]).tsv", [t[2][1][1:min_time_samples] map(v -> v[1:min_time_samples], t[4])...])
    DF.writedlm("$prefix/$(t[1])_headers.tsv", ["time", "stddev (1 column per value of mu)"])
    @info "Saved time sampled stddevs to $prefix/$(t[1]).tsv"
  end

  DF.writedlm("$prefix/rates.tsv", [mu_d_a -rates_d_a[:, 1] -rates_mfl_a[:, 1] -rates_mfl_mono_a[:, 1]])
  DF.writedlm("$prefix/rates_headers.tsv", ["mu" "rates_micro" "rates_mfl_multi" "rates_mfl_single"])
  @info "Saved convergence rates to to $prefix/rates.tsv"

  M.save("$prefix/comparison.png", fig)

  @info "Plot saved at:"
  @info "$prefix/comparison.png"
  if backend == :cairo
    M.save("$prefix/comparison.svg", fig)
    @info "$prefix/comparison.svg"
  end

  try
    display(fig)
  catch
    @error "Cannot display plot window, are you logged in over SSH?"
  end

end
