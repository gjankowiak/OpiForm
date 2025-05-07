import OpiForm as OF
import TOML

function compare_variance_lfr_multi(
  micro_dirs::Vector{String},
  meanfield_dirs::Vector{String},
  meanfield_mono_dirs::Vector{String},
  cutoff_factor::Float64=1.0,
  t_max::Real=0,
  stddev_min::Real=0
)

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

  aggregates_path = joinpath(prefix, "aggregates.toml")
  aggregates = nothing
  if isfile(aggregates_path)
    aggregates = TOML.parsefile(aggregates_path)["data"]
  end

  aggregates = sort(aggregates, by=x -> x["param"])

  if K_mfl > 0
    # labels_mfl = map(dn -> endswith("/", dn) ? basename(dirname(dn)) : basename(dn), meanfield_dirs)
    i_mfl_a = map(dn -> OF.load_hdf5_data(joinpath(dn, "data.hdf5"), "i"), meanfield_dirs)
    f_a = map(dn -> OF.load_hdf5_data(joinpath(dn, "data.hdf5"), "f"), meanfield_dirs)
    params_mfl_a = map(OF.Params.from_toml, meanfield_dirs)
    mu_mfl_a = map(x -> x.init_lfr_kwargs.mixing_parameter, params_mfl_a)

    N_a = map(f -> size(f, 1), f_a)

    f_stddev_a = map(dn -> sqrt.(OF.load_hdf5_data(joinpath(dn, "data.hdf5"), "f_var")), meanfield_dirs)
    rates_mfl_a = [OF.compute_rate(i_mfl_a[k], f_stddev_a[k], params_mfl_a[k].δt; cutoff_time=cutoff_factor * params_mfl_a[k].δt * i_mfl_a[k][end]) for k in 1:K_mfl]

    split_dir_names = OF.split_run_path(meanfield_dirs)
    prev_prefix = split_dir_names[1][1]

    aggregates_mfl = Dict{Symbol,Any}[]
    aggregate = Dict{Symbol,Any}()
    rates_acc = Float64[]
    stddev_acc = Vector{Float64}[]
    prefix_n = 0

    for (i, p) in enumerate(split_dir_names)
      if (p[1] != prev_prefix && i != 1) || (i == lastindex(split_dir_names) && prefix_n > 0)
        aggregate[:rates] = sum(rates_acc) / prefix_n
        aggregate[:param] = mu_mfl_a[i-1]
        aggregate[:time] = i_mfl_a[i-1] * params_mfl_a[i-1].δt
        aggregate[:stddev] = sum(stddev_acc) / prefix_n
        push!(aggregates_mfl, aggregate)
        aggregate = Dict{Symbol,Any}()
        empty!(rates_acc)
        empty!(stddev_acc)
        prefix_n = 0
        prev_prefix = p[1]
      end
      push!(rates_acc, rates_mfl_a[i])
      push!(stddev_acc, f_stddev_a[i])
      prefix_n += 1
    end


  if K_mfl_mono > 0
    # labels_mfl = map(dn -> endswith("/", dn) ? basename(dirname(dn)) : basename(dn), meanfield_mono_dirs)
    i_mfl_mono_a = map(dn -> OF.load_hdf5_data(joinpath(dn, "data.hdf5"), "i"), meanfield_mono_dirs)
    f_mono_a = map(dn -> OF.load_hdf5_data(joinpath(dn, "data.hdf5"), "f"), meanfield_mono_dirs)
    params_mfl_mono_a = map(OF.Params.from_toml, meanfield_mono_dirs)
    mu_mfl_mono_a = map(x -> x.init_lfr_kwargs.mixing_parameter, params_mfl_mono_a)

    N_mono_a = map(f -> size(f, 1), f_mono_a)

    f_stddev_mono_a = map(dn -> sqrt.(OF.load_hdf5_data(joinpath(dn, "data.hdf5"), "f_var")), meanfield_mono_dirs)
    rates_mfl_mono_a = [OF.compute_rate(i_mfl_mono_a[k], f_stddev_mono_a[k], params_mfl_mono_a[k].δt; cutoff_time=cutoff_factor * params_mfl_mono_a[k].δt * i_mfl_mono_a[k][end]) for k in 1:K_mfl_mono]

    split_dir_names = OF.split_run_path(meanfield_mono_dirs)
    prev_prefix = split_dir_names[1][1]

    aggregates_mfl = Dict{Symbol,Any}[]
    aggregate = Dict{Symbol,Any}()
    rates_acc = Float64[]
    stddev_acc = Vector{Float64}[]
    prefix_n = 0

    for (i, p) in enumerate(split_dir_names)
      if (p[1] != prev_prefix && i != 1) || (i == lastindex(split_dir_names) && prefix_n > 0)
        aggregate[:rates] = sum(rates_acc) / prefix_n
        aggregate[:param] = mu_mfl_mono_a[i-1]
        aggregate[:time] = i_mfl_mono_a[i-1] * params_mfl_mono_a[i-1].δt
        aggregate[:stddev] = sum(stddev_acc) / prefix_n
        push!(aggregates_mfl, aggregate)
        aggregate = Dict{Symbol,Any}()
        empty!(rates_acc)
        empty!(stddev_acc)
        prefix_n = 0
        prev_prefix = p[1]
      end
      push!(rates_acc, rates_mfl_mono_a[i])
      push!(stddev_acc, f_stddev_mono_a[i])
      prefix_n += 1
    end

  end


  if K_d > 0
    labels_d = map(dn -> endswith("/", dn) ? basename(dirname(dn)) : basename(dn), micro_dirs)
    i_d_a = map(dn -> load_hdf5_data(joinpath(dn, "data.hdf5"), "i"), micro_dirs)
    ω_a = map(dn -> load_hdf5_data(joinpath(dn, "data.hdf5"), "omega"), micro_dirs)
    params_d_a = map(Params.from_toml, micro_dirs)
    indep_param_d_a = map(parameter_extractor, params_d_a)

    adj_matrix_a = map(dn -> load_hdf5_sparse(joinpath(dn, "data.hdf5"), "adj_matrix"), micro_dirs)
    #adj_matrix_a = map(adj_matrix_full -> isnothing(adj_matrix_full) ? nothing : SpA.sparse(adj_matrix_full), adj_matrix_full_a)
    N_micro_a = map(ω -> size(ω, 1), ω_a)

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
    p2p_d_a = [peak2peak(ω; dims=1) for ω in ω_a]
    extrema_d_a = [extrema(ω; dims=1) for ω in ω_a]
    stddevs_d_a = [compute_stddev(ω_a[k], ω_inf_d_a[k]) for k in 1:K_d]
    rates_d_a = [compute_rate(i_d_a[k], stddevs_d_a[k], params_d_a[k].δt; cutoff_time=cutoff_factor * params_d_a[k].δt * i_d_a[k][end]) for k in 1:K_d]

    split_dir_names = split_run_path(micro_dirs)
    prev_prefix = split_dir_names[1][1]

    aggregates_d = Dict{Symbol,Any}[]
    aggregate = Dict{Symbol,Any}()
    rates_acc = Float64[]
    stddev_acc = Vector{Float64}[]
    prefix_n = 0

    for (i, p) in enumerate(split_dir_names)
      if (p[1] != prev_prefix && i != 1) || (i == lastindex(split_dir_names) && prefix_n > 0)
        aggregate[:rates] = sum(rates_acc) / prefix_n
        aggregate[:param] = indep_param_d_a[i-1]
        aggregate[:time] = i_d_a[i-1] * params_d_a[i-1].δt
        aggregate[:stddev] = sum(stddev_acc) / prefix_n
        push!(aggregates_d, aggregate)
        aggregate = Dict{Symbol,Any}()
        empty!(rates_acc)
        empty!(stddev_acc)
        prefix_n = 0
        prev_prefix = p[1]
      end
      push!(rates_acc, rates_d_a[i])
      push!(stddev_acc, stddevs_d_a[i])
      prefix_n += 1
    end

  end

  mfl_param = map(x -> x[:param], aggregates_mfl)
  mfl_rates = map(x -> x[:rates], aggregates_mfl)

  d_param = map(x -> x[:param], aggregates_d)
  d_rates = map(x -> x[:rates], aggregates_d)

  mean = (v) -> sum(v) / length(v)

  t_stars = [mean(a["t_star"]) for a in aggregates]
  clustering = [mean(a["clustering"]) for a in aggregates]

  fig = M.Figure(size=(1920, 1080))
  ax1 = M.Axis(fig[1, 1], yscale=log10, xlabel="time", title="Standard deviations")
  ax2 = M.Axis(fig[1, 2], xlabel="$(parameter_name)", title="Convergence rates", xscale=log10, yscale=M.Makie.pseudolog10)
  ax3 = M.Axis(fig[1, 3], xlabel="T*", ylabel="Convergence rate", xscale=log10)
  ax4 = M.Axis(fig[1, 4], xlabel="Clustering coeff.", xscale=identity)

  M.linkyaxes!(ax2, ax3, ax4)

  mfl_cutoff = cutoff_factor * params_mfl_a[1].δt * i_mfl_a[1][end]

  M.vspan!(ax1, 0.0, mfl_cutoff; color=(:blue, 0.1))

  for (i, a_d) in enumerate(aggregates_d)
    label = i == 1 ? "Micro" : nothing
    M.lines!(ax1, a_d[:time], a_d[:stddev], label=label, linestyle=:dot,
      color=p_to_color(a_d[:param]; mapping=log10, pmin=minimum(d_param), pmax=maximum(d_param)))
  end

  for (i, a_mfl) in enumerate(aggregates_mfl)
    label = i == 1 ? "MFL" : nothing
    M.lines!(ax1, a_mfl[:time], a_mfl[:stddev], label=label,
      color=p_to_color(a_mfl[:param]; mapping=log10, pmin=minimum(mfl_param), pmax=maximum(mfl_param)))
  end

  M.lines!(ax2, d_param, d_rates, label="Micro", linestyle=:dot, color=:blue)
  M.lines!(ax2, mfl_param, mfl_rates, label="MFL", color=:blue)

  M.lines!(ax3, t_stars, d_rates, label="Micro", linestyle=:dot, color=:blue)
  M.lines!(ax3, t_stars, mfl_rates, label="MFL", color=:blue)

  M.lines!(ax4, clustering, d_rates, label="Micro", linestyle=:dot, color=:blue)
  M.lines!(ax4, clustering, mfl_rates, label="MFL", color=:blue)

  if t_max > 0
    M.xlims!(ax1, low=0, high=t_max)
  end

  if stddev_min > 0
    M.ylims!(ax1, low=stddev_min)
  end

  M.axislegend(ax1)
  M.axislegend(ax2, position=:lt)
  M.axislegend(ax3)
  M.axislegend(ax4)

  M.save("$prefix/comparison.png", fig)
  M.save("$prefix/comparison.svg", fig)
  @info "Plot saved at:"
  @info "$prefix/comparison.svg"
  @info "$prefix/comparison.png"

  try
    display(fig)
  catch
    @error "Cannot display plot window, are you logged in over SSH?"
  end

  return (aggregates_micro=aggregates_d, aggregates_mfl=aggregates_mfl, mfl_cutoff=mfl_cutoff)

end
