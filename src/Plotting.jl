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

function p_to_color(p; pmin=0, pmax=1, mapping=identity, color_min=M.RGBf(0.25, 0.25, 0.25), color_max=M.RGBf(0.0, 0.0, 1.0), reverse=false)
  λ = (mapping(p) - mapping(pmin)) / (mapping(pmax) - mapping(pmin))
  if reverse
    return λ * color_min + (1 - λ) * color_max
  else
    return (1 - λ) * color_min + λ * color_max
  end
end

function compute_rate(i_a::Vector{Int64}, p2p_a::Vector{Float64}, δt::Float64; cutoff_time::Float64=5.0)
  idc = searchsortedfirst(i_a * δt, cutoff_time)
  return log(p2p_a[idc] / p2p_a[1]) / (δt * i_a[idc])
end

function compute_stddev(ω, centers)
  N = size(ω, 1)
  return sqrt.(vec(sum((ω .- centers') .^ 2; dims=1)) / N)
end

function plot_result(; output_filename::String="", meanfield_dir::Union{String,Nothing}=nothing, micro_dir::Union{String,Nothing}=nothing, kwargs...)
  mfl_dirs = isnothing(meanfield_dir) ? String[] : [meanfield_dir]
  d_dirs = isnothing(micro_dir) ? String[] : [micro_dir]
  return plot_results(; output_filename=output_filename, meanfield_dirs=mfl_dirs, micro_dirs=d_dirs, kwargs...)
end

function plot_results(; output_filename::String="",
  meanfield_dirs::Vector{String}=String[],
  micro_dirs::Vector{String}=String[],
  kwargs...)

  # check that the dirs actually exist
  @assert all(isdir, meanfield_dirs)
  @assert all(isdir, micro_dirs)

  has_mfl, has_d = length(meanfield_dirs) > 0, length(micro_dirs) > 0
  @assert length(micro_dirs) <= 1 "currently only a single micro result is supported"

  K_mfl = length(meanfield_dirs)

  if !(has_mfl || has_d)
    @error "No dir provided"
    return
  end


  half_connection_matrix = get(kwargs, :half_connection_matrix, false)
  center_histogram = get(kwargs, :center_histogram, false)

  if center_histogram
    if K_mfl > 1
      throw("Can center histogram with only one meanfield solution, you provided $K_mfl")
    end
    @warn "Histogram for the micro solution will be centered on ω_∞ given by the initial meanfield data!"
  end

  obs_i = M.Observable(1)
  obs_iter = M.Observable(0)

  if has_mfl
    labels = map(dn -> endswith("/", dn) ? basename(dirname(dn)) : basename(dn), meanfield_dirs)
    i_mfl_a = map(dn -> load_hdf5_data(joinpath(dn, "data.hdf5"), "i"), meanfield_dirs)
    f_a = map(dn -> load_hdf5_data(joinpath(dn, "data.hdf5"), "f"), meanfield_dirs)
    params_mfl_a = map(load_metadata, meanfield_dirs)

    α_init_a = map(dn -> load_hdf5_data(joinpath(dn, "data.hdf5"), "alpha"), meanfield_dirs)
    has_alpha = map(α -> !isnothing(α), α_init_a)

    N_micro_mfl_a = map(dn -> (
        meta = TOML.parsefile(joinpath(dn, "metadata.toml"));
        return meta["N_micro"]
      ), meanfield_dirs)

    N_a = map(f -> size(f, 1), f_a)

    obs_g_k = nothing
    function get_g_iter(dn, iter)
      try
        return load_hdf5_data(joinpath(dn, "data.hdf5"), "g/$iter")
      catch
        return nothing
      end
    end

    obs_g_k = map(dn -> (M.@lift get_g_iter(dn, $obs_iter)), meanfield_dirs)

    constant_g = isnothing(obs_g_k[1][])

    x_a = map(build_x, N_a)
  end

  if has_d
    micro_dir = micro_dirs[1]
    ω = load_hdf5_data(joinpath(micro_dir, "data.hdf5"), "omega")
    i_d = load_hdf5_data(joinpath(micro_dir, "data.hdf5"), "i")
    adj_matrix = load_hdf5_sparse(joinpath(micro_dir, "data.hdf5"), "adj_matrix")
    params_d = Params.from_toml(micro_dir)

    N_micro = size(ω, 1)
    if !isnothing(adj_matrix)
      graph = Graphs.SimpleGraphs.SimpleGraph(adj_matrix)

      @assert SpA.is_hermsym(adj_matrix, identity)

      adj_matrix_nnz = SpA.nnz(adj_matrix)

      sharp_I = vec(sum(adj_matrix; dims=2))
      ω_inf_d = sum(ω[:, 1] .* sharp_I) ./ sum(sharp_I)
      xs = Vector{Float64}(undef, adj_matrix_nnz)
      ys = Vector{Float64}(undef, adj_matrix_nnz)
      ones_vector_d = ones(adj_matrix_nnz)

      obs_xs = M.Observable(view(xs, 1:adj_matrix_nnz))
      obs_ys = M.Observable(view(ys, 1:adj_matrix_nnz))
      obs_ones_vector_d = M.Observable(view(ones_vector_d, 1:adj_matrix_nnz))

      function get_xy(opis)
        rows = SpA.rowvals(adj_matrix)
        got = 0
        for j in 1:N_micro
          nzr = SpA.nzrange(adj_matrix, j)
          nnz_in_col = length(nzr)

          if half_connection_matrix
            opis_x = view(opis, rows[nzr])

            idc = opis_x .>= opis[j]
            nnz_in_col = length(filter(isone, idc))

            xs[got+1:got+nnz_in_col] .= opis_x[idc]
            ys[got+1:got+nnz_in_col] .= opis[j]

            got += nnz_in_col
          else
            xs[got+1:got+nnz_in_col] .= opis[rows[nzr]]
            ys[got+1:got+nnz_in_col] .= opis[j]

            got += nnz_in_col
          end
        end

        if half_connection_matrix
          obs_xs.val = view(xs, 1:got)
          obs_ys[] = view(ys, 1:got)
          obs_ones_vector_d[] = view(ones_vector_d, 1:got)
        end

      end

      get_xy(ω[:, 1])
    else
      ω_inf_d = sum(ω[:, 1]) / N_micro
    end
  end


  set_makie_backend(:gl)

  warning = center_histogram ? " !!! The ω_i have been centered to ω_∞ (from MF initial data)" : ""
  N_mfl = isempty(N_a) ? 300 : N_a[1]

  fig = M.Figure(size=(1920, 1080))
  ax1 = M.Axis(fig[1:4, 1])
  ax1.title = "f / ω_i"

  if has_mfl
    ax2 = M.Axis(fig[1:4, 2], aspect=1)
    ax2.title = "g(ω,m)"
    ax2.xlabel = warning

    graph_source = if params_d.init_method_adj_matrix == :from_file
      "(from file)"
    elseif params_d.init_method_adj_matrix == :from_sampling_α_init
      "(from α_init, connection_density=$(params_d.connection_density))"
    elseif params_d.init_method_adj_matrix == :from_graph
      "($(params_d.init_micro_graph_type)($(params_d.init_micro_graph_args); $(params_d.init_micro_graph_kwargs)))"
    else
      ""
    end


    ax3 = M.Axis(fig[1:4, 3], aspect=1)
    ax3.title = "graph $graph_source"
    M.hidedecorations!(ax3)

    g_bottom_left = fig[5, 1:1] = M.GridLayout()
    g_bottom_mid = fig[5, 2:2] = M.GridLayout()
    g_bottom_right = fig[5, 3:3] = M.GridLayout()
  else
    g_bottom_left = fig[5, 1:1] = M.GridLayout()
    g_bottom_right = fig[5, 2:2] = M.GridLayout()
  end

  if has_mfl
    obs_f_a = [M.@lift f_a[k][:, $obs_i] for k in 1:K_mfl]
    # DEFINITION G
    # obs_fαf_a = [M.@lift f_a[k][:, $obs_i] .* α_init_a[k] .* f_a[k][:, $obs_i]' / params_mfl_a[k]["connection_density"] for k in 1:K_mfl]

    function find_support(f_a)
      left_idc = map(f -> (idx = findfirst(x -> x > 1e-5, f[]); return (isnothing(idx) ? 1 : idx)), f_a)
      right_idc = map(f -> (idx = findlast(x -> x > 1e-5, f[]); return (isnothing(idx) ? length(f[]) : idx)), f_a)

      left_x = [x_a[i][left_idc[i]] for i in 1:K_mfl]
      right_x = [x_a[i][right_idc[i]] for i in 1:K_mfl]

      if has_d
        return (left=min(minimum(obs_ω[]), minimum(left_x)), right=max(maximum(obs_ω[]), maximum(right_x)))
      else
        return (left=minimum(left_x), right=maximum(right_x))
      end
    end

    function find_max(f_a)
      return maximum(map(f -> maximum(f[]), f_a))
    end

    if !constant_g
      obs_gff_a = [M.Observable(obs_g_k[k][]) for k in 1:K_mfl]
      max_g_a = maximum(obs_gff_a[1][])
    else
      max_g_a = 0.0
    end

    obs_max_g_a = M.Observable((0.0, 1.05 * max_g_a))

    for k in 1:K_mfl
      M.lines!(ax1, x_a[k], obs_f_a[k], label=labels[k])

      if !constant_g
        M.heatmap!(ax2, x_a[k], x_a[k], obs_gff_a[k], colorrange=obs_max_g_a, colormap=:ice)
        if k == 1
        end
      end
    end

  end

  if has_d
    if center_histogram
      obs_ω = M.@lift (ω[:, $obs_i] .+ ω_inf_mfl_a[1] .- ω_inf_d)
    else
      obs_ω = M.@lift ω[:, $obs_i]
    end
    obs_extrema_ω = M.@lift extrema($obs_ω)

    ext_ω = M.@lift [-1; 1; $obs_ω...]

    cmap(x) = x < 0 ? M.Makie.RGB{Float64}(1.0, 1 + x, 1 + x) : M.Makie.RGB{Float64}(1 - x, 1 - x, 1.0)

    node_colors = M.@lift cmap.($obs_ω)

    M.hist!(ax1, ext_ω; bins=2 * N_mfl, normalization=:pdf)
    M.vlines!(ax1, ω_inf_d, color=:grey, ls=0.5)

    if !isnothing(adj_matrix)

      # The size (width) of the hexagons, which we try to scale along the evolution
      cs_obs = M.@lift 1.0 / N_mfl + 10.0 / ($obs_iter + 100)

      # The area of the hexagons A = sqrt(3) / 2 * width^2
      # https://en.wikipedia.org/wiki/Hexagon#Parameters (d = width in the article)
      hex_area = M.@lift 0.5 * sqrt(3) * ($cs_obs)^2

      # Scale the number of observations per hexagon to match g
      # If the distribution is homogeneous, we have
      # avg_obs = adj_matrix_nnz * A / |Ω²| particles per cell (hexagon)
      # This should correspond to the average value of g, ie ∫∫g/|Ω²|
      # So the correct mapping from number of observation per cell to value is
      # i -> i / avg_obs * ∫∫g/|Ω²| = i/(adj_matrix_nnz * A)
      obs_particle_weight = M.@lift 1 / ($hex_area * adj_matrix_nnz)
      # obs_particle_weight_rounded = M.@lift round($obs_particle_weight; digits=3)
      weights = M.@lift $obs_particle_weight * $obs_ones_vector_d

      hb = M.hexbin!(ax2, obs_xs, obs_ys,
        cellsize=cs_obs, strokewidth=0.5, strokecolor=:gray75, threshold=1,
        #colormap=:ice,
        colormap=[M.to_color(:transparent); M.to_colormap(:ice)],
        weights=weights, colorrange=obs_max_g_a)
      M.Colorbar(g_bottom_mid[1, 1], hb; vertical=false)

      GraphMakie.graphplot!(ax3, graph, node_color=node_colors)

      M.limits!(ax2, (-1, 1), (-1, 1))
    end
  end


  stride = get(kwargs, :stride, 1)
  first_idx = get(kwargs, :first_idx, 1)
  last_idx = get(kwargs, :last_idx, lastindex(i_mfl_a[1]))
  i_range = enumerate([first_idx:stride:last_idx; last_idx])

  function step_i(ii_i_tuple)

    ii, i = ii_i_tuple

    iter = i_mfl_a[1][i]

    pct = lpad(Int(round(100 * ii / length(i_range))), 3, " ")
    print("  Creating movie: $pct%" * "\b"^50 * ", current iteration: $iter")

    if has_mfl && any(f -> any(isnan, f[:, i]), f_a)
      return
    end

    obs_i[] = i
    obs_iter[] = iter
    if has_mfl
      first_mass = 2 / N_a[1] * sum(f_a[1][:, i])
      ax1.title = "$iter, M[1] = $(round(first_mass; digits=6))"
    else
      ax1.title = string(iter)
    end

    if has_d && !isnothing(adj_matrix)
      get_xy(obs_ω[])
    end

    if !constant_g
      for k in 1:K_mfl
        obs_gff_a[k][] = obs_g_k[k][]
      end
      max_g_a = maximum(obs_gff_a[1][])
    else
      max_g_a = 0.0
    end

    #max_g_a = max(max_g_a, maximum(obs_fαf_a[1][]))
    obs_max_g_a[] = (0.0, 1.05 * max_g_a)

    int_g = sum(obs_g_k[1][]) * (2 / N_mfl)^2
    # int_fαf = sum(obs_fαf_a[1][]) * (2 / N_mfl)^2
    ax2.title = "g(ω,m), ∫∫g = $(round(int_g; digits=3))"
    # ax3.title = "fαf(ω,m), ∫∫fαf = $(round(int_fαf; digits=3))"

    if has_mfl
      support = find_support(obs_f_a)
      max_f = find_max(obs_f_a)
      M.ylims!(ax1, low=-1, high=1.3 * max_f)
      M.xlims!(ax1, low=support.left, high=support.right)

      M.xlims!(ax2, low=support.left, high=support.right)
      M.ylims!(ax2, low=support.left, high=support.right)

      #M.xlims!(ax3, low=support.left, high=support.right)
      #M.ylims!(ax3, low=support.left, high=support.right)

    else
      M.autolimits!(ax1)
    end

  end

  effective_output_filename = if output_filename != ""
    output_filename
  else
    anydir = vcat(meanfield_dirs, micro_dirs)[1]
    prefix = if basename(anydir) == 0
      dirname(dirname(anydir))
    else
      dirname(anydir)
    end
    "$prefix/movie.mp4"
  end

  M.record(step_i, fig, effective_output_filename, i_range)
  @info ("movie saved at $effective_output_filename")

  println()

end

function get_ω_inf_mfl(dir::String)
  p = joinpath(dir, "metadata.toml")
  r = TOML.parsefile(p)["omega_inf_mfl"]
  return r
end

function compare_variance(
  meanfield_dir::String,
  micro_dir::String,
)
  return compare_variance([meanfield_dir], [micro_dir])
end

function compare_variance(
  meanfield_dirs::Vector{String},
  micro_dirs::Vector{String},
)

  # check that the dirs actually exist
  @assert all(isdir, meanfield_dirs)
  @assert all(isdir, micro_dirs)

  K_mfl, K_d = length(meanfield_dirs), length(micro_dirs)

  if K_mfl == 0 && K_d == 0
    @error "No dir provided"
    return
  end

  if K_mfl > 0
    labels_mfl = map(dn -> endswith("/", dn) ? basename(dirname(dn)) : basename(dn), meanfield_dirs)
    i_mfl_a = map(dn -> load_hdf5_data(joinpath(dn, "data.hdf5"), "i"), meanfield_dirs)
    f_a = map(dn -> load_hdf5_data(joinpath(dn, "data.hdf5"), "f"), meanfield_dirs)
    params_mfl_a = map(load_metadata, meanfield_dirs)

    N_a = map(f -> size(f, 1), f_a)

    x_a = map(build_x, N_a)

    # ω_inf_mfl_init_a = map(dn -> get_ω_inf_mfl(dn), meanfield_dirs)

    #support_bounds_mfl_a = find_support_bounds(f_a, x_a)
    #support_width_mfl_a = [map(x -> x[2] - x[1], s) for s in support_bounds_mfl_a]

    g_M1_a = map(dn -> load_hdf5_data(joinpath(dn, "data.hdf5"), "g_M1_n"), meanfield_dirs)
    f_var_a = map(dn -> load_hdf5_data(joinpath(dn, "data.hdf5"), "f_var"), meanfield_dirs)

  end

  if K_d > 0
    labels_d = map(dn -> endswith("/", dn) ? basename(dirname(dn)) : basename(dn), micro_dirs)
    i_d_a = map(dn -> load_hdf5_data(joinpath(dn, "data.hdf5"), "i"), micro_dirs)
    ω_a = map(dn -> load_hdf5_data(joinpath(dn, "data.hdf5"), "omega"), micro_dirs)
    params_d_a = map(load_metadata, micro_dirs)

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

    function compute_variance(ω, centers)
      N = size(ω, 1)
      return vec(sum((ω .- centers') .^ 2; dims=1)) / N
    end

    ω_inf_d_a = [compute_weighted_avg(k) for k in 1:K_d]
    p2p_d_a = [peak2peak(ω; dims=1) for ω in ω_a]
    extrema_d_a = [extrema(ω; dims=1) for ω in ω_a]
    rates_d_a = [compute_rate(i_d_a[k], p2p_d_a[k], params_d_a[k]["delta_t"]; cutoff_time=0.25 * params_d_a[k]["delta_t"] * i_d_a[k][end]) for k in 1:K_d]
    variances_d_a = [compute_variance(ω_a[k], ω_inf_d_a[k]) for k in 1:K_d]

  end

  set_makie_backend(:gl)

  fig = M.Figure(size=(1920, 1080))
  ax1 = M.Axis(fig[1, 1], yscale=log10, xlabel="time", title="Variances")
  ax2 = M.Axis(fig[1, 2], xlabel="time", title=M.L"\int\int \omega g / \int\int g \pm \sqrt{\text{variance}} \qquad \sum_i \omega_i # I_i / \sum_i # I_i \pm \sqrt{\text{variance}}")

  for k in 1:K_d
    r = i_d_a[k] * params_d_a[k]["delta_t"]
    # p2p = p2p_d_a[k]
    M.lines!(ax1, r, variances_d_a[k], label=labels_d[k] * " rate: $(round(rates_d_a[k]; digits=3))")
    #M.lines!(ax1, r, p2p, label=labels_d[k] * " rate: $(round(rates_d_a[k]; digits=3))")

    M.lines!(ax2, r, ω_inf_d_a[k], label=labels_d[k])
    # left_bounds = vec(map(v -> v[1], extrema_d_a[k]))
    # right_bounds = vec(map(v -> v[2], extrema_d_a[k]))
    M.band!(ax2, r, ω_inf_d_a[k] .- sqrt.(variances_d_a[k]), ω_inf_d_a[k] .+ sqrt.(variances_d_a[k]), alpha=0.2)
  end

  for k in 1:K_mfl
    r = i_mfl_a[k] * params_mfl_a[k]["delta_t"]
    M.lines!(ax1, r, f_var_a[k], label=labels_mfl[k], linestyle=:dot)

    #M.lines!(ax1, r, p2p, label=labels_mfl[k] * " rate: $(round(rates_mfl_a[k]; digits=3))")
    #M.hlines!(ax2, params_mfl_a[k]["omega_inf_mfl"])

    M.lines!(ax2, r, g_M1_a[k])
    M.band!(ax2, r, g_M1_a[k] - sqrt.(f_var_a[k]), g_M1_a[k] + sqrt.(f_var_a[k]), alpha=0.2)
  end

  M.axislegend(ax1)

  dirs = vcat(meanfield_dirs, micro_dirs)
  prefix = longest_prefix(dirs, existing_dir=true)
  M.save("$prefix/comparison.png", fig)
  M.save("$prefix/comparison.svg", fig)
  @info "Plot saved at $prefix/comparison.png"

  try
    display(fig)
  catch
    @error "Cannot display plot window, are you logged in over SSH?"
  end

end

function compare_variance_er(
  meanfield_dirs::Vector{String},
  micro_dirs::Vector{String},
)

  # check that the dirs actually exist
  @assert all(isdir, meanfield_dirs)
  @assert all(isdir, micro_dirs)

  K_mfl, K_d = length(meanfield_dirs), length(micro_dirs)

  if K_mfl == 0 && K_d == 0
    @error "No dir provided"
    return
  end

  if K_mfl > 0
    labels_mfl = map(dn -> endswith("/", dn) ? basename(dirname(dn)) : basename(dn), meanfield_dirs)
    i_mfl_a = map(dn -> load_hdf5_data(joinpath(dn, "data.hdf5"), "i"), meanfield_dirs)
    f_a = map(dn -> load_hdf5_data(joinpath(dn, "data.hdf5"), "f"), meanfield_dirs)
    params_mfl_a = map(Params.from_toml, meanfield_dirs)
    er_mfl_p_a = map(x -> x.init_micro_graph_args[2], params_mfl_a)

    N_a = map(f -> size(f, 1), f_a)

    x_a = map(build_x, N_a)

    g_M1_a = map(dn -> load_hdf5_data(joinpath(dn, "data.hdf5"), "g_M1_n"), meanfield_dirs)
    f_var_a = map(dn -> load_hdf5_data(joinpath(dn, "data.hdf5"), "f_var"), meanfield_dirs)
    rates_mfl_a = [compute_rate(i_mfl_a[k], f_var_a[k], params_mfl_a[k].δt; cutoff_time=(1 / 6) * params_mfl_a[k].δt * i_mfl_a[k][end]) for k in 1:K_mfl]
    # FIXME: be less lazy and get the actual value of N_micro
    mfl_scaling_a = binomial(params_mfl_a[1].N_micro, 2) * er_mfl_p_a / params_mfl_a[1].N_micro^2
    rates_mfl_rescaled_a = mfl_scaling_a .* rates_mfl_a

  end


  if K_d > 0
    labels_d = map(dn -> endswith("/", dn) ? basename(dirname(dn)) : basename(dn), micro_dirs)
    i_d_a = map(dn -> load_hdf5_data(joinpath(dn, "data.hdf5"), "i"), micro_dirs)
    ω_a = map(dn -> load_hdf5_data(joinpath(dn, "data.hdf5"), "omega"), micro_dirs)
    params_d_a = map(Params.from_toml, micro_dirs)
    er_d_p_a = map(x -> x.init_micro_graph_args[2], params_d_a)

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

    function compute_variance(ω, centers)
      N = size(ω, 1)
      return vec(sum((ω .- centers') .^ 2; dims=1)) / N
    end


    ω_inf_d_a = [compute_weighted_avg(k) for k in 1:K_d]
    p2p_d_a = [peak2peak(ω; dims=1) for ω in ω_a]
    extrema_d_a = [extrema(ω; dims=1) for ω in ω_a]
    variances_d_a = [compute_variance(ω_a[k], ω_inf_d_a[k]) for k in 1:K_d]
    rates_d_a = [compute_rate(i_d_a[k], variances_d_a[k], params_d_a[k].δt; cutoff_time=(1 / 6) * params_d_a[k].δt * i_d_a[k][end]) for k in 1:K_d]

  end

  fig = M.Figure(size=(1920, 1080))
  ax1 = M.Axis(fig[1, 1], yscale=log10, xlabel="time", title="Variances")
  ax2 = M.Axis(fig[1, 2], xlabel="p (Erdos-Renyi)", title="Convergence rates", xscale=log10, yscale=identity)

  M.ylims!(ax1, (1e-5, 1))
  M.xlims!(ax1, (nothing, 10.0))

  for k in 1:K_d
    r = i_d_a[k] * params_d_a[k].δt
    # p2p = p2p_d_a[k]
    M.lines!(ax1, r, variances_d_a[k], label=labels_d[k] * " p = $(round(er_d_p_a[k]; digits=3))", color=p_to_color(er_d_p_a[k]; mapping=log10, pmin=1e-3, pmax=1))
    #M.lines!(ax1, r, p2p, label=labels_d[k] * " rate: $(round(rates_d_a[k]; digits=3))")

    M.lines!(ax2, er_d_p_a, rates_d_a)
  end

  for k in 1:K_mfl
    r = i_mfl_a[k] * params_mfl_a[k].δt
    M.lines!(ax1, r, f_var_a[k], label=labels_mfl[k] * " p = $(round(er_mfl_p_a[k]; digits=3))", linestyle=:dot, color=p_to_color(er_mfl_p_a[k]; mapping=log10, pmin=1e-3, pmax=1))
    M.lines!(ax2, er_mfl_p_a, rates_mfl_a, label="MFL")
    # M.lines!(ax2, er_mfl_p_a, rates_mfl_rescaled_a, label="MFL (rescaled)")
  end

  #M.axislegend(ax1)

  dirs = vcat(meanfield_dirs, micro_dirs)
  prefix = longest_prefix(dirs, existing_dir=true)
  M.save("$prefix/comparison2.png", fig)
  M.save("$prefix/comparison2.svg", fig)
  @info "Plot saved at $prefix/comparison2.png"

  try
    display(fig)
  catch
    @error "Cannot display plot window, are you logged in over SSH?"
  end
end

function compare_variance_ws(
  meanfield_dirs::Vector{String},
  micro_dirs::Vector{String},
)

  # check that the dirs actually exist
  @assert all(isdir, meanfield_dirs)
  @assert all(isdir, micro_dirs)

  K_mfl, K_d = length(meanfield_dirs), length(micro_dirs)

  if K_mfl == 0 && K_d == 0
    @error "No dir provided"
    return
  end

  if K_mfl > 0
    labels_mfl = map(dn -> endswith("/", dn) ? basename(dirname(dn)) : basename(dn), meanfield_dirs)
    i_mfl_a = map(dn -> load_hdf5_data(joinpath(dn, "data.hdf5"), "i"), meanfield_dirs)
    f_a = map(dn -> load_hdf5_data(joinpath(dn, "data.hdf5"), "f"), meanfield_dirs)
    params_mfl_a = map(Params.from_toml, meanfield_dirs)
    ws_mfl_k_a = map(x -> round(Int64, 999 * x.init_micro_graph_args[2]), params_mfl_a)

    N_a = map(f -> size(f, 1), f_a)

    x_a = map(build_x, N_a)

    g_M1_a = map(dn -> load_hdf5_data(joinpath(dn, "data.hdf5"), "g_M1_n"), meanfield_dirs)
    f_var_a = map(dn -> load_hdf5_data(joinpath(dn, "data.hdf5"), "f_var"), meanfield_dirs)
    rates_mfl_a = [compute_rate(i_mfl_a[k], f_var_a[k], params_mfl_a[k].δt; cutoff_time=(1 / 6) * params_mfl_a[k].δt * i_mfl_a[k][end]) for k in 1:K_mfl]
    # FIXME: be less lazy and get the actual value of N_micro
    # mfl_scaling_a = binomial(params_mfl_a[1].N_micro, 2) * er_mfl_p_a / params_mfl_a[1].N_micro^2
    # rates_mfl_rescaled_a = mfl_scaling_a .* rates_mfl_a

  end


  if K_d > 0
    labels_d = map(dn -> endswith("/", dn) ? basename(dirname(dn)) : basename(dn), micro_dirs)
    i_d_a = map(dn -> load_hdf5_data(joinpath(dn, "data.hdf5"), "i"), micro_dirs)
    ω_a = map(dn -> load_hdf5_data(joinpath(dn, "data.hdf5"), "omega"), micro_dirs)
    params_d_a = map(Params.from_toml, micro_dirs)
    ws_d_k_a = map(x -> round(Int64, (params_d_a[1].N_micro - 1) * x.init_micro_graph_args[2]), params_d_a)

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

    function compute_variance(ω, centers)
      N = size(ω, 1)
      return vec(sum((ω .- centers') .^ 2; dims=1)) / N
    end


    ω_inf_d_a = [compute_weighted_avg(k) for k in 1:K_d]
    p2p_d_a = [peak2peak(ω; dims=1) for ω in ω_a]
    extrema_d_a = [extrema(ω; dims=1) for ω in ω_a]
    variances_d_a = [compute_variance(ω_a[k], ω_inf_d_a[k]) for k in 1:K_d]
    rates_d_a = [compute_rate(i_d_a[k], variances_d_a[k], params_d_a[k].δt; cutoff_time=(1 / 6) * params_d_a[k].δt * i_d_a[k][end]) for k in 1:K_d]

  end

  fig = M.Figure(size=(1920, 1080))
  ax1 = M.Axis(fig[1, 1], yscale=log10, xlabel="time", title="Variances")
  ax2 = M.Axis(fig[1, 2], xlabel="k (Watts-Strogatz)", title="Convergence rates", xscale=log10, yscale=identity)

  M.ylims!(ax1, (1e-5, 1))
  M.xlims!(ax1, (nothing, 10.0))

  for k in 1:K_d
    r = i_d_a[k] * params_d_a[k].δt
    # p2p = p2p_d_a[k]
    M.lines!(ax1, r, variances_d_a[k], label=labels_d[k] * " k = $ws_d_k_a[k]", color=p_to_color(ws_d_k_a[k]; mapping=log10, pmin=1e-3, pmax=1))
    #M.lines!(ax1, r, p2p, label=labels_d[k] * " rate: $(round(rates_d_a[k]; digits=3))")

    M.lines!(ax2, ws_d_k_a, rates_d_a)
  end

  for k in 1:K_mfl
    r = i_mfl_a[k] * params_mfl_a[k].δt
    M.lines!(ax1, r, f_var_a[k], label=labels_mfl[k] * " k = $ws_mfl_k_a[k]", linestyle=:dot, color=p_to_color(ws_mfl_k_a[k]; mapping=log10, pmin=1e-3, pmax=1))
    M.lines!(ax2, ws_mfl_k_a, rates_mfl_a, label="MFL")
    # M.lines!(ax2, er_mfl_p_a, rates_mfl_rescaled_a, label="MFL (rescaled)")
  end

  #M.axislegend(ax1)

  dirs = vcat(meanfield_dirs, micro_dirs)
  prefix = longest_prefix(dirs, existing_dir=true)
  M.save("$prefix/comparison2.png", fig)
  M.save("$prefix/comparison2.svg", fig)
  @info "Plot saved at $prefix/comparison2.png"

  try
    display(fig)
  catch
    @error "Cannot display plot window, are you logged in over SSH?"
  end

end

function compare_variance_ws_all(
  meanfield_dirs::Vector{String},
  micro_dirs::Vector{String};
  cutoff_factor::Float64=1.0
)

  # check that the dirs actually exist
  @assert all(isdir, meanfield_dirs)
  @assert all(isdir, micro_dirs)

  K_mfl, K_d = length(meanfield_dirs), length(micro_dirs)

  if K_mfl == 0 && K_d == 0
    @error "No dir provided"
    return
  end

  if K_mfl > 0
    labels_mfl = map(dn -> endswith("/", dn) ? basename(dirname(dn)) : basename(dn), meanfield_dirs)
    i_mfl_a = map(dn -> load_hdf5_data(joinpath(dn, "data.hdf5"), "i"), meanfield_dirs)
    f_a = map(dn -> load_hdf5_data(joinpath(dn, "data.hdf5"), "f"), meanfield_dirs)
    params_mfl_a = map(Params.from_toml, meanfield_dirs)
    ws_mfl_k_a = map(x -> round(Int64, 999 * x.init_micro_graph_args[2]), params_mfl_a)

    N_a = map(f -> size(f, 1), f_a)

    x_a = map(build_x, N_a)

    # g_M1_a = map(dn -> load_hdf5_data(joinpath(dn, "data.hdf5"), "g_M1_n"), meanfield_dirs)
    f_var_a = map(dn -> load_hdf5_data(joinpath(dn, "data.hdf5"), "f_var"), meanfield_dirs)
    rates_mfl_a = [compute_rate(i_mfl_a[k], f_var_a[k], params_mfl_a[k].δt; cutoff_time=cutoff_factor * params_mfl_a[k].δt * i_mfl_a[k][end]) for k in 1:K_mfl]

    split_dir_names = split_run_path(meanfield_dirs)
    prev_prefix = split_dir_names[1][1]

    aggregates_mfl = Dict{Symbol,Any}[]
    aggregate = Dict{Symbol,Any}()
    rates_acc = Float64[]
    var_acc = Vector{Float64}[]
    prefix_n = 0

    for (i, p) in enumerate(split_dir_names)
      if (p[1] != prev_prefix && i != 1) || (i == lastindex(split_dir_names) && prefix_n > 0)
        aggregate[:rates] = sum(rates_acc) / prefix_n
        aggregate[:k] = ws_mfl_k_a[i-1]
        aggregate[:time] = i_mfl_a[i-1] * params_mfl_a[i-1].δt
        aggregate[:variance] = sum(var_acc) / prefix_n
        push!(aggregates_mfl, aggregate)
        aggregate = Dict{Symbol,Any}()
        empty!(rates_acc)
        empty!(var_acc)
        prefix_n = 0
        prev_prefix = p[1]
      end
      push!(rates_acc, rates_mfl_a[i])
      push!(var_acc, f_var_a[i])
      prefix_n += 1
    end

  end


  if K_d > 0
    labels_d = map(dn -> endswith("/", dn) ? basename(dirname(dn)) : basename(dn), micro_dirs)
    i_d_a = map(dn -> load_hdf5_data(joinpath(dn, "data.hdf5"), "i"), micro_dirs)
    ω_a = map(dn -> load_hdf5_data(joinpath(dn, "data.hdf5"), "omega"), micro_dirs)
    params_d_a = map(Params.from_toml, micro_dirs)
    ws_d_k_a = map(x -> round(Int64, (params_d_a[1].N_micro - 1) * x.init_micro_graph_args[2]), params_d_a)

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

    function compute_variance(ω, centers)
      N = size(ω, 1)
      return vec(sum((ω .- centers') .^ 2; dims=1)) / N
    end


    ω_inf_d_a = [compute_weighted_avg(k) for k in 1:K_d]
    p2p_d_a = [peak2peak(ω; dims=1) for ω in ω_a]
    extrema_d_a = [extrema(ω; dims=1) for ω in ω_a]
    variances_d_a = [compute_variance(ω_a[k], ω_inf_d_a[k]) for k in 1:K_d]
    rates_d_a = [compute_rate(i_d_a[k], variances_d_a[k], params_d_a[k].δt; cutoff_time=cutoff_factor * params_d_a[k].δt * i_d_a[k][end]) for k in 1:K_d]

    split_dir_names = split_run_path(micro_dirs)
    prev_prefix = split_dir_names[1][1]

    aggregates_d = Dict{Symbol,Any}[]
    aggregate = Dict{Symbol,Any}()
    rates_acc = Float64[]
    var_acc = Vector{Float64}[]
    prefix_n = 0

    for (i, p) in enumerate(split_dir_names)
      if (p[1] != prev_prefix && i != 1) || (i == lastindex(split_dir_names) && prefix_n > 0)
        aggregate[:rates] = sum(rates_acc) / prefix_n
        aggregate[:k] = ws_d_k_a[i-1]
        aggregate[:time] = i_d_a[i-1] * params_d_a[i-1].δt
        aggregate[:variance] = sum(var_acc) / prefix_n
        push!(aggregates_d, aggregate)
        aggregate = Dict{Symbol,Any}()
        empty!(rates_acc)
        empty!(var_acc)
        prefix_n = 0
        prev_prefix = p[1]
      end
      push!(rates_acc, rates_d_a[i])
      push!(var_acc, variances_d_a[i])
      prefix_n += 1
    end

  end

  mfl_k = map(x -> x[:k], aggregates_mfl)
  mfl_rates = map(x -> x[:rates], aggregates_mfl)

  d_k = map(x -> x[:k], aggregates_d)
  d_rates = map(x -> x[:rates], aggregates_d)

  fig = M.Figure(size=(1920, 1080))
  # ax1 = M.Axis(fig[1, 1], yscale=log10, xlabel="time", title="Variances")
  ax2 = M.Axis(fig[1, 1], xlabel="k (Watts-Strogatz)", title="Convergence rates", xscale=log10, yscale=identity)

  M.lines!(ax2, mfl_k, mfl_rates, label="MFL")
  M.lines!(ax2, d_k, d_rates, label="Micro")

  #M.axislegend(ax1)

  dirs = vcat(meanfield_dirs, micro_dirs)
  prefix = longest_prefix(dirs, existing_dir=true)
  M.save("$prefix/comparison2.png", fig)
  M.save("$prefix/comparison2.svg", fig)
  @info "Plot saved at $prefix/comparison2.png"

  try
    display(fig)
  catch
    @error "Cannot display plot window, are you logged in over SSH?"
  end

end

function compare_variance_ba_all(
  meanfield_dirs::Vector{String},
  micro_dirs::Vector{String};
  cutoff_factor::Float64=1.0
)

  # check that the dirs actually exist
  @assert all(isdir, meanfield_dirs)
  @assert all(isdir, micro_dirs)

  K_mfl, K_d = length(meanfield_dirs), length(micro_dirs)

  if K_mfl == 0 && K_d == 0
    @error "No dir provided"
    return
  end

  if K_mfl > 0
    labels_mfl = map(dn -> endswith("/", dn) ? basename(dirname(dn)) : basename(dn), meanfield_dirs)
    i_mfl_a = map(dn -> load_hdf5_data(joinpath(dn, "data.hdf5"), "i"), meanfield_dirs)
    f_a = map(dn -> load_hdf5_data(joinpath(dn, "data.hdf5"), "f"), meanfield_dirs)
    params_mfl_a = map(Params.from_toml, meanfield_dirs)
    ba_mfl_k_a = map(x -> x.init_micro_graph_args[2], params_mfl_a)

    N_a = map(f -> size(f, 1), f_a)

    f_var_a = map(dn -> load_hdf5_data(joinpath(dn, "data.hdf5"), "f_var"), meanfield_dirs)
    rates_mfl_a = [compute_rate(i_mfl_a[k], f_var_a[k], params_mfl_a[k].δt; cutoff_time=cutoff_factor * params_mfl_a[k].δt * i_mfl_a[k][end]) for k in 1:K_mfl]

    split_dir_names = split_run_path(meanfield_dirs)
    prev_prefix = split_dir_names[1][1]

    aggregates_mfl = Dict{Symbol,Any}[]
    aggregate = Dict{Symbol,Any}()
    rates_acc = Float64[]
    var_acc = Vector{Float64}[]
    prefix_n = 0

    for (i, p) in enumerate(split_dir_names)
      if (p[1] != prev_prefix && i != 1) || (i == lastindex(split_dir_names) && prefix_n > 0)
        aggregate[:rates] = sum(rates_acc) / prefix_n
        aggregate[:k] = ba_mfl_k_a[i-1]
        aggregate[:time] = i_mfl_a[i-1] * params_mfl_a[i-1].δt
        aggregate[:variance] = sum(var_acc) / prefix_n
        push!(aggregates_mfl, aggregate)
        aggregate = Dict{Symbol,Any}()
        empty!(rates_acc)
        empty!(var_acc)
        prefix_n = 0
        prev_prefix = p[1]
      end
      push!(rates_acc, rates_mfl_a[i])
      push!(var_acc, f_var_a[i])
      prefix_n += 1
    end

  end


  if K_d > 0
    labels_d = map(dn -> endswith("/", dn) ? basename(dirname(dn)) : basename(dn), micro_dirs)
    i_d_a = map(dn -> load_hdf5_data(joinpath(dn, "data.hdf5"), "i"), micro_dirs)
    ω_a = map(dn -> load_hdf5_data(joinpath(dn, "data.hdf5"), "omega"), micro_dirs)
    params_d_a = map(Params.from_toml, micro_dirs)
    ba_d_k_a = map(x -> x.init_micro_graph_args[2], params_d_a)

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

    function compute_variance(ω, centers)
      N = size(ω, 1)
      return vec(sum((ω .- centers') .^ 2; dims=1)) / N
    end


    ω_inf_d_a = [compute_weighted_avg(k) for k in 1:K_d]
    p2p_d_a = [peak2peak(ω; dims=1) for ω in ω_a]
    extrema_d_a = [extrema(ω; dims=1) for ω in ω_a]
    variances_d_a = [compute_variance(ω_a[k], ω_inf_d_a[k]) for k in 1:K_d]
    rates_d_a = [compute_rate(i_d_a[k], variances_d_a[k], params_d_a[k].δt; cutoff_time=cutoff_factor * params_d_a[k].δt * i_d_a[k][end]) for k in 1:K_d]

    split_dir_names = split_run_path(micro_dirs)
    prev_prefix = split_dir_names[1][1]

    aggregates_d = Dict{Symbol,Any}[]
    aggregate = Dict{Symbol,Any}()
    rates_acc = Float64[]
    var_acc = Vector{Float64}[]
    prefix_n = 0

    for (i, p) in enumerate(split_dir_names)
      if (p[1] != prev_prefix) || (i == lastindex(split_dir_names) && prefix_n > 0)
        aggregate[:rates] = sum(rates_acc) / prefix_n
        aggregate[:k] = ba_d_k_a[i-1]
        aggregate[:time] = i_d_a[i-1] * params_d_a[i-1].δt
        aggregate[:variance] = sum(var_acc) / prefix_n
        push!(aggregates_d, aggregate)
        aggregate = Dict{Symbol,Any}()
        empty!(rates_acc)
        empty!(var_acc)
        prefix_n = 0
        prev_prefix = p[1]
      end
      push!(rates_acc, rates_d_a[i])
      push!(var_acc, variances_d_a[i])
      prefix_n += 1
    end

  end

  mfl_k = map(x -> x[:k], aggregates_mfl)
  mfl_rates = map(x -> x[:rates], aggregates_mfl)

  d_k = map(x -> x[:k], aggregates_d)
  d_rates = map(x -> x[:rates], aggregates_d)

  fig = M.Figure(size=(1920, 1080))
  ax1 = M.Axis(fig[1, 1], yscale=log10, xlabel="time", title="Variances")
  ax2 = M.Axis(fig[1, 2], xlabel="k (Barabasi-Albert)", title="Convergence rates", xscale=log10, yscale=identity)

  M.vspan!(ax1, 0.0, cutoff_factor * params_mfl_a[1].δt * i_mfl_a[1][end]; color=(:blue, 0.1))

  for (i, a_d) in enumerate(aggregates_d)
    label = i == 1 ? "Micro" : nothing
    M.lines!(ax1, a_d[:time], a_d[:variance], label=label, linestyle=:dot,
      color=p_to_color(a_d[:k]; mapping=log10, pmin=minimum(d_k), pmax=maximum(d_k)))
  end

  for (i, a_mfl) in enumerate(aggregates_mfl)
    label = i == 1 ? "MFL" : nothing
    M.lines!(ax1, a_mfl[:time], a_mfl[:variance], label=label,
      color=p_to_color(a_mfl[:k]; mapping=log10, pmin=minimum(mfl_k), pmax=maximum(mfl_k)))
  end

  M.lines!(ax2, d_k, d_rates, label="Micro", linestyle=:dot, color=:blue)
  M.lines!(ax2, mfl_k, mfl_rates, label="MFL", color=:blue)

  M.axislegend(ax1)
  M.axislegend(ax2)

  dirs = vcat(meanfield_dirs, micro_dirs)
  prefix = longest_prefix(dirs, existing_dir=true)
  M.save("$prefix/comparison2.png", fig)
  M.save("$prefix/comparison2.svg", fig)
  @info "Plot saved at $prefix/comparison2.png"

  try
    display(fig)
  catch
    @error "Cannot display plot window, are you logged in over SSH?"
  end

end

function compare_variance_lfr_all(
  meanfield_dirs::Vector{String},
  micro_dirs::Vector{String};
  cutoff_factor::Float64=1.0
)

  # check that the dirs actually exist
  @assert all(isdir, meanfield_dirs)
  @assert all(isdir, micro_dirs)

  K_mfl, K_d = length(meanfield_dirs), length(micro_dirs)

  if K_mfl == 0 && K_d == 0
    @error "No dir provided"
    return
  end

  if K_mfl > 0
    labels_mfl = map(dn -> endswith("/", dn) ? basename(dirname(dn)) : basename(dn), meanfield_dirs)
    i_mfl_a = map(dn -> load_hdf5_data(joinpath(dn, "data.hdf5"), "i"), meanfield_dirs)
    f_a = map(dn -> load_hdf5_data(joinpath(dn, "data.hdf5"), "f"), meanfield_dirs)
    params_mfl_a = map(Params.from_toml, meanfield_dirs)
    lfr_mfl_μ_a = map(x -> x.init_lfr_kwargs.mixing_parameter, params_mfl_a)

    N_a = map(f -> size(f, 1), f_a)

    f_var_a = map(dn -> load_hdf5_data(joinpath(dn, "data.hdf5"), "f_var"), meanfield_dirs)
    rates_mfl_a = [compute_rate(i_mfl_a[k], f_var_a[k], params_mfl_a[k].δt; cutoff_time=cutoff_factor * params_mfl_a[k].δt * i_mfl_a[k][end]) for k in 1:K_mfl]

    split_dir_names = split_run_path(meanfield_dirs)
    prev_prefix = split_dir_names[1][1]

    aggregates_mfl = Dict{Symbol,Any}[]
    aggregate = Dict{Symbol,Any}()
    rates_acc = Float64[]
    var_acc = Vector{Float64}[]
    prefix_n = 0

    for (i, p) in enumerate(split_dir_names)
      if (p[1] != prev_prefix && i != 1) || (i == lastindex(split_dir_names) && prefix_n > 0)
        aggregate[:rates] = sum(rates_acc) / prefix_n
        aggregate[:μ] = lfr_mfl_μ_a[i-1]
        aggregate[:time] = i_mfl_a[i-1] * params_mfl_a[i-1].δt
        aggregate[:variance] = sum(var_acc) / prefix_n
        push!(aggregates_mfl, aggregate)
        aggregate = Dict{Symbol,Any}()
        empty!(rates_acc)
        empty!(var_acc)
        prefix_n = 0
        prev_prefix = p[1]
      end
      push!(rates_acc, rates_mfl_a[i])
      push!(var_acc, f_var_a[i])
      prefix_n += 1
    end

  end


  if K_d > 0
    labels_d = map(dn -> endswith("/", dn) ? basename(dirname(dn)) : basename(dn), micro_dirs)
    i_d_a = map(dn -> load_hdf5_data(joinpath(dn, "data.hdf5"), "i"), micro_dirs)
    ω_a = map(dn -> load_hdf5_data(joinpath(dn, "data.hdf5"), "omega"), micro_dirs)
    params_d_a = map(Params.from_toml, micro_dirs)
    lfr_d_μ_a = map(x -> x.init_lfr_kwargs.mixing_parameter, params_d_a)

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

    function compute_variance(ω, centers)
      N = size(ω, 1)
      return vec(sum((ω .- centers') .^ 2; dims=1)) / N
    end

    ω_inf_d_a = [compute_weighted_avg(k) for k in 1:K_d]
    p2p_d_a = [peak2peak(ω; dims=1) for ω in ω_a]
    extrema_d_a = [extrema(ω; dims=1) for ω in ω_a]
    variances_d_a = [compute_variance(ω_a[k], ω_inf_d_a[k]) for k in 1:K_d]
    rates_d_a = [compute_rate(i_d_a[k], variances_d_a[k], params_d_a[k].δt; cutoff_time=cutoff_factor * params_d_a[k].δt * i_d_a[k][end]) for k in 1:K_d]

    split_dir_names = split_run_path(micro_dirs)
    prev_prefix = split_dir_names[1][1]

    aggregates_d = Dict{Symbol,Any}[]
    aggregate = Dict{Symbol,Any}()
    rates_acc = Float64[]
    var_acc = Vector{Float64}[]
    prefix_n = 0

    for (i, p) in enumerate(split_dir_names)
      if (p[1] != prev_prefix && i != 1) || (i == lastindex(split_dir_names) && prefix_n > 0)
        aggregate[:rates] = sum(rates_acc) / prefix_n
        aggregate[:μ] = lfr_d_μ_a[i-1]
        aggregate[:time] = i_d_a[i-1] * params_d_a[i-1].δt
        aggregate[:variance] = sum(var_acc) / prefix_n
        push!(aggregates_d, aggregate)
        aggregate = Dict{Symbol,Any}()
        empty!(rates_acc)
        empty!(var_acc)
        prefix_n = 0
        prev_prefix = p[1]
      end
      push!(rates_acc, rates_d_a[i])
      push!(var_acc, variances_d_a[i])
      prefix_n += 1
    end

  end

  mfl_µ = map(x -> x[:µ], aggregates_mfl)
  mfl_rates = map(x -> x[:rates], aggregates_mfl)

  d_µ = map(x -> x[:µ], aggregates_d)
  d_rates = map(x -> x[:rates], aggregates_d)

  fig = M.Figure(size=(1920, 1080))
  ax1 = M.Axis(fig[1, 1], yscale=log10, xlabel="time", title="Variances")
  ax2 = M.Axis(fig[1, 2], xlabel="μ (LFR)", title="Convergence rates", xscale=log10, yscale=identity)

  M.vspan!(ax1, 0.0, cutoff_factor * params_mfl_a[1].δt * i_mfl_a[1][end]; color=(:blue, 0.1))

  for (i, a_d) in enumerate(aggregates_d)
    label = i == 1 ? "Micro" : nothing
    M.lines!(ax1, a_d[:time], a_d[:variance], label=label, linestyle=:dot,
      color=p_to_color(a_d[:µ]; mapping=log10, pmin=minimum(d_µ), pmax=maximum(d_µ)))
  end

  for (i, a_mfl) in enumerate(aggregates_mfl)
    label = i == 1 ? "MFL" : nothing
    M.lines!(ax1, a_mfl[:time], a_mfl[:variance], label=label,
      color=p_to_color(a_mfl[:µ]; mapping=log10, pmin=minimum(mfl_µ), pmax=maximum(mfl_µ)))
  end

  M.lines!(ax2, d_µ, d_rates, label="Micro", linestyle=:dot, color=:blue)
  M.lines!(ax2, mfl_µ, mfl_rates, label="MFL", color=:blue)

  M.axislegend(ax1)
  M.axislegend(ax2)

  dirs = vcat(meanfield_dirs, micro_dirs)
  prefix = longest_prefix(dirs, existing_dir=true)
  M.save("$prefix/comparison2.png", fig)
  M.save("$prefix/comparison2.svg", fig)
  @info "Plot saved at $prefix/comparison2.svg"

  try
    display(fig)
  catch
    @error "Cannot display plot window, are you logged in over SSH?"
  end

end

function compare_variance_ensemble_average(
  meanfield_dirs::Vector{String},
  micro_dirs::Vector{String},
  parameter_name::String,
  parameter_extractor::Function
  ;
  cutoff_factor::Float64=1.0,
)

  # check that the dirs actually exist
  @assert all(isdir, meanfield_dirs)
  @assert all(isdir, micro_dirs)

  K_mfl, K_d = length(meanfield_dirs), length(micro_dirs)

  if K_mfl == 0 && K_d == 0
    @error "No dir provided"
    return
  end

  if K_mfl > 0
    labels_mfl = map(dn -> endswith("/", dn) ? basename(dirname(dn)) : basename(dn), meanfield_dirs)
    i_mfl_a = map(dn -> load_hdf5_data(joinpath(dn, "data.hdf5"), "i"), meanfield_dirs)
    f_a = map(dn -> load_hdf5_data(joinpath(dn, "data.hdf5"), "f"), meanfield_dirs)
    params_mfl_a = map(Params.from_toml, meanfield_dirs)
    indep_param_mfl_a = map(parameter_extractor, params_mfl_a)

    N_a = map(f -> size(f, 1), f_a)

    f_stddev_a = map(dn -> sqrt.(load_hdf5_data(joinpath(dn, "data.hdf5"), "f_var")), meanfield_dirs)
    rates_mfl_a = [compute_rate(i_mfl_a[k], f_stddev_a[k], params_mfl_a[k].δt; cutoff_time=cutoff_factor * params_mfl_a[k].δt * i_mfl_a[k][end]) for k in 1:K_mfl]

    split_dir_names = split_run_path(meanfield_dirs)
    prev_prefix = split_dir_names[1][1]

    aggregates_mfl = Dict{Symbol,Any}[]
    aggregate = Dict{Symbol,Any}()
    rates_acc = Float64[]
    stddev_acc = Vector{Float64}[]
    prefix_n = 0

    for (i, p) in enumerate(split_dir_names)
      if (p[1] != prev_prefix && i != 1) || (i == lastindex(split_dir_names) && prefix_n > 0)
        aggregate[:rates] = sum(rates_acc) / prefix_n
        aggregate[:param] = indep_param_mfl_a[i-1]
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

  fig = M.Figure(size=(1920, 1080))
  ax1 = M.Axis(fig[1, 1], yscale=log10, xlabel="time", title="Standard deviations")
  ax2 = M.Axis(fig[1, 2], xlabel="$(parameter_name)", title="Convergence rates", xscale=log10, yscale=M.Makie.pseudolog10)

  M.vspan!(ax1, 0.0, cutoff_factor * params_mfl_a[1].δt * i_mfl_a[1][end]; color=(:blue, 0.1))

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

  M.axislegend(ax1)
  M.axislegend(ax2)

  dirs = vcat(meanfield_dirs, micro_dirs)
  prefix = longest_prefix(dirs, existing_dir=true)
  M.save("$prefix/comparison.png", fig)
  M.save("$prefix/comparison.svg", fig)
  @info "Plot saved at $prefix/comparison.svg"

  try
    display(fig)
  catch
    @error "Cannot display plot window, are you logged in over SSH?"
  end

end

function compare_variance_er_EA(
  meanfield_dirs::Vector{String},
  micro_dirs::Vector{String};
  cutoff_factor::Float64=1.0
)
  compare_variance_ensemble_average(
    meanfield_dirs,
    micro_dirs,
    "p (Erdos-Renyi)",
    x -> x.init_micro_graph_args[2];
    cutoff_factor=cutoff_factor
  )
end

function compare_variance_ws_EA(
  meanfield_dirs::Vector{String},
  micro_dirs::Vector{String};
  cutoff_factor::Float64=1.0
)
  compare_variance_ensemble_average(
    meanfield_dirs,
    micro_dirs,
    "k (Watts-Strogatz)",
    x -> round(Int64, 999 * x.init_micro_graph_args[2]);
    cutoff_factor=cutoff_factor
  )
end

function compare_variance_ba_EA(
  meanfield_dirs::Vector{String},
  micro_dirs::Vector{String};
  cutoff_factor::Float64=1.0
)
  compare_variance_ensemble_average(
    meanfield_dirs,
    micro_dirs,
    "k (Barabasi-Albert)",
    x -> x.init_micro_graph_args[2];
    cutoff_factor=cutoff_factor
  )
end

function compare_variance_lfr_EA(
  meanfield_dirs::Vector{String},
  micro_dirs::Vector{String};
  cutoff_factor::Float64=1.0
)
  compare_variance_ensemble_average(
    meanfield_dirs,
    micro_dirs,
    "µ (LFR)",
    x -> x.init_lfr_kwargs.mixing_parameter;
    cutoff_factor=cutoff_factor
  )
end
