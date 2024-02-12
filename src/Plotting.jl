function plot_result(output_filename::String; meanfield_dir::Union{String,Nothing}=nothing, discrete_dir::Union{String,Nothing}=nothing, kwargs...)
  mf_dirs = isnothing(meanfield_dir) ? String[] : [meanfield_dir]
  d_dirs = isnothing(discrete_dir) ? String[] : [discrete_dir]
  return plot_results(output_filename; meanfield_dirs=mf_dirs, discrete_dirs=d_dirs, kwargs...)
end

function plot_results(output_filename::String;
  meanfield_dirs::Vector{String}=String[],
  discrete_dirs::Vector{String}=String[],
  kwargs...)

  # check that the dirs actually exist
  @assert all(isdir, meanfield_dirs)
  @assert all(isdir, discrete_dirs)

  has_mf, has_d = length(meanfield_dirs) > 0, length(discrete_dirs) > 0
  @assert length(discrete_dirs) <= 1 "currently only a single discrete result is supported"

  K_mf = length(meanfield_dirs)

  if !(has_mf || has_d)
    @error "No dir provided"
    return
  end


  half_connection_matrix = get(kwargs, :half_connection_matrix, false)

  center_histogram = get(kwargs, :center_histogram, false)
  if center_histogram
    @warn "Centering histograms, plots are biased!"
  end

  obs_i = M.Observable(1)
  obs_iter = M.Observable(0)

  if has_mf
    labels = map(dn -> endswith("/", dn) ? basename(dirname(dn)) : basename(dn), meanfield_dirs)
    i_mf_a = map(dn -> load_hdf5_data(joinpath(dn, "data_meanfield.h5"), "i"), meanfield_dirs)
    f_a = map(dn -> load_hdf5_data(joinpath(dn, "data_meanfield.h5"), "f"), meanfield_dirs)

    α_init_a = map(dn -> load_hdf5_data(joinpath(dn, "data_meanfield.h5"), "α_init"), meanfield_dirs)

    N_discrete_mf_a = map(dn -> (
        meta = TOML.parsefile(joinpath(dn, "metadata.toml"));
        return meta["N_discrete"]
      ), meanfield_dirs)

    N_a = map(f -> size(f, 1), f_a)

    obs_g_k = nothing
    function get_g_iter(dn, iter)
      try
        return load_hdf5_data(joinpath(dn, "data_meanfield.h5"), "g/$iter")
      catch
        return nothing
      end
    end

    obs_g_k = map(dn -> (M.@lift get_g_iter(dn, $obs_iter)), meanfield_dirs)

    constant_g = isnothing(obs_g_k[1][])

    function build_x(N)
      δx = 2 / N
      x_l, x_r = -1 + 0.5δx, 1 - 0.5δx

      return range(x_l, x_r, length=N)
    end

    x_a = map(build_x, N_a)
  end

  if has_d
    discrete_dir = discrete_dirs[1]
    ops = load_hdf5_data(joinpath(discrete_dir, "data_discrete.h5"), "ops")
    i_d_a = map(dn -> load_hdf5_data(joinpath(dn, "data_discrete.h5"), "i"), discrete_dirs)
    adj_matrix_full = load_hdf5_data(joinpath(discrete_dir, "data_discrete.h5"), "adj_matrix")
    adj_matrix = isnothing(adj_matrix_full) ? nothing : SpA.sparse(adj_matrix_full)
    N_discrete = size(ops, 1)
    if !isnothing(adj_matrix)
      @assert SpA.is_hermsym(adj_matrix, identity)

      adj_matrix_nnz = SpA.nnz(adj_matrix)

      sharp_I = vec(sum(adj_matrix; dims=2))
      ω_inf_d = sum(ops[:, 1] .* sharp_I) ./ sum(sharp_I)
      xs = Vector{Float64}(undef, adj_matrix_nnz)
      ys = Vector{Float64}(undef, adj_matrix_nnz)

      obs_xs = M.Observable(view(xs, 1:adj_matrix_nnz))
      obs_ys = M.Observable(view(ys, 1:adj_matrix_nnz))

      function get_xy(opis)
        rows = SpA.rowvals(adj_matrix)
        got = 0
        for j in 1:N_discrete
          nzr = SpA.nzrange(adj_matrix, j)
          nnz_in_col = length(nzr)

          if half_connection_matrix
            opis_x = view(opis, rows[nzr])

            idc = opis_x .> opis[j]
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

        if center_histogram
          xs .-= ω_inf_d
          ys .-= ω_inf_d
        end

        if half_connection_matrix
          obs_xs[] = view(xs, 1:got)
          obs_ys[] = view(ys, 1:got)
        end

      end

      get_xy(ops[:, 1])
    else
      ω_inf_d = sum(ops[:, 1]) / N_discrete
    end
  end


  set_makie_backend(:gl)

  fig = M.Figure(size=(1024, 720))
  ax1 = M.Axis(fig[1:4, 1])
  ax1.title = "f / ω_i"

  if has_mf
    ax2 = M.Axis(fig[1:4, 2])
    ax2.title = "g(ω,m)"

    ax3 = M.Axis(fig[1:4, 3])
    ax3.title = "f α f"

    g_bottom = fig[5, 1:3] = M.GridLayout()
  else
    g_bottom = fig[5, 1:2] = M.GridLayout()
  end



  if has_mf
    obs_f_a = [M.@lift f_a[k][:, $obs_i] for k in 1:K_mf]
    obs_fαf_a = [M.@lift 0.5 * N_discrete_mf_a[k] * f_a[k][:, $obs_i] .* α_init_a[k] .* f_a[k][:, $obs_i]' for k in 1:K_mf]

    function find_support(f_a)
      left_idc = map(f -> (idx = findfirst(x -> x > 1e-5, f[]); return (isnothing(idx) ? 1 : idx)), f_a)
      right_idc = map(f -> (idx = findlast(x -> x > 1e-5, f[]); return (isnothing(idx) ? length(f[]) : idx)), f_a)

      left_x = [x_a[i][left_idc[i]] for i in 1:K_mf]
      right_x = [x_a[i][right_idc[i]] for i in 1:K_mf]

      if has_d
        return (left=min(minimum(obs_ops[]), minimum(left_x)), right=max(maximum(obs_ops[]), maximum(right_x)))
      else
        return (left=minimum(left_x), right=maximum(right_x))
      end
    end

    function find_max(f_a)
      return maximum(map(f -> maximum(f[]), f_a))
    end

    if !constant_g
      obs_gff_a = [M.Observable(obs_g_k[k][]) for k in 1:K_mf]
    end

    for k in 1:K_mf
      M.lines!(ax1, x_a[k], obs_f_a[k], label=labels[k])

      #M.barplot!(ax3, x_a[k], obs_f_a[k], gap=0)

      if !constant_g
        M.heatmap!(ax2, x_a[k], x_a[k], obs_gff_a[k])
      end
      M.heatmap!(ax3, x_a[k], x_a[k], obs_fαf_a[k])
    end

    legend = M.Legend(g_bottom[1, 1], ax1)

  end

  if has_d
    if center_histogram
      obs_ops = M.@lift (ops[:, $obs_i] .- ω_inf_d)
    else
      obs_ops = M.@lift ops[:, $obs_i]
    end
    obs_extrema_ops = M.@lift extrema($obs_ops)

    M.hist!(ax1, obs_ops; bins=50, normalization=:pdf)
    M.vlines!(ax1, ω_inf_d, color=:grey, ls=0.5)
    # M.hist!(ax3, obs_ops; bins=50, normalization=:pdf)
    # M.vlines!(ax3, ω_inf_d, color=:grey, ls=0.5)

    if !isnothing(adj_matrix)
      M.scatter!(ax2, obs_xs, obs_ys, alpha=0.0005, markersize=4, color=:black, strokewidth=0.05, transparency=true)
      M.limits!(ax2, (-1, 1), (-1, 1))
      M.limits!(ax3, (-1, 1), (-1, 1))
    end
  end

  function step_i(tuple)
    i, iter = tuple

    if has_mf && any(f -> any(isnan, f[:, i]), f_a)
      return
    end

    obs_i[] = i
    obs_iter[] = iter
    if has_mf
      first_mass = 2 / N_a[1] * sum(f_a[1][:, i])
      ax1.title = "$iter, M[1] = $(round(first_mass; digits=6))"
    else
      ax1.title = string(iter)
    end

    if has_d && !isnothing(adj_matrix)
      get_xy(obs_ops[])
    end

    if !constant_g
      for k in 1:K_mf
        obs_gff_a[k][] = obs_g_k[k][]
      end
    end

    if has_mf
      support = find_support(obs_f_a)
      max_f = find_max(obs_f_a)
      M.ylims!(ax1, low=-1, high=1.3 * max_f)
      M.xlims!(ax1, low=support.left, high=support.right)
      # M.xlims!(ax3, low=support.left, high=support.right)
      # M.ylims!(ax3, low=-1, high=1.3 * max_f)

      M.xlims!(ax2, low=support.left, high=support.right)
      M.ylims!(ax2, low=support.left, high=support.right)

    else
      M.autolimits!(ax1)
      # M.autolimits!(ax3)
      # M.xlims!(ax3, low=obs_extrema_ops[][:left], high=obs_extrema_ops[][:right])
    end

  end

  i_range = enumerate(i_mf_a[1])
  M.record(step_i, fig, output_filename, i_range)
  @info ("movie saved at $output_filename")

  println()

end

function get_ω_inf_mf(dir::String)
  p = joinpath(dir, "metadata.toml")
  r = TOML.parsefile(p)["omega_inf_mf"]
  return r
end

function compare_peak2peak(
  meanfield_dir::String,
  discrete_dir::String,
)
  return compare_peak2peak([meanfield_dir], [discrete_dir])
end

function compare_peak2peak(
  meanfield_dirs::Vector{String},
  discrete_dirs::Vector{String},
)

  # check that the dirs actually exist
  @assert all(isdir, meanfield_dirs)
  @assert all(isdir, discrete_dirs)

  K_mf, K_d = length(meanfield_dirs), length(discrete_dirs)

  if K_mf == 0 && K_d == 0
    @error "No dir provided"
    return
  end

  if K_mf > 0
    labels_mf = map(dn -> endswith("/", dn) ? basename(dirname(dn)) : basename(dn), meanfield_dirs)
    i_mf_a = map(dn -> load_hdf5_data(joinpath(dn, "data_meanfield.h5"), "i"), meanfield_dirs)
    f_a = map(dn -> load_hdf5_data(joinpath(dn, "data_meanfield.h5"), "f"), meanfield_dirs)

    N_a = map(f -> size(f, 1), f_a)

    function build_x(N)
      δx = 2 / N
      x_l, x_r = -1 + 0.5δx, 1 - 0.5δx

      return range(x_l, x_r, length=N)
    end

    x_a = map(build_x, N_a)

    ω_inf_mf_init_a = map(dn -> get_ω_inf_mf(dn), meanfield_dirs)

    support_bounds_mf_a = find_support_bounds(f_a, x_a)
    support_width_mf_a = [map(x -> x[2] - x[1], s) for s in support_bounds_mf_a]

    # function get_g_iter(dn, iter)
    #   try
    #     return load_hdf5_data(joinpath(dn, "data_meanfield.h5"), "g/$iter")
    #   catch
    #     return nothing
    #   end
    # end

    # function compute_weighted_avgs(k)
    #   dn = meanfield_dirs[k]
    #   ω_inf_mf = zeros(length(i_mf_a[k]))
    #   for (i, iter) in enumerate(i_mf_a[k])
    #     g = get_g_iter(dn, iter)
    #     @assert !isnothing(g)
    #     ω_inf_mf[i] = sum(g .* x_a[k]) / sum(g)
    #   end
    #   return ω_inf_mf
    # end

    g_M1_a = map(dn -> load_hdf5_data(joinpath(dn, "data_meanfield.h5"), "g_M1"), meanfield_dirs)

  end

  if K_d > 0
    labels_d = map(dn -> endswith("/", dn) ? basename(dirname(dn)) : basename(dn), discrete_dirs)
    i_d_a = map(dn -> load_hdf5_data(joinpath(dn, "data_discrete.h5"), "i"), discrete_dirs)
    ops_a = map(dn -> load_hdf5_data(joinpath(dn, "data_discrete.h5"), "ops"), discrete_dirs)

    adj_matrix_full_a = map(dn -> load_hdf5_data(joinpath(dn, "data_discrete.h5"), "adj_matrix"), discrete_dirs)
    adj_matrix_a = map(adj_matrix_full -> isnothing(adj_matrix_full) ? nothing : SpA.sparse(adj_matrix_full), adj_matrix_full_a)
    N_discrete_a = map(ops -> size(ops, 1), ops_a)

    function compute_weighted_avg(k)
      adj_matrix = adj_matrix_a[k]
      ops = ops_a[k]
      N_discrete = N_discrete_a[k]

      if !isnothing(adj_matrix)
        sharp_I = vec(sum(adj_matrix; dims=2))
        n_connections = sum(sharp_I)

        return [sum(ops[:, k] .* sharp_I) ./ n_connections for k in axes(ops, 2)]
      else
        return [sum(ops[:, k]) / (N_discrete - 1) for k in axes(ops, 2)]
      end
    end

    ω_inf_d_a = [compute_weighted_avg(k) for k in 1:K_d]
    p2p_d_a = [peak2peak(ops; dims=1) for ops in ops_a]
    extrema_d_a = [extrema(ops; dims=1) for ops in ops_a]
  end

  set_makie_backend(:gl)

  fig = M.Figure(size=(1024, 720))
  ax1 = M.Axis(fig[1, 1], yscale=log10, xlabel="iterations", title=M.L"\max_i\;\omega_i - \min_i\;\omega_i")
  ax2 = M.Axis(fig[1, 2], xlabel="iterations", title=M.L"\omega_\inf \quad \min_i\; \omega_i \quad \max_i\; \omega_i")

  for k in 1:K_d
    r = i_d_a[k]
    p2p = p2p_d_a[k]
    M.lines!(ax1, r, p2p, label=labels_d[k])

    M.lines!(ax2, r, ω_inf_d_a[k], label=labels_d[k])
    left_bounds = vec(map(v -> v[1], extrema_d_a[k]))
    right_bounds = vec(map(v -> v[2], extrema_d_a[k]))
    M.band!(ax2, r, left_bounds, right_bounds, alpha=0.2)
  end

  for k in 1:K_mf
    p2p = support_width_mf_a[k]
    r = i_mf_a[k]
    M.lines!(ax1, r, p2p, label=labels_mf[k])

    M.hlines!(ax2, ω_inf_mf_init_a[k])
    M.lines!(ax2, r, g_M1_a[k])
    M.band!(ax2, r, map(v -> v[1], support_bounds_mf_a[k]), map(v -> v[2], support_bounds_mf_a[k]), alpha=0.2)
  end

  #M.axislegend()

  display(fig)

end

