import OpiForm
import CairoMakie as M
import Graphs
import TOML

function extract(dir_name::String, parameter_extractor::Function)
  params = OpiForm.Params.from_toml(dir_name)
  adj_matrix = OpiForm.load_hdf5_sparse(joinpath(dir_name, "data.hdf5"), "adj_matrix")
  t_star = OpiForm.Micro.compute_T_star(params.δt, adj_matrix)
  avg_degree = sum(adj_matrix) / size(adj_matrix, 1)
  degrees = vec(sum(adj_matrix; dims=1))
  p = parameter_extractor(params)
  g = Graphs.SimpleGraph(adj_matrix)
  assortativity = Graphs.assortativity(g)
  clustering = Graphs.global_clustering_coefficient(g)
  return (parameter=p, avg_degree=avg_degree, degrees=degrees, assortativity=assortativity, clustering=clustering, t_star=t_star)
end

function getbboxatPoint(ax, coords, extent=(30, 30))
  width, height = extent
  xscale, yscale = ax.xscale[], ax.yscale[]
  bbox = M.lift(ax.scene.camera.projectionview, ax.scene.viewport) do _, pxa
    p = M.Makie.project(ax.scene, M.Point(xscale(coords[1]), yscale(coords[2])))

    c = p + pxa.origin
    R = M.Rect2f(c .- M.Point2f(width, height), (2 * width, 2 * height))
  end
  return bbox
end

function insetAtPoint(fig, ax, Point, extent=(30, 30); kwargs...)

  bbox = getbboxatPoint(ax, Point, extent)

  ax2 = M.Axis(fig, bbox=bbox; kwargs...)
  M.translate!(ax2.blockscene, 0, 0, 100)
  return ax2
end

function plot1(parameter_values, avg_degrees, aggregate_degrees)
  fig = M.Figure(size=(1920, 1080))
  ax1 = M.Axis(fig[1, 1], xlabel="parameter", ylabel="average degree", xscale=log10, backgroundcolor=(:white, 0.5))
  M.scatter!(ax1, parameter_values, avg_degrees)

  for (i, p) in enumerate(sort(collect(keys(aggregate_degrees))))
    inset_ax = insetAtPoint(fig, ax1, (p, unique_avg_degrees[i] + 50), (50, 50); xscale=identity, backgroundcolor=(:blue, 0.1))
    M.hideydecorations!(inset_ax)
    M.hidespines!(inset_ax)
    M.hist!(inset_ax, aggregate_degrees[p], offset=0, direction=:y, normalization=:density, bins=range(0, 1000), scale_to=1.0, color=:blue)
  end

  fig
end

function plot_degree(aggregates, param_name::String;
  xscale=identity, stride::Int64=1, offset_factor::Float64=4e-2, max_degree::Int64=1000, max_assortativity::Float64=1.0)

  idx_sort = sortperm(aggregates; by=x -> x["param"])
  sorted_aggregates = aggregates[idx_sort]

  p_vals = [a["param"] for a in sorted_aggregates]
  labels = map(x -> "$(param_name) = $(round(x; digits=3))", p_vals[1:stride:end])

  fig = M.Figure(size=(1920, 1080))
  offset = offset_factor / stride
  ax1 = M.Axis(fig[1, 1], xlabel="degree distribution", backgroundcolor=(:white, 0.5), yticks=((1:length(labels)) .* offset, labels), xscale=xscale)
  ax2 = M.Axis(fig[1, 2], xlabel="assortativity distribution", backgroundcolor=(:white, 0.5), yticks=((1:length(labels)) .* offset, labels), xscale=xscale)
  ax3 = M.Axis(fig[1, 3], xlabel="T*", backgroundcolor=(:white, 0.5), yticks=((1:length(labels)) .* offset, labels), xscale=log10)
  ax4 = M.Axis(fig[1, 4], xlabel="Clustering coefficitent", backgroundcolor=(:white, 0.5), yticks=((1:length(labels)) .* offset, labels))

  M.hideydecorations!(ax2, grid=false)
  M.hideydecorations!(ax3, grid=false)
  M.hideydecorations!(ax4, grid=false)

  M.linkyaxes!(ax1, ax2, ax3, ax4)

  for (i, a) in enumerate(sorted_aggregates)

    #
    # Degree
    #
    if ax1.xscale[] == log10
      bins = [5; 6; 7; 8; 9; 10 .^ range(1, 3; length=1000)...]
      bins = range(1, max_degree)
      d = M.hist!(ax1, a["degrees"], direction=:y, offset=i * offset, bins=bins, color=(:blue, (0.25 + 0.75 / (i + 0.5))), weights=fill(1e-4, length(a["degrees"])))
    else
      bins = range(0, max_degree)
      d = M.hist!(ax1, a["degrees"], direction=:y, offset=i * offset, bins=bins, color=(:blue, (0.25 + 0.75 / (i + 0.5))), weights=fill(1e-4, length(a["degrees"])))
    end
    avg_degree = sum(a["degrees"] / length(a["degrees"]))

    M.scatter!(ax1, avg_degree, offset * i; color=:black)
    M.lines!(ax1, fill(avg_degree, 2), offset * [i + 0.1; i + 0.9]; color=:black)

    #
    # Assortativity
    #
    avg_assortativity = sum(a["assortativity"] / length(a["assortativity"]))

    M.scatter!(ax2, a["assortativity"], fill(offset * i, length(a["assortativity"])); color=(:blue, 0.3))
    M.lines!(ax2, fill(avg_assortativity, 2), offset * [i - 0.5; i + 0.5]; color=:black)

    #
    # Clustering
    #
    avg_clustering = sum(a["clustering"] / length(a["clustering"]))

    M.scatter!(ax4, a["clustering"], fill(offset * i, length(a["clustering"])); color=(:blue, 0.3))
    M.lines!(ax4, fill(avg_clustering, 2), offset * [i - 0.5; i + 0.5]; color=:black)

    #
    # T_star
    #
    avg_t_star = sum(a["t_star"] / length(a["t_star"]))

    M.scatter!(ax3, a["t_star"], fill(offset * i, length(a["t_star"])); color=(:blue, 0.3))
    M.lines!(ax3, fill(avg_t_star, 2), offset * [i - 0.5; i + 0.5]; color=:black)
  end

  return fig
end

function check_degree(base_dir::String, extractor::Symbol;
  xscale::Function=identity, stride::Int64=1, offset_factor::Float64=4e-2, max_degree::Int64=1000)
  dirs = walkdir(base_dir; follow_symlinks=true)

  regex_micro = r"micro-[0-9]+$"

  micro_dirs = String[]

  parameter_values = Float64[]
  avg_degrees = Float64[]
  degrees = Int64[]
  assortativity = Float64[]
  clustering = Float64[]
  t_star = Float64[]

  aggregates = Dict{String,Any}[]

  extractors = (
    ER=x -> x.init_micro_graph_args[2], # ER
    WS=x -> round(Int64, (x.N_micro - 1) * x.init_micro_graph_args[2]), # WS
    BA=x -> x.init_micro_graph_args[2], # BA
    LFR=x -> x.init_lfr_kwargs.mixing_parameter # LFR TODO
  )

  param_names = (
    ER="p",
    WS="k",
    BA="k",
    LFR="µ"
  )

  for d_tuple in dirs
    d = d_tuple[1]
    if endswith(d, regex_micro) && isfile(joinpath(d, "data.hdf5"))
      push!(micro_dirs, d)
    end
  end

  p_prev = NaN

  # for (i, d) in enumerate(micro_dirs)
  #   # get the new parameter value and degrees
  #   p_val, avg_d, ds = get_param_degree(d, extractors[extractor])
  #
  #   # store the new values
  #   push!(parameter_values, p_val)
  #   push!(avg_degrees, avg_d)
  #
  #   append!(degrees, ds)
  #   append!(parameter_values_per_node, fill(p_val, length(ds)))
  # end

  n_val = 1

  for (i, d) in enumerate(micro_dirs)
    # get the new parameter value and degrees
    p_val, avg_d, ds, assort, clust, t_s = extract(d, extractors[extractor])

    # if the parameter changed, store the aggregate degrees and empty the temporary store
    if !isempty(degrees) && p_val != p_prev

      push!(aggregates, Dict{String,Any}())
      n_val = length(aggregates)

      aggregates[n_val]["degrees"] = copy(degrees)
      aggregates[n_val]["assortativity"] = copy(assortativity)
      aggregates[n_val]["clustering"] = copy(clustering)
      aggregates[n_val]["t_star"] = copy(t_star)
      aggregates[n_val]["param"] = p_prev

      empty!(degrees)
      empty!(assortativity)
      empty!(clustering)
      empty!(t_star)
    end
    p_prev = p_val

    # store the new values
    push!(parameter_values, p_val)
    push!(avg_degrees, avg_d)
    push!(assortativity, assort)
    push!(clustering, clust)
    push!(t_star, t_s)
    append!(degrees, ds)

    # if this is the last iteration in the loop, store the aggregate
    if i == lastindex(micro_dirs) && !isempty(degrees)
      push!(aggregates, Dict{String,Any}())
      n_val = length(aggregates)

      aggregates[n_val]["degrees"] = copy(degrees)
      aggregates[n_val]["assortativity"] = copy(assortativity)
      aggregates[n_val]["clustering"] = copy(clustering)
      aggregates[n_val]["t_star"] = copy(t_star)
      aggregates[n_val]["param"] = p_prev
    end
  end

  open(joinpath(base_dir, "aggregates.toml"), "w") do toml
    TOML.print(toml, Dict("data" => aggregates))
  end

  fig = plot_degree(aggregates, param_names[extractor];
    xscale=xscale, stride=stride, offset_factor=offset_factor, max_degree=max_degree)
  suffix = splitpath(base_dir)[end]
  # M.save("plots/check_degree_$(extractor)_$(suffix)_$(xscale).pdf", fig)
  # M.save("plots/check_degree_$(extractor)_$(suffix)_$(xscale).svg", fig)
  M.save("$(base_dir)/graph_analysis.pdf", fig)
  M.save("$(base_dir)/graph_analysis.svg", fig)
end
