import OpiForm
import CairoMakie as C
import GLMakie as GL

MM = C
MM.activate!()

function get_param_degree(dir_name::String, parameter_extractor::Function)
  params = OpiForm.Params.from_toml(dir_name)
  adj_matrix = OpiForm.load_hdf5_sparse(joinpath(dir_name, "data.hdf5"), "adj_matrix")
  avg_degree = sum(adj_matrix) / size(adj_matrix, 1)
  degrees = vec(sum(adj_matrix; dims=1))
  p = parameter_extractor(params)
  return (parameter=p, avg_degree=avg_degree, degrees=degrees)
end

function getbboxatPoint(ax, coords, extent=(30, 30))
  width, height = extent
  xscale, yscale = ax.xscale[], ax.yscale[]
  bbox = MM.lift(ax.scene.camera.projectionview, ax.scene.viewport) do _, pxa
    p = MM.Makie.project(ax.scene, MM.Point(xscale(coords[1]), yscale(coords[2])))

    c = p + pxa.origin
    R = MM.Rect2f(c .- MM.Point2f(width, height), (2 * width, 2 * height))
  end
  return bbox
end

function insetAtPoint(fig, ax, Point, extent=(30, 30); kwargs...)

  bbox = getbboxatPoint(ax, Point, extent)

  ax2 = MM.Axis(fig, bbox=bbox; kwargs...)
  MM.translate!(ax2.blockscene, 0, 0, 100)
  return ax2
end

function plot1(parameter_values, avg_degrees, aggregate_degrees)
  fig = MM.Figure(size=(3000, 1500))
  ax1 = MM.Axis(fig[1, 1], xlabel="parameter", ylabel="average degree", xscale=log10, backgroundcolor=(:white, 0.5))
  MM.scatter!(ax1, parameter_values, avg_degrees)

  for (i, p) in enumerate(sort(collect(keys(aggregate_degrees))))
    inset_ax = insetAtPoint(fig, ax1, (p, unique_avg_degrees[i] + 50), (50, 50); xscale=identity, backgroundcolor=(:blue, 0.1))
    MM.hideydecorations!(inset_ax)
    MM.hidespines!(inset_ax)
    MM.hist!(inset_ax, aggregate_degrees[p], offset=0, direction=:y, normalization=:density, bins=range(0, 1000), scale_to=1.0, color=:blue)
  end

  fig
end

function plot2(parameter_values, avg_degrees, aggregate_degrees, param_name::String;
  xscale=identity, stride::Int64=1, offset_factor::Float64=4e-2, max_degree::Int64=1000)

  unique_avg_degrees = unique(avg_degrees)

  p_vals = sort(collect(keys(aggregate_degrees)))
  labels = map(x -> "$(param_name) = $(round(x; digits=3))", p_vals[1:stride:end])

  fig = MM.Figure(size=(2000, 1000))
  offset = offset_factor / stride
  ax1 = MM.Axis(fig[1, 1], xlabel="degree distribution", backgroundcolor=(:white, 0.5), yticks=((1:length(labels)) .* offset, labels), xscale=xscale)

  for (i, p) in enumerate(p_vals[1:stride:end])
    # d = MM.density!(ax1, aggregate_degrees[p], offset=i, color=(:blue, 0.4))
    # d = MM.stephist!(ax1, aggregate_degrees[p], offset=i * offset, normalization=:pdf, bins=range(0, 1000), color=(:blue, 0.4))
    if ax1.xscale[] == log10
      bins = [5; 6; 7; 8; 9; 10 .^ range(1, 3; length=1000)...]
      bins = range(1, max_degree)
      d = MM.hist!(ax1, aggregate_degrees[p], direction=:y, offset=i * offset, bins=bins, color=(:blue, (0.25 + 0.75 / (i + 0.5))), weights=fill(1e-4, length(aggregate_degrees[p])))
    else
      bins = range(0, max_degree)
      d = MM.hist!(ax1, aggregate_degrees[p], direction=:y, offset=i * offset, bins=bins, color=(:blue, (0.25 + 0.75 / (i + 0.5))), weights=fill(1e-4, length(aggregate_degrees[p])))
    end
    avg_deg = sum(aggregate_degrees[p] / length(aggregate_degrees[p]))
    MM.scatter!(ax1, avg_deg, offset * i; color=:black)
    MM.lines!(ax1, fill(avg_deg, 2), offset * [i + 0.1; i + 0.9]; color=:black)
  end

  return fig
end

function main(base_dir::String, extractor::Symbol;
  xscale::Function=identity, stride::Int64=1, offset_factor::Float64=4e-2, max_degree::Int64=1000)
  dirs = walkdir(base_dir)

  regex_micro = r"micro-[0-9]+$"

  micro_dirs = String[]

  parameter_values = Float64[]
  avg_degrees = Float64[]
  degrees = Int64[]
  parameter_values_per_node = Float64[]
  aggregate_degrees = Dict{Float64,Vector{Int64}}()

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

  for (i, d) in enumerate(micro_dirs)
    # get the new parameter value and degrees
    p_val, avg_d, ds = get_param_degree(d, extractors[extractor])

    # if the parameter changed, store the aggregate degrees and empty the temporary store
    if !isempty(degrees) && p_val != p_prev
      aggregate_degrees[p_prev] = copy(degrees)
      empty!(degrees)
    end
    p_prev = p_val

    # store the new values
    push!(parameter_values, p_val)
    push!(avg_degrees, avg_d)
    append!(degrees, ds)

    # if this is the last iteration in the loop, store the aggregate
    if i == lastindex(micro_dirs) && !isempty(degrees)
      aggregate_degrees[p_prev] = copy(degrees)
    end
  end

  unique_avg_degrees = unique(avg_degrees)

  fig = plot2(parameter_values, avg_degrees, aggregate_degrees, param_names[extractor];
    xscale=xscale, stride=stride, offset_factor=offset_factor, max_degree=max_degree)
  suffix = splitpath(base_dir)[end]
  MM.save("plots/check_degree_$(extractor)_$(suffix)_$(xscale).pdf", fig)
  MM.save("plots/check_degree_$(extractor)_$(suffix)_$(xscale).svg", fig)
end
