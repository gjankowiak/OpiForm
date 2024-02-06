module Plotting
import UnicodePlots, GLMakie
import Statistics


function plot_meanfield()
  if params.plot_backend == :makie
    i_obs = M.Observable(i)
    t_obs = M.Observable(i * params.δt)

    fig = M.Figure()
    axis_group = fig[1, 1] = M.GridLayout()
    title_label = M.Label(axis_group[1, 1:3, M.Top()], "i=0, t=0", valign=:bottom,
      font=:bold,
      padding=(0, 0, 5, 0))
    ticks = M.MultiplesTicks(1 + Int(ceil(params.Ω_width / params.EC_ρ)), params.EC_ρ, "ρ")
    if params.full_g
      ax_f = M.Axis(axis_group[1, 1], title="f", xticks=ticks, yscale=params.plot_scale)
      ax_a = M.Axis(axis_group[1, 2], title="a", xticks=ticks)
      ax_µ = M.Axis(axis_group[1, 3], title="µ/µC", xticks=ticks)
    else
      ax_f = M.Axis(axis_group[1:2, 1], title="f", xticks=ticks, yscale=params.plot_scale)
      ax_a = M.Axis(axis_group[1:2, 2], title="a", xticks=ticks)
      ax_µ = M.Axis(axis_group[1, 3], title="µ/µC", xticks=ticks)
      ax_g = M.Axis(axis_group[2, 3], title="g")
    end
    f_obs = M.Observable(f)
    a_obs = M.Observable(a)
    af_obs = M.Observable(a .* f)
    µ_obs = M.Observable(µ - params.x)
    µC_obs = M.Observable(µC - params.x)

    M.lines!(ax_f, x, f_obs)

    if !params.full_g
      g_obs = M.Observable(Matrix(g))
      M.heatmap!(ax_g, x, x, g_obs)
    end


    M.lines!(ax_a, x, a_obs, label=M.L"a")
    M.lines!(ax_a, x, af_obs, label=M.L"a f")
    M.vlines!(ax_a, [params.Ω_right - params.EC_ρ, params.Ω_left + params.EC_ρ], width=0.5, color=:grey)
    M.band!(ax_µ, x, -params.EC_ρ, params.EC_ρ, color=:grey, alpha=0.2)
    M.lines!(ax_µ, x, µ_obs, gap=0.0, label="µ - ω")
    M.lines!(ax_µ, x, µC_obs, gap=0.0, label="µC - ω")
    M.vlines!(ax_μ, [params.Ω_right - params.EC_ρ, params.Ω_left + params.EC_ρ], width=0.5, color=:grey)
    M.axislegend(ax_µ)
    M.axislegend(ax_a)

    glfw_window = GLMakie.to_native(display(fig.scene)) # display returns the screen, to_native the glfw window
    should_terminate = false

    # Abort when q is pressed
    GLMakie.on(GLMakie.events(fig.scene).keyboardbutton) do event
      if "q" == GLMakie.GLFW.GetKeyName(event.key)
        GLMakie.GLFW.SetWindowShouldClose(glfw_window, true) # this will close the window after all callbacks are finished
        should_terminate = true
      end
    end

    #display(fig)
  else
    println(UnicodePlots.lineplot(f_init, width=100, height=30, yscale=params.plot_scale))
    println(UnicodePlots.lineplot(a_init, width=100, height=30))
    if !params.full_g
      println(UnicodePlots.heatmap(params.g_init, width=100))
    end
  end
end

function update_meanfield()
  if (i % params.plot_every == 0)
    if params.plot_backend == :makie
      title_label.text = "i=$(i), t=$(round(i*params.δt, digits=2)), M=$(round(params.δx*sum(f),digits=2))"
      f_obs[] = f
      if !params.full_g
        g_obs[] = Matrix(g)
      end
      a_obs[] = a
      af_obs[] = a .* f
      µ_obs[] = µ - params.x
      µC_obs[] = µC - params.x
      M.autolimits!(ax_f)
      M.autolimits!(ax_a)
      sleep(0.01)
    else
      # p = UnicodePlots.lineplot(a, width=100, height=30, yscale=params.plot_scale)
      p = UnicodePlots.lineplot(f, width=100, height=30, yscale=params.plot_scale,
        title="i = $i, t = $(round(i*params.δt, digits=2)), M=$(round(params.δx*sum(f),digits=2))")
      # UnicodePlots.lineplot!(p, a)
      println(p)

      if !params.full_g
        p = UnicodePlots.heatmap(g, width=100)
        # UnicodePlots.lineplot!(p, a)
        println(p)
      end

      # println(UnicodePlots.heatmap(f' .* D_matrix))
    end
  end
end

colors = [
  (lines="blue", area="lightblue"),
  (lines="red", area="pink"),
  (lines="green", area="lightgreen"),
  (lines="orange", area="yellow")]

function plot_hist(Ns, hist, params; old_plot=Missing)
  throw("Not implemented, need to switch to Makie.jl")
  local n_pa = 0 # number of previous agents
  local p
  for i in 1:length(Ns)
    current_group = view(hist, 1:params.max_iter, 1+n_pa:Ns[i]+n_pa)

    mean = vec(Statistics.mean(current_group; dims=2))
    std = vec(Statistics.std(current_group; dims=2))

    if i == 1
      p_func = Plots.plot
    else
      p_func = Plots.plot!
    end
    p_func(current_group, color=colors[i].lines, opacity=0.2, labels="")
    Plots.plot!(mean, ribbon=std, color=colors[i].area, opacity=0.2, lw=0, label="")

    title = ""

    if params.mode == :potential
      Plots.plot!(mean .+ params.u.x_crit, lw=1, color="black", ls=:dot, label="")
      Plots.plot!(mean .- params.u.x_crit, lw=1, color="black", ls=:dot, label="")
      title = "$(title)Potential: $(params.u.a)*x - $(params.u.b)*x²/2, "
    end

    title = "$(title)Coeffs: EB = $(params.coeffs.EB), EC = $(params.coeffs.EC_a), $(params.coeffs.EC_r))"

    p = Plots.plot!(mean, color=colors[i].lines, lw=2, label="Group $i (#$(Ns[i]))", thickness_scaling=1, ylims=(0, 1), title=title)

    n_pa += Ns[i]
  end
  p
end

end
