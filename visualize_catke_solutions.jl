using Oceananigans
using OceanTurbulenceParameterEstimation
using Oceananigans.Units
using DataDeps
using GLMakie
using Printf

suite = "four_day_suite"
case_path(case) = @datadep_str("$(suite)_1m/$(case)_instantaneous_statistics.jld2")

cases = [
         "strong_wind_no_rotation",
         "free_convection",
         "weak_wind_strong_cooling",
         "strong_wind_weak_cooling",
         "strong_wind",
        ]

observations = [SyntheticObservations(case_path(case), field_names=(:u, :v, :T, :e)) for case in cases]

u_catke_time_series = FieldTimeSeries("catke_simulation_$suite.jld2", "u", boundary_conditions=nothing)
v_catke_time_series = FieldTimeSeries("catke_simulation_$suite.jld2", "v", boundary_conditions=nothing)
T_catke_time_series = FieldTimeSeries("catke_simulation_$suite.jld2", "T", boundary_conditions=nothing)
e_catke_time_series = FieldTimeSeries("catke_simulation_$suite.jld2", "e", boundary_conditions=nothing)

#####
##### Animate!
#####

fig = Figure(resolution=(1800, 800))

T_axs = []
u_axs = []
e_axs = []

for (i, case) in enumerate(cases)
    ax_l = Label(fig[1, i+1], replace(case, "_" => "\n"), tellwidth=false)
    ax_T = Axis(fig[2, i+1])
    ax_u = Axis(fig[3, i+1])
    ax_e = Axis(fig[4, i+1])

    xlims!(ax_e, 0, 0.005) 
    xlims!(ax_u, -0.2, 0.3)

    hidedecorations!(ax_T)
    hidedecorations!(ax_u)
    hidedecorations!(ax_e)

    hidespines!(ax_T)
    hidespines!(ax_u)
    hidespines!(ax_e)

    push!(T_axs, ax_T)
    push!(u_axs, ax_u)
    push!(e_axs, ax_e)
end

Nt = length(first(observations).times)
slider = Slider(fig[5, :], range=1:Nt, startvalue=1)
n = slider.value

###
### Vertical velocity
###

z = znodes(T_catke_time_series)

for (j, case) in enumerate(cases)
    obs = observations[j]

    u_les = @lift interior(obs.field_time_serieses.u[$n])[1, 1, :]
    v_les = @lift interior(obs.field_time_serieses.v[$n])[1, 1, :]
    T_les = @lift interior(obs.field_time_serieses.T[$n])[1, 1, :]
    e_les = @lift interior(obs.field_time_serieses.e[$n])[1, 1, :]

    ax_T = T_axs[j]
    ax_u = u_axs[j]
    ax_e = e_axs[j]

    linewidth = 5
    color = :indigo
    lines!(ax_T, T_les, z; linewidth, color=(color, 0.8))
    lines!(ax_u, u_les, z; linewidth, color=(color, 0.8))
    lines!(ax_u, v_les, z; linewidth, color=(color, 0.5))
    lines!(ax_e, e_les, z; linewidth, color=(color, 0.8))

    linewidth = 2
    colors = [:darkorange3, :deeppink, :lightseagreen]
    for i = 1:3
        u_catke = @lift interior(u_catke_time_series[$n])[i, j, :]
        v_catke = @lift interior(v_catke_time_series[$n])[i, j, :]
        T_catke = @lift interior(T_catke_time_series[$n])[i, j, :]
        e_catke = @lift interior(e_catke_time_series[$n])[i, j, :]

        color = colors[i]
        lines!(ax_T, T_catke, z; linewidth, color)
        lines!(ax_u, u_catke, z; linewidth, color)
        lines!(ax_u, v_catke, z; linewidth, color=(color, 0.5))
        lines!(ax_e, e_catke, z; linewidth, color)
    end
end

Label(fig[2, 1], "Temperature (ᵒC)", tellheight=false)
Label(fig[3, 1], "Velocity (m s⁻¹)", tellheight=false)
Label(fig[4, 1], "Turbulent \n kinetic \n energy", tellheight=false)

display(fig)
