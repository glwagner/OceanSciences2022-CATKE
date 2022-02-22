using Oceananigans
using Oceananigans.Units
using OceanTurbulenceParameterEstimation
using OceanTurbulenceParameterEstimation: Transformation
using Distributions
using DataDeps
using GLMakie
using Printf

using Oceananigans.TurbulenceClosures.CATKEVerticalDiffusivities:
    CATKEVerticalDiffusivity,
    SurfaceTKEFlux,
    MixingLength

include("best_catke_parameters.jl")

function perturbation_prior(θ★, ϵ=0.7)
    L = 0.1θ★
    U = 2θ★
    return ScaledLogitNormal(bounds=(L, U))
end

θ★ = θ_constant_Ri

priors = Dict()
priors[:CᵂwΔ] = ScaledLogitNormal(bounds=(4, 12))  # 8.66
priors[:Cᵂu★] = ScaledLogitNormal(bounds=(1, 4))   # 2.89
priors[:Cᴰ]   = ScaledLogitNormal(bounds=(0, 1))   # 0.489
priors[:Cᴸᵇ]  = ScaledLogitNormal(bounds=(0, 0.1)) # 0.0286
priors[:Cᴷu⁻] = ScaledLogitNormal(bounds=(0, 0.2)) # 0.0776
priors[:Cᴷc⁻] = ScaledLogitNormal(bounds=(0, 1))   # 0.567
priors[:Cᴷe⁻] = ScaledLogitNormal(bounds=(7, 11))  # 9.00

free_parameters = FreeParameters(priors, names = tuple(keys(priors)...))

case_path(case) = @datadep_str("two_day_suite_1m/$(case)_instantaneous_statistics.jld2")

case = "weak_wind_strong_cooling"

times = range(2hours, step=10minutes, stop=48hours)
Nt = length(times)
transformation = Transformation(time=TimeIndices([2, round(Int, Nt/2), Nt]))
field_names = (:u, :v, :T, :e)
regrid_size = (1, 1, 128)
observations = SyntheticObservations(case_path(case); field_names, regrid_size, times, transformation)

α = observations.metadata.parameters.thermal_expansion_coefficient
equation_of_state = LinearEquationOfState(; α)

mixing_length = MixingLength(Cᴬu   = 0.0,
                             Cᴬc   = 0.0,
                             Cᴬe   = 0.0,
                             CᴷRiᶜ = 2.0,
                             Cᴷuʳ  = 0.0,
                             Cᴷcʳ  = 0.0,
                             Cᴷeʳ  = 0.0)

catke = CATKEVerticalDiffusivity(; mixing_length)

Nensemble = 200

simulation = ensemble_column_model_simulation(observations;
                                              Nensemble,
                                              architecture = GPU(),
                                              buoyancy = SeawaterBuoyancy(; equation_of_state, constant_salinity=true),
                                              tracers = (:T, :e),
                                              closure = catke)

simulation.Δt = 2.0

Qᵘ = simulation.model.velocities.u.boundary_conditions.top.condition
Qᵀ = simulation.model.tracers.T.boundary_conditions.top.condition
dTdz = simulation.model.tracers.T.boundary_conditions.bottom.condition

Qᵘ .= observations.metadata.parameters.momentum_flux
Qᵀ .= observations.metadata.parameters.temperature_flux
dTdz .= observations.metadata.parameters.dθdz_deep

calibration = InverseProblem(observations, simulation, free_parameters)

output_paths = []

model = simulation.model

simulation.output_writers[:fields] =
    JLD2OutputWriter(model, merge(model.velocities, model.tracers, model.diffusivity_fields),
                     prefix = "catke_calibration_0",
                     schedule = SpecifiedTimes(times...),
                     force = true)

push!(output_paths, simulation.output_writers[:fields].filepath)

eki = EnsembleKalmanInversion(calibration; convergence_rate=0.5)

for i = 1:4
    simulation.output_writers[:fields] =
        JLD2OutputWriter(model, merge(model.velocities, model.tracers, model.diffusivity_fields),
                         prefix = string("catke_calibration_", i),
                         schedule = SpecifiedTimes(times...),
                         force = true)

    push!(output_paths, simulation.output_writers[:fields].filepath)

    iterate!(eki)
end

#=
u = []
v = []
T = []
e = []

for i = 0:5
    @show i
    path = string("catke_calibration_", i, ".jld2")
    push!(u, FieldTimeSeries(path, "u", boundary_conditions=nothing))
    push!(v, FieldTimeSeries(path, "v", boundary_conditions=nothing))
    push!(T, FieldTimeSeries(path, "T", boundary_conditions=nothing))
    push!(e, FieldTimeSeries(path, "e", boundary_conditions=nothing))
end

z = znodes(first(u))

fig = Figure(resolution=(1200, 600))

ax_T = Axis(fig[1, 1], xlabel = "Temperature (ᵒC)", ylabel = "z (m)")
ax_u = Axis(fig[1, 2], xlabel = "Velocity (m s⁻¹)", ylabel = "z (m)")
ax_e = Axis(fig[1, 3], xlabel = "Turbulent Kinetic Energy (m² s⁻²)", ylabel = "z (m)")

Nt = length(observations.times)
time_slider = Slider(fig[2, :], range=1:Nt, horizontal=true, startvalue=1)
n = time_slider.value

iter_slider = Slider(fig[3, :], range=1:length(u), horizontal=true, startvalue=1)
iter = iter_slider.value

profile_max(u, i, n) = Point2f.(maximum(interior(u[n])[:, i, :], dims=1)[:], z)
profile_min(u, i, n) = Point2f.(minimum(interior(u[n])[:, i, :], dims=1)[:], z)

#=
u_max = @lift profile_max(u[$iter], 1, $n)
v_max = @lift profile_max(v[$iter], 1, $n)
T_max = @lift profile_max(T[$iter], 1, $n)
e_max = @lift profile_max(e[$iter], 1, $n)
    
u_min = @lift profile_min(u[$iter], 1, $n)
v_min = @lift profile_min(v[$iter], 1, $n)
T_min = @lift profile_min(T[$iter], 1, $n)
e_min = @lift profile_min(e[$iter], 1, $n)

color = (:pink, 0.4)
band!(ax_T, T_max, T_min; color)
band!(ax_e, e_max, e_min; color)
band!(ax_u, u_max, u_min; color)
band!(ax_u, v_max, v_min; color)
=#

for k = 1:Nensemble
    uk = @lift interior(u[$iter][$n])[k, 1, :]
    vk = @lift interior(v[$iter][$n])[k, 1, :]
    Tk = @lift interior(T[$iter][$n])[k, 1, :]
    ek = @lift interior(e[$iter][$n])[k, 1, :]

    lines!(ax_T, Tk, z, color=(:royalblue1, 0.8), linewidth=1,   label="CATKE")
    lines!(ax_u, uk, z, color=(:royalblue1, 0.8), linewidth=1,   label="u, CATKE")
    lines!(ax_u, vk, z, color=(:orange,     0.8), linewidth=1, label="v, CATKE")
    lines!(ax_e, ek, z, color=(:royalblue1, 0.8), linewidth=1,   label="CATKE")
end

u_truth = @lift interior(observations.field_time_serieses.u[$n])[1, 1, :]
v_truth = @lift interior(observations.field_time_serieses.v[$n])[1, 1, :]
T_truth = @lift interior(observations.field_time_serieses.T[$n])[1, 1, :]
e_truth = @lift interior(observations.field_time_serieses.e[$n])[1, 1, :]

lines!(ax_T, T_truth, z, color=(:gray23,  0.4),  linewidth=6, label="LES")
lines!(ax_u, u_truth, z, color=(:gray23,  0.4),  linewidth=6, label="u, LES")
lines!(ax_u, v_truth, z, color=(:darkred, 0.4),  linewidth=6, label="v, LES")
lines!(ax_e, e_truth, z, color=(:gray23,  0.4),  linewidth=6, label="LES")

xlims!(ax_u, -0.2, 0.3)
xlims!(ax_e, -0.0002, 0.0016)
xlims!(ax_T, 19.6, 20.0)

axislegend(ax_T, position=:rb, merge=true)
axislegend(ax_u, position=:rb, merge=true)
axislegend(ax_e, position=:rb, merge=true)

title = @lift @sprintf("LES and CATKE solutions at %.1f hours with iteration %d parameters", u[1].times[$n] / hour, $iter - 1)
Label(fig[0, :], title)

display(fig)

nswitch = floor(Int, Nt/length(output_paths))

#record(fig, "calibration_demo.mp4", 1:Nt; framerate=16) do nn
#    @info "Drawing frame $nn of $Nt..."
#    if nn % nswitch == 0 && iter.val < 2
#        iter[] += iter.val
#    end 
#
#    n[] = nn
#end
=#
