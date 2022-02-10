using Oceananigans
using OceanTurbulenceParameterEstimation
using Distributions
using DataDeps
using GLMakie

using Oceananigans.TurbulenceClosures.CATKEVerticalDiffusivities:
    CATKEVerticalDiffusivity,
    SurfaceTKEFlux,
    MixingLength

θ_constant_Ri = (; 
                  CᵂwΔ  = 8.38,
                  Cᵂu★  = 9.06,
                  Cᴰ    = 1.37,
                  Cᴸᵇ   = 1.13,
                  Cᴷu⁻  = 0.0887,
                  Cᴷc⁻  = 0.164,
                  Cᴷe⁻  = 2.13,
                 )

function perturbation_prior(θ★, ϵ=0.5)
    L = θ★ / 2
    U = 2θ★
    return ScaledLogitNormal(bounds=(L, U))
end

θ★ = θ_constant_Ri
priors = NamedTuple(name => perturbation_prior(θ★[name]) for name in keys(θ★))
free_parameters = FreeParameters(priors)

case_path(case) = @datadep_str("two_day_suite_1m/$(case)_instantaneous_statistics.jld2")

cases = [#"free_convection",
         "weak_wind_strong_cooling",
         #"strong_wind_weak_cooling",
         #"strong_wind",
         #"strong_wind_no_rotation"]
         
case = "weak_wind_strong_cooling"

times = 6hours:10minutes:44hours
field_names = (:u, :v, :T, :e)
regrid_size = nothing
observations = SyntheticObservations(case_path(case); field_names, regrid_size, times)

observation = first(observations)
α = observation.metadata.parameters.thermal_expansion_coefficient
equation_of_state = LinearEquationOfState(; α)

mixing_length = MixingLength(Cᴬu   = 0.0,
                             Cᴬc   = 0.0,
                             Cᴬe   = 0.0,
                             CᴷRiᶜ = 2.0,
                             Cᴷuʳ  = 0.0,
                             Cᴷcʳ  = 0.0,
                             Cᴷeʳ  = 0.0)

catke = CATKEVerticalDiffusivity(; mixing_length)

Nensemble = 10

simulation = ensemble_column_model_simulation(observations;
                                              Nensemble,
                                              architecture = CPU(),
                                              buoyancy = SeawaterBuoyancy(; equation_of_state, constant_salinity=true),
                                              tracers = (:T, :e),
                                              closure = catke)

simulation.Δt = 30.0

progress(sim) = @info "Iter: $(iteration(sim)), time: $(prettytime(sim))"
simulation.callbacks[:progress] = Callback(progress, IterationInterval(100))

Qᵘ = simulation.model.velocities.u.boundary_conditions.top.condition
Qᵀ = simulation.model.tracers.T.boundary_conditions.top.condition
dTdz = simulation.model.tracers.T.boundary_conditions.bottom.condition

for (case, obs) in enumerate(observations)
    @show case cases[case]
    @show obs.metadata.parameters.momentum_flux
    @show obs.metadata.parameters.temperature_flux
    @show obs.metadata.parameters.buoyancy_flux
    @show f = obs.metadata.parameters.coriolis_parameter

    view(Qᵘ, :, case) .= obs.metadata.parameters.momentum_flux
    view(Qᵀ, :, case) .= obs.metadata.parameters.temperature_flux
    view(dTdz, :, case) .= obs.metadata.parameters.dθdz_deep
    view(simulation.model.coriolis, :, case) .= Ref(FPlane(f=f))
end

calibration = InverseProblem(observations, simulation, free_parameters)

model = simulation.model

# either

output_paths = []

simulation.output_writers[:fields] =
    JLD2OutputWriter(model, merge(model.velocities, model.tracers, model.diffusivity_fields),
                     prefix = "catke_calibration_0",
                     schedule = SpecifiedTimes(round.(observation_times(obs))...),
                     force = true)

push!(output_paths, simulation.output_writers[:fields].path)

eki = EnsembleKalmanInversion(calibration; noise_covariance=1e-2)

for i = 1:3
    iterate!(eki)
    
    simulation.output_writers[:fields] =
        JLD2OutputWriter(model, merge(model.velocities, model.tracers, model.diffusivity_fields),
                         prefix = string("catke_calibration_", eki.iteration),
                         schedule = SpecifiedTimes(round.(observation_times(obs))...),
                         force = true)

    push!(output_paths, simulation.output_writers[:fields].path)
end

u = []
v = []
T = []
e = []

for path in output_paths
    push!(u, FieldTimeSeries(output_path, "u"))
    push!(v, FieldTimeSeries(output_path, "v"))
    push!(T, FieldTimeSeries(output_path, "T"))
    push!(e, FieldTimeSeries(output_path, "e"))
end

z = znodes(u)

fig = Figure(resolution=(1200, 1200))

ax_T = Axis(fig[1, 1], xlabel = "Temperature (ᵒC)", ylabel = "z (m)")
ax_u = Axis(fig[1, 2], xlabel = "Velocity (m s⁻¹)", ylabel = "z (m)")
ax_e = Axis(fig[1, 3], xlabel = "Turbulent Kinetic Energy (m² s⁻²)", ylabel = "z (m)")

Nt = length(observations.times)
time_slider = Slider(fig[2, :], range=1:Nt, horizontal=true, startvalue=1)
n = time_slider.value

iter_slider = Slider(fig[3, :], range=1:Nt, horizontal=true, startvalue=1)
iter = iter_slider.value

profile_max(u, i, n) = Point2f.(maximum(interior(u[n])[:, i, :], dims=1)[:], z)
profile_min(u, i, n) = Point2f.(minimum(interior(u[n])[:, i, :], dims=1)[:], z)

u_truth = @lift interior(observations.field_time_serieses.u[$n])[1, 1, :]
v_truth = @lift interior(observations.field_time_serieses.v[$n])[1, 1, :]
T_truth = @lift interior(observations.field_time_serieses.T[$n])[1, 1, :]
e_truth = @lift interior(observations.field_time_serieses.e[$n])[1, 1, :]

lines!(ax_T, T_truth, z, color=(:gray23, 0.6), linewidth=5,   label="LES")
lines!(ax_u, u_truth, z, color=(:gray23, 0.6), linewidth=5,   label="u, LES")
lines!(ax_u, v_truth, z, color=(:darkred, 0.6), linewidth=1.5, label="v, LES")
lines!(ax_e, e_truth, z, color=(:gray23, 0.6), linewidth=5,   label="LES")

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

    lines!(ax_T, Tk, z, color=(:royalblue1, 0.2), linewidth=1,   label="CATKE")
    lines!(ax_u, uk, z, color=(:royalblue1, 0.2), linewidth=1,   label="u, CATKE")
    lines!(ax_u, vk, z, color=(:orange,     0.2), linewidth=1, label="v, CATKE")
    lines!(ax_e, ek, z, color=(:royalblue1, 0.2), linewidth=1,   label="CATKE")
end

xlims!(ax_T, 19.6, 20.05)
xlims!(ax_u, -0.6, 0.6)
xlims!(ax_e, -0.0005, 0.005)
axislegend(ax_T, position=:rb)
axislegend(ax_u, position=:rb)
axislegend(ax_e, position=:rb)

display(fig)

