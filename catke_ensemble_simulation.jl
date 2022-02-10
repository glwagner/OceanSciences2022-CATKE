using Oceananigans
using OceanTurbulenceParameterEstimation
using Distributions
using DataDeps
using GLMakie

using Oceananigans.TurbulenceClosures.CATKEVerticalDiffusivities:
    CATKEVerticalDiffusivity,
    SurfaceTKEFlux,
    MixingLength

θ_constant_Ri = (; Cᴸᵇ   = 1.36,
                   Cᴷu⁻  = 0.101,
                   Cᴷc⁻  = 0.0574,
                   Cᴷe⁻  = 3.32,
                   Cᵟu   = 0.296,
                   Cᵟc   = 1.32,
                   Cᵟe   = 1.49,
                   CᵂwΔ  = 4.74,
                   Cᵂu★  = 2.76,
                   Cᴰ    = 1.78)

function perturbation_prior(θ★, ϵ=0.5)
    L = (1 - ϵ) * θ★
    U = (1 + ϵ) * θ★
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

field_names = (:u, :v, :T, :e)
regrid_size = nothing
observations = [SyntheticObservations(case_path(case); field_names, regrid_size)
                for case in cases]

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

Nensemble = 20

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

simulation.output_writers[:fields] =
    JLD2OutputWriter(model, merge(model.velocities, model.tracers, model.diffusivity_fields),
                     prefix = "catke_simulation",
                     schedule = SpecifiedTimes(round.(observation_times(obs))...),
                     force = true)

θ_perturbed = [NamedTuple(name => rand(priors[name]) for name in keys(priors)) for k=1:Nensemble-1]
θ = [θ★, θ_perturbed...]

forward_run!(calibration, θ; suppress=false)

output_path = simulation.output_writers[:fields].filepath
u = FieldTimeSeries(output_path, "u")
v = FieldTimeSeries(output_path, "v")
T = FieldTimeSeries(output_path, "T")
e = FieldTimeSeries(output_path, "e")

z = znodes(u)

fig = Figure(resolution=(1200, 1200))

ax_T = [Axis(fig[1, i], xlabel = "Temperature (ᵒC)", ylabel = "z (m)") for i = 1:2]
ax_u = [Axis(fig[2, i], xlabel = "Velocity (m s⁻¹)", ylabel = "z (m)") for i = 1:2]
ax_e = [Axis(fig[3, i], xlabel = "Turbulent Kinetic Energy (m² s⁻²)", ylabel = "z (m)") for i = 1:2]

Nt = minimum(length(obs.times) for obs in observations) - 5
slider = Slider(fig[0, :], range=1:Nt, horizontal=true, startvalue=1)
n = slider.value

profile_max(u, i, n) = Point2f.(maximum(interior(u[n])[:, i, :], dims=1)[:], z)
profile_min(u, i, n) = Point2f.(minimum(interior(u[n])[:, i, :], dims=1)[:], z)

for i = 1:5
    u★ = @lift interior(u[$n])[1, i, :]
    v★ = @lift interior(v[$n])[1, i, :]
    T★ = @lift interior(T[$n])[1, i, :]
    e★ = @lift interior(e[$n])[1, i, :]

    u_max = @lift profile_max(u, i, $n)
    v_max = @lift profile_max(v, i, $n)
    T_max = @lift profile_max(T, i, $n)
    e_max = @lift profile_max(e, i, $n)

    u_min = @lift profile_min(u, i, $n)
    v_min = @lift profile_min(v, i, $n)
    T_min = @lift profile_min(T, i, $n)
    e_min = @lift profile_min(e, i, $n)

    obs = observations[i]
    u_truth = @lift interior(obs.field_time_serieses.u[$n])[1, 1, :]
    v_truth = @lift interior(obs.field_time_serieses.v[$n])[1, 1, :]
    T_truth = @lift interior(obs.field_time_serieses.T[$n])[1, 1, :]
    e_truth = @lift interior(obs.field_time_serieses.e[$n])[1, 1, :]

              lines!(ax_T[i], T_truth, z, color=(:gray23, 0.6), linewidth=5,   label="LES")
              lines!(ax_u[i], u_truth, z, color=(:gray23, 0.6), linewidth=5,   label="u, LES")
    i == 1 && lines!(ax_u[i], v_truth, z, color=(:gray23, 0.6), linewidth=1.5, label="v, LES")
              lines!(ax_e[i], e_truth, z, color=(:gray23, 0.6), linewidth=5,   label="LES")

    color = (:pink, 0.4)
    band!(ax_T[i], T_max, T_min; color)
    band!(ax_e[i], e_max, e_min; color)
    band!(ax_u[i], u_max, u_min; color)
    i == 1 && band!(ax_u[i], v_max, v_min; color)

              lines!(ax_T[i], T★, z, color=:purple1, linewidth=3,   label="CATKE")
              lines!(ax_u[i], u★, z, color=:purple1, linewidth=3,   label="u, CATKE")
    i == 1 && lines!(ax_u[i], v★, z, color=:purple1, linewidth=1.5, label="v, CATKE")
              lines!(ax_e[i], e★, z, color=:purple1, linewidth=3,   label="CATKE")
end

for i = 1:2
    xlims!(ax_T[i], 19.6, 20.05)
    xlims!(ax_u[i], -0.6, 0.6)
    xlims!(ax_e[i], -0.0005, 0.005)
    axislegend(ax_T[i], position=:rb)
    axislegend(ax_u[i], position=:rb)
    axislegend(ax_e[i], position=:rb)
end

display(fig)

