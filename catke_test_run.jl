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

θ★ = θ_constant_Ri
priors = NamedTuple(name => perturbation_prior(θ_constant_Ri[name]) for name in keys(θ★))
free_parameters = FreeParameters(priors)

case_path(case) = @datadep_str("two_day_suite_1m/$(case)_instantaneous_statistics.jld2")

cases = ["free_convection",
         "weak_wind_strong_cooling",
         "strong_wind_weak_cooling",
         "strong_wind",
         "strong_wind_no_rotation"]

field_names = (:u, :v, :T, :e)
regrid_size = nothing
observation = SyntheticObservations(data_path; field_names, regrid_size)

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

Nensemble = 3

simulation = ensemble_column_model_simulation(observations;
                                              Nensemble,
                                              architecture = CPU(),
                                              buoyancy = SeawaterBuoyancy(; equation_of_state, constant_salinity=true),
                                              tracers = (:T, :e),
                                              closure = catke)

simulation.Δt = 2.0
simulation.stop_time = 3days

progress(sim) = @info "Iter: $(iteration(sim)), time: $(prettytime(sim))"
simulation.callbacks[:progress] = Callback(progress, IterationInterval(100))

model = simulation.model
simulation.output_writers[:fields] =
    JLD2OutputWriter(model, merge(model.velocities, model.tracers, model.diffusivity_fields),
                     prefix = "catke_simulation",
                     schedule = TimeInterval(10minutes),
                     force = true)

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
forward_run!(calibration, θ★; suppress=false)

output_path = simulation.output_writers[:fields].filepath
u = FieldTimeSeries(output_path, "u")
v = FieldTimeSeries(output_path, "v")
T = FieldTimeSeries(output_path, "T")
e = FieldTimeSeries(output_path, "e")

z = znodes(u)

fig = Figure(resolution=(2400, 1200))

ax_T = [Axis(fig[1, j], xlabel = "Temperature (ᵒC)", ylabel = "z (m)") for j = 1:5]
ax_u = [Axis(fig[2, j], xlabel = "Velocity (m s⁻¹)", ylabel = "z (m)") for j = 1:5]
ax_e = [Axis(fig[3, j], xlabel = "Turbulent Kinetic Energy (m² s⁻²)", ylabel = "z (m)") for j = 1:5]

Nt = length(u.times)
slider = Slider(fig[0, :], range=1:Nt, horizontal=true, startvalue=1)
n = slider.value

for j = 1:length(observations)
    for i = 1:Nensemble
        u★ = @lift interior(u[$n])[1, j, :]
        v★ = @lift interior(v[$n])[1, j, :]
        T★ = @lift interior(T[$n])[1, j, :]
        e★ = @lift interior(e[$n])[1, j, :]

                  lines!(ax_T[j], T★, z, color=:purple1, linestyle=:dash, linewidth=3,   label="$i CATKE")
                  lines!(ax_u[j], u★, z, color=:purple1, linestyle=:dash, linewidth=3,   label="u, $i CATKE")
        i == 3 && lines!(ax_u[j], v★, z, color=:purple1, linestyle=:dash, linewidth=1.5, label="v, $i CATKE")
                  lines!(ax_e[j], e★, z, color=:purple1, linestyle=:dash, linewidth=3,   label="$i CATKE")
   end
end

for j = 1:2
    xlims!(ax_T[j], 19.6, 20.05)
    xlims!(ax_u[j], -0.6, 0.6)
    xlims!(ax_e[j], -0.01, 0.01)
    axislegend(ax_T[j], position=:rb)
    axislegend(ax_u[j], position=:rb)
    axislegend(ax_e[j], position=:rb)
end

display(fig)
