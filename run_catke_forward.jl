using Oceananigans
using Oceananigans.Units
using OceanTurbulenceParameterEstimation
using Distributions
using DataDeps
using GLMakie

using Oceananigans.TurbulenceClosures.CATKEVerticalDiffusivities:
    CATKEVerticalDiffusivity,
    SurfaceTKEFlux,
    MixingLength

include("best_catke_parameters.jl")

θ = [θ_constant_Pr, θ_variable_Pr, θ_conv_variable_Pr]
Nensemble = length(θ)

function perturbation_prior(θ★, ϵ=0.1)
    L = (1 - ϵ) * θ★
    U = (1 + ϵ) * θ★
    return ScaledLogitNormal(bounds=(L, U))
end

θ★ = θ_constant_Pr
priors = NamedTuple(name => perturbation_prior(θ★[name]) for name in keys(θ★))
free_parameters = FreeParameters(priors)

suites = [
          "two_day_suite_1m", 
          "four_day_suite_1m", 
          "six_day_suite_1m", 
         ]

end_times = [
             2days,
             4days,
             6days
            ]

for (suite, end_time) in zip(suites, end_times)
    case_path(case) = @datadep_str("$suite/$(case)_instantaneous_statistics.jld2")

    cases = [
             "free_convection",
             "weak_wind_strong_cooling",
             "strong_wind_weak_cooling",
             "strong_wind",
             "strong_wind_no_rotation",
            ]

    Δz = 8
    times = [6hours, end_time]
    field_names = (:u, :v, :T, :e)
    regrid_size = (1, 1, Int(256/Δz))
    observations = [SyntheticObservations(case_path(case); field_names, regrid_size, times) for case in cases]
    obs = first(observations)
    α = obs.metadata.parameters.thermal_expansion_coefficient
    equation_of_state = LinearEquationOfState(; α)
    buoyancy = SeawaterBuoyancy(; equation_of_state, constant_salinity=true)

    mixing_length = MixingLength(Cᴬu   = 0.0,
                                 Cᴬc   = 0.0,
                                 Cᴬe   = 0.0,
                                 CᴷRiᶜ = 2.0,
                                 Cᴷuʳ  = 0.0,
                                 Cᴷcʳ  = 0.0,
                                 Cᴷeʳ  = 0.0)

    catke = CATKEVerticalDiffusivity(; mixing_length)

    simulation = ensemble_column_model_simulation(observations; Nensemble, buoyancy,
                                                  architecture = GPU(),
                                                  tracers = (:T, :e),
                                                  closure = catke)

    simulation.Δt = 10.0
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
                         prefix = string("catke_simulation_", suite[1:end-2], Δz, "m"),
                         schedule = SpecifiedTimes(times[1]:10minutes:end_time...),
                         force = true)

    @show simulation.output_writers[:fields]

    forward_run!(calibration, θ; suppress=false)
end

