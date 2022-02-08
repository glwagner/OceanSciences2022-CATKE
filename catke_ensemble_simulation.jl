using Oceananigans
using OceanTurbulenceParameterEstimation
using Distributions
using DataDeps

prior_library = Dict()
prior_library[:Cᴰ]    = ScaledLogitNormal(bounds=(1, 3), interval=(1.8, 1.9), mass=0.8)
prior_library[:CᵂwΔ]  = ScaledLogitNormal(bounds=(3, 7), interval=(4.7, 4.8), mass=0.8)
prior_library[:Cᵂu★]  = ScaledLogitNormal(bounds=(2, 5), interval=(2.7, 2.9), mass=0.8)
prior_library[:Cᴸᵇ]   = ScaledLogitNormal(bounds=(0, 3), interval=(1.9, 2.1), mass=0.8)

prior_library[:Cᴷu⁻]  = ScaledLogitNormal(bounds=(0, 0.5), interval=(0.15, 0.25), mass=0.8)
prior_library[:Cᴷc⁻]  = ScaledLogitNormal(bounds=(0, 0.5), interval=(0.1, 0.2), mass=0.8)
prior_library[:Cᴷe⁻]  = ScaledLogitNormal(bounds=(0, 5), interval=(3.2, 3.6), mass=0.8)

prior_library[:Cᵟu]  = ScaledLogitNormal(bounds=(0, 2), interval=(0.3, 0.4), mass=0.8)
prior_library[:Cᵟc]  = ScaledLogitNormal(bounds=(0, 2), interval=(1.1, 1.4), mass=0.8)
prior_library[:Cᵟe]  = ScaledLogitNormal(bounds=(0, 4), interval=(1.2, 1.6), mass=0.8)

prior_library[:Cᴷuʳ]  = Normal(-1.5, 0.5)
prior_library[:Cᴷcʳ]  = Normal(-1.5, 0.5)
prior_library[:Cᴷeʳ]  = Normal(-0.6, 0.2)

prior_library[:CᴷRiʷ] = ScaledLogitNormal(bounds=(0.0, 2.0), interval=(0.4, 0.6), mass=0.8)
prior_library[:CᴷRiᶜ] = Normal(0.3, 0.1)

prior_library[:Cᴬu]   = ScaledLogitNormal(bounds=(0, 0.1))
prior_library[:Cᴬc]   = ScaledLogitNormal(bounds=(0, 10))
prior_library[:Cᴬe]   = ScaledLogitNormal(bounds=(0, 0.1))


# No convective adjustment:
constant_Ri_parameters = (:Cᴰ, :CᵂwΔ, :Cᵂu★, :Cᴸᵇ, :Cᴷu⁻, :Cᴷc⁻, :Cᴷe⁻, :Cᵟu, :Cᵟc, :Cᵟe)
variable_Ri_parameters = tuple(constant_Ri_parameters..., :Cᴷuʳ, :Cᴷcʳ, :Cᴷeʳ, :CᴷRiʷ, :CᴷRiᶜ)
constant_Ri_convective_adjustment_parameters = tuple(constant_Ri_parameters..., :Cᴬu, :Cᴬc, :Cᴬe)
variable_Ri_convective_adjustment_parameters = tuple(variable_Ri_parameters..., :Cᴬu, :Cᴬc, :Cᴬe)

free_parameters = FreeParameters(prior_library, names=variable_Ri_parameters)

θ_variable_Ri = (Cᴰ    = 1.846,
                 CᵂwΔ  = 4.753,
                 Cᵂu★  = 2.758,
                 Cᴸᵇ   = 2.05,
                 Cᴷu⁻  = 0.2198,
                 Cᴷc⁻  = 0.7627,
                 Cᴷe⁻  = 3.617,
                 Cᵟu   = 0.3256,
                 Cᵟc   = 1.312,
                 Cᵟe   = 1.184,
                 Cᴷuʳ  = -1.239,
                 Cᴷcʳ  = -0.8775,
                 Cᴷeʳ  = -0.2499,
                 CᴷRiʷ = 0.5530,
                 CᴷRiᶜ = 0.1175)

data_path = datadep"two_day_suite_1m/weak_wind_strong_cooling_instantaneous_statistics.jld2"

error("hi")
observations = SyntheticObservations(data_path, field_names=(:u, :v, :T, :e))

catke = CATKEVerticalDiffusivity()

simulation = ensemble_column_model_simulation(observations;
                                              Nensemble = 10,
                                              architecture = CPU(),
                                              equation_of_state = LinearEquationOfState(; α),
                                              buoyancy = SeawaterBuoyancy(; equation_of_state, constant_salinity=true),
                                              tracers = (:T, :e),
                                              closure = catke)

simulation.Δt = 10.0

Qᵘ = simulation.model.velocities.u.boundary_conditions.top.condition
Qᵀ = simulation.model.tracers.T.boundary_conditions.top.condition
dTdz = simulation.model.tracers.T.boundary_conditions.bottom.condition

observations = [observations]
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

