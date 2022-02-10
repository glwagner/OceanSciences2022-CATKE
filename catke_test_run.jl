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

θ_CA_constant_Pr = (CᴷRiʷ = 0.0,
                    CᴷRiᶜ = 0.0,
                    Cᴷu⁻ = 0.942,
                    Cᴷc⁻ = 0.738,
                    Cᴷe⁻ = 0.26,
                    Cᴷuʳ = 0.0,
                    Cᴷcʳ = 0.0,
                    Cᴷeʳ = 0.0,
                    Cᵟu  = 0.5,
                    Cᵟc  = 0.642,
                    Cᵟe  = 5.0,
                    Cᴬu  = 0.0,
                    Cᴬc  = 0.0,
                    Cᴬe  = 0.186,
                    Cᴰ   = 2.97,
                    Cᴸᵇ  = 1.62,
                    Cᵂu★ = 0.95,
                    CᵂwΔ = 2.51)

θ_constant_Pr = (
                 CᵂwΔ  = 8.38,
                 Cᵂu★  = 9.06,
                 Cᴰ    = 1.37,
                 Cᴸᵇ   = 1.13,
                 Cᴷu⁻  = 0.0887,
                 Cᴷc⁻  = 0.164,
                 Cᴷe⁻  = 2.13,
                 Cᴷuʳ  = 0.0,
                 Cᴷcʳ  = 0.0,
                 Cᴷeʳ  = 0.0,
                 CᴷRiʷ = 0.0,
                 CᴷRiᶜ = 0.0,
                 Cᴬu   = 0.0,
                 Cᴬc   = 0.0,
                 Cᴬe   = 0.0,
                 Cᵟu   = 0.5,
                 Cᵟc   = 0.5,
                 Cᵟe   = 0.5,
                )

θ_variable_Pr = (
                 CᵂwΔ  = 8.20,
                 Cᵂu★  = 6.77,
                 Cᴰ    = 1.69,
                 Cᴸᵇ   = 1.15,
                 Cᴷu⁻  = 0.0924,
                 Cᴷc⁻  = 0.288,
                 Cᴷe⁻  = 2.46,
                 Cᴷuʳ  = 1.32,
                 Cᴷcʳ  = 0.0404,
                 Cᴷeʳ  = 4.27,
                 CᴷRiʷ = 3.15,
                 CᴷRiᶜ = 2.08,
                 Cᴬu   = 0.0,
                 Cᴬc   = 0.0,
                 Cᴬe   = 0.0,
                 Cᵟu   = 0.5,
                 Cᵟc   = 0.5,
                 Cᵟe   = 0.5,
                )

Cᴷu⁻ = 0.2739
Cᴷu⁺ = 1.8625
Cᴷc⁻ = 0.2573
Cᴷc⁺ = 1.4961
Cᴷe⁻ = 0.2941
Cᴷe⁺ = 1.3852

# 1. C⁻ + C⁺ * step
# 2. C⁻ (1 + Cʳ * step)
# => Cʳ = C⁺ / C⁻
Cᴷuʳ = Cᴷu⁺ / Cᴷu⁻
Cᴷcʳ = Cᴷc⁺ / Cᴷc⁻
Cᴷeʳ = Cᴷe⁺ / Cᴷe⁻

# 1. x/w - c/w
# 2. x/w - c
# => c = w*c
CᴷRiʷ = 0.6487

θ_variable_Ri = (CᴷRiʷ = 0.6487,
                 CᴷRiᶜ = 1.9231 * CᴷRiʷ,
                 Cᴷu⁻ = 0.2739,
                 Cᴷuʳ = Cᴷu⁺ / Cᴷu⁻,
                 Cᴷc⁻ = 0.2573,
                 Cᴷcʳ = Cᴷc⁺ / Cᴷc⁻,
                 Cᴷe⁻ = 0.2941,
                 Cᴷeʳ = Cᴷe⁺ / Cᴷe⁻,
                 Cᵟu  = 0.5,
                 Cᵟc  = 0.5,
                 Cᵟe  = 0.5,
                 Cᴰ   = 3.0376,
                 Cᴸᵇ  = 1.6998,
                 Cᵂu★ = 3.5992,
                 CᵂwΔ = 1.8507)

θ_constant_Ri = (; Cᴸᵇ   = 1.36,
                   Cᴷu⁻  = 0.101,
                   Cᴷc⁻  = 0.0574,
                   Cᴷe⁻  = 3.32,
                   Cᵟu   = 0.296,
                   Cᵟc   = 1.32,
                   Cᵟe   = 1.49,
                   CᵂwΔ  = 4.74,
                   Cᵂu★  = 2.76,
                   Cᴰ    = 1.78,
                   Cᴷuʳ  = 0.0,
                   Cᴷcʳ  = 0.0,
                   Cᴷeʳ  = 0.0,
                   CᴷRiʷ = 0.0,
                   CᴷRiᶜ = 0.0)

mixing_length = MixingLength(Cᴬu   = 0.0,
                             Cᴬc   = 0.0,
                             Cᴬe   = 0.0,
                             CᴷRiᶜ = 2.0,
                             Cᵟu   = 0.5,
                             Cᵟc   = 0.5,
                             Cᵟe   = 0.5,
                             Cᴷuʳ  = 0.0,
                             Cᴷcʳ  = 0.0,
                             Cᴷeʳ  = 0.0)

catke = CATKEVerticalDiffusivity(; mixing_length)

function perturbation_prior(θ★, ϵ=0.5)
    L = (1 - ϵ) * θ★
    U = (1 + ϵ) * θ★
    return ScaledLogitNormal(bounds=(L, U))
end

priors = NamedTuple(name => perturbation_prior(θ_constant_Ri[name]) for name in keys(θ_variable_Ri))
free_parameters = FreeParameters(priors)

case_path(case) = @datadep_str("six_day_suite_4m/$(case)_instantaneous_statistics.jld2")

cases = ["free_convection",
         "weak_wind_strong_cooling",
         "strong_wind_weak_cooling",
         "strong_wind",
         "strong_wind_no_rotation"]

field_names = (:u, :v, :T, :e)
regrid_size = nothing
observations = [SyntheticObservations(case_path(case); field_names, regrid_size)
                for case in cases]

observation = first(observations)
α = observation.metadata.parameters.thermal_expansion_coefficient
equation_of_state = LinearEquationOfState(; α)

Nensemble = 3
buoyancy = SeawaterBuoyancy(; equation_of_state, constant_salinity=true)

simulation = ensemble_column_model_simulation(observations;
                                              Nensemble,
                                              buoyancy,
                                              architecture = CPU(),
                                              tracers = (:T, :e),
                                              closure = catke)

simulation.Δt = 1.0
simulation.stop_time = 2days

progress(sim) = @info "Iter: $(iteration(sim)), time: $(prettytime(sim))"
simulation.callbacks[:progress] = Callback(progress, IterationInterval(100))

model = simulation.model
simulation.output_writers[:fields] =
    JLD2OutputWriter(model, merge(model.velocities, model.tracers, model.diffusivity_fields),
                     prefix = "catke_test_simulation",
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

forward_run!(calibration, [θ_variable_Ri, θ_constant_Ri]; suppress=false)

output_path = simulation.output_writers[:fields].filepath
u = FieldTimeSeries(output_path, "u")
v = FieldTimeSeries(output_path, "v")
T = FieldTimeSeries(output_path, "T")
e = FieldTimeSeries(output_path, "e")

z = znodes(u)

fig = Figure(resolution=(1200, 800))

ax_T = [Axis(fig[2, j], ylabel = "z (m)") for j = 1:5]
ax_u = [Axis(fig[3, j], ylabel = "z (m)") for j = 1:5]
ax_e = [Axis(fig[4, j], ylabel = "z (m)") for j = 1:5]

[Label(fig[1, j], replace(cases[j], "_" => " ")) for j=1:length(cases)]

Label(fig[2, 6], "Temperature", tellheight=false)
Label(fig[3, 6], "Velocity", tellheight=false)
Label(fig[4, 6], "Turbulent \n Kinetic \n Energy \n", tellheight=false)

Nt = length(u.times)
slider = Slider(fig[0, :], range=1:Nt, horizontal=true, startvalue=1)
n = slider.value

for j = 1:length(observations)
    obs = observations[j]
    u_truth = @lift interior(obs.field_time_serieses.u[$n])[1, 1, :]
    v_truth = @lift interior(obs.field_time_serieses.v[$n])[1, 1, :]
    T_truth = @lift interior(obs.field_time_serieses.T[$n])[1, 1, :]
    e_truth = @lift interior(obs.field_time_serieses.e[$n])[1, 1, :]

    lines!(ax_T[j], T_truth, z, color=(:gray23, 0.4), linewidth=6,   label="LES")
    lines!(ax_u[j], u_truth, z, color=(:gray23, 0.4), linewidth=6,   label="u, LES")
    lines!(ax_u[j], v_truth, z, color=(:gray23, 0.4), linewidth=2, label="v, LES")
    lines!(ax_e[j], e_truth, z, color=(:gray23, 0.4), linewidth=6,   label="LES")

    i = 1
    #for i = 1:Nensemble
        u★ = @lift interior(u[$n])[i, j, :]
        v★ = @lift interior(v[$n])[i, j, :]
        T★ = @lift interior(T[$n])[i, j, :]
        e★ = @lift interior(e[$n])[i, j, :]

        lines!(ax_T[j], T★, z, color=:purple1, linewidth=2,   label="$i CATKE")
        lines!(ax_u[j], u★, z, color=:purple1, linewidth=2,   label="u, $i CATKE")
        lines!(ax_u[j], v★, z, color=:purple1, linewidth=1, label="v, $i CATKE")
        lines!(ax_e[j], e★, z, color=:purple1, linewidth=2,   label="$i CATKE")
   #end
end

for j = 1:length(observations)

    if j == 1
        hidespines!(ax_T[j], :t, :r, :b)
        hidespines!(ax_u[j], :t, :r, :b)
        hidespines!(ax_e[j], :t, :r, :b)
        hidexdecorations!(ax_T[j])
        hidexdecorations!(ax_u[j])
        hidexdecorations!(ax_e[j])
        ax_T[j].ygridvisible[] = false
        ax_u[j].ygridvisible[] = false
        ax_e[j].ygridvisible[] = false
    else
        hidedecorations!(ax_T[j])
        hidedecorations!(ax_u[j])
        hidedecorations!(ax_e[j])
        hidespines!(ax_T[j])
        hidespines!(ax_u[j])
        hidespines!(ax_e[j])
    end

    xlims!(ax_T[j], 19.6, 20.05)
    xlims!(ax_u[j], -0.6, 0.6)
    xlims!(ax_e[j], -0.01, 0.01)
    #axislegend(ax_T[j], position=:rb)
    #axislegend(ax_u[j], position=:rb)
    #axislegend(ax_e[j], position=:rb)
end

display(fig)
