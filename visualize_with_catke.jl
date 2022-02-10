using Oceananigans
using Oceananigans.Units

using Oceananigans.TurbulenceClosures.CATKEVerticalDiffusivities:
    CATKEVerticalDiffusivity,
    SurfaceTKEFlux,
    MixingLength

using OceanTurbulenceParameterEstimation.Observations: FieldTimeSeriesCollector
using JLD2
using GLMakie
using Printf
using Statistics

dir = "six_day_suite"
name = "weak_wind_strong_cooling"
xz_filepath = joinpath(dir, name * "_xz_slice.jld2")
yz_filepath = joinpath(dir, name * "_yz_slice.jld2")
xy_filepath = joinpath(dir, name * "_xy_slice.jld2")
statistics_filepath = joinpath(dir, name * "_instantaneous_statistics.jld2")

#####
##### Metadata
#####

xy_file = jldopen(xy_filepath)

Nx = xy_file["grid/Nx"]
Ny = xy_file["grid/Ny"]
Nz = xy_file["grid/Nz"]
Hx = xy_file["grid/Hx"]
Hy = xy_file["grid/Hy"]
Hz = xy_file["grid/Hz"]
Lx = xy_file["grid/Lx"]
Ly = xy_file["grid/Ly"]
Lz = xy_file["grid/Lz"]

iterations = parse.(Int, keys(xy_file["timeseries/t"]))
times = [xy_file["timeseries/t/$i"] for i in iterations]
Nt = length(times)

Qᵀ = xy_file["parameters/temperature_flux"]
Qᵘ = xy_file["parameters/momentum_flux"]
f  = xy_file["parameters/coriolis_parameter"]
α  = xy_file["parameters/thermal_expansion_coefficient"]
f  = xy_file["parameters/coriolis_parameter"]
dTdz = xy_file["parameters/dθdz_deep"]

close(xy_file)

H = Hz
chop(a::AbstractArray{<:Any, 1}) = a[1+H:end-H]
chop(a::AbstractArray{<:Any, 2}) = a[1+H:end-H, 1+H:end-H]

function extract_slices(filepath; dims, name="w")
    file = jldopen(filepath)
    slices = [chop(dropdims(file["timeseries/$name/$i"]; dims)) for i in iterations]
    close(file)
    return slices
end

#####
##### Construct 1D model
#####

# Data
u_avg_series = extract_slices(statistics_filepath, dims=(1, 2), name="u")
v_avg_series = extract_slices(statistics_filepath, dims=(1, 2), name="v")
T_avg_series = extract_slices(statistics_filepath, dims=(1, 2), name="T")

observations = SyntheticObservations(statistics_filepath, field_names=(:u, :v, :T, :e))

catke_mixing_length = MixingLength(Cᴬu=0.0, Cᴬc=0.0, Cᴬe=0.0,
                                   Cᴷuʳ=0.0, Cᴷcʳ=0.0, Cᴷeʳ=0.0)

catke = CATKEVerticalDiffusivity(mixing_length=catke_mixing_length)

simulation = ensemble_column_model_simulation(observations;
                                              Nensemble = 100,
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

set!(simulation.model, observations)

times = round.(times)
collector = FieldTimeSeriesCollector(merge(model.velocities, model.tracers), times)
simulation.callbacks[:collector] = Callback(collector, SpecifiedTimes(times...))

progress(sim) = @info "Sim time: " * prettytime(sim) * ", wall time: " * prettytime(sim.run_wall_time)
simulation.callbacks[:progress] = Callback(progress, IterationInterval(100))

@info "Running 1D simulation..."
start = time_ns()
run!(simulation)
elapsed = 1e-9 * (time_ns() - start)
@info "    ... done (" * prettytime(elapsed) * ")."
=#

#####
##### 3D visualization
#####

grid = RectilinearGrid(size=(Nx, Ny, Nz), halo=(Hx, Hy, Hz), x=(0, Lx), y=(0, Ly), z=(-Lz, 0))

u_yz_series = extract_slices(yz_filepath, dims=1, name="u")
u_xz_series = extract_slices(xz_filepath, dims=2, name="u")
u_xy_series = extract_slices(xy_filepath, dims=3, name="u")

w_yz_series = extract_slices(yz_filepath, dims=1, name="w")
w_xz_series = extract_slices(xz_filepath, dims=2, name="w")
w_xy_series = extract_slices(xy_filepath, dims=3, name="w")

T_yz_series = extract_slices(yz_filepath, dims=1, name="T")
T_xz_series = extract_slices(xz_filepath, dims=2, name="T")
T_xy_series = extract_slices(xy_filepath, dims=3, name="T")

x, y, z = nodes((Center, Center, Center), grid)

x_xz = repeat(x, 1, Nz)
y_xz = 0.995 * Ly * ones(Nx, Nz)
z_xz = repeat(reshape(z, 1, Nz), Nx, 1)

x_yz = 0.995 * Lx * ones(Ny, Nz)
y_yz = repeat(y, 1, Nz)
z_yz = repeat(reshape(z, 1, Nz), grid.Ny, 1)

# Slight displacements to "stitch" the cube together
x_xy = x
y_xy = y
z_xy = - 0.001 * Lz * ones(grid.Nx, grid.Ny)

#####
##### Animate!
#####

fig = Figure(resolution=(1400, 1200))

n = Observable(1)

###
### Vertical velocity
###

w_yz = @lift w_yz_series[$n]
w_xz = @lift w_xz_series[$n]
w_xy = @lift w_xy_series[$n]

u_yz = @lift u_yz_series[$n]
u_xz = @lift u_xz_series[$n]
u_xy = @lift u_xy_series[$n]

T_yz = @lift T_yz_series[$n]
T_xz = @lift T_xz_series[$n]
T_xy = @lift T_xy_series[$n]

u_catke = @lift interior(collector.field_time_serieses.u[$n])[1, 1, :]
v_catke = @lift interior(collector.field_time_serieses.v[$n])[1, 1, :]
T_catke = @lift interior(collector.field_time_serieses.T[$n])[1, 1, :]

u_LES = @lift u_avg_series[$n]
v_LES = @lift v_avg_series[$n]
T_LES = @lift T_avg_series[$n]

colormap_T = :oslo
colormap_u = :balance
colormap_w = :balance

w_max = maximum(maximum.(abs, w_xz_series[end]))
colorrange_w = (-w_max/2, w_max/2)

u_max = maximum(maximum.(abs, u_xz_series[end]))
colorrange_u = (-u_max/2, u_max/2)

# Compute maximum temperature for colorrange with a moving average
ΔT = 0.04 # Absolute range
Δn = 2 # Half-width of moving average
Tf_max = maximum(T_xy_series[end])
T0_max = maximum(T_xy_series[1])

colorrange_T = @lift begin
    if $n < Δn + 1
        T_max = T0_max
    elseif $n < Nt - Δn
        T_max = mean([maximum(T_xy_series[nn]) for nn in $n-Δn:$n+Δn])
    else
        T_max = Tf_max
    end

    T_min = T_max - ΔT

    (T_min, T_max)
end

ax_T = Axis3(fig[1:8,  2], aspect=:data, xlabel="x (m)", ylabel="y (m)", zlabel="z (m)", azimuth=0.86, elevation=0.38)
ax_u = Axis3(fig[9:16, 2], aspect=:data, xlabel="x (m)", ylabel="y (m)", zlabel="z (m)", azimuth=0.86, elevation=0.38)

#sfc_w = surface!(ax_w, x_xz, y_xz, z_xz; color=w_xz, colormap=colormap_w, colorrange=colorrange_w)
#        surface!(ax_w, x_yz, y_yz, z_yz; color=w_yz, colormap=colormap_w, colorrange=colorrange_w)
#        surface!(ax_w, x_xy, y_xy, z_xy; color=w_xy, colormap=colormap_w, colorrange=colorrange_w)

sfc_u = surface!(ax_u, x_xz, y_xz, z_xz; color=u_xz, colormap=colormap_u, colorrange=colorrange_u)
        surface!(ax_u, x_yz, y_yz, z_yz; color=u_yz, colormap=colormap_u, colorrange=colorrange_u)
        surface!(ax_u, x_xy, y_xy, z_xy; color=u_xy, colormap=colormap_u, colorrange=colorrange_u)

sfc_T = surface!(ax_T, x_xz, y_xz, z_xz; color=T_xz, colormap=colormap_T, colorrange=colorrange_T)
        surface!(ax_T, x_yz, y_yz, z_yz; color=T_yz, colormap=colormap_T, colorrange=colorrange_T)
        surface!(ax_T, x_xy, y_xy, z_xy; color=T_xy, colormap=colormap_T, colorrange=colorrange_T)

cp_T = fig[3:6,   1] = Colorbar(fig, sfc_T, flipaxis=false, label="Temperature (ᵒC)")
cp_u = fig[11:14, 1] = Colorbar(fig, sfc_u, flipaxis=false, label="u (x-velocity component, m s⁻¹)")

ax_T_avg = Axis(fig[3:7, 3],
                xlabel = "Horizontally-averaged \n temperature (ᵒC)",
                ylabel = "z (m)",
                yaxisposition = :right)

ax_u_avg = Axis(fig[11:15, 3],
                xlabel = "Horizontally-averaged velocity \n components (m s⁻¹)",
                ylabel = "z (m)",
                yaxisposition = :right)

T_catke_lines = lines!(ax_T_avg, T_catke, z, label="CATKE turbulence model")
T_LES_lines = lines!(ax_T_avg, T_LES, z, label="Large eddy simulation data")

model_color = T_catke_lines.attributes.color.val
LES_color = T_LES_lines.attributes.color.val

lines!(ax_u_avg, u_catke, z, color=model_color, label="u")
lines!(ax_u_avg, v_catke, z, color=model_color, linestyle=:dash, label="v")

lines!(ax_u_avg, u_LES, z, color=LES_color, label="u")
lines!(ax_u_avg, v_LES, z, color=LES_color, linestyle=:dash, label="v")

xlims!(ax_u_avg, -0.2, 0.2)

Legend(fig[8:9, 3], ax_T_avg)
axislegend(ax_u_avg, position=:rb, merge=true)

title = @lift "Ocean surface boundary layer at t = " * prettytime(times[$n])
lab = Label(fig[0, :], title, textsize=18)

record(fig, joinpath(dir, "$name.mp4"), 1:Nt; framerate=12) do nn
    @info "Drawing frame $nn of $Nt..."
    n[] = nn
end

