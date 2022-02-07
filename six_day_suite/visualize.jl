using Oceananigans

using Oceananigans.TurbulenceClosures.CATKEVerticalDiffusivities:
    CATKEVerticalDiffusivity,
    MixingLength

using OceanTurbulenceParameterEstimation.EnsembleSimulations: ensemble_column_model_simulation
using OceanTurbulenceParameterEstimation.Observations: FieldTimeSeriesCollector
using JLD2
using GLMakie
using Printf
using Statistics

name = "weak_wind_strong_cooling"
xz_filepath = name * "_xz_slice.jld2"
yz_filepath = name * "_yz_slice.jld2"
xy_filepath = name * "_xy_slice.jld2"
statistics_filepath = name * "_instantaneous_statistics.jld2"

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

# column_ensemble_size = ColumnEnsembleSize(Nz=Nz, ensemble=(1, 1))
# column_ensemble_halo_size = ColumnEnsembleSize(Nz=0, Hz=Hz)

one_dimensional_grid = RectilinearGrid(size=Nz, topology=(Flat, Flat, Bounded), z=(-Lz, 0))
                           
boundary_conditions = (
    u = FieldBoundaryConditions(top = FluxBoundaryCondition(Qᵘ)),
    T = FieldBoundaryConditions(top = FluxBoundaryCondition(Qᵀ), bottom = GradientBoundaryCondition(dTdz))
)

coriolis = FPlane(; f)
equation_of_state = LinearEquationOfState(; α)
buoyancy = SeawaterBuoyancy(; equation_of_state, constant_salinity=true)
tracers = (:T, :e)

mixing_length = MixingLength(Cᴬu   = 0.0,
                             Cᴬc   = 1.0,
                             Cᴬe   = 0.0,
                             Cᴷu⁻  = 0.072,
                             Cᴷc⁻  = 0.088,
                             Cᴷe⁻  = 0.50,
                             Cᴷuʳ  = 1.0,
                             Cᴷcʳ  = 0.069,
                             Cᴷeʳ  = 0.5,
                             CᴷRiʷ = 0.05,
                             CᴷRiᶜ = 0.1)

surface_tke_flux = SurfaceTKEFlux(CᵂwΔ=4.99, Cᵂu★=2.5)

closure = CATKEVerticalDiffusivity(; Cᴰ=2.44, surface_tke_flux, mixing_length)

model = HydrostaticFreeSurfaceModel(; tracers, buoyancy, coriolis, closure,
                                    boundary_conditions, grid = one_dimensional_grid)

# Initial condition
model.tracers.T .= reshape(T_avg_series[1], 1, 1, Nz)

times = round.(times)
simulation = Simulation(model; Δt=2.0, stop_time=times[end])

collector = FieldTimeSeriesCollector(merge(model.velocities, model.tracers), times)
simulation.callbacks[:collector] = Callback(collector, SpecifiedTimes(times...))

progress(sim) = @info "Sim time: " * prettytime(sim) * ", wall time: " * prettytime(sim.run_wall_time)
simulation.callbacks[:progress] = Callback(progress, IterationInterval(100))

@info "Running 1D simulation..."
start = time_ns()
run!(simulation)
elapsed = 1e-9 * (time_ns() - start)
@info "    ... done (" * prettytime(elapsed) * ")."

#####
##### 3D visualization
#####

grid = RectilinearGrid(size=(Nx, Ny, Nz), halo=(Hx, Hy, Hz), x=(0, Lx), y=(0, Ly), z=(-Lz, 0))

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

fig = Figure(resolution=(1800, 600))

n = Observable(1)

###
### Vertical velocity
###

w_yz = @lift w_yz_series[$n]
w_xz = @lift w_xz_series[$n]
w_xy = @lift w_xy_series[$n]

w_max = maximum(maximum.(abs, w_xz_series[end]))

colormap = :balance
colorrange = (-w_max/2, w_max/2)

ax_w = Axis3(fig[1:7, 1:4], aspect=:data, xlabel="x (m)", ylabel="y (m)", zlabel="z (m)", azimuth=0.85, elevation=0.70)
ax_T = Axis3(fig[1:7, 5:8], aspect=:data, xlabel="x (m)", ylabel="y (m)", zlabel="z (m)", azimuth=0.86, elevation=0.38)

sfc_w = surface!(ax_w, x_xz, y_xz, z_xz; color=w_xz, colormap, colorrange)
        surface!(ax_w, x_yz, y_yz, z_yz; color=w_yz, colormap, colorrange)
        surface!(ax_w, x_xy, y_xy, z_xy; color=w_xy, colormap, colorrange)

###
### Temperature
###

T_yz = @lift T_yz_series[$n]
T_xz = @lift T_xz_series[$n]
T_xy = @lift T_xy_series[$n]

colormap = :oslo

# Compute maximum temperature for colorrange with a moving average
ΔT = 0.02 # Absolute range
Δn = 5 # Half-width of moving average
Tf_max = maximum(T_xy_series[end])
T0_max = maximum(T_xy_series[1])

colorrange = @lift begin
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

sfc_T = surface!(ax_T, x_xz, y_xz, z_xz; color=T_xz, colormap, colorrange)
        surface!(ax_T, x_yz, y_yz, z_yz; color=T_yz, colormap, colorrange)
        surface!(ax_T, x_xy, y_xy, z_xy; color=T_xy, colormap, colorrange)

cp = fig[8, 1:4] = Colorbar(fig, sfc_w, vertical=false, flipaxis=false, label="Vertical velocity (m s⁻¹)")
cp = fig[8, 5:8] = Colorbar(fig, sfc_T, vertical=false, flipaxis=false, label="Temperature (ᵒC)")

ax_T_avg = Axis(fig[2:6, 9:10], xlabel = "Average temperature (ᵒC)", ylabel = "z (m)")
ax_u_avg = Axis(fig[2:6, 11:12], xlabel = "Average velocity \n components (m s⁻¹)", ylabel = "z (m)")

u_model = @lift interior(collector.field_time_serieses.u[$n])[1, 1, :]
v_model = @lift interior(collector.field_time_serieses.v[$n])[1, 1, :]
T_model = @lift interior(collector.field_time_serieses.T[$n])[1, 1, :]

u_LES = @lift u_avg_series[$n]
v_LES = @lift v_avg_series[$n]
T_LES = @lift T_avg_series[$n]

T_model_lines = lines!(ax_T_avg, T_model, z, label="Calibrated CATKE model")
T_LES_lines = lines!(ax_T_avg, T_LES, z, label="Large eddy simulation data")

model_color = T_model_lines.attributes.color.val
LES_color = T_LES_lines.attributes.color.val

lines!(ax_u_avg, u_model, z, color=model_color, label="u")
lines!(ax_u_avg, v_model, z, color=model_color, linestyle=:dash, label="v")

lines!(ax_u_avg, u_LES, z, color=LES_color, label="u")
lines!(ax_u_avg, v_LES, z, color=LES_color, linestyle=:dash, label="v")

xlims!(ax_u_avg, -0.1, 0.1)

Legend(fig[7:8, 9:12], ax_T_avg)
axislegend(ax_u_avg, position=:rb, merge=true)

title = @lift "Ocean surface boundary layer at t = " * prettytime(times[$n])
lab = Label(fig[0, :], title, textsize=18)

record(fig, "$name.mp4", 1:Nt; framerate=16) do nn
    @info "Drawing frame $nn of $Nt..."
    n[] = nn
end

