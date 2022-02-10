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

#dir = "two_day_suite/highres"
dir = "six_day_suite"
filename = "weak_wind_strong_cooling"
xy_filepath = joinpath(dir, filename * "_xy_slice.jld2")

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
@show Nt = length(times)

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

function extract_slices(dir, filename; name="w")
    xz_filepath = joinpath(dir, filename * "_xz_slice.jld2")
    yz_filepath = joinpath(dir, filename * "_yz_slice.jld2")
    xy_filepath = joinpath(dir, filename * "_xy_slice.jld2")
    statistics_filepath = joinpath(dir, filename * "_instantaneous_statistics.jld2")

    all_iterations = []
    slices = []
    for (filepath, dims) in zip((yz_filepath, xz_filepath, xy_filepath), (1, 2, 3))
        file = jldopen(filepath)
        iterations = parse.(Int, keys(file["timeseries/t"]))
        push!(slices, [chop(dropdims(file["timeseries/$name/$i"]; dims)) for i in iterations])
        push!(all_iterations, iterations)
        close(file)
    end

    file = jldopen(statistics_filepath)
    iterations = parse.(Int, keys(file["timeseries/t"]))
    push!(all_iterations, iterations)
    statistics = [chop(file["timeseries/$name/$i"][1, 1, :]) for i in iterations]
    close(file)

    return (; yz=slices[1], xz=slices[2], xy=slices[3], statistics, all_iterations)
end

#####
##### 3D visualization
#####

grid = RectilinearGrid(size=(Nx, Ny, Nz), halo=(Hx, Hy, Hz), x=(0, Lx), y=(0, Ly), z=(-Lz, 0))

e = extract_slices(dir, filename, name="e")
u = extract_slices(dir, filename, name="u")
v = extract_slices(dir, filename, name="v")
T = extract_slices(dir, filename, name="T")

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

fig = Figure(resolution=(1800, 800))

azimuth = 0.86
elevation = 0.38
perspectiveness = 0.6 
aspect = :data
xlabel = "x (m)"
ylabel = "y (m)"
zlabel = "z (m)"

row = 1:6
bottom = row[end] + 2
ax_e = Axis3(fig[row, 1:4]; aspect, xlabel, ylabel, zlabel, azimuth, elevation, perspectiveness)
ax_T = Axis3(fig[row, 5:8]; aspect, xlabel, ylabel, zlabel, azimuth, elevation, perspectiveness)

ax_T_avg = Axis(fig[3:6, 9:10], xlabel="Temperature (ᵒC)", ylabel="z (m)")
ax_u_avg = Axis(fig[3:6, 11:12], xlabel="Velocity components (m s⁻¹)", ylabel="z (m)")

slider = Slider(fig[bottom+2, :], range=1:Nt, startvalue=1)
n = slider.value #Observable(1)

title = @lift @sprintf("Ocean surface boundary layer turbulence forced by cooling and light winds for %.1f hours",
                       times[$n] / hour)
lab = Label(fig[0, :], title, textsize=24)

###
### Vertical velocity
###

e_yz = @lift e.yz[$n]
e_xz = @lift e.xz[$n]
e_xy = @lift e.xy[$n]

T_yz = @lift T.yz[$n]
T_xz = @lift T.xz[$n]
T_xy = @lift T.xy[$n]

T_avg = @lift T.statistics[$n]
e_avg = @lift e.statistics[$n]
u_avg = @lift u.statistics[$n]
v_avg = @lift v.statistics[$n]

colormap_T = :oslo
colormap_e = :solar

e_max = maximum(maximum.(abs, e.xz))
colorrange_e = (0, e_max/20)

# Compute maximum temperature for colorrange with a moving average
ΔT = 0.04 # Absolute range
Δn = 20 # Half-width of moving average
Tf_max = maximum(T.xy[end])
T0_max = maximum(T.xy[1])

colorrange_T = @lift begin
    if $n < Δn + 1
        T_max = T0_max
    elseif $n < Nt - Δn
        T_max = mean([maximum(T.xy[nn]) for nn in $n-Δn:$n+Δn])
    else
        T_max = Tf_max
    end

    T_min = T_max - ΔT

    (T_min, T_max)
end

function box!(ax, yz, xz, xy; colormap, colorrange)
    sfc = surface!(ax, x_xz, y_xz, z_xz; color=xz, colormap, colorrange)
          surface!(ax, x_yz, y_yz, z_yz; color=yz, colormap, colorrange)
          surface!(ax, x_xy, y_xy, z_xy; color=xy, colormap, colorrange)
    return sfc
end

sfc_e = box!(ax_e, e_yz, e_xz, e_xy, colormap=colormap_e, colorrange=colorrange_e)
sfc_T = box!(ax_T, T_yz, T_xz, T_xy, colormap=colormap_T, colorrange=colorrange_T)

cp_e = fig[bottom, 2:3] = Colorbar(fig, sfc_e, flipaxis=false, vertical=false, label="Turbulent kinetic energy (m² s⁻²)")
cp_T = fig[bottom, 6:7] = Colorbar(fig, sfc_T, flipaxis=false, vertical=false, label="Temperature (ᵒC)")

lines!(ax_T_avg, T_avg, z, linewidth=5, color=(:royalblue1, 0.6))

lines!(ax_u_avg, u_avg, z, linewidth=5, color=(:royalblue1, 0.6), label="u")
lines!(ax_u_avg, v_avg, z, linewidth=3, color=(:orange, 0.9), label="v")
xlims!(ax_u_avg, -0.2, 0.3)
axislegend(ax_u_avg, position=:rb)

hideydecorations!(ax_u_avg, grid=false)
hidespines!(ax_u_avg, :r, :t, :l)
hidespines!(ax_T_avg, :r, :t)

colgap!(fig.layout, Relative(0))
colgap!(fig.layout, 10, Relative(0.03))
rowgap!(fig.layout, Relative(0))

display(fig)

record(fig, joinpath(dir, filename * "_one_row_intro.mp4"), 1:Nt; framerate=16) do nn
    @info "Drawing frame $nn of $Nt..."
    n[] = nn
end

