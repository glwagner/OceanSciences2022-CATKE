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

dir = "two_day_suite"
filename = "free_convection"
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

function extract_slices(dir, filename; name="w")
    xz_filepath = joinpath(dir, filename * "_xz_slice.jld2")
    yz_filepath = joinpath(dir, filename * "_yz_slice.jld2")
    xy_filepath = joinpath(dir, filename * "_xy_slice.jld2")

    slices = []
    for (filepath, dims) in zip((yz_filepath, xz_filepath, xy_filepath), (1, 2, 3))
        file = jldopen(filepath)
        push!(slices, [chop(dropdims(file["timeseries/$name/$i"]; dims)) for i in iterations])
        close(file)
    end

    return (yz=slices[1], xz=slices[2], xy=slices[3])
end

#####
##### 3D visualization
#####

grid = RectilinearGrid(size=(Nx, Ny, Nz), halo=(Hx, Hy, Hz), x=(0, Lx), y=(0, Ly), z=(-Lz, 0))

e = extract_slices(dir, filename, name="e")
u = extract_slices(dir, filename, name="u")
w = extract_slices(dir, filename, name="w")
T = extract_slices(dir, filename, name="T")
ϵ = extract_slices(dir, filename, name="ϵ")

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

fig = Figure(resolution=(2600, 1000))

n = Observable(1)

###
### Vertical velocity
###

w_yz = @lift w.yz[$n]
w_xz = @lift w.xz[$n]
w_xy = @lift w.xy[$n]

e_yz = @lift e.yz[$n]
e_xz = @lift e.xz[$n]
e_xy = @lift e.xy[$n]

ϵ_yz = @lift log10.(ϵ.yz[$n])
ϵ_xz = @lift log10.(ϵ.xz[$n])
ϵ_xy = @lift log10.(ϵ.xy[$n])

u_yz = @lift u.yz[$n]
u_xz = @lift u.xz[$n]
u_xy = @lift u.xy[$n]

T_yz = @lift T.yz[$n]
T_xz = @lift T.xz[$n]
T_xy = @lift T.xy[$n]

colormap_T = :oslo
colormap_u = :balance
colormap_w = :balance
colormap_e = :solar
colormap_ϵ = :viridis

w_max = maximum(maximum.(abs, w.xz))
colorrange_w = (-w_max/2, w_max/2)

u_max = maximum(maximum.(abs, u.xz))
colorrange_u = (-u_max/2, u_max/2)

e_max = maximum(maximum.(abs, e.xz))
colorrange_e = (0, e_max/6)

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

function box!(ax, yz, xz, xy; colormap, colorrange)
    sfc = surface!(ax, x_xz, y_xz, z_xz; color=xz, colormap, colorrange)
          surface!(ax, x_yz, y_yz, z_yz; color=yz, colormap, colorrange)
          surface!(ax, x_xy, y_xy, z_xy; color=xy, colormap, colorrange)
    return sfc
end

ax_T = Axis3(fig[1, 1:4], aspect=:data, xlabel="x (m)", ylabel="y (m)", zlabel="z (m)", azimuth=0.86, elevation=0.38)
ax_e = Axis3(fig[1, 5:8], aspect=:data, xlabel="x (m)", ylabel="y (m)", zlabel="z (m)", azimuth=0.86, elevation=0.38)

sfc_e = box!(ax_e, e_yz, e_xz, e_xy, colormap=colormap_e, colorrange=colorrange_e)
sfc_T = box!(ax_T, T_yz, T_xz, T_xy, colormap=colormap_T, colorrange=colorrange_T)

cp_T = fig[2, 2:3] = Colorbar(fig, sfc_T, flipaxis=false, vertical=false, label="Temperature (ᵒC)")
cp_e = fig[2, 6:7] = Colorbar(fig, sfc_e, flipaxis=false, vertical=false, label="Turbulent kinetic energy (m² s⁻²)")

title = @lift "Ocean surface boundary layer at t = " * prettytime(times[$n])
lab = Label(fig[0, :], title, textsize=30)

record(fig, joinpath(dir, name * "_LES.mp4"), 1:Nt; framerate=12) do nn
    @info "Drawing frame $nn of $Nt..."
    n[] = nn
end

