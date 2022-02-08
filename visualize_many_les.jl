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

    all_iterations = []
    all_times = []
    slices = []
    for (filepath, dims) in zip((yz_filepath, xz_filepath, xy_filepath), (1, 2, 3))
        file = jldopen(filepath)
        iterations = parse.(Int, keys(file["timeseries/t"]))
        push!(all_iterations, iterations)
        push!(all_times, [file["timeseries/t/$i"] for i in iterations])
        push!(slices, [chop(dropdims(file["timeseries/$name/$i"]; dims)) for i in iterations])
        close(file)
    end

    return (; yz=slices[1], xz=slices[2], xy=slices[3], iterations=all_iterations, times=all_times)
end

#####
##### 3D visualization
#####

grid = RectilinearGrid(size=(Nx, Ny, Nz), halo=(Hx, Hy, Hz), x=(0, Lx), y=(0, Ly), z=(-Lz, 0))

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

function box!(ax, yz, xz, xy; colormap, colorrange)
    sfc = surface!(ax, x_xz, y_xz, z_xz; color=xz, colormap, colorrange)
          surface!(ax, x_yz, y_yz, z_yz; color=yz, colormap, colorrange)
          surface!(ax, x_xy, y_xy, z_xy; color=xy, colormap, colorrange)
    return sfc
end

fig = Figure(resolution=(1600, 1200))

#slider = Slider(fig[5, :], range=1:Nt, horizontal=true, startvalue=1)
#n = slider.value #Observable(1)
n = Observable(1)

#####
##### Animate!
#####

function plot_case!(case, i, ΔT=0.02, r_e_max=1/6)

    e = extract_slices(dir, case, name="e")
    T = extract_slices(dir, case, name="T")


    if isnothing(r_e_max)
        for i = 1:length(e.iterations)
            e_max = max(1e-6, maximum(e.xz[i]))
            e.xy[i] ./= e_max
            e.xz[i] ./= e_max
            e.yz[i] ./= e_max
        end
        colorrange_e = (0, 1/20)
    else
        e_max = maximum(maximum.(abs, e.xz))
        colorrange_e = (0, r_e_max * e_max)
    end

    e_yz = @lift e.yz[$n]
    e_xz = @lift e.xz[$n]
    e_xy = @lift e.xy[$n]

    T_yz = @lift T.yz[$n]
    T_xz = @lift T.xz[$n]
    T_xy = @lift T.xy[$n]

    colormap_T = :oslo
    colormap_e = :solar

    # Compute maximum temperature for colorrange with a moving average
    Δn = 2 # Half-width of moving average
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

    ax_T = Axis3(fig[2, 4i+1:4i+4], aspect=:data, xlabel="x (m)", ylabel="y (m)", zlabel="z (m)", azimuth=0.86, elevation=0.38)
    ax_e = Axis3(fig[3, 4i+1:4i+4], aspect=:data, xlabel="x (m)", ylabel="y (m)", zlabel="z (m)", azimuth=0.86, elevation=0.38)

    sfc_e = box!(ax_e, e_yz, e_xz, e_xy, colormap=colormap_e, colorrange=colorrange_e)
    sfc_T = box!(ax_T, T_yz, T_xz, T_xy, colormap=colormap_T, colorrange=colorrange_T)

    cp_T = fig[1, 4i+2:4i+3] = Colorbar(fig, sfc_T, flipaxis=true, vertical=false, label="Temperature (ᵒC)")
    cp_e = fig[4, 4i+2:4i+3] = Colorbar(fig, sfc_e, flipaxis=false, vertical=false, label="Turbulent kinetic energy (m² s⁻²)")

    return e, T
end

cases = ["free_convection",
         "weak_wind_strong_cooling",
         #"strong_wind_weak_cooling",
         #"strong_wind",
         #"strong_wind_no_rotation",
         ]

ΔT = [0.02, 0.02, 0.01, 0.02, 0.02]
re = [1/6 1/10 1/40]

for (i, case) in enumerate(cases)
    plot_case!(case, i-1, ΔT[i], re[i])
end

title = @lift @sprintf("Turbulent evolution of ocean surface boundary layers after %.1f hours", times[$n] / hours)
lab = Label(fig[0, :], title, textsize=30)

display(fig)

record(fig, joinpath(dir, "five_cases_LES.mp4"), 1:Nt; framerate=6) do nn
    @info "Drawing frame $nn of $Nt..."
    n[] = nn
end
