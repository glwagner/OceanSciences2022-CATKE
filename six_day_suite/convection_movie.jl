using Oceananigans
using JLD2
using GLMakie
using Printf

xz_file = jldopen("free_convection_xz_slice.jld2")
yz_file = jldopen("free_convection_yz_slice.jld2")
xy_file = jldopen("free_convection_xy_slice.jld2")

Nx = xz_file["grid/Nx"]
Ny = xz_file["grid/Ny"]
Nz = xz_file["grid/Nz"]
Lx = xz_file["grid/Lx"]
Ly = xz_file["grid/Ly"]
Lz = xz_file["grid/Lz"]

grid = RectilinearGrid(size=(Nx, Ny, Nz), x=(0, Lx), y=(0, Ly), z=(-Lz, 0))

# Coordinates

x, y, z = nodes((Center, Center, Face), grid)

slice_height = -8
k_xy_slice = searchsortedfirst(grid.zF[:], slice_height)

z = z[1:k_xy_slice]
Nz′ = length(z)

x_xz = repeat(x, 1, Nz′)
y_xz = 511 * ones(grid.Nx, Nz′)
z_xz = repeat(reshape(z, 1, Nz′), grid.Nx, 1)

x_yz = 511 * ones(grid.Ny, Nz′)
y_yz = repeat(y, 1, Nz′)
z_yz = repeat(reshape(z, 1, Nz′), grid.Ny, 1)

# Slight displacements to "stitch" the cube together
x_xy = x
y_xy = y
z_xy = (slice_height - 0.5) * ones(grid.Nx, grid.Ny)

# Analyze

iterations = parse.(Int, keys(xz_file["timeseries/t"]))

w_max = maximum([maximum(abs, xz_file["timeseries/w/$iter"]) for iter in iterations])
@show w_max

w_lim = 0.5 * w_max

# Animate

iter = Node(0)

wxz = @lift reverse(xz_file["timeseries/w/" * string($iter)][:, 1, 1:k_xy_slice], dims=1)
wyz = @lift reverse(yz_file["timeseries/w/" * string($iter)][1, :, 1:k_xy_slice], dims=1)
wxy = @lift rot180(xy_file["timeseries/w/" * string($iter)][:, :, 1])

fig = Figure(resolution=(1000, 800))

ax = fig[1:10, 1:3] = LScene(fig, aspect=:data, xlabel="x (meters)", ylabel = "y (meters)", zlabel = "height (meters)")

# Three surfaces
pl = surface!(ax, x_xz, y_xz, z_xz, color=wxz, colormap=:bwr, colorrange=(-w_lim, w_lim))
surface!(ax, x_yz, y_yz, z_yz, color=wyz, colormap=:bwr, colorrange=(-w_lim, w_lim))
surface!(ax, x_xy, y_xy, z_xy, color=wxy, colormap=:bwr, colorrange=(-w_lim, w_lim))

# Colorbar
cb = fig[3:8, 4] = Colorbar(fig, pl) #, label="Vertical velocity", height=Relative(0.5))

# Title
title_str = @lift begin
    primitive_title = @sprintf("t = %-16s", prettytime(xz_file["timeseries/t/" * string($iter)]))
    @sprintf("% 52s", primitive_title)
end

title_label = fig[1, 1:3] = Label(fig, title_str, textsize=30)

display(fig)

record(fig, "convection.mp4", iterations; framerate = 16) do i
    iter[] = i
end
