#!/bin/bash

name=weak_wind_strong_cooling
case=three_layer_constant_fluxes_linear_hr48_Qu3.3e-04_Qb1.1e-07_f1.0e-04_Nh256_Nz256_weak_wind_strong_cooling
tartarusdir=/home/greg/Projects/LESbrary.jl/idealized/data

xyname=weak_wind_strong_cooling_xy_slice
xzname=weak_wind_strong_cooling_xz_slice
yzname=weak_wind_strong_cooling_yz_slice
stname=weak_wind_strong_cooling_instantaneous_statistics

scp tartarus:$tartarusdir/$case/$xyname.jld2 ./
scp tartarus:$tartarusdir/$case/$xzname.jld2 ./
scp tartarus:$tartarusdir/$case/$yzname.jld2 ./
scp tartarus:$tartarusdir/$case/$stname.jld2 ./
