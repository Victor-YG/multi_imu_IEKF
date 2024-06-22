clear all
clc

rng("default")

% camera calibration info

% imaging noise ~ 3 pixels stddev for 1000x1000 image
var_n_x = 1e-4;
var_n_y = 1e-4;
var_n_z = 1e-4;

% load trajectory data
load("data_trajectory.mat")

pos_obs_all = zeros(N, 3);  % pos_obs = (x, y, z)

for k = 1 : N
    r_vi_i  = r_vi_i_all(k, :);
    pos_obs = zeros(1, 3);
    pos_obs(1) = r_vi_i(1) + 2 * (rand() - 0.5) * sqrt(var_n_x);
    pos_obs(2) = r_vi_i(2) + 2 * (rand() - 0.5) * sqrt(var_n_y);
    pos_obs(3) = r_vi_i(3) + 2 * (rand() - 0.5) * sqrt(var_n_z);
    pos_obs_all(k, :) = pos_obs;
end

save("data_gps.mat", "pos_obs_all", "var_n_x", "var_n_y", "var_n_z");

disp("done generating GPS measurement");