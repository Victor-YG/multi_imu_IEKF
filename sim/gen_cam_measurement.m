clear all
clc

rng("default")

% camera calibration info
C_vc   = eye(3);
r_cv_v = zeros(1, 3);
fx = 1.0;
fy = 1.0;
cx = 0.5;
cy = 0.5;

% imaging noise ~ 3 pixels stddev for 1000x1000 image
var_n_u = 1e-5;
var_n_v = 1e-5;

% load kinematic data
load("data_kinematics.mat")

obs_all = zeros(N, 7);  % obs = (x, y, z, u, v, n_u, n_v)

for k = 1 : N
    C_iv    = C_all(:, :, k);
    r_vi_i  = pos_all(k, :);

    % sample a random (u, v, d)
    u = rand();
    v = rand();
    d = (rand() + 0.2) * 10.0;

    % reproject to 3D
    r_pc_c = zeros(3, 1);
    r_pc_c(1) = d * (u - cx) / fx;
    r_pc_c(2) = d * (v - cy) / fy;
    r_pc_c(3) = d;

    % transform to inertial frame
    r_pi_i = C_iv * (C_vc * r_pc_c + r_cv_v') + r_vi_i';

    % add noise to observation
    n_u = rand() * sqrt(var_n_u);
    n_v = rand() * sqrt(var_n_v);

    obs = zeros(1, 7);
    obs(1) = r_pi_i(1); obs(2) = r_pi_i(2); obs(3) = r_pi_i(3);
    obs(4) = u; obs(5) = v; obs(6) = n_u; obs(7) = n_v;
    obs_all(k, :) = obs;
end

save("data_cam.mat", "C_vc", "r_cv_v", "fx", "fy", "cx", "cy", ...
                     "var_n_u", "var_n_v", "obs_all");

disp("done generating camera measurement");