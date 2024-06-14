clear all
clc

% simulation settings
sim_time = 200;
sim_freq = 1000;
N = sim_time * sim_freq;
dt = 1.0 / sim_freq;

% state variables
C_all       = zeros(3, 3, N);
omega_all   = zeros(N, 3);
omega_i_all = zeros(N, 3);
alpha_all   = zeros(N, 3);
pos_all     = zeros(N, 3);
vel_all     = zeros(N, 3);
accel_all   = zeros(N, 3);
accel_i_all = zeros(N, 3);

% initial value
C = eye(3);
omega = zeros(1, 3);
alpha = zeros(1, 3);
pos   = zeros(1, 3);
vel   = zeros(1, 3);
accel = zeros(1, 3);

% trajectory function coefficients
w_x_0 = 0.3;
w_x_1 = 0.3;
w_x_2 = 0.05;
w_x_3 = 3;
w_x_4 = 0.15;
w_x_5 = 0.1;
w_x_6 = 2;

w_y_0 = 0.23;
w_y_1 = 0.33;
w_y_2 = 0.06;
w_y_3 = 0;
w_y_4 = 0.19;
w_y_5 = 0.11;
w_y_6 = 4;

w_z_0 = 0.05;
w_z_1 = 0.27;
w_z_2 = 0.04;
w_z_3 = 9;
w_z_4 = 0.12;
w_z_5 = 0.15;
w_z_6 = 2;

a_x_0 = 0;
a_x_1 = 0.45;
a_x_2 = 0.09;
a_x_3 = 4;
a_x_4 = 0.2;
a_x_5 = 0.15;
a_x_6 = 3;

a_y_0 = 0;
a_y_1 = 0.35;
a_y_2 = 0.12;
a_y_3 = 7;
a_y_4 = 0.3;
a_y_5 = 0.07;
a_y_6 = 5;

a_z_0 = 0;
a_z_1 = 0.4;
a_z_2 = 0.1;
a_z_3 = 7;
a_z_4 = 0.25;
a_z_5 = 0.14;
a_z_6 = 2;

vehicle = kinematicTrajectory('SampleRate', sim_freq, 'Orientation', C);

for k = 1 : N
    t = k / sim_freq;

    % artificial function for angular velocity (vehicle frame)
    omega(1) = w_x_0 + w_x_1 * sin(w_x_2 * t + w_x_3) + w_x_4 * cos(w_x_5 * t + w_x_6);
    omega(2) = w_y_0 + w_y_1 * sin(w_y_2 * t + w_y_3) + w_y_4 * cos(w_y_5 * t + w_y_6);
    omega(3) = w_z_0 + w_z_1 * sin(w_z_2 * t + w_z_3) + w_z_4 * cos(w_z_5 * t + w_z_6);
    omega_all(k, :) = omega;

    % derived function for angular acceleration (vehicle frame)
    alpha(1) = w_x_1 * w_x_2 * cos(w_x_2 * t + w_x_3) - w_x_4 * w_x_5 * sin(w_x_5 * t + w_x_6);
    alpha(2) = w_y_1 * w_y_2 * cos(w_y_2 * t + w_y_3) - w_y_4 * w_y_5 * sin(w_y_5 * t + w_y_6);
    alpha(3) = w_z_1 * w_z_2 * cos(w_z_2 * t + w_z_3) - w_z_4 * w_z_5 * sin(w_z_5 * t + w_z_6);
    alpha_all(k, :) = alpha;

    % artificial function for acceleration (body frame)
    accel(1) = a_x_0 + a_x_1 * sin(a_x_2 * t + a_x_3) + a_x_4 * cos(a_x_5 * t + a_x_6);
    accel(2) = a_y_0 + a_y_1 * sin(a_y_2 * t + a_y_3) + a_y_4 * cos(a_y_5 * t + a_y_6);
    accel(3) = a_z_0 + a_z_1 * sin(a_z_2 * t + a_z_3) + a_z_4 * cos(a_z_5 * t + a_z_6);
    accel_all(k, :) = accel;

    % compute trajectory
    [pos, C, vel, accel_i, omega_i] = vehicle(accel, omega); % here C := C_{vi}
    pos_all(k, :) = pos;
    vel_all(k, :) = vel;
    accel_i_all(k, :) = accel_i;
    C_all(:, :, k) = inv(C); % we want C_{iv}
    omega_i_all(k, :) = omega_i;
end

save("data_kinematics.mat", "C_all", "omega_all", "omega_i_all", "alpha_all", ...
                            "pos_all", "vel_all", "accel_all", "accel_i_all", ...
                            "N", "dt");

disp("done generating trajectory");