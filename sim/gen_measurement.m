clear all
clc

% IMU calibration info
C_sv   = eye(3);
r_sv_v = zeros(1, 3);
r_sv_v = [1, 0, 0];

% create imu
imu = imuSensor;
fn = fullfile('mpu6050.json');
loadparams(imu, fn, 'mpu6050_6axis_calibration');

imu.Gyroscope
imu.Accelerometer

% load kinematic data
load("data_kinematics.mat")

y_accel_all   = zeros(N, 3);
y_omega_all   = zeros(N, 3);
y_accel_i_all = zeros(N, 3);

for k = 1 : N
    C_iv    = C_all(:, :, k);
    alpha   = alpha_all(k, :);
    omega   = omega_all(k, :);
    omega_i = omega_i_all(k, :);
    accel_i = accel_i_all(k, :);
    
    % compute acceleration and omega at IMU location (inertial frame)
    accel_t = cross(alpha, r_sv_v);
    accel_r = cross(omega, cross(omega, r_sv_v));
    accel_i = accel_i + (C_iv * (accel_t + accel_r)')';

    % take measurement
    % negative sign is needed as the IMU model output negative accel measurement
    C_si = C_sv * inv(C_iv);
    [y_accel, y_omega] = imu(-accel_i, omega_i, C_si);

    y_accel_all(k, :) = y_accel;
    y_omega_all(k, :) = y_omega;

    % store transformed measurement in inertial frame with gravity removed
    y_accel_i = C_iv * y_accel' - [0; 0; 9.81];
    y_accel_i_all(k, :) = y_accel_i';
end

save("data_imu.mat", "C_all", "omega_all", "omega_i_all", "alpha_all", ...
                     "pos_all", "vel_all", "accel_all", "accel_i_all", ...
                     "y_accel_all", "y_omega_all", "y_accel_i_all", ...
                     "N", "dt");

disp("done generating imu measurement");