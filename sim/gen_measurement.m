clear all
clc

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
    % take measurement
    accel_i = accel_i_all(k, :);
    omega_i = omega_i_all(k, :);
    C       = C_all(:, :, k);

    % negative sign is needed as the IMU model output negative accel measurement
    [y_accel, y_omega] = imu(-accel_i, omega_i, inv(C));

    y_accel_all(k, :) = y_accel;
    y_omega_all(k, :) = y_omega;

    % store transformed measurement in inertial frame with gravity removed
    y_accel_i = C * y_accel' - [0; 0; 9.81];
    y_accel_i_all(k, :) = y_accel_i';
end

save("data_imu.mat", "C_all", "omega_all", "omega_i_all", "alpha_all", ...
                     "pos_all", "vel_all", "accel_all", "accel_i_all", ...
                     "y_accel_all", "y_omega_all", "y_accel_i_all", ...
                     "N", "dt");

disp("done generating imu measurement");