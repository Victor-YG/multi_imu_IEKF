clear all
clc

load("data_kinematics.mat")
load("data_imu.mat")

% plot measurement angular velocity (body frame)
figure;
subplot(3, 1, 1);
plot(y_omega_all(:, 1));
hold on
plot(omega_v_all(:, 1));
title('y-omega_x');

subplot(3, 1, 2);
plot(y_omega_all(:, 2));
hold on
plot(omega_v_all(:, 2));
title('y-omega_y');

subplot(3, 1, 3);
plot(y_omega_all(:, 3));
hold on
plot(omega_v_all(:, 3));
title('y-omega_z');

% plot measurement acceleration (inertial frame)
figure;
subplot(3, 1, 1);
plot(y_accel_i_all(:, 1));
hold on
plot(accel_i_all(:, 1));
title('y-accel_i_x');

subplot(3, 1, 2);
plot(y_accel_i_all(:, 2));
hold on
plot(accel_i_all(:, 2));
title('y-accel_i_y');

subplot(3, 1, 3);
plot(y_accel_i_all(:, 3));
hold on
plot(accel_i_all(:, 3));
title('y-accel_i_z');