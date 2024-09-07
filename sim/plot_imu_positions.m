function plot_axes(axs, q, r)
    C_sv = quat2rotm(q);
    T_sv = eye(4);
    T_sv(1:3, 1:3) = C_sv;
    T_sv(1:3, 4) = r;
    T_sv

    h = triad('Parent', axs, 'Scale', 0.2, 'LineWidth', 1, 'Tag', '1', 'Matrix', T_sv);
    H = get(h, 'Matrix');
    set(h, 'Matrix', H);
end

axs = axes;
view(3);
daspect([1 1 1]);

h0 = triad('Parent', axs, 'Scale', 3, 'LineWidth', 1, 'Tag', '1', 'Matrix', eye(4));
H0 = get(h0, 'Matrix');
set(h0, 'Matrix', H0);

plot_axes(axs, [ 0.63332,  0.06541, -0.02471,  0.77073], [ 1,  0,  0])
plot_axes(axs, [ 0.39538,  0.79543,  0.32189,  0.32765], [-1,  0,  0])
plot_axes(axs, [ 0.53324, -0.31409,  0.67079,  0.40872], [ 0,  1,  0])
plot_axes(axs, [-0.59964,  0.32251,  0.62000,  0.38990], [ 0, -1,  0])
plot_axes(axs, [ 0.23130,  0.49528,  0.34275, -0.76402], [ 0,  0,  1])
plot_axes(axs, [ 0.74433, -0.47340,  0.35456,  0.31007], [ 0,  0, -1])
drawnow

% plot_axes(axs, [ 0.04461, -0.11193,  0.66997,  0.73254], [ 0.9,  0.0,  0.0])
% plot_axes(axs, [ 0.22307, -0.05903,  0.92923, -0.28859], [-1.2,  0.0,  0.0])
% plot_axes(axs, [ 0.11921,  0.96165,  0.24508, -0.03087], [ 0.3,  1.1,  0.0])
% plot_axes(axs, [ 0.60520,  0.70834,  0.33483,  0.14097], [ 0.2, -0.7,  0.0])
% plot_axes(axs, [ 0.00367, -0.29900,  0.89422,  0.33309], [-0.1,  0.2,  1.3])
% plot_axes(axs, [ 0.59757, -0.09380, -0.66917, -0.43166], [ 0.4, -0.3, -1.1])
% drawnow