%BEFORE RUNNING
%MAKE SURE TO HAVE RECIPROCAL SPACE UNIT CELL ADDED TO FIGURE
%AND MAKE SURE TO HAVE RS LATTICE POINT GRID AND LINES ADDED

h = get(gca, 'Children');

% These are relative to the top of the graphics array
% so adjust indices as needed based on your actual order
% Usually, the last added object is h(1)

% Identify the specific lines visually by their order
% For example:
% h(5) = linea, h(6) = lineb, h(7) = linec (if this matches what you see)
% h(2) = grid_hkl

unitcellLines = findobj(h, '-regexp', 'Tag', 'myrecicell');
set(unitcellLines, 'LineWidth', 2.2);

% Make linea, lineb, and linec thicker
%set(findobj(h, 'Tag', 'linea'), 'LineWidth', 7.5);  % linea
%set(findobj(h, 'Tag', 'lineb'), 'LineWidth', 7.5);  % lineb
%set(findobj(h, 'Tag', 'linec'), 'LineWidth', 7.5);  % linec
set(findobj(h, 'Tag', 'linea'), 'LineWidth', 7.5);  % linea
set(findobj(h, 'Tag', 'lineb'), 'LineWidth', 7.5);  % lineb
set(findobj(h, 'Tag', 'linec'), 'LineWidth', 7.5);  % linec

% Make grid_hkl markers larger
set(findobj(h, 'Tag', 'grid_hkl'), 'MarkerSize', 6);
set(findobj(h, 'Tag', 'grid_hkl'), 'Linewidth', 1.0);



% Optionally add axis labels
%xlabel('q_x', 'FontSize', 16);
%xlabel('$q_x\,(\mathrm{\AA}^{-1})$', 'FontSize', 16,'Interpreter', 'latex');
%ylabel('$q_y\,(\mathrm{\AA}^{-1})$', 'FontSize', 16, 'Interpreter', 'latex');
%zlabel('$q_z\,(\mathrm{\AA}^{-1})$', 'FontSize', 16, 'Interpreter', 'latex');
xlabel('$q_x\,(\AA^{-1})$', 'FontSize', 16,'Interpreter', 'latex');
ylabel('$q_y\,(\AA^{-1})$', 'FontSize', 16,'Interpreter', 'latex');
zlabel('$q_z\,(\AA^{-1})$', 'FontSize', 16,'Interpreter', 'latex');
axis vis3d

axis(gca,'padded')
set(gca, 'Position', [0.2, 0.2, 0.65, 0.65]);  % [left, bottom, width, height]

% Set axes labels
set(gca, 'XLim', [-0.03 0.03], ...
         'YLim', [-0.03 0.03], ...
         'ZLim', [-0.03 0.03])
% Customize axes appearance
set(gca, ...
    'FontSize', 18, ...                    % Tick font size
    'LineWidth', 3.5, ...                    % Axis line width
    'LabelFontSizeMultiplier', 1.2);       % Multiplier for axis labels
    %'LabelFontSizeMultiplier', 1.7);       % Multiplier for axis labels

% Set hkl view direction
hr=1;
kr=0;
lr=1;
v1=cellinfo.recimat(1:3);
v2=cellinfo.recimat(4:6);
v3=cellinfo.recimat(7:9);
camdir = hr*v1 + kr*v2 + lr*v3;
view(camdir);