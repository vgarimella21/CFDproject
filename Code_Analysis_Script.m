%% Code_Analysis_Script.m
% Post-processing for ONE cavity run in the current folder.
%
% Assumes this folder contains:
%   - history.dat
%   - cavity.dat
%
% Run separately in each case folder (PJ/SGS, coarse/fine, MMS on/off).

clear; clc; close all;

%% ===================== USER SETTINGS FOR THIS RUN =====================

caseName = 'PJ_65_MMS1_02visc';   % <-- change per run (used in titles & filenames)
isMMS    = true;            % true  for MMS case (imms=1)
                            % false for physical cavity (imms=0)
Re_value = 10;              % Re for this run (set 100 for Re=100 cavity)

% Grid size for THIS run (must match Fortran inputs)
imax = 65;
jmax = 65;

% MMS flag (same as imms in Fortran: 1 = MMS, 0 = physical)
imms = 1;

% Domain extents (match set_inputs in Fortran)
xmin = 0.0;  xmax = 0.05;
ymin = 0.0;  ymax = 0.05;

% Save figures as PNGs?
saveFigs = true;

% Directory where history.dat and cavity.dat live
runDir = '.';   % '.' = current folder

%% ===================== READ history.dat ===============================
fprintf('\n=== Reading history.dat for case: %s ===\n', caseName);

histFile = fullfile(runDir,'history.dat');
if ~isfile(histFile)
    error('history.dat not found in %s', runDir);
end

% First two lines are headers
H = readmatrix(histFile,'NumHeaderLines',2);

iter = H(:,1);
time = H(:,2);
res1 = H(:,3);
res2 = H(:,4);
res3 = H(:,5);

maxRes = max([res1,res2,res3],[],2);

figure(1); clf;
semilogy(iter, res1,'-','LineWidth',1.2); hold on;
semilogy(iter, res2,'-','LineWidth',1.2);
semilogy(iter, res3,'-','LineWidth',1.2);
semilogy(iter, maxRes,'k--','LineWidth',1.2);
grid on;
xlabel('Iteration');
ylabel('Scaled residual');
legend('Res p','Res u','Res v','Max','Location','southwest');
title(sprintf('Residual history: %s', caseName));

if saveFigs
    saveas(gcf, sprintf('residuals_%s.png', caseName));
end

%% ===================== READ cavity.dat (final zone) ===================
fprintf('=== Reading cavity.dat (final zone) ===\n');

[dataCav, hasExact] = readCavitySingle(runDir, imax, jmax);

nNodes = imax*jmax;

% For MMS=1 and hasExact=true: columns
% [x y p u v p_exact u_exact v_exact DE_p DE_u DE_v]
% Otherwise: [x y p u v]

X = reshape(dataCav(1:nNodes,1), imax, jmax).';
Y = reshape(dataCav(1:nNodes,2), imax, jmax).';
P = reshape(dataCav(1:nNodes,3), imax, jmax).';
U = reshape(dataCav(1:nNodes,4), imax, jmax).';
V = reshape(dataCav(1:nNodes,5), imax, jmax).';

%% ========== FIELD PLOTS: p, u, v, speed + streamlines ================
fprintf('=== Making field plots (p, u, v, speed + streamlines) ===\n');

Speed = sqrt(U.^2 + V.^2);

figure(2); clf;

subplot(2,2,1);
contourf(X,Y,P,30); colorbar;
title(sprintf('%s: pressure p', caseName));
xlabel('x'); ylabel('y'); axis equal tight;

subplot(2,2,2);
contourf(X,Y,U,30); colorbar;
title('u-velocity');
xlabel('x'); ylabel('y'); axis equal tight;

subplot(2,2,3);
contourf(X,Y,V,30); colorbar;
title('v-velocity');
xlabel('x'); ylabel('y'); axis equal tight;

subplot(2,2,4);
contourf(X,Y,Speed,30); colorbar; hold on;
streamslice(X,Y,U,V);   % 2-D call
title('Speed and streamlines');
xlabel('x'); ylabel('y'); axis equal tight;

sgtitle(sprintf('Field plots: %s', caseName));

if saveFigs
    saveas(gcf, sprintf('fields_%s.png', caseName));
end

%% ===================== MMS-SPECIFIC ANALYSIS ==========================
if isMMS && hasExact
    fprintf('=== MMS ON: doing numerical vs exact comparison and norms ===\n');

    % Extract exact solution columns (assumed Tecplot order)
    nCols = size(dataCav,2);
    if nCols < 8
        error('MMS exact data requested but cavity.dat has only %d columns.', nCols);
    end

    P_exact = reshape(dataCav(1:nNodes,6), imax, jmax).';
    U_exact = reshape(dataCav(1:nNodes,7), imax, jmax).';
    V_exact = reshape(dataCav(1:nNodes,8), imax, jmax).';

    % Discretization error fields
    DE_p = P - P_exact;
    DE_u = U - U_exact;
    DE_v = V - V_exact;

    % --- Centerline comparison for u (y ~ mid) ---
    [~, jmid] = min(abs(Y(:,1) - 0.5*(ymin+ymax)));
    xline = X(jmid,:);
    unum  = U(jmid,:);
    uex   = U_exact(jmid,:);

    figure(3); clf;

    subplot(2,2,1);
    contourf(X,Y,U,30); colorbar;
    title('u numerical'); axis equal tight;

    subplot(2,2,2);
    contourf(X,Y,U_exact,30); colorbar;
    title('u exact (MMS)'); axis equal tight;

    subplot(2,2,3);
    plot(xline, unum,'b-o', xline, uex,'r--','LineWidth',1.2);
    legend('Numerical','Exact','Location','best');
    xlabel('x @ mid-height'); ylabel('u');
    title('Centerline u: numerical vs exact'); grid on;

    subplot(2,2,4);
    contourf(X,Y,abs(DE_u),30); colorbar;
    title('|u - u_{exact}|'); axis equal tight;

    sgtitle(sprintf('MMS u-velocity comparison: %s', caseName));

    if saveFigs
        saveas(gcf, sprintf('MMS_u_compare_%s.png', caseName));
    end

    % --- Global discretization error norms (L1, L2, Linf) --------------
    err_p = abs(DE_p(:));
    err_u = abs(DE_u(:));
    err_v = abs(DE_v(:));

    L1_p   = mean(err_p);
    L2_p   = sqrt(mean(err_p.^2));
    Linf_p = max(err_p);

    L1_u   = mean(err_u);
    L2_u   = sqrt(mean(err_u.^2));
    Linf_u = max(err_u);

    L1_v   = mean(err_v);
    L2_v   = sqrt(mean(err_v.^2));
    Linf_v = max(err_v);

    nNodesD = double(nNodes);
    % Representative spacing h ~ 1/sqrt(#nodes)
    h_eff = 1.0 / sqrt(nNodesD);

    fprintf('\nMMS Discretization error norms for case: %s\n', caseName);
    fprintf('  Effective grid spacing h_eff ≈ %.4e\n', h_eff);
    fprintf('  Variable    L1            L2            Linf\n');
    fprintf('  p        %11.4e  %11.4e  %11.4e\n', L1_p, L2_p, Linf_p);
    fprintf('  u        %11.4e  %11.4e  %11.4e\n', L1_u, L2_u, Linf_u);
    fprintf('  v        %11.4e  %11.4e  %11.4e\n\n', L1_v, L2_v, Linf_v);

    % Append to CSV for later log-log & p-hat plots
    normsFile = 'MMS_error_norms.csv';
    if ~isfile(normsFile)
        fid = fopen(normsFile,'w');
        fprintf(fid,'caseName,imax,jmax,h_eff,');
        fprintf(fid,'L1_p,L2_p,Linf_p,L1_u,L2_u,Linf_u,L1_v,L2_v,Linf_v\n');
    else
        fid = fopen(normsFile,'a');
    end

    fprintf(fid,'%s,%d,%d,%.6e,%.6e,%.6e,%.6e,%.6e,%.6e,%.6e,%.6e,%.6e,%.6e\n', ...
        caseName, imax, jmax, h_eff, ...
        L1_p, L2_p, Linf_p, ...
        L1_u, L2_u, Linf_u, ...
        L1_v, L2_v, Linf_v);
    fclose(fid);

    % --- DE_p(x) slice near top wall (for C(4) study) -------------------
    [jmaxLoc, imaxLoc] = size(P);
    j_slice = jmaxLoc - 1;        % one row below top

    x_slice   = X(j_slice, :);
    DEp_slice = abs(P(j_slice, :) - P_exact(j_slice, :));

    figure(); clf; hold on;
    plot(x_slice, DEp_slice, 'o-', 'LineWidth', 1.5);
    xlabel('x [m]');
    ylabel('|p - p_{exact}|');
    title(sprintf('DE_p vs x at y \\approx %.4f m  (%s)', Y(j_slice,1), caseName));
    grid on;
    xlim([0, 0.01]);  % requested window
    set(gca,'FontSize',12);

    if saveFigs
        saveas(gcf, sprintf('DEp_slice_%s.png', caseName));
    end

    % Save numeric slice for comparison between different Cx/Cy
    sliceFile = sprintf('DEp_slice_%s.csv', caseName);
    T_slice = table(x_slice(:), DEp_slice(:), ...
                    'VariableNames', {'x', 'DE_p'});
    writetable(T_slice, sliceFile);

else
    fprintf('MMS off or exact fields not present; skipping MMS-specific plots.\n');
end

%% === CAVITY: CENTERLINE u PROFILE (Re = 100) ==========================
if ~isMMS && abs(Re_value - 100) < 1e-6
    [jmaxLoc, imaxLoc] = size(U);

    % Vertical line through center x = L/2
    i_center = round((imaxLoc + 1)/2);

    y_center = Y(:, i_center);
    u_center = U(:, i_center);

    % Plot u vs y (for Richardson + literature comparison)
    figure(); clf;
    plot(u_center, y_center, 'o-', 'LineWidth', 1.5);
    set(gca,'YDir','normal');           % y increasing upward
    xlabel('u [m/s]');
    ylabel('y [m]');
    title(sprintf('Centerline u profile (x = L/2), %s', caseName));
    grid on;
    set(gca,'FontSize',12);

    if saveFigs
        saveas(gcf, sprintf('centerline_u_%s.png', caseName));
    end

    % Save data for Richardson extrapolation & Ghia comparison
    outFile = sprintf('centerline_u_%s.csv', caseName);
    T_center = table(y_center(:), u_center(:), ...
                     'VariableNames', {'y', 'u'});
    writetable(T_center, outFile);

    fprintf('\nCenterline u data written to %s\n', outFile);
end

fprintf('\nPost-processing for THIS run is complete.\n');
fprintf('Now change caseName/imax/jmax/imms/isMMS/Re_value as needed for the next run.\n');

%% ======================================================================
%% Helper function: readCavitySingle
%% ======================================================================
function [dataFinal, hasExact] = readCavitySingle(runDir, imax, jmax)
    % Reads the LAST Tecplot zone from cavity.dat, infers number of columns,
    % and returns it as an (nNodes x nCols) numeric array.

    cavFile = fullfile(runDir,'cavity.dat');
    if ~isfile(cavFile)
        error('cavity.dat not found in %s', runDir);
    end

    fid = fopen(cavFile,'r');
    if fid < 0
        error('Could not open cavity.dat');
    end

    % Read entire file as strings first
    raw = textscan(fid,'%s','Delimiter','\n');
    fclose(fid);
    lines = raw{1};
    nLines = numel(lines);

    % Find lines that start a ZONE (very minimal Tecplot parsing)
    zoneIdx = [];
    for ii = 1:nLines
        if contains(upper(lines{ii}),'ZONE')
            zoneIdx(end+1) = ii; %#ok<AGROW>
        end
    end

    if isempty(zoneIdx)
        % No explicit ZONE lines found: treat entire file as one block
        firstDataLine = 1;
    else
        % Assume last ZONE is the final solution
        firstDataLine = zoneIdx(end) + 3; 
        % (roughly skipping ZONE + I/J lines + DATAPACKING line if present)
        % This is heuristic but works for the typical Tecplot output in this project.
    end

    % Collect numeric data from firstDataLine to end
    numLines = lines(firstDataLine:end);
    numBlock = strjoin(numLines, '\n');
    A = sscanf(numBlock, '%f');
    M = numel(A);

    nNodes = imax*jmax;

    if M < nNodes
        error(['Not enough numeric data in cavity.dat for imax*jmax = %d. ', ...
               'Found only %d numbers. Check that imax/jmax match this run.'], ...
               nNodes, M);
    end

    % Infer number of columns per node
    nColsCandidate = floor(M / nNodes);

    if nColsCandidate < 4
        error('Inferred only %d columns per node (need at least 4).', nColsCandidate);
    end

    totalNeeded = nNodes * nColsCandidate;

    if totalNeeded > M
        error('Inferred nCols leads to more values than available. Check imax/jmax.');
    end

    % Take the last full block of size nNodes*nColsCandidate
    startIdx = M - totalNeeded + 1;
    dataVec  = A(startIdx : M);

    dataFinal = reshape(dataVec, [nColsCandidate, nNodes]).';
    hasExact  = (nColsCandidate >= 8);  % 8+ cols ⇒ MMS exact data likely present
end
