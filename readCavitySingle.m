function [data, hasExact] = readCavitySingle(folder, imax, jmax, imms)
% READCAVITYSINGLE  Read final zone from Tecplot-style cavity.dat
%
% Inputs
%   folder  - directory containing cavity.dat
%   imax    - grid points in x
%   jmax    - grid points in y
%   imms    - MMS flag (1 = MMS, 0 = physical)
%
% Outputs
%   data    - numeric matrix for final zone (nNodes x nCols)
%   hasExact - true if MMS exact columns appear to be present.

    fname = fullfile(folder,'cavity.dat');
    if ~isfile(fname)
        error('cavity.dat not found in %s', folder);
    end

    fid = fopen(fname,'r');
    if fid < 0
        error('Could not open %s', fname);
    end

    rows = [];
    while true
        tline = fgetl(fid);
        if ~ischar(tline)
            break;
        end
        tline = strtrim(tline);
        if isempty(tline)
            continue;
        end
        % Data lines start with a number, +, -, or .
        if ~isempty(regexp(tline,'^[\+\-0-9\.]','once'))
            vals = sscanf(tline,'%f').';
            rows = [rows; vals]; %#ok<AGROW>
        end
    end
    fclose(fid);

    if isempty(rows)
        error('No numeric data found in %s', fname);
    end

    dataAll = rows;
    nNodes = imax*jmax;

    if mod(size(dataAll,1), nNodes) ~= 0
        warning('Rows in cavity.dat (%d) are not a multiple of imax*jmax (%d).', ...
                size(dataAll,1), nNodes);
    end

    nZones = floor(size(dataAll,1)/nNodes);
    % Take the LAST zone (final iteration)
    data = dataAll( (nZones-1)*nNodes + 1 : nZones*nNodes , : );

    hasExact = (size(data,2) >= 7) && (imms == 1);
end
