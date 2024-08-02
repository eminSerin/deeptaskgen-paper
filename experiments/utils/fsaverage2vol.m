function [] = fsaverage2vol(gifti_dir)
    % Add FreeSurfer MATLAB toolbox to MATLAB path.
    % Set FREESURFER_HOME environment variable to FreeSurfer installation directory.
    freesurfer_path = getenv('FREESURFER_HOME');
    if ~isempty(freesurfer_path)
        matlab_path = fullfile(freesurfer_path, 'matlab');
        addpath(genpath(matlab_path));
    else
        error('FREESURFER_HOME environment variable not set');
    end
    % Add regfusion toolbox to MATLAB path.
    if exist('regfusion', 'dir')   
        addpath(genpath('regfusion'));
    else
        error('regfusion toolbox not found');
    end
    out_name = fullfile(gifti_dir, 'proj_img.mat');
    cd(gifti_dir)
    try
        if ~exist(out_name, 'file')
            % Load gifti files
            hemi_files = {'hemi_L_fsaverage.func.gii', 'hemi_R_fsaverage.func.gii'};
            hemi_data = cell(1, numel(hemi_files));
            for i = 1:numel(hemi_files)
                % file = fullfile(gifti_dir, hemi_files{i});
                file = hemi_files{i};
                fprintf("Loading %s\n", file)
                hemi_data{i} = gifti(file).cdata';
            end

            % Convert gifti files to volume space
            fprintf('Converting fsaverage to volume space...\n');
            convert_img(hemi_data{1}, hemi_data{2}, out_name);
        end
        fprintf("Done!\n");
        quit;
    catch ME
        fprintf("Error converting fsaverage to volume space!\n");
        fprintf("%s\n", ME.message);
        quit;
    end
end

function [] = convert_img(hemi_l, hemi_r, out_name)
    ts = size(hemi_l, 1);
    proj_img = ones(256, 256, 256, ts);
    % Project gifti files to volume space
    for i = 1:ts
        [projected, ~] = CBIG_RF_projectfsaverage2Vol_single(hemi_l(i, :), ...
            hemi_r(i, :), 'nearest', ...
            'allSub_fsaverage_to_FSL_MNI152_FS4.5.0_RF_ANTs_avgMapping.prop.mat', 'FSL_MNI152_FS4.5.0_cortex_estimate.nii');
        proj_img(:, :, :, i) = projected.vol;
    end

    % Save projected images
    save(out_name, 'proj_img', '-v7.3');
end
