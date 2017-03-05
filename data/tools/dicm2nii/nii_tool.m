function varargout = nii_tool(cmd, varargin)
% Basic function to create, load and save NIfTI file.
% 
% rst = NII_TOOL('cmd', para);
% 
% To list all command, type
%  NII_TOOL ?
% 
% To get help information for each command, include '?' in cmd, for example:
%  NII_TOOL init?
%  NII_TOOL('init?')
% 
% Here is a list of all command:
% 
% NII_TOOL('default', 'version', 1, 'rgb_dim', 1);
% nii = NII_TOOL('init', img);
% nii = NII_TOOL('update', nii);
% NII_TOOL('save', nii, filename, force_3D);
% hdr = NII_TOOL('hdr', filename);
% img = NII_TOOL('img', filename_or_hdr);
% ext = NII_TOOL('ext', filename_or_hdr);
% nii = NII_TOOL('load', filename_or_hdr);
% nii = NII_TOOL('cat3D', filenames);
% NII_TOOL('RGBStyle', 'afni');
% 
% Detail for each command is described below.
% 
% oldVal = NII_TOOL('default', 'version', 1, 'rgb_dim', 1);
% oldVal = NII_TOOL('default', struct('version', 1, 'rgb_dim', 1));
% 
% - Set/query default NIfTI version and/or rgb_dim. To check the setting, run
% NII_TOOL('default') without other input. The input for 'default' command can
% be either a struct with fields of 'version' and/or 'rgb_dim', or
% parameter/value pairs. See NII_TOOL('RGBstyle') for meaning of rgb_dim.
% 
% Note that the setting will be saved for future use. If one wants to change the
% settting temporarily, it is better to return the oldVal, and to restore it
% after done:
% 
%  oldVal = NII_TOOL('default', 'version', 2); % set version 2 as default
%  % 'init' and 'save' NIfTI using above version
%  NII_TOOL('default', oldVal); % restore default setting
% 
% The default version setting affects 'init' command only. If you 'load' a NIfTI
% file, modify it, and then 'save' it, the version will be the same as the
% original file, unless it is changed explicitly (see help for 'save' command).
% All 'load' command ('load', 'hdr', 'ext', 'img') will read any version
% correctly, regardless of version setting.
% 
% 
% nii = NII_TOOL('init', img, RGB_dim);
% 
% - Initialize nii struct based on img, normally 3D or 4D array. Most fields in
% the returned nii.hdr contain default values, and need to be updated based on
% dicom or other information. Important ones include pixdim, s/qform_code and
% related parameters.
% 
% The NIfTI datatype will depend on data type of img. Most Matlab data types are
% supported, including 8/16/32/64 bit signed and unsigned integers, single and
% double floating numbers. Single/double complex and logical array are also
% supported.
% 
% The optional third input, RGB_dim, is needed only if img contains RGB/RGBA
% data. It specifies which dimension in img encodes RGB or RGBA. In other words,
% if a non-empty RGB_dim is provided, img will be interpreted as RGB/RGBA data.
% 
% Another way to signify RGB/RGBA data is to permute color dim to 8th-dim of img
% (RGB_dim of 8 can be omitted then). Since NIfTI img can have up to 7 dim,
% NII_TOOL chooses to store RGB/RGBA in 8th dim. Although this looks lengthy
% (4th to 7th dim are often all ones), NII_TOOL can deal with up to 7 dim
% without causing any confusion. This is why the returned nii.img always stores
% RGB in 8th dim.
% 
% 
% nii = NII_TOOL('update', nii);
% 
% - Update nii.hdr according to nii.img. This is useful if one changes nii.img
% type or dimension. The 'save' command calls this internally, so it is not
% necessary to call this before 'save'. A useful case to call 'update' is that
% one likes to use nii struct without saving it to a file, and 'update' will
% make nii.hdr.dim correct.
% 
% 
% hdr = NII_TOOL('hdr', filename);
% 
% - Return hdr struct of the provided NIfTI file. This is useful to check NIfTI
% hdr, and it is much faster than 'load', especially for .gz file. 
% 
% 
% img = NII_TOOL('img', filename_or_hdr);
% 
% - Return image data in a NIfTI file. The second input can be NIfTI file name,
% or hdr struct returned by NII_TOOL('hdr', filename).
% 
% 
% ext = NII_TOOL('ext', filename_or_hdr);
% 
% - Return NIfTI extension in a NIfTI file. The second input can be NIfTI file
% name, or hdr struct returned by NII_TOOL('hdr', filename). The returned ext
% will have field 'edata_decoded' if 'ecode' is of known type, such as dicom
% (2), text (4 or 6) or Matlab (40).
% 
% Here is an example to add data in myFile.mat as extension to nii struct, which
% can be from 'init' or 'load':
% 
%  fid = fopen('myFile.mat'); % open the MAT file
%  myEdata = fread(fid, inf, '*uint8'); % load all bytes as byte column
%  fclose(fid);
%  len = int32(numel(myEdata)); % number of bytes in int32
%  myEdata = [typecast(len, 'uint8')'; myEdata]; % include len in myEdata
%  nii.ext.ecode = 40; % 40 for Matlab extension
%  nii.ext.edata = myEdata; % myEdata must be uint8 array
% 
% NII_TOOL will take care of rest when you 'save' nii to a file.
% 
% In case a NIfTI ext causes problem (for example, some FSL builds have problem
% in reading NIfTI img with ecode>30), one can remove the ext easily:
% 
%  nii = NII_TOOL('load', 'file_with_ext.nii'); % load the file with ext
%  nii.ext = []; % or nii = rmfield(nii, 'ext'); % remove ext
%  NII_TOOL('save', nii, 'file_without_ext.nii'); % save it
%
% 
% nii = NII_TOOL('load', filename_or_hdr);
% 
% - Load NIfTI file into nii struct. The returned struct includes NIfTI 'hdr'
% and 'img', as well as 'ext' if the file contains NIfTI extension.
% 
% NII_TOOL returns nii.img with the same data type as stored in the file, while
% numeric values in hdr are in double precision for convenience.
% 
% 
% NII_TOOL('save', nii, filename, force_3D);
% 
% - Save struct nii into filename. The format of the file is determined by the
% file extension, such as .img, .nii, .img.gz, .nii.gz etc. If filename is not
% provided, nii.hdr.file_name must contain a file name. Note that 'save' command
% always overwrites file in case of name conflict.
% 
% If filename has no extension, '.nii' will be used as default.
% 
% If the 4th input, force_3D, is true (default false), the output file will be
% 3D only, which means multiple volume data will be split into multiple files.
% This is the format SPM likes. You can use this command to convert 4D into 3D
% by 'load' a 4D file, then 'save' it as 3D files. The 3D file names will have
% 5-digit like '_00001' appended to indicate volume index.
% 
% The NIfTI version can be set by NII_TOOL('default'). One can override the
% default version by specifying it in nii.hdr.version. To convert between
% versions, load a NIfTI file, specify new version, and save it. For example:
% 
%  nii = NII_TOOL('load', 'file_nifti1.nii'); % load version 1 file
%  nii.hdr.version = 2; % force to NIfTI-2
%  NII_TOOL('save', nii, 'file_nifti2.nii'); % save as version 2 file
% 
% Following example shows how to change data type of a nii file:
%  nii = NII_TOOL('load', 'file_int16.nii'); % load int16 type file
%  nii.img = single(nii.img); % change data type to single/float32
%  NII_TOOL('save', nii, 'file_float.nii'); % NII_TOOL will take care of hdr
% 
% 
% nii = NII_TOOL('cat3D', files);
% 
% - Concatenate SPM 3D files into a 4D dataset. The input 'files' can be cellstr
% with file names, or char with wildcards (* or ?). If it is cellstr, the volume
% order in the 4D data corresponds to those files. If wildcards are used, the
% volume order is based on alphabetical order of file names.
% 
% Note that the files to be concatenated must have the same datatype, dim, voxel
% size, scaling slope and intercept, transformation matrix, etc. This is
% normally true if files are for the same dicom series.
% 
% Following example shows how to convert a series of 3D files into a 4D file:
% 
%  nii = NII_TOOL('cat3D', './data/fSubj2-0003*.nii'); % load files for series 3 
%  NII_TOOL('save', nii, './data/fSubj2-0003_4D.nii'); % save as a 4D file
% 
% 
% oldStyle = NII_TOOL('RGBStyle', 'afni');
% 
% - Set/query the method to save RGB or RGBA NIfTI file. The default method can
% be set by NII_TOOL('default', 'rgb_dim', dimN), where dimN can be 1, 3 or 4,
% or 'afni', 'mricron' or 'fsl', as explained below.
% 
% The default is 'afni' style (or 1), which is defined by NIfTI standard, but is
% not well supported by fslview till v5.0.8 or mricron till v20140804.
% 
% If the second input is set to 'mricron' (or 3), NII_TOOL will save file using
% the old RGB fashion (dim 3 for RGB). This works for mricron v20140804 or
% earlier. nii_viewer.m tool and the later mricron work for both rgb_dim of 1
% and 3.
% 
% If the second input is set to 'fsl' (or 4), NII_TOOL will save RGB or RGBA
% layer into 4th dimension, and the file is not encoded as RGB data, but as
% normal NIfTI. This violates the NIfTI rule, but it seems it is the only way
% to work for fslview (at least till fsl v5.0.8).
% 
% If no new style (second input) is provided, it means to query the current
% style (one of 'afni', 'mricron' and 'fsl').
% 
% The GUI mehtod to convert between different RGB style can be found in
% nii_viewer. Following shows how to convert other style into fsl style:
% 
%  nii = NII_TOOL('load', 'non_fsl_style.nii'); % load RGB file
%  NII_TOOL('RGBStyle', 'fsl'); % switch to fsl style for later save
%  NII_TOOL('save', nii, 'fslRGB.nii'); % fsl can read it as RGB
% 
% Note that, if one wants to convert fsl style (non-RGB file by NIfTI standard)
% to other styles, an extra step is needed to change the RGB dim from 4th to 8th
% dim before 'save':
% 
%  nii = NII_TOOL('load', 'fslStyleFile.nii'); % it is normal NIfTI
%  nii.img = permute(nii.img, [1:3 5:8 4]); % force it to be RGB data
%  NII_TOOL('RGBStyle', 'afni'); % switch to NIfTI RGB style
%  NII_TOOL('save', nii, 'afni_RGB.nii'); % now AFNI can read it as RGB
% 
% Also note that the setting by NII_TOOL('RGBStyle') is effective only for
% current Matlab session. If one clears all or starts a new Matlab session, the
% default style by NII_TOOL('default') will take effect.
%  
% See also NII_VIEWER, NII_XFORM, DICM2NII

% More information for NIfTI format:
% Official NIfTI website: http://nifti.nimh.nih.gov/
% Another excellent site: http://brainder.org/2012/09/23/the-nifti-file-format/

% History (yymmdd)
% 150109 Write it based on Jimmy Shen's NIfTI tool (xiangrui.li@gmail.com)
% 150202 Include renamed pigz files for Windows to trick Matlab Central
% 150203 Fix closeFile and deleteTmpFile order
% 150205 Add hdr.machine: needed for .img fopen
% 150208 Add 4th input for 'save', allowing to save SPM 3D files
% 150210 Add 'cat3D' to load SPM 3D files
% 150226 Assign all 8 char for 'magic' (version 2 needs it)
% 150321 swapbytes(nByte) for ecode=40 with big endian
% 150401 Add 'default' to set/query version and rgb_dim default setting
% 150514 read_ext: decode txt edata by dicm2nii.m
% 150517 func_handle: provide a way to use gunzipOS etc from outside
% 150617 auto detect rgb_dim 1&3 for 'load' etc using ChrisR method
% 151025 Change subfunc img2datatype as 'update' for outside access
% 151109 Include dd.win (exe) from WinAVR-20100110 for partial gz unzip
% 151205 Partial gunzip: fix fname with space & unknown pigz | dd error.
% 151222 Take care of img for intent_code 2003/2004: anyone uses it?
% 160110 Use matlab pref method to replace para file.
% 160120 check_gzip: use "" for included pigz; ignore dd error if err is false.
% 160326 fix setpref for older Octave: set each parameter separately.
% 160531 fopen uses 'W' for 'w': performance benefit according to Yair.
% 160701 subFuncHelp: bug fix for mfile case.
% 161018 gunzipOS: use unique name for outName, to avoid problem with parfor.
% 161025 Make included linux pigz executible; fix "dd" for windows.
% 161031 gunzip_mem(), nii_bytes() for hdr/ext read: read uint8 then parse;
%        Replace hdr.machine with hdr.swap_endian.
% 170202 check_gzip/check_dd: cd .. if m_dir is pwd, so get full path for m_dir
% 170212 Extract decode_ext() from 'ext' cmd so call it in 'update' cmd.
% 170215 gunzipOS: use -c > rather than copyfile for performance.

persistent C para; % C columns: name, length, format, value, offset
if isempty(C)
    [C, para] = niiHeader;
    if exist('OCTAVE_VERSION', 'builtin')
        warning('off', 'Octave:fopen-mode'); % avoid 'W' warning
        more off;
    end
end

if ~ischar(cmd)
    error('Provide a string command as the first input for nii_tool');
end
if any(cmd=='?'), subFuncHelp(mfilename, cmd); return; end

if strcmpi(cmd, 'init')
    if nargin<2, error('nii_tool(''%s'') needs second input', cmd); end
    for i = 1:size(C,1), nii.hdr.(C{i,1}) = C{i,4}; end
    nii.img = varargin{1};
    if numel(size(nii.img))>8
        error('NIfTI img can have up to 7 dimension');
    end
    if nargin>2
        i = varargin{2};
        if i<0 || i>8 || mod(i,1)>0, error('Invalid RGB_dim number'); end
        nii.img = permute(nii.img, [1:i-1 i+1:8 i]); % RGB to dim8
    end
    varargout{1} = nii_tool('update', nii); % set datatype etc
    
elseif strcmpi(cmd, 'save')
    if nargin<2, error('nii_tool(''%s'') needs second input', cmd); end
    nii = varargin{1};
    if ~isstruct(nii) || ~isfield(nii, 'hdr') || ~isfield(nii, 'img') 
        error(['nii_tool(''save'') needs a struct from nii_tool(''init'')' ...
            ' or nii_tool(''load'') as the second input']);
    end
    
    % Check file name to save
    if nargin>2
        fname = varargin{2};
        if numel(fname)<5 || ~ischar(fname)
            error('Invalid name for NIfTI file: %s', fname);
        end
    elseif isfield(nii.hdr, 'file_name')
        fname = nii.hdr.file_name;
    else
        error('Provide a valid file name as the third input');
    end
    if ~ispc && strncmp(fname, '~/', 2) % matlab may err with this abbrevation
        fname = [getenv('HOME') fname(2:end)];
    end
    [pth, fname, fext] = fileparts(fname);
    do_gzip = strcmpi(fext, '.gz');
    if do_gzip
        [~, fname, fext] = fileparts(fname); % get .nii .img .hdr
    end
    if isempty(fext), fext = '.nii'; end % default single .nii file
    fname = fullfile(pth, fname); % without file ext
    isNii = strcmpi(fext, '.nii'); % will use .img/.hdr if not .nii
    
    % Deal with NIfTI version and sizeof_hdr
    niiVer = para.version;
    if isfield(nii.hdr, 'version'), niiVer = nii.hdr.version; end
    if niiVer == 1
        nii.hdr.sizeof_hdr = 348; % in case it was loaded from other version
    elseif niiVer == 2
        nii.hdr.sizeof_hdr = 540; % version 2
    else 
        error('Unsupported NIfTI version: %g', niiVer);
    end
    
    if niiVer ~= para.version
        C0 = niiHeader(niiVer);
    else
        C0 = C;
    end
    
    % Update datatype/bitpix/dim in case nii.img is changed
    [nii, fmt] = nii_tool('update', nii);
        
    % This 'if' block: lazy implementation to split into 3D SPM files
    if nargin>3 && ~isempty(varargin{3}) && varargin{3} && nii.hdr.dim(5)>1
        if do_gzip, fext = [fext '.gz']; end
        nii0 = nii;
        for i = 1:nii.hdr.dim(5)
            fname0 = sprintf('%s_%05g%s', fname, i, fext);
            nii0.img = nii.img(:,:,:,i,:,:,:,:); % one vol
            if i==1 && isfield(nii, 'ext'), nii0.ext = nii.ext;
            elseif i==2 && isfield(nii0, 'ext'), nii0 = rmfield(nii0, 'ext'); 
            end
            nii_tool('save', nii0, fname0);
        end
        return;
    end
        
    % re-arrange img for special datatype: RGB/RGBA/Complex.
    if any(nii.hdr.datatype == [128 511 2304]) % RGB or RGBA
        if para.rgb_dim == 1 % AFNI style
            nii.img = permute(nii.img, [8 1:7]);
        elseif para.rgb_dim == 3 % old mricron style
            nii.img = permute(nii.img, [1 2 8 3:7]);
        elseif para.rgb_dim == 4 % for fslview
            nii.img = permute(nii.img, [1:3 8 4:7]); % violate nii rule
            dim = size(nii.img);
            if numel(dim)>6 % dim7 is not 1
                i = find(dim(5:7)==1, 1, 'last') + 4;
                nii.img = permute(nii.img, [1:i-1 i+1:8 i]);
            end
            nii = nii_tool('update', nii);  % changed to non-RGB datatype
        end
    elseif any(nii.hdr.datatype == [32 1792]) % complex single/double
        nii.img = [real(nii.img(:))'; imag(nii.img(:))'];
    end
    
    % Check nii extension: update esize to x16
    nExt = 0; esize = 0;
    nii.hdr.extension = [0 0 0 0]; % no nii ext
    if isfield(nii, 'ext') && isstruct(nii.ext) ...
            && isfield(nii.ext(1), 'edata') && ~isempty(nii.ext(1).edata)
        nExt = numel(nii.ext);
        nii.hdr.extension = [1 0 0 0]; % there is nii ext
        for i = 1:nExt
            if ~isfield(nii.ext(i), 'ecode') || ~isfield(nii.ext(i), 'edata')
                error('NIfTI header ext struct must have ecode and edata');
            end
            
            n0 = numel(nii.ext(i).edata) + 8; % 8 byte for esize and ecode
            n1 = ceil(n0/16) * 16; % esize: multiple of 16
            nii.ext(i).esize = n1;
            nii.ext(i).edata(end+(1:n1-n0)) = 0; % pad zeros
            esize = esize + n1;
        end
    end
    
    % Set magic, vox_offset, and open file for .nii or .hdr
    if isNii
        % version 1 will take only the first 4
        nii.hdr.magic = sprintf('n+%g%s', niiVer, char([0 13 10 26 10]));
        nii.hdr.vox_offset = nii.hdr.sizeof_hdr + 4 + esize;
        fid = fopen([fname fext], 'W');
    else
        nii.hdr.magic = sprintf('ni%g%s', niiVer, char([0 13 10 26 10]));
        nii.hdr.vox_offset = 0;
        fid = fopen([fname '.hdr'], 'W');
    end
    
    % Write nii hdr
    for i = 1:size(C0,1)
        if isfield(nii.hdr, C0{i,1})
            val = nii.hdr.(C0{i,1});
        else % niiVer=2 omit some fields, also take care of other cases
            val = C0{i,4};
        end
        n = numel(val);
        len = C0{i,2};
        if n>len
            val(len+1:n) = []; % remove extra, normally for char
        elseif n<len
            val(n+1:len) = 0; % pad 0, normally for char
        end
        fwrite(fid, val, C0{i,3});
    end
    
    % Write nii ext: extension is in hdr
    for i = 1:nExt % nExt may be 0
        fwrite(fid, nii.ext(i).esize, 'int32');
        fwrite(fid, nii.ext(i).ecode, 'int32');
        fwrite(fid, nii.ext(i).edata, 'uint8');
    end
    
    if ~isNii
        fclose(fid); % done with .hdr
        fid = fopen([fname '.img'], 'W');
    end

    % Write nii image
    fwrite(fid, nii.img, fmt);
    fclose(fid); % all written

    % gzip if asked
    if do_gzip
        if isNii
            gzipOS([fname '.nii']);
        else
            gzipOS([fname '.hdr']); % better not to compress .hdr
            gzipOS([fname '.img']);
        end
    end
    
elseif strcmpi(cmd, 'hdr')
    if nargin<2, error('nii_tool(''%s'') needs second input', cmd); end
    if ~ischar(varargin{1})
        error('nii_tool(''hdr'') needs nii file name as second input'); 
    end
    
    fname = nii_name(varargin{1}, '.hdr'); % get .hdr if it is .img
    [b, fname] = nii_bytes(fname, 600); % v2: 544+10 gzip header
    varargout{1} = read_hdr(b, C, fname);
   
elseif any(strcmpi(cmd, {'img' 'load'})) 
    if nargin<2, error('nii_tool(''%s'') needs second input', cmd); end
    
    if ischar(varargin{1}) % file name input
        fname = nii_name(varargin{1}, '.hdr');
        nii = struct;
    elseif isstruct(varargin{1}) && isfield(varargin{1}, 'file_name')
        nii.hdr = varargin{1};
        fname = nii.hdr.file_name;
    else        
        error(['nii_tool(''%s'') needs a file name or hdr struct from ' ...
            'nii_tool(''hdr'') as second input'], cmd); 
    end
    
    if strcmpi(cmd, 'load')
    	[ext, nii.hdr] = nii_tool('ext', varargin{1});
        if ~isempty(ext), nii.ext = ext; end
    elseif ~isfield(nii, 'hdr')
    	nii.hdr = nii_tool('hdr', fname);        
    end

    nii.img = read_img(nii.hdr, para);
    if strcmpi(cmd, 'load')
        varargout{1} = nii;
    else % img
        varargout{1} = nii.img;
    end
    
elseif strcmpi(cmd, 'ext') 
    if nargin<2, error('nii_tool(''%s'') needs second input', cmd); end
    
    if ischar(varargin{1}) % file name input
        fname = nii_name(varargin{1}, '.hdr');
        hdr = nii_tool('hdr', fname);
    elseif isstruct(varargin{1}) && isfield(varargin{1}, 'file_name')
        hdr = varargin{1};
        fname = hdr.file_name;
    else        
        error(['nii_tool(''%s'') needs a file name or hdr struct from ' ...
            'nii_tool(''hdr'') as second input'], cmd); 
    end
    
    if isempty(hdr.extension) || hdr.extension(1)==0
        varargout{1} = [];
    else
        if hdr.vox_offset>0, nByte = hdr.vox_offset + 64; % .nii arbituary +64
        else nByte = inf;
        end
        b = nii_bytes(fname, nByte);
        varargout{1} = read_ext(b, hdr);
    end
    if nargout>1, varargout{2} = hdr; end
    
elseif strcmpi(cmd, 'RGBStyle')
    styles = {'afni' '' 'mricron' 'fsl'};
    curStyle = styles{para.rgb_dim};
    if nargin<2, varargout{1} = curStyle; return; end % query only
    irgb = varargin{1};
    if isempty(irgb), irgb = 1; end % default as 'afni'
    if ischar(irgb)
        if strncmpi(irgb, 'fsl', 3), irgb = 4;
        elseif strncmpi(irgb, 'mricron', 4), irgb = 3;
        else irgb = 1;
        end
    end
    if ~any(irgb == [1 3 4])
        error('nii_tool(''RGBStyle'') can have 1, 3, or 4 as second input'); 
    end
    if nargout, varargout{1} = curStyle; end % return old one
    para.rgb_dim = irgb; % no save to pref
    
elseif strcmpi(cmd, 'cat3D')
    if nargin<2, error('nii_tool(''%s'') needs second input', cmd); end
    fnames = varargin{1};
    if ischar(fnames) % guess it is like run1*.nii
        f = dir(fnames);
        f = sort({f.name});
        fnames = strcat([fileparts(fnames) '/'], f);
    end
    
    n = numel(fnames);
    if n<2 || ~iscellstr(fnames)
        error('Invalid input for nii_tool(''cat3D''): %s', varargin{1});
    end

    nii = nii_tool('load', fnames{1}); % all for first file
    nii.img(:,:,:,2:n) = 0; % pre-allocate
    % For now, omit all consistence check between files
    for i = 2:n, nii.img(:,:,:,i) = nii_tool('img', fnames{i}); end
    varargout{1} = nii_tool('update', nii); % update dim
    
elseif strcmpi(cmd, 'default')
    flds = {'version' 'rgb_dim'}; % may add more in the future
    pf = getpref('nii_tool_para');
    for i = 1:numel(flds), val.(flds{i}) = pf.(flds{i}); end
    if nargin<2, varargout{1} = val; return; end % query only
    if nargout, varargout{1} = val; end % return old val
    in2 = varargin;
    if ~isstruct(in2), in2 = struct(in2{:}); end
    nam = fieldnames(in2);
    for i = 1:numel(nam)
        ind = strcmpi(nam{i}, flds);
        if isempty(ind), continue; end
        para.(flds{ind}) = in2.(nam{i});
        setpref('nii_tool_para', flds{ind}, in2.(nam{i}));
    end
    if val.version ~= para.version, C = niiHeader(para.version); end
    
elseif strcmpi(cmd, 'update') % old img2datatype subfunction
    if nargin<2, error('nii_tool(''%s'') needs second input', cmd); end
    nii = varargin{1};
    if ~isstruct(nii) || ~isfield(nii, 'hdr') || ~isfield(nii, 'img') 
        error(['nii_tool(''save'') needs a struct from nii_tool(''init'')' ...
            ' or nii_tool(''load'') as the second input']);
    end
    
    dim = size(nii.img);
    ndim = numel(dim);
    dim(ndim+1:7) = 1;
    
    if ndim == 8 % RGB/RGBA data. Change img type to uint8/single if needed
        valpix = dim(8);
        if valpix == 4 % RGBA
            typ = 'RGBA'; % error info only
            nii.img = uint8(nii.img); % NIfTI only support uint8 for RGBA
        elseif valpix == 3 % RGB, must be single or uint8
            typ = 'RGB';
            if max(nii.img(:))>1, nii.img = uint8(nii.img);
            else nii.img = single(nii.img);
            end
        else
            error('Color dimension must have length of 3 for RGB and 4 for RGBA');
        end
        
        dim(8) = []; % remove color-dim so numel(dim)=7 for nii.hdr
        ndim = find(dim>1, 1, 'last'); % update it
    elseif isreal(nii.img)
        typ = 'real';
        valpix = 1;
    else
        typ = 'complex';
        valpix = 2;
    end
    
    if islogical(nii.img), imgFmt = 'ubit1';
    else imgFmt = class(nii.img);
    end
    ind = find(strcmp(para.format, imgFmt) & para.valpix==valpix);
    
    if isempty(ind) % only RGB and complex can have this problem
        error('nii_tool does not support %s image of ''%s'' type', typ, imgFmt);
    elseif numel(ind)>1 % unlikely
        error('Non-unique datatype found for %s image of ''%s'' type', typ, imgFmt);
    end
    
    fmt = para.format{ind};
    nii.hdr.datatype = para.datatype(ind);
    nii.hdr.bitpix = para.bitpix(ind);
    nii.hdr.dim = [ndim dim];
    
    if nii.hdr.sizeof_hdr == 348
        nii.hdr.glmax = round(double(max(nii.img(:)))); % we may remove these
        nii.hdr.glmin = round(double(min(nii.img(:))));
    end
    
    if isfield(nii, 'ext')
        try swap = nii.hdr.swap_endian; catch, swap = false; end
        nii.ext = decode_ext(nii.ext, swap);
    end
    
    varargout{1} = nii;
    if nargout>1, varargout{2} = fmt; end
elseif strcmpi(cmd, 'func_handle') % make a local function avail to outside 
    varargout{1} = str2func(varargin{1});
else
    error('Invalid command for nii_tool: %s', cmd);
end
% End of main function

%% Subfunction: all nii header in the order in NIfTI-1/2 file
function [C, para] = niiHeader(niiVer)
pf = getpref('nii_tool_para');
if isempty(pf)
    setpref('nii_tool_para', 'version', 1);
    setpref('nii_tool_para', 'rgb_dim', 1);
    pf = getpref('nii_tool_para');
end
if nargin<1 || isempty(niiVer), niiVer = pf.version; end

if niiVer == 1
    C = {
    % name              len  format     value           offset
    'sizeof_hdr'        1   'int32'     348             0
    'data_type'         10  'char*1'    ''              4                                          
    'db_name'           18  'char*1'    ''              14
    'extents'           1   'int32'     16384           32
    'session_error'     1   'int16'     0               36
    'regular'           1   'char*1'    'r'             38
    'dim_info'          1   'uint8'     0               39
    'dim'               8   'int16'     ones(1,8)       40
    'intent_p1'         1   'single'    0               56
    'intent_p2'         1   'single'    0               60
    'intent_p3'         1   'single'    0               64
    'intent_code'       1   'int16'     0               68
    'datatype'          1   'int16'     0               70
    'bitpix'            1   'int16'     0               72
    'slice_start'       1   'int16'     0               74
    'pixdim'            8   'single'    zeros(1,8)      76
    'vox_offset'        1   'single'    352             108
    'scl_slope'         1   'single'    1               112
    'scl_inter'         1   'single'    0               116
    'slice_end'         1   'int16'     0               120
    'slice_code'        1   'uint8'     0               122
    'xyzt_units'        1   'uint8'     0               123
    'cal_max'           1   'single'    0               124
    'cal_min'           1   'single'    0               128
    'slice_duration'    1   'single'    0               132
    'toffset'           1   'single'    0               136
    'glmax'             1   'int32'     0               140
    'glmin'             1   'int32'     0               144
    'descrip'           80  'char*1'    ''              148
    'aux_file'          24  'char*1'    ''              228
    'qform_code'        1   'int16'     0               252
    'sform_code'        1   'int16'     0               254
    'quatern_b'         1   'single'    0               256
    'quatern_c'         1   'single'    0               260
    'quatern_d'         1   'single'    0               264
    'qoffset_x'         1   'single'    0               268
    'qoffset_y'         1   'single'    0               272
    'qoffset_z'         1   'single'    0               276
    'srow_x'            4   'single'    [1 0 0 0]       280
    'srow_y'            4   'single'    [0 1 0 0]       296
    'srow_z'            4   'single'    [0 0 1 0]       312
    'intent_name'       16  'char*1'    ''              328
    'magic'             4   'char*1'    ''              344
    'extension'         4   'uint8'     [0 0 0 0]       348
    };

elseif niiVer == 2
    C = {
    'sizeof_hdr'        1   'int32'     540             0
    'magic'             8   'char*1'    ''              4
    'datatype'          1   'int16'     0               12
    'bitpix'            1   'int16'     0               14
    'dim'               8   'int64'     ones(1,8)       16
    'intent_p1'         1   'double'    0               80
    'intent_p2'         1   'double'    0               88
    'intent_p3'         1   'double'    0               96
    'pixdim'            8   'double'    zeros(1,8)      104
    'vox_offset'        1   'int64'     544             168
    'scl_slope'         1   'double'    1               176
    'scl_inter'         1   'double'    0               184
    'cal_max'           1   'double'    0               192
    'cal_min'           1   'double'    0               200
    'slice_duration'    1   'double'    0               208
    'toffset'           1   'double'    0               216
    'slice_start'       1   'int64'     0               224
    'slice_end'         1   'int64'     0               232
    'descrip'           80  'char*1'    ''              240
    'aux_file'          24  'char*1'    ''              320
    'qform_code'        1   'int32'     0               344
    'sform_code'        1   'int32'     0               348
    'quatern_b'         1   'double'    0               352
    'quatern_c'         1   'double'    0               360
    'quatern_d'         1   'double'    0               368
    'qoffset_x'         1   'double'    0               376
    'qoffset_y'         1   'double'    0               384
    'qoffset_z'         1   'double'    0               392
    'srow_x'            4   'double'    [1 0 0 0]       400
    'srow_y'            4   'double'    [0 1 0 0]       432
    'srow_z'            4   'double'    [0 0 1 0]       464
    'slice_code'        1   'int32'     0               496
    'xyzt_units'        1   'int32'     0               500
    'intent_code'       1   'int32'     0               504
    'intent_name'       16  'char*1'    ''              508
    'dim_info'          1   'uint8'     0               524
    'unused_str'        15  'char*1'    ''              525
    'extension'         4   'uint8'     [0 0 0 0]       540
    };
else
    error('Nifti version %g is not supported', niiVer);
end
if nargout<2, return; end

%   class      datatype bitpix  valpix
D = {
    'ubit1'     1       1       1 % neither mricron nor fsl support this
    'uint8'     2       8       1
    'int16'     4       16      1
    'int32'     8       32      1
    'single'    16      32      1
    'single'    32      64      2 % complex
    'double'    64      64      1
    'uint8'     128     24      3 % RGB
    'int8'      256     8       1
    'single'    511     96      3 % RGB, not in NIfTI standard?
    'uint16'    512     16      1
    'uint32'    768     32      1
    'int64'     1024    64      1
    'uint64'    1280    64      1
%     'float128'  1536    128     1 % long double, for 22nd century?
    'double'    1792    128     2 % complex
%     'float128'  2048    256     2 % long double complex
    'uint8'     2304    32      4 % RGBA
    };

para.format   =  D(:,1)';
para.datatype = [D{:,2}];
para.bitpix   = [D{:,3}];
para.valpix   = [D{:,4}];
para.rgb_dim  = pf.rgb_dim; % dim of RGB/RGBA in NIfTI FILE
para.version  = niiVer;

%% Subfunction: use pigz or system gzip if available (faster)
function gzipOS(fname)
persistent cmd; % command to gzip
if isempty(cmd)
    cmd = check_gzip;
    if ischar(cmd)
    	cmd = @(nam) sprintf('%s -nf "%s"', cmd, nam); % overwrite if exist
    elseif islogical(cmd) && ~cmd
        fprintf(2, ['None of system pigz, gzip or Matlab gzip available. ' ...
            'Files are not compressed into gz.\n']);
    end
end

if islogical(cmd)
    if cmd, gzip(fname); deleteFile(fname); end
    return;
end

if ispc
	[err, str] = system(['start "" /B ' cmd(fname)]); % background
else
    [err, str] = system([cmd(fname) ' &']);
end
if err
    try
        gzip(fname); deleteFile(fname);
    catch
        fprintf(2, 'Error during compression: %s\n', str);
    end
end

% Deal with pigz/gzip on path or in nii_tool folder, and matlab gzip/gunzip
function cmd = check_gzip
m_dir = fileparts(which(mfilename));
% away from pwd, so use OS pigz if both exist. Avoid error if pwd changed later
if strcmpi(pwd, m_dir), cd ..; clnObj = onCleanup(@() cd(m_dir)); end

% first, try system pigz
[err, ~] = system('pigz -V 2>&1');
if ~err, cmd = 'pigz'; return; end

% next, try pigz included with nii_tool
if ismac % pigz for mac is not included in the package
    fprintf(2, [' Please install pigz for fast compression: ' ...
        'http://macappstore.org/pigz/\n']);
elseif ispc % rename back pigz for Windows. Renamed to trick Matlab Central
    try %#ok<TRYNC>
        fname = [m_dir '\pigz.win'];
        if exist(fname, 'file')
            movefile(fname, [m_dir '\pigz.exe'], 'f');
        end
        fname = [m_dir '\pthreadGC2.win'];
        if exist(fname, 'file')
            movefile(fname, [m_dir '\pthreadGC2.dll'], 'f');
        end
    end
else % linux
    [st, val] = fileattrib([m_dir '/pigz']);
    if st && ~val.UserExecute, fileattrib([m_dir '/pigz'], '+x'); end
end

cmd = fullfile(m_dir, 'pigz');
cmd = ['"' cmd '"'];
[err, ~] = system([cmd ' -V 2>&1']);
if ~err, return; end

% Third, try system gzip
[err, ~] = system('gzip -V 2>&1'); % gzip on system path?
if ~err, cmd = 'gzip'; return; end

% Lastly, try to use Matlab gzip/gunzip. Check only one, since they are paired
if isempty(which('gunzip')) || ~usejava('jvm')
    cmd = false; % none of de/compress tools available
    return;
end 
cmd = true; % use slower matlab gzip/gunzip

%% check dd command, return empty if not available
function dd = check_dd
[err, ~] = system('dd --version 2>&1');
if ~err, dd = 'dd'; return; end % dd with linix/mac, and maybe windows

if ispc % rename it as exe
    m_dir = fileparts(which(mfilename));
    if strcmpi(pwd, m_dir), cd ..; clnObj = onCleanup(@() cd(m_dir)); end
    fname = [m_dir '\dd.win'];
    if exist(fname, 'file') % first time after download
        try movefile(fname, [m_dir '\dd.exe'], 'f'); catch, end
    end
    dd = ['"' fullfile(m_dir, 'dd') '"'];
    [err, ~] = system([dd ' --version 2>&1']);
    if ~err, return; end
end
dd = '';

%% Try to use in order of pigz, system gunzip, then matlab gunzip
function outName = gunzipOS(fname, nByte)
persistent cmd dd pth; % command to run gupzip, dd tool, and temp_path
if isempty(cmd)
    cmd = check_gzip;
    if ischar(cmd)
        if strcmp(cmd, 'gzip'), cmd = 'gunzip'; end % fix gzip -c problem in PC
        cmd = @(nam)sprintf('%s -nfdc "%s" ', cmd, nam); % f for overwrite
    elseif islogical(cmd) && ~cmd
        cmd = [];
        error('None of system pigz, gunzip or Matlab gunzip is available');
    end
    dd = check_dd;
    if ~isempty(dd)
        dd = @(n,out)sprintf('| %s count=%g of="%s"', dd, ceil(n/512), out);
    end
    
    if ispc % matlab tempdir could be slow due to cd in and out
        pth = getenv('TEMP');
        if isempty(pth), pth = pwd; end
    else
        pth = getenv('TMP');
        if isempty(pth), pth = '/tmp'; end
    end
end

if islogical(cmd)
    outName = gunzip(fname, pth);
    return;
end

[~, outName, ext] = fileparts(fname);
uid = [datestr(now, '_yymmddHHMMSSfff') sprintf('_%03x', randi(999))];
if strcmpi(ext, '.gz') % likely always true
    [~, outName, ext1] = fileparts(outName);
    outName = [outName uid ext1];
else
    outName = [outName uid];
end
outName = fullfile(pth, outName);
if ~isempty(dd) && nargin>1 && ~isinf(nByte) % unzip only part of data
    try %#ok
        [err, ~] = system([cmd(fname) dd(nByte, outName)]);
        if err==0, return; end
    end
end

[err, str] = system([cmd(fname) '> "' outName '"']);
if err
    try
    	outName = gunzip(fname, pth);
    catch
        error('Error during gunzip:\n%s', str);
    end
end

%% cast bytes into a type, swapbytes if needed
function out = cast_swap(b, typ, swap)
out = typecast(b, typ);
if swap, out = swapbytes(out); end
out = double(out); % for convenience

%% subfunction: read hdr
function hdr = read_hdr(b, C, fname)
n = typecast(b(1:4), 'int32');
if     n==348, niiVer = 1; swap = false;
elseif n==540, niiVer = 2; swap = false;
else
    n = swapbytes(n);
    if     n==348, niiVer = 1; swap = true;
    elseif n==540, niiVer = 2; swap = true;
    else error('Not valid NIfTI file: %s', fname);
    end
end

if niiVer>1, C = niiHeader(niiVer); end % C defaults for version 1
for i = 1:size(C,1)
    try a = b(C{i,5}+1 : C{i+1,5}); 
    catch
        if C{i,5}==numel(b), a = [];
        else a = b(C{i,5} + (1:C{i,2})); % last item extension is in bytes
        end
    end
    if strcmp(C{i,3}, 'char*1')
        a = deblank(char(a));
    else
        a = cast_swap(a, C{i,3}, swap);
        a = double(a);
    end
    hdr.(C{i,1}) = a;
end
  
hdr.version = niiVer; % for 'save', unless user asks to change
hdr.swap_endian = swap;
hdr.file_name = fname;

%% subfunction: read ext, and decode it if known ecode
function ext = read_ext(b, hdr)
ext = []; % avoid error if no ext but hdr.extension(1) was set
nEnd = hdr.vox_offset;
if nEnd == 0, nEnd = numel(b); end % .hdr file

swap = hdr.swap_endian;
j = hdr.sizeof_hdr + 4; % 4 for hdr.extension
while j < nEnd
    esize = cast_swap(b(j+(1:4)), 'int32', swap); j = j+4; % x16
    if isempty(esize) || mod(esize,16), return; end % just to be safe
    i = numel(ext) + 1;
    ext(i).esize = esize; %#ok<*AGROW>
    ext(i).ecode = cast_swap(b(j+(1:4)), 'int32', swap); j = j+4; 
    ext(i).edata = b(j+(1:esize-8))'; % -8 for esize & ecode
    j = j + esize - 8;
end
ext = decode_ext(ext, swap);

%% subfunction
function ext = decode_ext(ext, swap)
% Decode edata if we know ecode
for i = 1:numel(ext)
    if isfield(ext(i), 'edata_decoded'), continue; end % done
    if ext(i).ecode == 40 % Matlab: any kind of matlab variable
        nByte = cast_swap(ext(i).edata(1:4), 'int32', swap); % MAT data
        tmp = [tempname '.mat']; % temp MAT file to save edata
        fid1 = fopen(tmp, 'W');
        fwrite(fid1, ext(i).edata(5:nByte+4)); % exclude padded zeros
        fclose(fid1);
        deleteMat = onCleanup(@() deleteFile(tmp)); % delete temp file after done
        ext(i).edata_decoded = load(tmp); % load into struct
    elseif ext(i).ecode == 6 % plain text
        str = char(ext(i).edata(:)');
        if isempty(strfind(str, 'dicm2nii.m'))
            ext(i).edata_decoded = deblank(str);
        else % created by dicm2nii.m
            ss = struct;
            ind = strfind(str, [';' char([0 10])]); % strsplit error in Octave
            ind = [-2 ind]; % -2+3=1: start of first para
            for k = 1:numel(ind)-1
                a = str(ind(k)+3 : ind(k+1));
                a(a==0) = []; % to be safe. strtrim wont remove null
                a = strtrim(a);
                if isempty(a), continue; end
                try
                    eval(['ss.' a]); % put all into struct
                catch me
                    fprintf(2, '%s\n', me.message);
                    fprintf(2, 'Unrecognized text: %s\n', a);
                end
            end
            flds = fieldnames(ss); % make all vector column
            for k = 1:numel(flds)
                val = ss.(flds{k});
                if isnumeric(val) && isrow(val), ss.(flds{k}) = val'; end
            end
            ext(i).edata_decoded = ss;
        end
    elseif ext(i).ecode == 4 % AFNI
        ext(i).edata_decoded = deblank(char(ext(i).edata(:))');
    elseif ext(i).ecode == 2 % dicom
        tmp = [tempname '.dcm'];
        fid1 = fopen(tmp, 'W');
        fwrite(fid1, ext(i).edata);
        fclose(fid1);
        deleteDcm = onCleanup(@() deleteFile(tmp));
        ext(i).edata_decoded = dicm_hdr(tmp);
    end
end

%% subfunction: read img
% memory gunzip may be slow and error for large img, so use file unzip
function img = read_img(hdr, para)
ind = para.datatype == hdr.datatype;
if ~any(ind)
    error('Datatype %g is not supported by nii_tool.', hdr.datatype);
end

dim = hdr.dim(2:8);
dim(hdr.dim(1)+1:7) = 1; % avoid some error in file
dim(dim<1) = 1;
valpix = para.valpix(ind);
n = prod(dim); % num of voxels

fname = nii_name(hdr.file_name, '.img'); % in case of .hdr/.img pair
fid = fopen(fname);
sig = fread(fid, 2, '*uint8')';
if isequal(sig, [31 139]) % .gz
    fclose(fid);
    fname = gunzipOS(fname);
    cln = onCleanup(@() deleteFile(fname)); % delete gunzipped file
    fid = fopen(fname);
end

if hdr.swap_endian % switch between LE and BE
    [~, ~, ed] = fopen(fid); % default endian: almost always ieee-le
    fclose(fid);
    if isempty(strfind(ed, '-le')), ed = strrep(ed, '-be', '-le');
    else ed = strrep(ed, '-le', '-be');
    end
    fid = fopen(fname, 'r', ed); % re-open with changed endian
end

fseek(fid, hdr.vox_offset, 'bof');
img = fread(fid, n*valpix, ['*' para.format{ind}]); % * to keep original class
fclose(fid);

if any(hdr.datatype == [128 511 2304]) % RGB or RGBA
    a = reshape(single(img), valpix, n); % assume rgbrgbrgb...
    d1 = abs(a - a(:,[2:end 1])); % how similar are voxels to their neighbor
    a = reshape(a, prod(dim(1:2)), valpix*prod(dim(3:7))); % rr...rgg...gbb...b
    d2 = abs(a - a([2:end 1],:));
    j = (sum(d1(:))>sum(d2(:)))*2 + 1; % 1 for afni, 3 for mricron
    dim = [dim(1:j-1) valpix dim(j:7)]; % length=8 now
    img = reshape(img, dim);
    img = permute(img, [1:j-1 j+1:8 j]); % put RGB(A) to dim8
elseif any(hdr.datatype == [32 1792]) % complex single/double
    img = reshape(img, [2 dim]);
    img = complex(permute(img(1,:,:,:,:,:,:,:), [2:8 1]), ... % real
                  permute(img(2,:,:,:,:,:,:,:), [2:8 1]));    % imag
else % all others: valpix=1
    if hdr.datatype==1, img = logical(img); end
    img = reshape(img, dim);
end

% RGB triplet in 5th dim OR RGBA quadruplet in 5th dim
c = hdr.intent_code;
if (c == 2003 && dim(5) == 3) || (c == 2004  && dim(5) == 4) 
    img = permute(img, [1:4 6:8 5]);
end

%% Return requested fname with ext, useful for .hdr and .img files
function fname = nii_name(fname, ext)
[~, f, e] = fileparts(fname);
n = numel(fname);
if strcmpi(e, '.gz')
    n = n - 3; % 3 is numel('.gz')
    [~, ~, e] = fileparts(f); % .nii/.hdr/.img
end
if strcmpi(e, '.nii') || strcmpi(e, ext), return; end
if ~strcmpi(e, '.hdr') && ~strcmpi(e, '.img')
    error(['Invalid NIfTI file name: ' fname]); 
end
fname(n+(-3:0)) = ext; % if not return or error, change it

%% Read NIfTI file as bytes, gunzip if needed, but ignore endian
function [b, fname] = nii_bytes(fname, nByte)
if nargin<2, nByte = inf; end
[fid, err] = fopen(fname); % system default endian
if fid<1, error([err ': ' fname]); end
b = fread(fid, nByte, '*uint8')';
fname = fopen(fid);
fclose(fid);
if isequal(b(1:2), [31 139]) % gz, tgz file
    b = gunzip_mem(b, fname, nByte)';
end

%% subfunction: get help for a command
function subFuncHelp(mfile, subcmd)
fid = fopen(which(mfile));
if fid<1, error(' %s not exists.', mfile); end
clnObj = onCleanup(@()fclose(fid));
while 1 % find first % line
    ln = strtrim(fgetl(fid));
    if feof(fid), fprintf(2, ' No help text found.\n'); return; end
    if ~isempty(ln) && ln(1) == '%', break; end
end

cr = char(10);
str = [ln(2:end) cr];
while 1
    ln = strtrim(fgetl(fid));
    if isempty(ln) || ln(1) ~= '%', break; end % first non % line
    str = [str ln(2:end) cr];
end

% detect topic line before formating the str: try each in order
topicChar = [cr ' - ']; % we rely on this for each topic: see help
str = strrep(str, [cr ' -- '], topicChar); % ' -- ' is also fine
ind = strfind(str, topicChar);
if isempty(ind), disp(str); return; end % no topicChar found. Show all help text

fakeChar = repmat(char(1), 1, numel(topicChar));
str = strrep(str, topicChar, fakeChar); % will restore later
str = strrep(str, upper(mfile), mfile);

% format for reliable syntax and paragraph detection (order is important):
cr1 = [cr ' ']; % cr with a space
chars = {[mfile '  ']   [mfile ' ']; % reduce multiple space after mfile to one
         [mfile ' (']   [mfile '(']; % remove space between mfile and (
         [mfile '( ']   [mfile '(']; % remove space after mfile(
         [cr '    ']    [cr char(9)]; % replace 4 space with tab for beauty
         cr1            cr; % remove space after cr
         };
for i = 1:size(chars, 1)
    while ~isempty(strfind(str, chars{i,1}))
        str = strrep(str, chars{i,1}, chars{i,2}); % regexprep error in Octave
    end
end
str = strrep(str, cr, cr1); % restore one space after cr

dashes = strfind(str, fakeChar); % index for ' - ' after formating
str = strrep(str, fakeChar, topicChar); % restore ' - '

prgrfs = strfind(str, [cr1 cr1]); % double blank lines
nTopic = numel(dashes);
topics = ones(1, nTopic+1); % syntax 'mfile(' before each '-' line
for i = 1:nTopic
    ind = strfind(str(1:dashes(i)), [mfile '(']); % syntax before ' - '
    if isempty(ind), continue; end % no syntax before ' - ', assume start with 1
    ind = find(prgrfs < ind(end), 1, 'last'); % previous paragraph
    if isempty(ind), continue; end
    topics(i) = prgrfs(ind) + 1; % start of this topic 
end
topics(nTopic+1) = numel(str); % set last topic to the end

cmd = strrep(subcmd, '?', ''); % remove ? in case it is in subcmd
if isempty(cmd) % help for main function
    disp(str(1:topics(1))); % text before first topic
    return;
end

% find a topic with cmd syntax, and the syntax is prior to ' - '
cmd = sprintf('%s(''%s''', mfile, cmd);
for i = 1:nTopic
    ind = strfind(lower(str(topics(i):dashes(i))), lower(cmd));
    if ~isempty(ind) % found the syntax in the topic
        disp(str(topics(i):topics(i+1)));
        return;
    end
end

% if we reach here, no subcmd found in syntax
fprintf(2, ' Unknown command for %s: %s\n', mfile, subcmd);

%% gunzip bytes in memory if possible. 2nd/3rd input for fallback file gunzip
% Trick: try-block avoid error for partial file unzip.
function bytes = gunzip_mem(gz_bytes, fname, nByte)
bytes = [];
try
    bais = java.io.ByteArrayInputStream(gz_bytes);
    gzis = java.util.zip.GZIPInputStream(bais);
    buff = java.io.ByteArrayOutputStream;
    try org.apache.commons.io.IOUtils.copy(gzis, buff); end %#ok
    gzis.close;
    bytes = typecast(buff.toByteArray, 'uint8'); % faster than uint8()
catch
    if nargin<3 || isempty(nByte), nByte = inf; end
    if nargin<2 || isempty(fname)
        fname = [tempname '.gz']; % temp gz file
        fid = fopen(fname, 'W');
        if fid<0, return; end
        cln = onCleanup(@() deleteFile(fname)); % delete temp gz file
        fwrite(fid, gz_bytes, 'uint8');
        fclose(fid);
    end
    
    try %#ok
        fname = gunzipOS(fname, nByte);
        fid = fopen(fname);
        bytes = fread(fid, nByte, '*uint8');
        fclose(fid);
        deleteFile(fname); % unzipped file
    end
end

%% Delete file in background
function deleteFile(fname)
if ispc, system(['start "" /B del "' fname '"']);
else system(['rm "' fname '" &']);
end
%%
