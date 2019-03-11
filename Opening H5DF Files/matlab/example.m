pyversion % To check which Python version Matlab is using

% use the following to set a specific version
% pyversion 3.6
% or set a specific Python path
% pyversion /path/to/anaconda/python

%% load a file just like in Python
file = py.lumicks.pylake.File('example.h5');

file % to print the contents of the file

py.list(file.scans)  % to list the names of the scans in a file

scan = file.scans{'30'};  % get one scan -- [] in Python is {} in Matlab

%% metadata
x_center_um = scan.json{'scan volume'}{'center point (um)'}{'x'};
y_center_um = scan.json{'scan volume'}{'center point (um)'}{'y'};

x_width_um = scan.json{'scan volume'}{'scan axes'}{1}{'scan width (um)'};
y_width_um = scan.json{'scan volume'}{'scan axes'}{2}{'scan width (um)'};

x_pixels = scan.json{'scan volume'}{'scan axes'}{1}{'num of pixels'};
y_pixels = scan.json{'scan volume'}{'scan axes'}{2}{'num of pixels'};

%% image
rgb_image = to_matlab(scan.rgb_image); % returns a matlab array of pixels
size(rgb_image);  % == num_frames x width x heigh x 3

blue_image = to_matlab(scan.blue_image);
size(blue_image);  % == num_frames x width x heigh

slice = timeslice(file.force1x, scan.start, scan.stop);
force_samples = to_matlab(slice.data);
force_timestaps = to_matlab(slice.timestamps);

%% multiple scans in a file
scans = py.list(file.scans.values());

images = zeros(length(scans), y_pixels, x_pixels, 3);
for i = 1:length(scans)
    images(i, :, :, :) = to_matlab(scans{i}.rgb_image);
end
