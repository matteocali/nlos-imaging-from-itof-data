function [ depth_out ] = depth_radial2z( depth_in, focal_length)
%Input
%   depth_in = radial depth map
%   focal_length [pixels]
%Output
%   depth_out = depth map wrt z axis
res_v = size(depth_in, 1);
res_ho = size(depth_in, 2);
axis_v = linspace(-res_v/2+1/2,res_v/2-1/2,res_v);
axis_ho = linspace(-res_ho/2+1/2,res_ho/2-1/2,res_ho);


conversion_matrix = zeros(size(depth_in,1),size(depth_in,2));
for i = 1:size(depth_in,1)
    for j = 1:size(depth_in,2)
        conversion_matrix(i,j) = 1./sqrt(1+[axis_v(i)/focal_length].^2+[axis_ho(j)/focal_length].^2);%1./sqrt(1+[tand(angle_v(i))].^2+[tand(angle_ho(j))].^2);
    end
end
depth_out = depth_in.*conversion_matrix;
end

