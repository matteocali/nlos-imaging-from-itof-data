
clear;
clc;
close all;
addpath(genpath('./'));

% parameters
filename              = 'cube';
folderData            = './data';
folderReconstruction  = './reconstructions';
load(fullfile(folderData, filename));

whetherConfocal   = true;       % whether confocal setting (colocated virtual source and detecotr)


%% ===== detecting discontinuities in transients =====
% ----- parameters -----
discontDetectionPara.expCoeff             = [0.3];       % model the exponential falloff of the SPAD signal
discontDetectionPara.sigmaBlur            = [1];         % Difference of Gaussian, standard deviation
discontDetectionPara.numOfDiscont         = 1;           % number of discontinuities per transient
discontDetectionPara.convolveTwoSides     = true;        % convolve transient with DoG filter in both sides (for detecting local minimum/maximum)
discontDetectionPara.whetherSortDisconts  = true;        % whether sort discontinuities

% ----- pathlength discontinuties visualization  -----
whetherVisualizePDSurface       = true;
whetherVisualizePDIndivisually  = false;
visualizaRange                  = 1:1785;

detectDiscontinuity;


%% ===== Fermat Flow (sphere-ray intersection) =====
% ----- parameters for computing x and y derivatives -----
pathSurfaceDerivativePara.detGridSize              = detGridSize;
pathSurfaceDerivativePara.planeFittingRange        = 5;           % local 5*5 patch for estimating x and y derivatives, odd number, at least 3
pathSurfaceDerivativePara.spatialSigma             = 8;           % bilateral filtering, spatial gaussian blur kernel size
pathSurfaceDerivativePara.diffSigma                = 10;          % bilateral filtering, range gaussian blur kernel size
pathSurfaceDerivativePara.fitErrorThreshold        = Inf;         % reconstruction threshold, larger the value, looser the constraint
pathSurfaceDerivativePara.curvatureRatioThreshold  = 0;           % reconstruction threshold, smaller the value, looser the constraint

% ----- x, y derivatives visualization -----
whetherVisualizexDyDzD = true;

fermatFlowReconstruction;
