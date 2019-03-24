# This file is used to do optimisation based on different metrics.
# We want to get three different hand poses through simulation.
# See: https://www.rheumtutor.com/rheumtutoring/approach-to-hand-x-rays/
# Author: Tianci Wen

import numpy as np
import random
import math
import matplotlib.pyplot as plt
from time import time
import pandas as pd
import csv
from scipy import optimize
import sys, argparse
import gvxrPython3 as gvxr
from rotation import poserior_anterior
import App
import cv2

from skimage.measure import compare_ssim as SSIM
from sklearn.metrics import mean_absolute_error, mean_squared_error

def setXRayParameters(SOD, SDD):
    # Compute the source position in 3-D from the SOD
    gvxr.setSourcePosition(SOD,  0.0, 0.0, "cm");
    gvxr.setDetectorPosition(SOD - SDD, 0.0, 0.0, "cm");
    gvxr.usePointSource();

def objective_function(params):

    SOD = params[0];
    SDD = params[1];
    angle = len(params)-2;
    angle_list = np.zeros(angle);

    for i in range(angle):

     angle_list[i] = params[i+2];

    setXRayParameters(SOD*SDD, SDD);

    pred_image = poserior_anterior(angle_list);

    rmse = root_mean_squared_error(ground_truth_image, pred_image);

    return rmse

def relative_error(y_true, y_pred):
    s = 0;
    pix1 = 1024;
    pix2 = 1536;

    for i in range(pix1):
        for j in range(pix2):
            s += abs((y_true[i,j] - y_pred[i,j])/y_true[i,j]);
    s = s/(pix1*pix2);

    return s

def zero_mean_normalised_cross_correlation(y_true, y_pred):
    '''
    ZNCC = (1/n)*(1/(std(target_image)*std(est_image)))* SUM_n_by_n{(target_image-
            mean(target_image))*(est_image-mean(est_image))}
    '''
    z = 0;
    pix1 = 1024;
    pix2 = 1536;
    mean1 = np.mean(y_true);
    mean2 = np.mean(y_pred);
    std1 = np.std(y_true);
    std2 = np.std(y_pred);

    for i in range(pix1):
        for j in range(pix2):
            z += (y_true[i,j]-mean1)*(y_pred[i,j]-mean2);
    z = z/(pix1*pix2*std1*std2);

    return z

def root_mean_squared_error(y_true, y_pred):

    return math.sqrt(mean_squared_error(y_true, y_pred));

def structural_similarity(y_pred, y_true):

    return SSIM(y_pred, y_true);

def random_search():

    error_low = 1

    start = time();
    print("Optimising... using Random Search!");

    random.seed();

    for j in range(1000):

        if j>99 and j%100 == 0:
            print("Iterations: %d " % j);
        params = [];
        SOD = random.uniform(0., 1.);
        SDD = random.randint(0, 1000);
        params.append(SOD);
        params.append(SDD);

        for i in range(40):
            angles = random.randint(-180, 180);
            params.append(angles);

        error = objective_function(params);

        if error_low > error:

            print("Better parameters found, updating...");
            error_low = error;
            result_params = params;
            print("Metrics: %.4f" % error_low);
            print("Parameters: ", result_params);

    end = time();

    print("Time: %d" % (end-start));

    result_angle_list = [];
    for a in range(len(result_params)-2):

        result_angle = result_params[a+2];
        result_angle_list.append(result_angle);

    pred_image = poserior_anterior(result_angle_list);

    return pred_image

ground_truth_SOD=100;
ground_truth_SDD=140;
ground_truth_angles = [-90, 20, -10, 0, 5, 0, 5, 0,
                        0, 0, 0, 0, 0, 0, 0, 0,
                        0, 0, 0, 0, 0, 0, 0, 0,
                        0, 0, 0, 0, 0, 0, 0, 0,
                        0, 0, 0, 0, 0, 0, 0, 0,]

# Create an OpenGL context
print("Create an OpenGL context")
gvxr.createWindow();
gvxr.setWindowSize(512, 512);

# Set up the beam
print("Set up the beam")
gvxr.usePointSource();
#gvxr.useParallelBeam();
gvxr.setMonoChromatic(0.08, "MeV", 1000);

# Set up the detector
print("Set up the detector");
gvxr.setDetectorUpVector(0, 0, -1);
gvxr.setDetectorNumberOfPixels(1024, 1536);
gvxr.setDetectorPixelSize(0.5, 0.5, "mm");

setXRayParameters(ground_truth_SOD, ground_truth_SDD);

# Load the data
print("Load the data");

gvxr.loadSceneGraph("/home/ti/Documents/gvxr-python3-gui/hand.dae", "m");
gvxr.moveToCentre('root');
gvxr.disableArtefactFiltering();

# Compute groud truth x-ray image
ground_truth_image = poserior_anterior(ground_truth_angles);
plt.imsave("ground-truth.png", ground_truth_image);
# Compute optimisation
image = random_search();
plt.imsave("prediction.png", image);

plt.subplot(1, 2, 1);
plt.imshow(ground_truth_image);

plt.subplot(1, 2, 2);
plt.imshow(image);
plt.show();

# gvxr.displayScene();

# app = App.App(0.08);
