#!/usr/bin/env python3

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
from rotation import bone_rotation
import App
import cv2

from skimage.measure import compare_ssim as structural_similarity
from sklearn.metrics import mean_absolute_error, mean_squared_error

sys.path.append('/home/ti/Documents/Optimisation-algorithm-examples/src')

import SimulatedAnnealing as SA
import EvolutionaryAlgorithm  as EA

global ground_truth_image;

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
    pred_image = np.array([]);
    for i in range(angle):

     angle_list[i] = params[i+2];

    setXRayParameters(SOD, SDD);

    pred_image = bone_rotation(angle_list);

    #RMSE = root_mean_squared_error(ground_truth_image, pred_image);
    #SSIM = structural_similarity(pred_image, ground_truth_image);
    ZNCC = zero_mean_normalised_cross_correlation(ground_truth_image, pred_image);
    #RE = relative_error(ground_truth_image, pred_image);
    #MAE = mean_absolute_error(ground_truth_image, pred_image);

    return ZNCC

def relative_error(y_true, y_pred):
    s = 0;
    pix1 = 1536;
    pix2 = 1024;

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
    pix1 = 1536;
    pix2 = 1024;
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

def random_search():

    error_low = 0;

    start = time();
    print("Optimising... using Random Search!");

    random.seed();

    for j in range(500):

        # if j>99 and j%100 == 0:
        #     print("Iterations: %d " % j);

        params = [];
        SOD = random.uniform(0.05, 1.);
        SDD = random.randint(10, 1000);
        SOD = round(SOD*SDD);
        params.append(SOD);
        params.append(SDD);

        for i in range(48):
            angles = random.randint(-180, 180);
            params.append(angles);

        error = objective_function(params);


        if error_low < error:

            # print("Better parameters found, updating...");
            error_low = error;
            result_params = params;
            # print("Metrics: %.6f" % error_low);
            # print("Parameters: ", result_params);

    end = time();

    print("Time: %d" % (end-start));
    print(error_low);

    result_angle_list = [];
    for a in range(48):

        result_angle = result_params[a+2];
        result_angle_list.append(result_angle);

    setXRayParameters(result_params[0], result_params[1]);
    pred_image = bone_rotation(result_angle_list);

    methods = 'Random Searching';
    SOD_error = abs(SOD-ground_truth_SOD);
    SDD_error = abs(SDD-ground_truth_SDD);
    SSIM = structural_similarity(pred_image, ground_truth_image);
    MAE = mean_absolute_error(ground_truth_image, pred_image);
    RMSE = root_mean_squared_error(ground_truth_image, pred_image);
    RE = relative_error(ground_truth_image, pred_image);
    ZNCC = zero_mean_normalised_cross_correlation(ground_truth_image, pred_image);
    computing_time = end-start;

    row = [[methods, SOD, SOD_error, SDD, SDD_error, result_angle_list,
            SSIM, MAE, RMSE, RE, ZNCC, computing_time]];

    df = pd.DataFrame(row, columns=['Methods', 'SOD', 'SOD Error','SDD',
                                    'SDD Error', 'Rotation Angles','SSIM',
                                    'MAE', 'RMSE','Relative Error', 'ZNCC',
                                    'Time']);

    return pred_image, df

def simulated_annealing():

    num_of_dims = 50;
    bounds = [];
    bounds.append([0.05, 1.]);
    bounds.append([10, 1000]);
    for b in range(48):
        bounds.append([-180, 180]);

    temp = 10000;
    cooling_rate = 0.02;
    print("Optimising... using Simulated Annealing!");

    start = time();
    optimiser = SA.SimulatedAnnealing(num_of_dims, bounds,objective_function,
                                        temp, cooling_rate);

    optimiser.run(False, False);
    end = time();
    pred_image = np.array([]);
    SOD = round(optimiser.best_solution[0]*optimiser.best_solution[1]);
    SDD = round(optimiser.best_solution[1]);
    setXRayParameters(SOD, SDD);
    result_angle_list = [];
    a=0;
    for a in range(48):

        result_angle = optimiser.best_solution[a+2];
        result_angle_list.append(result_angle);

    pred_image = bone_rotation(result_angle_list);

    print(end-start);
    print(optimiser.best_energy);
    methods = 'Simulated Annealing';
    SOD_error = abs(SOD-ground_truth_SOD);
    SDD_error = abs(SDD-ground_truth_SDD);
    SSIM = structural_similarity(pred_image, ground_truth_image);
    MAE = mean_absolute_error(ground_truth_image, pred_image);
    RMSE = root_mean_squared_error(ground_truth_image, pred_image);
    RE = relative_error(ground_truth_image, pred_image);
    ZNCC = zero_mean_normalised_cross_correlation(ground_truth_image, pred_image);
    computing_time = end-start;

    row = [[methods, SOD, SOD_error, SDD, SDD_error, result_angle_list,
            SSIM, MAE, RMSE, RE, ZNCC, computing_time]];
    df = pd.DataFrame(row, columns=['Methods', 'SOD', 'SOD Error','SDD',
                                    'SDD Error', 'Rotation Angles','SSIM',
                                    'MAE', 'RMSE','Relative Error', 'ZNCC',
                                    'Time']);

    return pred_image, df

def evolutionary_algorithm():

    # default 10 ,20
    g_number_of_individuals = 400;
    g_iterations            = 400;
    g_number_of_genes       = 50;

    g_max_mutation_sigma = 0.5;
    g_min_mutation_sigma = 0.01;

    bounds = [];
    bounds.append([0.05, 1.]);
    bounds.append([10, 1000]);
    for b in range(48):
        bounds.append([-180, 180]);

    print("Optimising... using Evolutionary Algorithm!");

    start = time();
    optimiser = EA.EvolutionaryAlgorithm(g_number_of_genes, bounds,
                                            objective_function,
                                            g_number_of_individuals);

    end = time();

    SOD = round(optimiser.best_individual.genes[0]*optimiser.best_individual.genes[1])
    SDD = round(optimiser.best_individual.genes[1]);

    setXRayParameters(SOD, SDD);
    result_angle_list = [];
    a=0;
    for a in range(48):

        result_angle = optimiser.best_individual.genes[a+2];
        result_angle_list.append(result_angle);

    pred_image = bone_rotation(result_angle_list);

    print(SOD, SDD, result_angle_list);

    print(end-start);
    print(optimiser.best_individual.fitness);

    methods = 'Evolutionary Algorithm';
    SOD_error = abs(SOD-ground_truth_SOD);
    SDD_error = abs(SDD-ground_truth_SDD);
    SSIM = structural_similarity(pred_image, ground_truth_image);
    MAE = mean_absolute_error(ground_truth_image, pred_image);
    RMSE = root_mean_squared_error(ground_truth_image, pred_image);
    RE = relative_error(ground_truth_image, pred_image);
    ZNCC = zero_mean_normalised_cross_correlation(ground_truth_image, pred_image);
    computing_time = end-start;

    row = [[methods, SOD, SOD_error, SDD, SDD_error, result_angle_list,
            SSIM, MAE, RMSE, RE, ZNCC, computing_time]];
    df = pd.DataFrame(row, columns=['Methods', 'SOD', 'SOD Error','SDD',
                                    'SDD Error', 'Rotation Angles','SSIM',
                                    'MAE', 'RMSE','Relative Error', 'ZNCC',
                                    'Time']);

    return pred_image, df

ground_truth_SOD=100;
ground_truth_SDD=140;
ground_truth_angles_pa = [-90, 20, 0, -10, 0, 0,
                            5, 0, 0, 5, 0, 0,
                            0, 0, 0, 0, 0, 0,
                            0, 0, 0, 0, 0, 0,
                            0, 0, 0, 0, 0, 0,
                            0, 0, 0, 0, 0, 0,
                            0, 0, 0, 0, 0, 0,
                            0, 0, 0, 0, 0, 0
                            ];

ground_truth_angles_bc = [-90, 0, 0, -5, 0, 0,
                           -2, 0, 0, -25, 0, 0,
                           17, 0, 0, 0, 0, 0,
                           0, 0, 0, 0, 0, 0,
                           0, 0, 0, 0, 0, 0,
                           0, 0, 0, 0, 0, 0,
                           0, 0, 0, -7, 0, 0,
                           0, 0, 0, 0, 0, 0
                           ];

ground_truth_angles_lat = [-90, 0, -90, 16, -20, 0,
                             0, 0, 5, 7, 0, 0,
                             0, 0, 0, 0, 0, 0,
                             0, 0, 0, 0, -2, 0,
                             0, 0, 0, 0, 0, 0,
                             0, 0, 0, 0, -15, 0,
                             0, 0, 0, 0, -50, 0,
                             0, -65, 0, 0, 10, 0
                             ];

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
ground_truth_image = bone_rotation(ground_truth_angles_bc);
plt.imsave("./ballcatchers/ZNCC/ground-truth", ground_truth_image, cmap='Greys_r');

# print("Save the last image into a file");
# gvxr.saveLastXRayImage();
# gvxr.saveLastLBuffer();
#
# print(ground_truth_image);
# print('Maximum pixel value in this image ', np.amax(ground_truth_image));
# print('Minimum pixel value in this image ', np.amin(ground_truth_image));
#
# pic = plt.imread("./ballcatchers/ZNCC/ground-truth.png", 0);
# print('Maximum RGB value in this image {}'.format(pic.max()));
# print('Minimum RGB value in this image {}'.format(pic.min()));


df = pd.DataFrame();
# Compute optimisation
i = 0;
for i in range(10):
    image, df2 = random_search();
    plt.imsave("./ballcatchers/ZNCC/prediction-rs-%d" % i, image, cmap='Greys_r');
    df = df.append(df2, ignore_index=True);
i = 0;
for i in range(10):
    image, df2 = simulated_annealing();
    plt.imsave("./ballcatchers/ZNCC/prediction-sa-%d" % i, image, cmap='Greys_r');
    df = df.append(df2, ignore_index=True);
i = 0;
for i in range(10):
    image, df2 = evolutionary_algorithm();
    plt.imsave("./ballcatchers/ZNCC/prediction-ea-%d" % i, image, cmap='Greys_r');
    df = df.append(df2, ignore_index=True);

print('Saving to csv file...');
df.to_csv('./ballcatchers/ZNCC/results.csv');

# gvxr.displayScene();
#
# app = App.App(0.08);
