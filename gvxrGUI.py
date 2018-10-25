#!/usr/bin/env python3


#from PIL import Image

import random
import matplotlib
import sys, argparse
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

import numpy as np

import gvxrPython3 as gvxr
import App

backend = matplotlib.get_backend()
if backend is 'agg':
    print ("Change Matplotlib backend from", backend, "to", "TkAgg");
    plt.switch_backend('TkAgg');
    backend = matplotlib.get_backend()


def main(argv):
    parser = argparse.ArgumentParser()
    parser.add_argument("-input", type=str, help="Input file (see http://assimp.sourceforge.net/main_features_formats.html for a list of supported file formats)")
    parser.add_argument("-unit",  type=str, help="Unit of length corresponding to the input", choices=["um", "mm", "cm", "dm", "m", "dam", "hm", "km"]);

    args = parser.parse_args()
    if args.input and args.unit:
        # Create an OpenGL context
        print("Create an OpenGL context")
        gvxr.createWindow();
        gvxr.setWindowSize(512, 512);


        # Set up the beam
        print("Set up the beam")
        gvxr.setSourcePosition(100.0, 0.0, 0.0, "cm");
        gvxr.usePointSource();
        #gvxr.useParallelBeam();
        gvxr.setMonoChromatic(0.08, "MeV", 1);

        # Set up the detector
        print("Set up the detector");
        gvxr.setDetectorPosition(-40.0, 0.0, 0.0, "cm");
        gvxr.setDetectorUpVector(0, 0, -1);
        gvxr.setDetectorNumberOfPixels(1024, 1024);
        gvxr.setDetectorPixelSize(0.5, 0.5, "mm");

        # Load the data
        print("Load the data");

        gvxr.loadSceneGraph(args.input, args.unit);

        gvxr.disableArtefactFiltering();

        #gvxr.loadMeshFile("chest", "./HVPTest/chest2.obj", "mm");
        #gvxr.invertNormalVectors("armR");
        #gvxr.invertNormalVectors("chest");

        for label in gvxr.getMeshLabelSet():
            print("Move ", label, " to the centre");
            #gvxr.moveToCentre(label);

            #print("Move the mesh to the center");
            #gvxr.moveToCenter(label);

            print("Set ", label, "'s Hounsfield unit");
            gvxr.setHU(label, 1000)

            Z = gvxr.getElementAtomicNumber("H");
            gvxr.setElement(label, gvxr.getElementName(Z));
            #gvxr.invertNormalVectors(label);

        gvxr.moveToCentre();


        # Compute an X-ray image
        #print("Compute an X-ray image");
        #gvxr.disableArtefactFiltering();
        #gvxr.enableArtefactFilteringOnGPU();
        # Not working anymore gvxr.enableArtefactFilteringOnGPU();
        # Not working anymore gvxr.enableArtefactFilteringOnCPU();
        '''x_ray_image = np.array(gvxr.computeXRayImage());
        x_ray_image -= 0.0799;
        x_ray_image /= 0.08 - 0.0799;
        plt.ioff();
        plt.imshow(x_ray_image, cmap="gray");
        plt.show()

        gvxr.setShiftFilter(-0.0786232874);
        gvxr.setScaleFilter(726.368958);
        '''


        gvxr.displayScene()
        app = App.App(0.08);


if __name__ == "__main__":
   main(sys.argv[1:])

   exit()