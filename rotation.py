# This file is used to rotate node of the hand model.
# We want to get three different hand poses through simulation.
# See: https://www.rheumtutor.com/rheumtutoring/approach-to-hand-x-rays/
# Author: Tianci Wen

import numpy as np
import cv2
import random

import gvxrPython3 as gvxr


def poserior_anterior(angles):

    node_label_set = [];
    node_label_set.append('root');

    # The list is not empty
    while (len(node_label_set)):

        # Get the last node
        node = node_label_set[-1];

        # Initialise the material properties
        Z = gvxr.getElementAtomicNumber("H");
        gvxr.setElement(node, gvxr.getElementName(Z));

        # Change the node colour to a random colour
        gvxr.setColour(node, random.uniform(0, 1), random.uniform(0, 1), random.uniform(0, 1), 1.0);

        # Remove it from the list
        node_label_set.pop();

        # Add its Children
        for i in range(gvxr.getNumberOfChildren(node)):
            node_label_set.append(gvxr.getChildLabel(node, i));

        if node == 'root':

            gvxr.rotateNode(node, angles[0], 1, 0, 0);
            gvxr.rotateNode(node, angles[1], 0, 1, 0);

        if node == 'node-Thu_Meta':
            gvxr.rotateNode(node, angles[2], 1, 0, 0);
            gvxr.rotateNode(node, angles[3], 0, 1, 0);

        if node == 'node-Thu_Prox':
            gvxr.rotateNode(node, angles[4], 1, 0, 0);
            gvxr.rotateNode(node, angles[5], 0, 1, 0);

        if node == 'node-Thu_Dist':
            gvxr.rotateNode(node, angles[6], 1, 0, 0);
            gvxr.rotateNode(node, angles[7], 0, 1, 0);

        if node == 'node-Lit_Meta':
            gvxr.rotateNode(node, angles[8], 1, 0, 0);
            gvxr.rotateNode(node, angles[9], 0, 1, 0);

        if node == 'node-Lit_Prox':
            gvxr.rotateNode(node, angles[10], 1, 0, 0);
            gvxr.rotateNode(node, angles[11], 0, 1, 0);

        if node == 'node-Lit_Midd':
            gvxr.rotateNode(node, angles[12], 1, 0, 0);
            gvxr.rotateNode(node, angles[13], 0, 1, 0);

        if node == 'node-Lit_Dist':
            gvxr.rotateNode(node, angles[14], 1, 0, 0);
            gvxr.rotateNode(node, angles[15], 0, 1, 0);

        if node == 'node-Thi_Meta':
            gvxr.rotateNode(node, angles[16], 1, 0, 0);
            gvxr.rotateNode(node, angles[17], 0, 1, 0);

        if node == 'node-Thi_Prox':
            gvxr.rotateNode(node, angles[18], 1, 0, 0);
            gvxr.rotateNode(node, angles[19], 0, 1, 0);

        if node == 'node-Thi_Midd':
            gvxr.rotateNode(node, angles[20], 1, 0, 0);
            gvxr.rotateNode(node, angles[21], 0, 1, 0);

        if node == 'node-Thi_Dist':
            gvxr.rotateNode(node, angles[22], 1, 0, 0);
            gvxr.rotateNode(node, angles[23], 0, 1, 0);

        if node == 'node-Mid_Meta':
            gvxr.rotateNode(node, angles[24], 1, 0, 0);
            gvxr.rotateNode(node, angles[25], 0, 1, 0);

        if node == 'node-Mid_Prox':
            gvxr.rotateNode(node, angles[26], 1, 0, 0);
            gvxr.rotateNode(node, angles[27], 0, 1, 0);

        if node == 'node-Mid_Midd':
            gvxr.rotateNode(node, angles[28], 1, 0, 0);
            gvxr.rotateNode(node, angles[29], 0, 1, 0);

        if node == 'node-Mid_Dist':
            gvxr.rotateNode(node, angles[30], 1, 0, 0);
            gvxr.rotateNode(node, angles[31], 0, 1, 0);

        if node == 'node-Ind_Meta':
            gvxr.rotateNode(node, angles[32], 1, 0, 0);
            gvxr.rotateNode(node, angles[33], 0, 1, 0);

        if node == 'node-Ind_Prox':
            gvxr.rotateNode(node, angles[34], 1, 0, 0);
            gvxr.rotateNode(node, angles[35], 0, 1, 0);

        if node == 'node-Ind_Midd':
            gvxr.rotateNode(node, angles[36], 1, 0, 0);
            gvxr.rotateNode(node, angles[37], 0, 1, 0);

        if node == 'node-Ind_Dist':
            gvxr.rotateNode(node, angles[0], 1, 0, 0);
            gvxr.rotateNode(node, angles[0], 0, 1, 0);

    x_ray_image = gvxr.computeXRayImage();
    image = np.array(x_ray_image);

    return image
