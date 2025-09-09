#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov  3 12:39:05 2020

@author: apramanik
"""


from torch.utils.data import DataLoader
import numpy as np
import os, torch
import matplotlib.pyplot as plt
import tkinter as tk
from tkinter import filedialog
from tst_dataset import cardiacdata
from UNET3D_D4 import UNet3D

# def select_test_data():
#     """
#     GUI to select test data directory.
#     Returns:
#         test_data_dir (str): Path to the selected test data directory.
#     """
#     root = tk.Tk()
#     root.withdraw()  # Hide the main tkinter window
#     test_data_dir = filedialog.askdirectory(title="Select Test Data Directory")
#     if not test_data_dir:
#         print("No directory selected. Exiting...")
#         exit()
#     print(f"Test data directory selected: {test_data_dir}")
#     return test_data_dir

# if __name__ == "__main__":
#     # Step 1: Use GUI to select test data
#     test_data_dir = select_test_data()

    # Step 2: Dynamically update the IMG_DIR in cardiacdata
    # cardiacdata.IMG_DIR = test_data_dir  # Dynamically set the test data directory

    # # Step 3: Load test dataset
    # tst_dataset = cardiacdata()
    # tst_loader = DataLoader(tst_dataset, batch_size=1, shuffle=False, num_workers=0)



def dice_comp(pred, gt):
    return (2. * (np.sum(pred.astype(float) * gt.astype(float))) + 1.) / (np.sum(pred.astype(float)) \
        + np.sum(gt.astype(float)) + 1.)



#%%
nImg=1
dispind=0
vol_slice=5
chunk_size=nImg
#%% Choose training model directory
############################## 3DUNET #########################
subDirectory='20Apr_1115am_70I_10000E_1B'

print(subDirectory)

#%%
cwd=os.getcwd()
PATH= cwd+'/savedModels/'+subDirectory #complete path

#%%
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

tst_dataset = cardiacdata()
tst_loader = DataLoader(tst_dataset, batch_size=1, shuffle=False, num_workers=0)
# network
net = UNet3D(num_classes=4, in_channels=1, depth=4, start_filts=32, res=True)
# net.load_state_dict(torch.load(os.path.join(PATH, "model_best.pth.tar"))['state_dict'])
net.load_state_dict(torch.load(os.path.join(PATH, "model_best.pth.tar"), map_location=torch.device('cpu'))['state_dict'])

normOrg=np.zeros((1,16,144,144),dtype=np.float32)
normGT=np.zeros((1,16,144,144),dtype=np.int16)
normSeg=np.zeros((1,16,144,144),dtype=np.int16)
dice = np.zeros((nImg, 3))
net.eval()
for step, (img, seg_gt) in enumerate(tst_loader, 0):
    img, seg_gt = img.to(device), seg_gt.to(device)
    pred = net(img)
    _, pred = torch.max(pred, 1)

    pred = pred.squeeze().detach().cpu().numpy().astype(np.int8)
    img = img.squeeze().detach().cpu().numpy()
    gt = seg_gt.squeeze().detach().cpu().numpy().astype(np.int8)
    for i in range(3):
            dice[step, i] = dice_comp(pred==i+1, gt==i+1)
    normOrg[step]=img
    normGT[step]=gt
    normSeg[step]=pred


    
print("DICE Right Ventricle: {0:.5f}".format(np.mean(dice[:,0])))
print("DICE Myocardium: {0:.5f}".format(np.mean(dice[:,1])))
print("DICE Left Ventricle: {0:.5f}".format(np.mean(dice[:,2])))



#%%%
normOrg=np.reshape(normOrg,[int(normOrg.shape[1]/8),8,144,144])
normGT=np.reshape(normGT,[int(normGT.shape[1]/8),8,144,144])
normSeg=np.reshape(normSeg,[int(normSeg.shape[1]/8),8,144,144])
normError=np.abs(normGT.astype(np.float32)-normSeg.astype(np.float32))
normOrg=normOrg-normOrg.min()
#%% Display the output images
plot= lambda x: plt.imshow(x,cmap=plt.cm.gray,interpolation='bilinear')
plot1= lambda x: plt.imshow(x,interpolation='bilinear')
plt.clf()
plt.subplot(141)
plot(np.abs(normOrg[dispind,vol_slice,:,:]))
plt.axis('off')
plt.title('Original')
plt.subplot(142)
plot1(np.abs(normGT[dispind,vol_slice,:,:]))
plt.axis('off')
plt.title('True labels')
plt.subplot(143)
plot1(np.abs(normSeg[dispind,vol_slice,:,:]))
plt.axis('off')
plt.title('Segmentation')
plt.subplot(144)
plot(np.abs(normError[dispind,vol_slice,:,:]))
plt.title('Error')
plt.axis('off')
plt.subplots_adjust(left=0, right=1, top=1, bottom=0,wspace=.01)
plt.show()



import numpy as np
from scipy.spatial.distance import cdist
from scipy.ndimage import distance_transform_edt

def calculate_thickness_epicardium_endocardium_distance_transform(seg_slice, pixel_spacing, endo_label=1, epi_label=2):
    """
    Calculate myocardial thickness using distance transform for the distance between epicardium and endocardium.
    
    Args:
        seg_slice (numpy.ndarray): 2D segmentation slice with labeled regions.
        pixel_spacing (float): Spacing between pixels in mm (assumes isotropic spacing).
        endo_label (int): Label for endocardium.
        epi_label (int): Label for epicardium.
        
    Returns:
        avg_thickness_mm (float): Average myocardial thickness in mm.
    """
    # Create binary masks for endocardium and epicardium
    endo_mask = (seg_slice == endo_label).astype(np.uint8)
    epi_mask = (seg_slice == epi_label).astype(np.uint8)
    
    if np.sum(endo_mask) == 0 or np.sum(epi_mask) == 0:
        # If one of the boundaries is missing, return 0
        return 0.0

    # Compute the distance transform for both endocardium and epicardium
    endo_dist = distance_transform_edt(endo_mask == 0)  # Distance from each pixel to the nearest endocardium
    epi_dist = distance_transform_edt(epi_mask == 0)    # Distance from each pixel to the nearest epicardium
    
    # Calculate the myocardial thickness as the difference between endocardial and epicardial distances
    # Epicardium should be further out, so endo_dist - epi_dist gives the actual thickness
    thickness = endo_dist - epi_dist
    
    # Mask the area where the myocardium exists
    myocardium_mask = (seg_slice == endo_label) | (seg_slice == epi_label)
    thickness_masked = thickness[myocardium_mask]
    
    # Convert to mm using pixel spacing
    thickness_mm = thickness_masked * pixel_spacing
    
    # Only positive values make sense, so filter out negative thicknesses
    thickness_mm = thickness_mm[thickness_mm > 0]
    
    if len(thickness_mm) == 0:
        return 0.0
    
    # Calculate average thickness
    avg_thickness_mm = np.mean(thickness_mm)

    return avg_thickness_mm

# Example usage
if __name__ == "__main__":
    # Example segmented slice with endocardium and epicardium
    seg_slice = np.array(np.abs(normSeg[dispind,vol_slice,:,:]))
     # Example data, replace with actual segmentation
    
    # Define pixel spacing in mm
    pixel_spacing = 0.5  # Example: 0.5 mm per pixel

    # Calculate myocardial thickness
    avg_thickness_mm = calculate_thickness_epicardium_endocardium_distance_transform(
        seg_slice, pixel_spacing, endo_label=1, epi_label=2
    )
    
    # Print the result
    print(f"Average Myocardial Thickness: {avg_thickness_mm:.2f} mm")
