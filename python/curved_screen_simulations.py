# -*- coding: utf-8 -*-
"""
Created on Wed Feb 27 08:57:33 2019

@author: robster
"""

from vision.camera import Camera
from vision.plot_tools import *
from vision.screen import *
from mayavi import mlab
import matplotlib.pyplot as plt
import numpy as np
import cv2


#%% Settings
""" Settings: """
change_curvature = True
change_cam_position = False
change_cam_rotation = False

vary_curvature = np.arange(1.8,0.5,-0.1)
vary_cam_tx = np.arange(0, 0, 0.1)
vary_cam_ty = np.arange(0, 0, 0.1)
vary_cam_tz =np.arange(0.3, 2, 0.1)

vary_parameter_length = 13






#%% Initialisation

""" Initial camera pose looking stratight down into the plane model """
cam = Camera()
cam.set_K(fx=1400, fy=1500, cx=640, cy=480)
cam.set_width_heigth(1280, 720)
cam.set_R_axisAngle(1.0,  0.0,  0.0, np.deg2rad(180))
initial_t = np.array([0.0, 0.0, 0.7])
cam.set_t(initial_t[0], initial_t[1], initial_t[2], frame='world')

""" Initial Camera Matrix for estimisation"""
camera_matrix = np.zeros((3, 3))
camera_matrix[0,0]= 500
camera_matrix[1,1]= 500
camera_matrix[0,2]=cam.img_width/2
camera_matrix[1,2]=cam.img_height/2
camera_matrix[2,2]=1.0
distCoeffs= np.zeros(4)

""" Inital Screen for the control points """
sc = Screen(width=1920,height=1080, pixel_pitch= 0.270, curvature_radius=1.0)
sc.update()

""" Initial Parameter list """ 
camera_matrix_list = []
dist_list = []
repo_error_list = []
curvature_list = []
c1_list = []
c2_list = []
c3_list = []
c4_list = []
c5_list = []
c6_list = []



""" Figures """ 
fig1 = plt.figure("Estimated Camera Parameters")
c1 = fig1.add_subplot(321)
c2 = fig1.add_subplot(322)
c3 = fig1.add_subplot(323)
c4 = fig1.add_subplot(324)
c5 = fig1.add_subplot(325)
c6 = fig1.add_subplot(326)

fig2 = plt.figure("Reprojection Error")
repo_error_ax = fig2.add_subplot(111)

#fig3 = plt.figure("Dist Coefs")
#d1 = fig3.add_subplot(321)
#d2 = fig3.add_subplot(322)
#d3 = fig3.add_subplot(323)
#d4 = fig3.add_subplot(324)

fig4 = plt.figure("Curvature")
curvature_ax = fig4.add_subplot(111)



#%% Begin of loop 

for i in range(vary_parameter_length):
    
    ##Generate points
    if(change_curvature):
        sc.curvature_radius = vary_curvature[i]
        sc.update()  
        
    if(change_cam_position):
        new_t = initial_t
        if(vary_cam_tx.size):
            new_t[0] = vary_cam_tx[i]
        if(vary_cam_ty.size):
            new_t[1] = vary_cam_ty[i]
        if(vary_cam_tz.size):
            new_t[2] = vary_cam_tz[i]        
        cam.set_t(new_t[0], new_t[1], new_t[2], frame='world')
    
    if(change_cam_rotation):
        cam.set_R_axisAngle(1.0,  0.0,  0.0, np.deg2rad(180))


    X = sc.get_points() # If you want to check the points
    X = X[:, 0::2000] # Use every 2000 point
    x = cam.project(X)
    x = cam.addnoise_imagePoints(x)
    
    mlab.points3d(X[0], X[1], X[2],scale_factor = 0.005, color = sc.get_color())
    plot3D_cam(cam, 0.05)
    cam.plot_image(x)
    
    #Calibrate with opencv
    objp1 = np.transpose(X[:3,:])
    imgp1 = np.transpose(x[:2,:])
    objpoints = [] # 3d point in real world space
    imgpoints = [] # 2d points in image plane.
    objpoints.append(objp1.astype(np.float32))
    imgpoints.append(imgp1.astype(np.float32))
    ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, (cam.img_width,cam.img_height), camera_matrix.astype('float32'),distCoeffs, flags=cv2.CALIB_USE_INTRINSIC_GUESS)
        
    
    #Append lists
    curvature_list.append(sc.curvature_radius)
    repo_error_list.append(ret)
    c1_list.append(mtx[0][0])
    c2_list.append(mtx[0][1])
    c3_list.append(mtx[0][2])
    c4_list.append(mtx[1][0])
    c5_list.append(mtx[1][1])
    c6_list.append(mtx[1][2])

    
    ##Plot Results    
    plt.sca(c1)
    plt.ion()
    c1.cla()
    c1.plot(c1_list)

    plt.sca(c2)
    plt.ion()
    c2.cla()
    c2.plot(c2_list)
    
    plt.sca(c3)
    plt.ion()
    c3.cla()
    c3.plot(c3_list)

    plt.sca(c4)
    plt.ion()
    c4.cla()
    c4.plot(c4_list)
    
    plt.sca(c5)
    plt.ion()
    c5.cla()
    c5.plot(c5_list)

    plt.sca(c6)
    plt.ion()
    c6.cla()
    c6.plot(c6_list)
    
    c1.set_title('fx')
    c2.set_title('sx')
    c3.set_title('cx')
    c4.set_title('sy')
    c5.set_title('fy')
    c6.set_title('cy')
    
    plt.show()
    plt.pause(0.001)
      
    plt.sca(curvature_ax)
    plt.ion()
    curvature_ax.cla()
    curvature_ax.plot(curvature_list)
    plt.show()
    curvature_ax.set_title("Screen Curvature in Meter")
    plt.pause(0.01)
    
    plt.sca(repo_error_ax)
    plt.ion()
    repo_error_ax.cla()
    repo_error_ax.plot(repo_error_list)
    repo_error_ax.set_title("Reprojection Error")
    plt.show()
    plt.pause(0.01)

    
    
display_figure()

