#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Oct 27 12:00:09 2023

@author: vincent
"""
import numpy as np
import glob


def convert_farfields(directory):
    """Convert E-field beams to .npy objects.
    
    This function takes in only one argument, which is the folder containing the E-field beams
    as exported in CST. In the current implementation, the beams need to be exported in linear
    scaling (no dB), and they need to be exported in E-field dimensions. If the beams are in power
    dimensions, no error will be returned but the results will not make physical sense."""
    fnames = glob.glob(f'{directory}*farfield*txt')
    
    freqs = [float(fname.split('=')[1].split(')')[0]) for fname in fnames]
    ports = [int(fname.split('[')[1].split(']')[0]) for fname in fnames]
    
    freqs = np.array(sorted(set(freqs)))
    nports = max(ports)
    
    dat = np.loadtxt(fnames[len(fnames)//2],skiprows=2)
    ntheta = len(np.unique(dat[:,0]))
    nphi = len(np.unique(dat[:,1]))
    assert(ntheta * nphi==dat.shape[0])
    
    theta = np.reshape(dat[:,0] , [nphi,ntheta])
    phi = np.reshape(dat[:,1] , [nphi,ntheta])
    nfiles = len(fnames)
    
    nfreq = freqs.shape[0]
    
    # e_field_magnitudes will hold the e-field magnitude
    e_field_magnitudes = np.zeros([nports,nfreq,nphi,ntheta],dtype=np.float32)#we can save on space with float32's; actual numbers are only a few digits 
    
    # e_field_components will hold the components in theta and phi, as complex numbers
    e_field_components = np.zeros([nports,nfreq,nphi,ntheta,2],dtype=np.complex64)
    
    # axial_ratios will hold the axial ratios; not sure this is useful...
    axial_ratios = np.zeros([nports,nfreq,nphi,ntheta],dtype=np.float32)
    
    for i in range(nfiles):
        # columns go:
        # 0 = theta
        # 1 = phi
        # 2 = e_field_magnitudes
        # 3 = theta_comp (abs)
        # 4 = theta_comp (phase)
        # 5 = phi_comp (abs)
        # 6 = phi_comp (phase)
        # 7 = axial_ratios
        if (i+1)%10==0:
            print('working on file '+str(i+1)+' of '+str(nfiles))
        freq = float((fnames[i].split('=')[1]).split(')')[0])
        i_freq = np.where(freqs == freq)
        del dat
        dat=np.loadtxt(fnames[i],skiprows=2, usecols = (2,3,4,5,6,7))
        e_field_magnitude = np.reshape(dat[:,0],[nphi,ntheta])
        theta_comp_abs = np.reshape(dat[:,1],[nphi,ntheta])
        theta_comp_phase = np.reshape(dat[:,2],[nphi,ntheta])
        phi_comp_abs = np.reshape(dat[:,3],[nphi,ntheta])
        phi_comp_phase = np.reshape(dat[:,4],[nphi,ntheta])
        axial_ratio = np.reshape(dat[:,5],[nphi,ntheta])
        if nports>1:
            i_port = int(fnames[i].split('[')[1].split(']')[0])-1
        else:
            i_port = 0
        e_field_magnitudes[i_port,i_freq,:,:] = e_field_magnitude
        e_field_components[i_port,i_freq,:,:,0]=theta_comp_abs * np.exp(1j * theta_comp_phase * np.pi / 180)
        e_field_components[i_port,i_freq,:,:,1]=phi_comp_abs * np.exp(1j * phi_comp_phase * np.pi / 180)
        axial_ratios[i_port,i_freq,:,:] = axial_ratio
        
    np.save(directory+'e_field_magnitudes.npy',np.array(e_field_magnitudes))
    np.save(directory+'theta.npy',theta)
    np.save(directory+'phi.npy',phi)
    np.save(directory+'freqs.npy',freqs)
    np.save(directory+'e_field_components',e_field_components)
    np.save(directory+'axial_ratios.npy',axial_ratios)
    print('Done!')
