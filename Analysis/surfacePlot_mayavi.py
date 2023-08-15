#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May 18 14:53:37 2022

Plotting script for 3D rendering using Mayavi

Reads *.config files from inDir, and creates Config objects containing the
geometric fields. 
A single DataFrame contains the Config objects and characteristic information, 
such as the labeling tag to each config

The tag:    '{helix}_{cS}_{H0}_{W0}'   is defined by a tuple with these values.
The specific configurations to be read/used are given by lists of tuples. 


@author: garcia-aguilar
"""



import pandas as pd
import numpy as np
from mayavi import mlab 

import cmap_rainforest 

colors_rainforest = cmap_rainforest.cm_data

################################################################################
################################################################################

##################  DATA / INPUT

################################################################################

def paramTag(h0, w0, cS, helix):
    ''' For now, based on Helix runs, can be changed to any format used in runs'''
    return '{3}_cS{2}_H{0}_W{1}'.format(h0, w0, cS, helix)


wDir = '../'
inputDir = wDir + 'Examples/'
outDir = wDir + 'Examples/Plots/'

files_results = [inputDir+'conf_{}.config'.format(paramTag(1.0,0.0,60.0,'C')),
                inputDir+'conf_{}.config'.format(paramTag(0.5,1.0,60.0,'A'))]
sets_results = [('C',1.0,0.0,60.0),('A',0.5,1.0,60.0)]


width = 1.06
l0 = 0.13249511385834717
minMax_dimK_manuscript = [-3.2,3.2]    
minMax_K_manuscript = [-28.11,28.11]   
mainDF = pd.DataFrame()

# Default plotting options
defSave = False
properPlot = False

################################################################################
################################################################################

##################  OBJECTS / DATA FUNCTIONS

################################################################################

def dimCurv(curv, power=1):
    return curv/pow(np.pi / width, power)

def renormalize(attr,value):
    if attr in ['H','W']:
        return dimCurv(value)
    elif attr in ['H2','K']:
        return dimCurv(value,2)
    else:
        return value

def get_attrBounds(confsList, attr):
    print("Getting attr bounds for {} configurations, for ".format(len(confsList))+attr)
    maxs = [np.max(getattr(cf,attr)) for cf in confsList]
    mins = [np.min(getattr(cf,attr)) for cf in confsList]
    return np.min(mins), np.max(maxs)

################################################################################   
def cf_area_average(cf, field='H',pw=1, skipBoundaries=True):   
    vals = pow(getattr(cf,field),pw)*cf.A
    if skipBoundaries:
        return vals.sum()/cf.netA
    else:
        return vals.sum()/cf.totA
        
def cf_tot_average(cf, field='H',pw=1, skipBoundaries=True):
    bdFactor = 1.
    num = cf.n
    if skipBoundaries:
        bdFactor = 1.-cf.bd
        num = np.count_nonzero(cf.bd==0)
    
    vals = pow(getattr(cf,field)*bdFactor,pw)        
    return vals.sum()/num  
    
def triangle_strain(cf,tr):
    i,j,k = tr
    ri = np.array([cf.x[i],cf.y[i],cf.z[i]])
    rj = np.array([cf.x[j],cf.y[j],cf.z[j]])
    rk = np.array([cf.x[k],cf.y[k],cf.z[k]])
    u1 =  (np.linalg.norm(ri-rj)-l0)/l0
    u2 =  (np.linalg.norm(rj-rk)-l0)/l0
    u3 =  (np.linalg.norm(ri-rk)-l0)/l0
    cf.u_ij[i] += [u1,u3]
    cf.u_ij[j] += [u1,u2]
    cf.u_ij[k] += [u2,u3]
    return 

def triangle_area_strain(cf,tr):
    a0 = np.sqrt(3)*l0*l0/4
    
    i,j,k = tr
    ri = np.array([cf.x[i],cf.y[i],cf.z[i]])
    rj = np.array([cf.x[j],cf.y[j],cf.z[j]])
    rk = np.array([cf.x[k],cf.y[k],cf.z[k]])
    e_ji =  rj-ri
    e_jk =  rj-rk
    base = np.linalg.norm(e_jk)
    height = np.linalg.norm(e_ji-np.dot(e_ji,e_jk)*e_jk/base/base)
    area_tr = base*height/2.
    
    diff = (area_tr-a0)/a0
    #arreglar u_ij2#
    cf.u_ij2[i] += [diff]
    cf.u_ij2[j] += [diff]
    cf.u_ij2[k] += [diff]
    return 


        

class Config:   
    def __init__(self,gname):
        self.type = gname    
  
    def getConf(self,filename):        
     # Open up the file        
        fil = open(filename, "r")
        lines = fil.readlines()
        paramNames = lines[0].split()
        paramValues = np.fromstring(lines[1],sep='\t')
        self.n = int(lines[3])
        self.n_tr = int(lines[6+self.n])
                
     # Read Parameters
        for pnum, p in enumerate(paramNames):
            setattr(self,p,paramValues[pnum])
              
     # Read vertex info
        cDF = pd.read_csv(filename,sep='\t',nrows=self.n, header=1,skiprows=3)
        self.x = cDF.x
        self.y = cDF.y
        self.z = cDF.z
        self.bd = cDF.bd
        self.A = cDF.A
        self.H2 = cDF.H2*(1-self.bd)
        self.H = cDF.H*(1-self.bd)
        self.W = cDF.W*(1-self.bd)
        self.L = cDF.L
        self.E = cDF.E
        self.E_H = cDF.E_H
        self.E_W = cDF.E_W
        self.E_L = self.cS*self.L
        self.dEx = cDF.dEx
        self.dEy = cDF.dEy
        self.dEz = cDF.dEz
        self.cst = np.array([1]*len(self.x))    
        self.K = self.H2-self.W*self.W
        self.AbsK = abs(self.K)
        #self.cst = [1]*self.n
        
      # Read triangle info        
        self.triangles = [np.fromstring(lines[l],sep='\t',dtype=int).tolist() for l in np.arange(7+self.n,len(lines))]
                 
    # Integral values  
        self.totE = self.E.sum()
        self.totEH = self.E_H.sum()
        self.totEW = self.E_W.sum()
        self.totEL = self.cS*self.L.sum()
        self.dimEL = self.totEL/self.cS/(self.l0*self.l0)
        self.totA = self.A.sum()
        self.netA = self.totA - np.sum(self.A*self.bd)
        self.intK = np.sum(self.K*self.A)
        self.sumK2 = np.sum(self.K*self.K)
        self.sumAbsK = np.sum(abs(self.K))
        
    # Double-check if len(tr) matches n_tr read, same as len(v). 
        if len(self.bd) != self.n:
            print("Only {} vertices read instead of {} in the file".format(len(self.x),self.n))
            return 1
        if len(self.triangles) != self.n_tr:
            print("Only {} triangles read instead of {} in the file".format(len(self.triangles),self.n_tr))           
            return 1
        
        
    def add_avgFields(self):
        self.navL = cf_tot_average(self, field='L',pw=1,skipBoundaries=False)
        
        self.avH = cf_area_average(self, field='H',pw=1)
        self.navH = cf_tot_average(self, field='H',pw=1)
        self.avH2 = cf_area_average(self, field='H',pw=2)
        self.navH2 = cf_tot_average(self, field='H',pw=2)
        
        self.avK = cf_area_average(self, field='K',pw=1)
        self.navK = cf_tot_average(self, field='K',pw=1)
        
        self.avK2 = cf_area_average(self, field='K',pw=2)
        self.navK2 = cf_tot_average(self, field='K',pw=2)
        
        self.avW = cf_area_average(self, field='W',pw=1)
        self.navW = cf_tot_average(self, field='W',pw=1)
        
    def set_strain(self):
        self.u_ij = [ [] for _ in range(self.n) ]
        for tr in self.triangles:
            triangle_strain(self, tr)
        self.u = np.array([np.sum(np.unique([self.u_ij[i]]))/len(np.unique(self.u_ij[i])) for i in range(self.n)])
        #arreglar#
        self.u_ij2 = [ [] for _ in range(self.n) ]
        for tr in self.triangles:
            triangle_area_strain(self, tr)
        self.u2 = np.array([np.sum([self.u_ij2[i]])/len(self.u_ij2[i]) for i in range(self.n)])
       
def read_conf(fil,tag):
    cf = Config(tag)
    cf.getConf(fil)
    cf.add_avgFields()
    cf.set_strain()
    return cf

def add_init(helix,tag='',appendTo=None):#'_H'):
    global mainDF
    confFile = inputDir+'conf'+tag+'_{}_INIT.config'.format(helix)    
    initConfig=read_conf(confFile,helix+'0')
    newRow = {'H0':0,'W0':0,'helix':helix,'init':1,'cf':initConfig,'tag':helix+'0'}
#    mainDF = mainDF.append(newRow, ignore_index=True)
    mainDF = pd.concat([mainDF,pd.DataFrame([newRow])], axis=0, ignore_index=True)
    
    if appendTo is not None:
        appendTo.append(initConfig)
    return


##################  GET DATA FUNCTIONS

################################################################################

def get_config(H0,W0,cS,helix):
    group = mainDF.groupby(['helix','H0','W0','cS'])
    return group.get_group((helix,H0,W0,cS)).cf.item()

def get_set(setList, DF=mainDF):
    group = mainDF.groupby(['helix','H0','W0','cS'])
    newList = [group.get_group(tup).cf.item() for tup in setList]
    return newList

def get_setDF(setList, DF=mainDF):
    #group = mainDF.groupby(['helix','H0','W0','cS'])
    configTags = [paramTag(h0, w0, cS, helix) for helix,h0,w0,cS in setList] 
    indxs = [mainDF[mainDF.tag==configTag].index[0] for configTag in configTags]
    thisDF = mainDF.iloc[indxs]
    #newList = [group.get_group(tup).cf.item() for tup in setList]
    return thisDF


def get_confs_inSetList(setList, appendTo=None):
    global mainDF
    for helix, h0, w0, cS in setList:
        configTag = paramTag(h0,w0,cS,helix)
        confFile = inputDir+'conf_'+configTag+'.config'  
        newConfig = read_conf(confFile, configTag)
        if appendTo is not None:
            appendTo.append(newConfig)
        newRow =  {'H0':h0,'W0':w0,'helix':helix,'init':0,'cf':newConfig,'tag':configTag,'cS':cS}
        mainDF = mainDF.append(newRow, ignore_index=True)
    return "{0} configurations read and currently {1} in mainDF".format(len(setList),len(mainDF))

def get_confs_inList(fileList, appendTo=None):
    listConfs = []
    for file in fileList:
        configTag = file[5:-7]
        #confFile = inputDir+'conf_'+configTag+'.config'  
        newConfig = read_conf(file, configTag)
        listConfs.append(newConfig)
        if appendTo is not None:
            appendTo.append(newConfig)
    return listConfs
    #     newRow =  {'H0':h0,'W0':w0,'helix':helix,'init':0,'cf':newConfig,'tag':configTag,'cS':cS}
    #     mainDF = mainDF.append(newRow, ignore_index=True)
    # return "{0} configurations read and currently {1} in mainDF".format(len(fileList),len(mainDF))


################################################################################
################################################################################

##################  PLOTTING FUNCTIONS

################################################################################
import matplotlib.pyplot as plt

##import cmasher as cmr   ---> not working on Lenovo
def create_cmr_rainforest():    
    '''From https://github.com/1313e/CMasher, when package is not installed
    # All declaration
        __all__ = ['cmap']
    # Author declaration
        __author__ = "Ellert van der Velden (@1313e)"
    # Package declaration
        __package__ = 'cmasher'
    # Type of this colormap
        cm_type = 'sequential'
'''
    
    from matplotlib.cm import register_cmap
    from matplotlib.colors import ListedColormap
    # %% GLOBALS AND DEFINITIONS 

    # RGB-values of this colormap
    cm_data = colors_rainforest
    #cm_data_r = cm_data.reverse()
    cm_data_r = cm_data[::-1]

    # Create ListedColormap object for this colormap
    cmap = ListedColormap(cm_data, name='cmr_rainforest', N=256)
    cmap_r = ListedColormap(cm_data_r, name='cmr_rainforest_r', N=256)
    # Register (reversed) cmap in MPL
    register_cmap(cmap=cmap)
    register_cmap(cmap=cmap_r)
    return cmap, cmap_r

cmr_rainforest, cmr_rainforest_r = create_cmr_rainforest()


def translate_cmap(cm=cmr_rainforest,alpha=255):#cmr.rainforest,alpha=255):
    ''' take colormaps like in matplotlib, and translate to LUT used by mayavi,
    which is a 255x4 array, with the columns representing RGBA
 (red, green, blue, alpha) coded with integers going from 0 to 255.
 set alpha to fully solid by default'''
    Array4 = [[r*255,g*255,b*255,alpha] for r,g,b in cm.colors]
    return Array4

def mayavi_plot_field(listConfs, attr='L', minMax=None, dimensionless=True, 
                        sameScale=True, cmap='viridis'):#,save=defSave):
    ''' Mayavi surface plot for a list of Conf objects, using colormap.
    Shapes are colored according to _attr_ (default: dim stretching E = E_L/cS).
    If no minMax tuple/list is given but using _sameScale_, read out the bounds from the provided confs.
    Plot dimensionless curvature if _dimensionless_. 
    To use costum colormap, pass on a colormap object (cmr.rainforest), instead of the string
    '''

    xOffset = 0.
        
    
    if sameScale:  #colormap with a single range
       #rm# allValues = []    
       #rm# for cf in listConfs:        
       #rm#     values = getattr(cf,attr)
       #rm#     allValues.append(values)
        #minVal, maxVal = get_attrBounds(listConfs, attr)   
        #print (minVal,maxVal)
        if minMax is not None:
            print('minmax is not None')
            minVal, maxVal = minMax[0], minMax[1]    
        else:
            print('noMinMax, get it')
            minVal, maxVal = get_attrBounds(listConfs, attr)   
        
        if dimensionless:
            minVal = renormalize(attr,minVal)
            maxVal = renormalize(attr,maxVal)
            
        colormap = cmap if isinstance(cmap,str) else 'viridis'

        print("Min and Max range values are: {0} and {1}".format(minVal,maxVal))     
        for cf in listConfs:
            field = np.array(getattr(cf,attr))
            if dimensionless:
                field = renormalize(attr,field)
            mesh = mlab.triangular_mesh(cf.x+xOffset, cf.y, cf.z, cf.triangles, 
                             scalars = field,representation='surface',
                             reset_zoom = False, vmin=minVal,vmax=maxVal,
                             colormap=colormap)
            if not isinstance(cmap, str):
                LUT = translate_cmap(cmap)
                mesh.module_manager.scalar_lut_manager.lut.table = LUT
            xOffset += (cf.x.max()-cf.x.min())*1.05    
        print("Min and Max range values are: {0} and {1}".format(minVal,maxVal)) 
    
    else:  #not same scale
        for cf in listConfs:
            field = np.array(getattr(cf,attr))
            mesh = mlab.triangular_mesh(cf.x+xOffset, cf.y, cf.z, cf.triangles, 
                             scalars = field,representation='surface',
                             reset_zoom = False,colormap=colormap)
            xOffset += (cf.x.max()-cf.x.min())*1.05    
            
    mlab.colorbar(title=attr, orientation='horizontal', nb_labels=5)
    mlab.show()
    return


def mayavi_one_plot_field(cf, attr='K', minMax=minMax_K_manuscript, 
                          dimensionless=True, cmap=cmr_rainforest, 
                          surfaceRep='surface',save=defSave, name='p1',
                          figSize=(1200,950),magnification=2,
                          showColorbar=False):#,save=defSave):
    ''' Mayavi surface plot for a list of SINGLE Conf obj, using colormap.
    Shapes are colored according to _attr_ (default: dim Gaussian curv K, as in manuscript).
    If no minMax tuple/list is given, plot default without.
    Plot dimensionless curvature if _dimensionless_. 
    To use costum colormap, pass on a colormap object (cmr.rainforest), instead of the string
    '''
    
    fig = mlab.figure(size=figSize, bgcolor=(1,1,1), fgcolor=(0.5,0.5,0.5))#mlab.figure()
    #colormap = cmap
    colormap = cmap if isinstance(cmap,str) else 'viridis'
    field = np.array(getattr(cf,attr))        
    if dimensionless:
        field = renormalize(attr,field)   

    if minMax is not None:
        print('minmax is not None')
        minVal, maxVal = minMax[0], minMax[1]  
        if dimensionless:
            minVal = renormalize(attr,minVal)
            maxVal = renormalize(attr,maxVal)
        mesh = mlab.triangular_mesh(cf.x, cf.y, cf.z, cf.triangles, 
                             scalars = field, representation=surfaceRep,
                             reset_zoom = False, vmin=minVal,vmax=maxVal,
                             colormap=colormap)        
    else:
        mesh = mlab.triangular_mesh(cf.x, cf.y, cf.z, cf.triangles, 
                             scalars = field, representation=surfaceRep,
                             reset_zoom = False,colormap=colormap)
        
    if not isinstance(cmap, str):
        LUT = translate_cmap(cmap)
        mesh.module_manager.scalar_lut_manager.lut.table = LUT
        
    
    
    if save: 
        mlab.savefig(outDir+name+'.png',magnification=magnification)
        #imgmap = mlab.screenshot(figure=fig, mode='rgba', antialiased=True)
        #mlab.close(fig)
        #fig2 = plt.figure(figsize=(7, 5))
        #plt.imshow(imgmap, zorder=4)
    if showColorbar:
        mlab.colorbar(title=attr, orientation='horizontal', nb_labels=5)
        
    return fig

def plot_flexible(attr='L', confsList=[], minMax=None, dim=True, 
                  save=defSave,txt='',cmap=cmr_rainforest_r):#'viridis'):
    ''' Equilibrium sheet configurations for low FvK (stretchy/flexible) surface.
    Uses the function: mayavi_plot_field() for the corresponding surfaces 
    with the parameters in the _setList_ given. 
    '''
    global mainDF    

    mlab.figure()
    mayavi_plot_field(confsList, attr, dimensionless=dim,cmap=cmap,minMax=minMax)
    if save:
        mlab.savefig(outDir + "fig2_"+attr+txt+".png")
    
    

def plot_and_save(cf, xtraName='',mag=2):
    fig = mayavi_one_plot_field(cf)
    mlab.show()
    ##input("Adjust figure... Press Enter to continue...")
    xtraName = xtraName+'_x{}'.format(mag)
    
    
################################################################################
################################################################################

##################  GET DATA

################################################################################    

# If adding init configurations from setup. In this case for helix. 
configs_Init = []
for helix in ['C']:#,'B','A']:
    add_init(helix,appendTo=configs_Init)    
    
configs_results = []
get_confs_inSetList(sets_results, configs_results)

################################################################################
##################  Go for it
################################################################################


plot_flexible('K',configs_results[:2],minMax=minMax_K_manuscript,save=False)