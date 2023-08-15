#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Aug 19 21:59:13 2021

Config object and local plotting functions using matplotlib 


@author: garcia-aguilar
"""

wDir = '../'


import pandas as pd
import matplotlib.pyplot as plt
import numpy as np


#### DATA ----------
#--------------------

###############################################################################    
# --------------------------------------------------------__#
    # This part is specific for the tags on HelixRuns/ #

confs=[]

helix='B'
dH0 = 1.0
dW0 = 1.0
cS = 60.0
cB = 1.0
width = 1.06

runValue = 309951
tag = '_{}_cS{}_H{}_W{}'.format(helix, cS, dH0, dW0)
longTag = '_{}_c1.0_cS{}_H{}_W{}'.format(helix, cS, dH0, dW0)
resDir = './Helix{}/Run/Helix{}_from300951_ok/'.format(helix,tag)

### Here, change to any configuration file naming, does not have to be helix

configFile = resDir+'Confs/conf{}_{}.config'.format(longTag, runValue)



# --------------------------------------------------------__#
################################################################################ 


#### CLASS & FUNCTIONS ----------
#---------------------------------


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
        self.H2 = cDF.H2
        self.H = cDF.H
        self.W = cDF.W
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
        self.totA = self.A.sum()
        self.netA = self.totA - np.sum(self.A*self.bd)
        self.intK = np.sum(self.K*self.A)
        self.sumK2 = np.sum(self.K*self.K)
        self.sumAbsK = np.sum(abs(self.K))
        self.dE = np.linalg.norm(np.array([self.dEx, self.dEy, self.dEz]))
        
        
    # Double-check if len(tr) matches n_tr read, same as len(v). 
        if len(self.bd) != self.n:
            print("Only {} vertices read instead of {} in the file".format(len(self.x),self.n))
            return 1
        if len(self.triangles) != self.n_tr:
            print("Only {} triangles read instead of {} in the file".format(len(self.triangles),self.n_tr))           
            return 1  
       
def read_conf(fil,tag):
    cf = Config(tag)
    cf.getConf(fil)
    return cf

def cf_area_average(cf, field='H',pw=1):   
        vals = pow(getattr(cf,field),pw)*cf.A
        return vals.sum()/cf.netA
        
def cf_tot_average(cf, field='H',pw=1, skipBoundaries=True):
        bdFactor = 1.
        num = cf.n
        if skipBoundaries:
            bdFactor = 1.-cf.bd
            num = np.count_nonzero(cf.bd==0)
        
        vals = pow(getattr(cf,field)*bdFactor,pw)        
        return vals.sum()/num      
    
def add_curvatureAverages(cfs=[]):
    for cf in cfs:    
        cf.navL = cf_tot_average(cf, field='L',pw=1,skipBoundaries=False)
        
        cf.avH = cf_area_average(cf, field='H',pw=1)
        cf.navH = cf_tot_average(cf, field='H',pw=1)
        cf.avH2 = cf_area_average(cf, field='H',pw=2)
        cf.navH2 = cf_tot_average(cf, field='H',pw=2)
        
        cf.avK = cf_area_average(cf, field='K',pw=1)
        cf.navK = cf_tot_average(cf, field='K',pw=1)
        
        cf.avK2 = cf_area_average(cf, field='K',pw=2)
        cf.navK2 = cf_tot_average(cf, field='K',pw=2)
        
        cf.avW = cf_area_average(cf, field='W',pw=1)
        cf.navW = cf_tot_average(cf, field='W',pw=1)

def get_attrBounds(cfs=[], attr='H'):
    maxs = [np.max(getattr(cf,attr)) for cf in cfs]
    mins = [np.min(getattr(cf,attr)) for cf in cfs]
    return np.min(mins), np.max(maxs)

# Get geometrical bounds of allData
def get_shapeBounds(cfs = []):
    lims=[]
    ptps = []
    for coord in ['x','y','z']:
        maxs = np.array([np.max(getattr(cf,coord)) for cf in cfs])
        mins = np.array([np.min(getattr(cf,coord)) for cf in cfs])
        lims.append([mins.min(),maxs.max()])
        ptps.append(lims[-1][1]-lims[-1][0])
    for l, lim in enumerate(lims):
        lim =np.array(lim)*1.01
        lims[l] = lim
    return lims[0],lims[1],lims[2],ptps
 
    
 ################################################################################   
 ##################  GET DATA   #######################################
 ################################################################################
###############################################################################    
# --------------------------------------------------------__#
    # This part is specific for the tags on HelixRuns/ #
    

hB = read_conf(configFile,'H{}_W{}'.format(dH0,dW0))
confs.append(hB)
    
# --------------------------------------------------------__#
################################################################################ 


xlims, ylims, zlims, ptps = get_shapeBounds(confs)
# Based on these, get fixed box aspect ratio
coordShort = np.array(ptps).argmin()
shortestSide = ptps[coordShort]
aspRatio = np.array(ptps)/shortestSide


#add_curvatureAverages(confs)
    

    
################################################################################   
##################  PLOT   #######################################
################################################################################    
import matplotlib.tri as tri
import mpl_toolkits.mplot3d.axes3d
import matplotlib.ticker as mticker


# == SHAPES ===============

def plot_attribute(config=None, attr='H', minmax=[], 
                   noGrid=False, sameScale=False, curvFactor=2*np.pi):      
'''
    Parameters
    ----------
    config : Config object
        From the class define in the script. The default is None.
    attr : str, optional
        Local field, or vertex attribute in Config class (for example other
        curvatures such as 'W', or 'K'.  The default is 'H'.
    minmax : list, optional
        if not empty, uses firs two entries as plot ranges. Else, it calculates them
        using get_attrBounds(). The default is [].
    noGrid : boolean, optional
        No plot grid. The default is False.
    sameScale : boolean, optional
        Colormap with fixed max/min. The default is False.
    curvFactor : float, optional
        Normalization factor to the field. For curvatures, this is related to using
        the dimensionless expressions. The default is 2*np.pi.

    Returns
    -------
    None.

    '''
    #config = confs[helix]
    width=1.06
    if not config:
        return "GET A CONFIGURATION TO RUN plot_attribute"
    fig1 = plt.figure(figsize=(7.5,6))   #i#plt.figure()
    ax1 = fig1.gca(projection='3d')        
    ax1.set_xlabel('x',fontsize=14)
    ax1.set_ylabel('y',fontsize=14)
    ax1.set_zlabel('z',fontsize=14)   
    
    if attr in ['H','W']:
        curvFactor = 1 * width/np.pi
    elif attr in ['H2', 'K', 'AbsK']:
        curvFactor = width**2/(np.pi)**2
    else:
        curvFactor = 1

    ###curvFactor=1    #CHECK TODO RM
    
    vcolor = list(getattr(config,attr)*curvFactor)       
    if len(minmax):
        aMin, aMax = minmax
    else:    
        aMin, aMax = get_attrBounds(confs,attr)
        
    tg = tri.Triangulation(config.x, config.y, triangles=config.triangles)
    ax1.plot_trisurf(tg, Z=config.z)   
    
    if not attr=='cst':
        if sameScale:
            #norm = matplotlib.colors.Normalize(vmin=aMin,vmax=aMax)
            #facecolors = plt.cm.viridis(norm(vcolor))    
            #sm = plt.cm.ScalarMappable(norm=norm,cmap=plt.cm.viridis)
            p = ax1.scatter(config.x, config.y, config.z, c=vcolor, vmin=aMin*curvFactor, vmax=aMax*curvFactor)
        else:
            p = ax1.scatter(config.x, config.y, config.z, c=vcolor)      
            
        fig1.colorbar(p,ax=ax1)

    ax1.set_xlim3d(*xlims)
    ax1.set_ylim3d(*ylims)
    ax1.set_zlim3d(*zlims)
    ax1.set_box_aspect((aspRatio[0],aspRatio[1],aspRatio[2]))
        
    plt.tight_layout()       
    if noGrid:
        ax1.set_axis_off() 
    return fig1    



