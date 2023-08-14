#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu April 25 21

v3.1
(Changes with respect to previous versions are highlighted below.
 Smaller changes are not kept track of)

v3.1 -> Setup from scratch

Simulation for triangulated surfaces for coordinate-dependent energies. 
The algorithm finds the vertex positions that minimize the elastic energy. The
stretching energy is given by standard Hooke springs, while the bending energy
is given by the Fischer energy 
(Fischer, TM, Journal de Physique II France, 3 (1993). DOI: 10.1051/jp2:1993230)


Older versions of the code (Zwaan, Garcia-Aguilar) in:
SurfaceWarper/Old/

Changes with respect to v2:
Compared to the initial code, here I tried to leave only the necessary functions
for the actual implementation in use. In other words, old functions, and other 
evolution algorithms implemented were removed. 

(**chk wrt. changes below)    
    
Changes with respect to v1 (S.Zwaan):

Furthermore, the end goal here is to express the energy as a function of a the
mesh coordinates given by a (3,N) array. Before, the energy was calculated by 
accesing the different Vertex objects containing the relevant vertex fields. 
Since functions are in terms of the index i of a particular Vertex object, 
a restructuring is needed to accomplish this goal. 

The vertex numbering and order is still important for the mesh connectivity
and for the correct calculation of the surface normal/curvature vector. 
The class Vertex will be kept, as it is an easy way to access the geometric
informaton when studying a particular shape, so it might be a bit of a mixed
Vertex objects/arrays scheme

(TODO)
Potentially, implement here the 3D plotting with Mayavi as well. 

"""



allCNT = 0

import numpy as np
import matplotlib.tri as tri
import mpl_toolkits.mplot3d.axes3d
import matplotlib.pyplot as plt
import time
import pandas as pd
import scipy.optimize as opt
import atexit
import os

outDir = './'

TOL = 1e-12
CONVTOL = 1e-9

# =============================================================================
# Input parameters
# =============================================================================
start_conf = 'h'     
bd_mode = 'u'     
grad_mode = 'fa'   #i# commented below that only 'a' greedy analyt works
lmin_method = 'fminb'   #fminb, min, mins, mins_b

step_size = 0.001    #Size of gradient descent update step for vertices
dx = 1e-8                   #For numerically determining the gradient 
noise = 0.01      #i# noise for vertex displacement in range [0,noise*l_0]

min_step_size = 1e-10#dx*10.   #for line minimization bracketing
max_f = 0.5#1 #*l_0   # for line min bracketing, use max = f*l_0 
max_step_size = max_f  #initial value, updated when calculating l_0

#i# gc_noise = 0.01   #i# not used anywhere else atm

#Starting with a (long) cylinder:
R = 1 #Also radius for hemisphere
h_cyl = 2#10#1

#Starting with a hexagonal sheet:
Lx_sheet = 3*R#1.22             #i# set like this originally but not sure why
n_x = 4                    #Number of vertices in a row
n_y = n_x*2                   #Number of vertices in a column
#l   = 1/(n_x-1.22)          #Standard edge length, can differ due to z-coord
l = Lx_sheet/(n_x-1.)


#Starting with a long cylinder:
h_seg = 4*R #Vertical distance between two triplets of vertices
n_seg = 5*4 #Number of segments to start with. Height of cylinder is h_seg*n_seg, n_seg>0.
cap = 0  #Put cap on cylinder? 0 or 1

#Starting with a helicoid:
rh = 1.0           #radius
wh  = 0.5          #width
hh  = 3            #heigth (of top line, such that total heigh is hh+wh)

#Energy constants:
cH = 0.1    #stretching relaxation
H0 = 0
cW = 0.1    #stretching relaxation
W0 = 0
cL = 1           #Constant for Hooke energy E_L = cL sum((|v_i-v_j|-l_0)/2)
l_0 = 0           #l_0 for Hooke Energy, default set to average edge length (of starting config)

fixed_v = []#[0,1,2]#list(range(n_x))#[]        #i# list of bd vertices which are fixed 
# [0,1,2]  one-sided cylinder
# list(range(n_x)) one-sided sheet. If aspectratio !=1, it fixes short side


# =============================================================================
# Vertex class
# =============================================================================
class Vertex:
    
    def __init__(self):
        self.c       = np.zeros(3) #Coordinates
        self.nb      = np.array([], dtype=int) #Neighbourhood vertices
        self.bd      = 0 #Boundary
        self.fixed   = 0
        self.A       = 0 #Area associated to vertex
        self.dA      = np.zeros(3)
        self.H2      = 0 #Squared mean curv
        self.dH2     = np.zeros(3)
        self.H       = 0 #Mean curv (with sign)
        self.dH      = np.zeros(3)
        self.K       = 0 #Guassian curv
        self.dK      = np.zeros(3)
        self.L       = 0 #Hooke energy
        self.dL      = np.zeros(3)
        self.W2      = 0 #Squared warp
        self.dW2     = np.zeros(3)    # Not used atm in current fxs
        self.W       = 0 #Warp
        self.dW      = np.zeros(3)
        self.E_H     = 0
        self.E_W     = 0
        self.E_L     = 0
        self.E       = 0 #Energy of vertex
        self.dE_H    = np.zeros(3)
        self.dE_W    = np.zeros(3)
        self.dE_L    = np.zeros(3)
        self.dE      = np.zeros(3)
        self.dE_prev = np.zeros(3) #Save previous gradient values for restoration
        self.dE_num  = np.zeros(3)
        self.dE_i    = np.zeros(3) 
        self.dE_js   = np.zeros(3)  #Gradient contribution from grad in neighborhood energy 
        self.norm    = np.zeros(3)
        self.Hnorm   = np.zeros(3)    #Normal mean curvature vector = S/(4A)
        
        #For full analytical gradient
        self.dA_i    = np.zeros(3)
        self.dH_i    = np.zeros(3)
        self.dK_i    = np.zeros(3)
        self.dW_i    = np.zeros(3)
        self.DS_i    = np.zeros((3,3))

# =============================================================================
# Initialize variables
# =============================================================================
n = 0               #Number of vertices
v = []              #List of Vertex class elements
tr = []             #Triangle list
n_tr = 0            #Number of triangles
bd_v = []          #i# list of boundary vertices
n_obtuse = 0        #i# number of obtuse triangles after analysis
curr_bd_c = []      #i# temporary list of boundary positions to assess free boundaries
tr_obtuse = [0]      #i# list of 0 or 1 if corresponding triangle in tr[] is obtuse

E_array = []
M_array = []
dE_array = []
S_array = []
Conv_array = []
fresh = 1           #If 1, recalculate before taking a step
#dx = 1e-9

DFcol = ['conf', 'E', 'E_H', 'E_W', 'E_L', 'A', 'Nm', 'Nbd']
dataDF = pd.DataFrame(columns=DFcol)
        
# =============================================================================
# Setup functions        
# =============================================================================
def summary():
    ''' Prints a summary of the user functions'''
    print('A summary of the user functions:')
    print('  reset() - resets the surface to its starting configuration')
    print('  choose_start_conf() - choose starting configuration [s,c,l,h,o,i]')
    print('  \t Sheet, Cylinder, Longcylinder, Helicoid, semisphere, Icosahedron')
    print('  choose_bd_mode() - choose boundary mode [u,f,p,a**,e**,h*]')
    print('  \tUnconstraint, Fixed, Partially-constrained')
    print('  choose_grad_mode() - choose gradient mode [fa,a,n,f]')
    print('  \tFullAnalytical, greedyAnalytical, greedyNumerical, Fullnumerical')
    print('  g(n,[mod],[False]) - move vertices in negative gradient direction')
    print('      n - number of steps, mod - step size modifier')
    print('  show([False]) - Show the current surface (does not update automatically)')
    print('  refine() - Refine the surface - creating more vertices and triangles')
    print('  analysis([False]) - Print various values regarding the shape of the surface')
    print('  export_conf(file) - Save to file current mesh information')
    print('  read_saved_conf(file) - Read from file some mesh')
    
    

def set_vc_ico():
    ''' Set vertex positions and triangulation of a regular icosahedron with
    edge length=2, rotated in a way that a vertex is in the top Z. '''
    global n, n_tr, tr, tr_obtuse
    for i in range(12):
        v.append(Vertex())
    n = 12
    n_tr = 20
    p = (1+np.sqrt(5))/2
    v[ 0].c = np.array([ p, 0, 1])
    v[ 1].c = np.array([ p, 0,-1])
    v[ 2].c = np.array([ 1, p, 0])
    v[ 3].c = np.array([ 0, 1, p])
    v[ 4].c = np.array([ 0,-1, p])
    v[ 5].c = np.array([ 1,-p, 0])
    v[ 6].c = np.array([-p, 0,-1])
    v[ 7].c = np.array([ 0, 1,-p])
    v[ 8].c = np.array([-1, p, 0])
    v[ 9].c = np.array([-p, 0 ,1])
    v[10].c = np.array([-1,-p, 0])
    v[11].c = np.array([ 0,-1,-p])
    
    tr = np.array([   [ 0, 1, 2],
                      [ 0, 2, 3],
                      [ 0, 3, 4],
                      [ 0, 4, 5],
                      [ 0, 5, 1],
                      [ 6, 8, 7],
                      [ 6, 9, 8],
                      [ 6,10, 9],
                      [ 6,11,10],
                      [ 6, 7,11],                      
                      [ 7, 1,11],
                      [ 1, 7, 2],
                      [ 2, 7, 8],
                      [ 2, 8, 3],
                      [ 3, 8, 9],
                      [ 3, 9, 4],
                      [ 4, 9,10],
                      [ 4,10, 5],
                      [ 5,10,11],
                      [ 5,11, 1] ])
    tr_obtuse=[0]*n_tr
    
    #i# using golden ratio permutation, there is an edge on the "top" z.
      # rotate the icosahedron so there is a vertex here. 
    rot_angle = np.arctan2(v[3].c[1],v[3].c[2])
    rot_matrix = np.array([ [1,0,0],
                            [0,np.cos(rot_angle),-1*np.sin(rot_angle)],
                            [0,np.sin(rot_angle),np.cos(rot_angle)]
                            ])
    for i in range(n):
        v[i].c = rot_matrix @ v[i].c 
    return 0

def get_positive_halfIco(threshold=-0.1):     
    ''' Call after constructing a full icosahedron/projection to take only a 
    z-defined portion of the vertices. When building a semisphere, this corresponds
    to the positive half.
    The threshold is set to be -0.1, because of numerical acc. getting rid of
    relevant vertices.'''
    
    
    for i in reversed(range(n)):
        if v[i].c[2] > threshold: #< threshold: # -0.1:
            remove_vert(i)  
    return 0

def remove_vert(i):
    ''' Removes vertex with index i in v-list. Removes triangles associated to it
    and relabels indeces for triangles with vertices of higher i. 
    Used only for geometry building. If ever used for anything else, clear also 
    bd_v-list in case that this is a boundary'''
    
    global v, tr, n, n_tr, tr_obtuse
    new_tr = np.copy(tr)
    for j in reversed(range(n_tr)):
        if i in tr[j,:]:
            new_tr = np.delete(new_tr,j,0)
            n_tr-=1
    del v[i]    
    n-=1
    tr=new_tr    
    tr_obtuse=[0]*n_tr
    for j in range(n_tr):
        for k in range(3):
            if tr[j,k]>i:
                tr[j,k]-=1 
    if i in bd_v:
        bd_v.remove(i)
    if i in fixed_v:
        fixed_v.remove(i)
    return 0
    

def set_vc_icoHemisphere(half=True,onlyIco=False):
    ''' Builds the hemisphere starting from a semi icosaheron, before refinement.
    This way the initial boundary points can be identified. '''
    
    global v, bd_v
    
    bd_v = []    
    set_vc_ico()
    if not onlyIco:
        refine()
    proj_hemi()
    if half:
        get_positive_halfIco()
    for i in range(n):
        constr_nb(i)
    
    

        
#set_z_coord(v):
#Sets z-coords of vertices, as function of x and y coords
# =============================================================================
def set_z_coord(z=0):
    ''' Set height (or z-coord) of points for a hexagonal sheet.
    For the moment, keep flat, z=0. But it can be changed'''
    for i in range(n):
#        x = v[i].c[0]
#        y = v[i].c[1]
#        z = ((1 - (x-0.5)**2) * (1 - (y/0.886-0.5)**2))
#        z = ((1 - (x-0.5)**2) * (1 - (y-0.5)**2))
#        z = np.sqrt(1-y*y)
#        z = (1 - np.abs(x-0.5)) * (1 - np.abs(y/0.886-0.5))
#        z = x*(1-x)*y/0.886*(1-y/0.886)
#        z*=18
        
        v[i].c[2] = z
#        if v[i].bd:
#            v[i].c[2] = 0
    return 0

#set_vc():
#Initialize the vertex coords in a hexagonal pattern
# =============================================================================
def set_vc():
    '''Generate vertices of flat meshed sheet. 
       For the moment, only called when start_conf='S'. 
       '''
    print('Generating vertices...')
    global n
    n = n_x*n_y
    count=0
    h_tri = np.sqrt(3)/2
    for i in np.arange(n_y):
        for j in np.arange(n_x):
            v.append(Vertex())
            v[count].c[0] = j*l
            v[count].c[1] = i*l*h_tri          
            if i % 2 == 1:
                v[count].c[0]+=l/2.0
            if i == 0 or i == n_y-1 or j==0 or j==n_x-1:
                v[count].bd = 1
                bd_v.append(count)
            count+=1
    set_z_coord(1)
    return 0

#set_tr():
#Create triangles in counterclockwise direction, works for hexagonal meshes 
#created by set_vc()
# =============================================================================
def set_tr():
    '''Generate triangles of flat meshed sheet. 
    Only called when start_conf='S' 
    It goes per y row, adding triangles and checking whether they are boundary terms.
    '''
    
    print('Generating triangles...')
    global tr, n_tr, tr_obtuse
    count=0
    for i in np.arange(n_y-1):
        for j in np.arange(n_x):
            offset=0
            #Check if the vertex is on the boundary (and where on it)
            check1, check2 = 1, 1
            if i==0 and j==0:
                tr = np.array([[0,1,n_x]])
                check1, check2 = 0, 0
            elif i%2 == 0:
                if j==0:
                    check2 = 0
                elif j==(n_x-1):
                    check1 = 0
            else:
                offset=1
                if j==(n_x-1):
                    #The second to last row of points' triangles are included 
                    #from the bulk or the boundary, so these are not considered
                    check1 = 0
                    check2 = 0
            #Add the appropriate triangle(s), considering boundary
            if check1:
                triangle = np.array([0,1,n_x+offset])+count
                triangle = triangle[np.newaxis,:]
                tr = np.concatenate((tr,triangle),axis=0) 
            if check2:
                triangle = np.array([0,n_x+offset,n_x-1+offset])+count
                triangle = triangle[np.newaxis,:]
                tr = np.concatenate((tr,triangle),axis=0)        
            count += 1
    n_tr = tr.shape[0]
    tr_obtuse=[0]*n_tr
    return 0


def pb_bottom(p):
    ''' Set a point with polar coords (R, p, 0)'''
    x = R*np.cos(p)
    y = R*np.sin(p)
    z = 0
    return np.array([x,y,z])
    
def pb_top(p):
    ''' Set a point with polar coords (R, p, h_cyl)'''
    x = R*np.cos(p)
    y = R*np.sin(p)
    z = h_cyl
    return np.array([x,y,z])

def set_vc_capcyl():
    ''' Set vertices and triangles for a capped cylinder  in a layered way by
    adding triplets of vertices forming a triangle, with each triplet pair
    corresponding to a segment.
    n_seg = number of these pairs, h_seg= distance between each, in a way that
    the total height of the cylinder=h_seg*n_seg. 
    if variable cap=1, add an additional vertex on top, at a distance rh from last
    triplet and face-centered.'''
    global tr, n_tr, n, bd_v, tr_obtuse
    print('Generating vertices...')
    
    n = (n_seg+1)*3 + cap
    bd_v = [0,1,2, n-3,n-2,n-1]  # boundaries are first and last vertex triplet set

    for i in range(n_seg+1):
        for j in range(3):
            v.append(Vertex())
            v[-1].c = pb_bottom(2*np.pi/3*(j+0.5*i))
            v[-1].c[2] = h_seg*(i - 0.5*n_seg)
    if cap:
        # add a vertex in the center of the cap, to later project it to a hemisphere. 
        v.append(Vertex())
        v[-1].c = np.array([0,0,0], dtype='float')
        v[-1].c[2] = 0.5*n_seg*h_seg+rh
        v[-1].cap = 1
        bd_v = bd_v[0:3]  # if there is a cap, only one boundary. 
            
    print('Setting triangles...')
    for i in range(n_seg):
        tr_piece = np.array([   [ 0, 1, 3],
                                [ 1, 4, 3],
                                [ 1, 2, 4],
                                [ 2, 5, 4],
                                [ 2, 0, 5],
                                [ 0, 3, 5],
                                ])
        tr_piece += 3*i
        if i==0:
            tr=tr_piece
        else:
            tr = np.concatenate((tr, tr_piece), axis=0)
    if cap:
        tr_piece = np.array([   [ 0, 1, 3],
                                [ 1, 2, 3],
                                [ 2, 0, 3],
                                ])
        tr_piece += 3*n_seg
        tr = np.concatenate((tr, tr_piece), axis=0)

    ###n = (n_seg+1)*3 + cap
    n_tr = tr.shape[0]    
    tr_obtuse=[0]*n_tr
    return 0

def set_vc_pyramid_hemisphere():       #i# not used in current code atm
    ''' Construct a semisphere by starting with a triangular pyramid. 
    An alternative method to constructing a semi-ico. This one however, leads to
    the top point (top of pyramid) with obtuse triangles around'''
    global tr, n_tr, n, v, bd_v, tr_obtuse
    v = []
    print('Generating vertices...')
    n=4
    bd_v = [0,1,2]
    for j in range(3):
        v.append(Vertex())
        v[-1].c = pb_bottom(2*np.pi/3*(j))

    v.append(Vertex())
    v[-1].c = np.array([0,0,R], dtype='float')
            
    print('Setting triangles...')
    tr = np.array([[ 0, 1, 3],
                   [ 1, 2, 3],
                   [ 2, 0, 3],
                   ])
        
    n_tr = tr.shape[0]   
    tr_obtuse=[0]*n_tr
    return 0

def proj_hemi():
    ''' Project into a sphere of radius R'''
    for i in range(n):
        rs = np.linalg.norm(v[i].c)
        v[i].c *= R/rs
        
def remove_hemi_highest():      #i# not used in the current code
    ''' This is a manual fix that was implemented for the hemispheres built from
    a triangular pyramid, where the top points ends surrounded by obtuse 
    triangles... Not used anymore'''
    global v, tr, n, n_tr, tr_obtuse
    del v[3]
                               #i# why those triangles, why the following lines?
    tr = np.delete(tr,160,0)   #i# delete(array,obj,axis)
    tr = np.delete(tr,96,0)
    tr = np.delete(tr,32,0)
    tr = np.append(tr, np.array([[49,51,78]]), 0)
    for i in range(n_tr-2):
        for j in range(3):
            if tr[i,j]>3:
                tr[i,j]-=1
    n -= 1
    n_tr -= 2
    tr_obtuse=[0]*n_tr
    

def set_vc_cyl():
    ''' Generate the initial vertices of a cylinder. Only called on "C" mode
    Build 2 triangles as the top/bottom boundaries, with the bottom at z=0 and
    the top at z = h_cyl defined as a initial parameter.'''
    global tr, n_tr, n, bd_v, tr_obtuse
    n=6
    print('Generating vertices...')
    for i in range(3):
        v.append(Vertex())
        v[-1].c = pb_bottom(2*np.pi/3*i)
    for i in range(3):
        v.append(Vertex())
        v[-1].c = pb_top(2*np.pi/3*(i+0.5))    
    bd_v = np.arange(6).tolist()
    
    print('Setting triangles...')
    tr = np.array([   [ 0, 1, 3],
                      [ 1, 4, 3],
                      [ 1, 2, 4],
                      [ 2, 5, 4],
                      [ 2, 0, 5],
                      [ 0, 3, 5], ])
    n_tr = tr.shape[0]   
    tr_obtuse=[0]*n_tr
    return 0

def proj_cyl():
    ''' Projects the coordinates of all vertices to have a fixed polar radius,
    for the cylinder and the capped cylinder of radius R. It also projects 
    this cap and, in general, any hemisphere, to a hemi-sphere of radius R.. 
    '''
    print('Projecting on cylinder...')
    global v
    if start_conf in ['H','h']:
        r = rh
    else:
        r = R
    for i in range(n):
        if start_conf not in ['L','l'] or v[i].c[2] <= 0.5*h_seg*n_seg:
            rs = np.sqrt(v[i].c[0]**2 + v[i].c[1]**2)
            v[i].c[0] *= r/rs
            v[i].c[1] *= r/rs
        #The cap of a long cylinder should be projected to a half-shpere
        else:
            v[i].c[2] -= 0.5*h_seg*n_seg
            rs = np.linalg.norm(v[i].c)
            v[i].c *= r/rs
            v[i].c[2] += 0.5*h_seg*n_seg

def pb_lowerStrip(p):
    ''' Set a point with polar coords (R, p, z(p)), therefore being the lower
    boundary of the helicoidal strip. 
    Called from helix where xy-projection is a pentagon and therefore p has 
    5-symmetry'''
    x = rh*np.cos(p)
    y = rh*np.sin(p)
    z = hh*p/2/np.pi
    return np.array([x,y,z])

def pb_higherStrip(p):
    ''' Set a point with polar coords (R, p, z(p)+wh), therefore being the
    "upper" boundary of a helicoidal strip of width wh. 
    Called from helix where xy-projection is a pentagon and therefore p has 
    5-symmetry'''
    x = rh*np.cos(p)
    y = rh*np.sin(p)
    z = hh*p/2/np.pi + wh
    return np.array([x,y,z])
   
def set_vc_helix():
    ''' Set vertex positions and triangles for an helocoidal strip of a single 
    turn, by defining the two strip boundaries separated by the width wh of the 
    helicoid. Vertices in a turn are defined in a pentagonal symmetry, and the 
    total height of the shape is then hh+wh. '''
    global tr, n_tr, n, bd_v, tr_obtuse
    n=12
    bd_v = np.arange(n).tolist()
    print('Generating vertices...')
    for i in range(6):
        v.append(Vertex())
        v[-1].c = pb_lowerStrip(2*np.pi/5*i)
    for i in range(6):
        v.append(Vertex())
        v[-1].c = pb_higherStrip(2*np.pi/5*i)
    
    print('Setting triangles...')
    tr = np.array([   [0,1,6],
                      [1,2,7],
                      [2,3,8],
                      [3,4,9],
                      [4,5,10],
                      
                      [6,1,7],
                      [7,2,8],
                      [8,3,9],
                      [9,4,10],
                      [10,5,11] ])
    n_tr = tr.shape[0]
    tr_obtuse=[0]*n_tr
    return 0 
        
def av_edge_len():
    ''' Returns the average length of the mesh edges'''
    length = 0
    count = 0
    for i in range(n):
        for j in range(len(v[i].nb)):
            length += np.linalg.norm(v[i].c-v[v[i].nb[j]].c)
            count += 1
    return length/count

def add_noise(p):
    '''Add random noise to the vertex position of maximum p times 
    the average rest length, to all vertices'''
    av_len = av_edge_len()
    for i in range(n):
        v[i].c += p*av_len*(1-2*np.random.rand(3))
        
def tr_index(vList):
    for ti in np.where(tr==vList[0])[0]:
        if (vList[1] in tr[ti]) & (vList[2] in tr[ti]):
            return ti    
    #print("({0},{1},{2}) don't form a triangle".format(*vList))
    return -1
        
        
def assign_bd_old():
    '''According to bd_v list, change flag bd of vertex object to 1. 
    This function might be fall out of use if everything is implemented in triangles'''
    global v
    fixed = 0
    if bd_mode in ['F','f']:
        fixed = 1 
    for i in bd_v:
        v[i].bd=1
        v[i].fixed = fixed

        
def assign_bd():
    '''According to bd_v list, change flag bd of vertex object to 1. 
    This function might be fall out of use if everything is implemented in triangles'''
    global v, fixed_v
    
    if bd_mode in ['F','f']:  #(F)ix the whole boundary
        fixed_v = bd_v.copy()
    elif bd_mode in ['U','u']:  # Let the whole boundary (U)nconstrained  (else,'P', use defined/read fixed_v)
        fixed_v = []
    for i in bd_v:
        v[i].bd=1
    for i in fixed_v:
        v[i].fixed = 1



def choose_start_conf():
    global start_conf
    print('Choose starting configuration:')
    print('S: Hexagonal Sheet')
    print('C: Cylinder')
    print('L: Long cylinder (with cap)')
    print('H: Helical Ribbon')
    print('O: Hemisphere')
    print('I: Icosahedron')
    
    start_conf = input('Your choice -> ')
    while start_conf not in ['S','s','C','c','H','h','L','l','O','o','I','i']:
        start_conf = input('Please input S, C, L, H, O or I -> ')
    return 0

def choose_bd_mode():
    global bd_mode
    print('Choose boundary mode (Currently not sure about E works):')
    print('F: Fixed')
    print('U: Unconstrained')
    print('A: Averaged')
    print('E: Introduces boundary energy (with analytical gradient)')
    bd_mode = input('Your choice -> ')
    while bd_mode not in ['F','f','U','u','A','a','E','e']:
        bd_mode = input('Please input F, A or H -> ')
    return 0

def choose_grad_mode():
    global grad_mode
    print('Choose gradient mode (Currently only Greedy Analytical works):')
    print('A: Greedy Analytical')
    print('N: Greedy Numerical')
    print('F: Full Numerical')
    grad_mode = input('Your choice -> ')   
    while grad_mode not in ['A','a','N','n','F','f']:
        grad_mode = input('Please input A, N or F -> ')
    return 0

def fix_longcyl():      #i# why this, and those in particular? It is being used ATM
    ''' it seems it manually eliminates top vertex in the cap, resulting in 
    one vertex less and two triangles less. It is clear to me why; ask Steven
   It won't be run for the time being.
   moreover, there is an error with the v-list and indeces out of bounds in the 
   minimization algorithms.'''
    global v, tr, n, n_tr, tr_obtuse
    del v[9]    #i# the v-list needs to be further reordered which is not done here, leading to future errors. 
    tr = np.delete(tr,232,0)
    tr = np.delete(tr,216,0)
    tr = np.delete(tr,200,0)
    tr = np.append(tr, np.array([[116,118,123]]), 0)
    for i in range(n_tr-2):
        for j in range(3):
            if tr[i,j]>9:
                tr[i,j]-=1
    n -= 1
    n_tr -= 2
    tr_obtuse=[0]*n_tr
    
def reset():
    ''' Restart shape and calculation from scratch according to start_conf, 
    bd_mode and grad_mode. There is only the mesh, no energy calculations at this stage'''
    global v,tr, start_conf, l_0, bd_mode, grad_mode, max_step_size
    
    if start_conf == '':
        choose_start_conf()
    if bd_mode == '':
        choose_bd_mode()
    if grad_mode == '':
        choose_grad_mode()

    v=[]
    tr=[]
  
    if start_conf in ['S','s']:
        set_vc()
        set_tr()

    elif start_conf in ['C','c']:
        set_vc_cyl()
        refine()
        refine()
        #refine()  #r#
        #refine()  #816
        proj_cyl()
        #refine()
    
    elif start_conf in ['L','l']:
        set_vc_capcyl()
        refine()
        refine()  #n_seg(10)->252, nseg(20)->975
        proj_cyl()

    
    elif start_conf in ['H','h']:
        set_vc_helix()       
        refine()
        refine()
        refine()
        proj_cyl()
    
    elif start_conf in ['O','o']:
        set_vc_icoHemisphere()    # starts with ico, there is another one starting from triangular pyramid       
        refine()
        proj_hemi()
        
    elif start_conf in ['I','i']:
        set_vc_icoHemisphere(half=False,onlyIco=True)
        #proj_hemi()  #if wanting to project to a given circumscribed radius
        
    for i in range(n):
        constr_nb(i)        
    assign_bd()

    l_0 = av_edge_len()
    max_step_size = max_f*l_0
    #i TODO TMP#
    add_noise(noise)
    store_bd_coords()
    print('Surface setup complete, showing surface...')
    show()
 
    
def read_saved_conf(fileName, oldFormat=False, show3d=True):
    ''' Read from a file, the current mesh configuration. 
    Only reads files in the format saved by export_config()'''
    
    global v, n, n_tr, tr, bd_v, fixed_v, n_obtuse, curr_bd_c, l_0, max_step_size, dataDF
    global E_array, M_array, dE_array, S_array, Conv_array, fresh, tr_obtuse
    global cH, H0, cW, W0, cL
    
    #print('npre',n)
    
  # Re-initialize variables
    n = 0               #Number of vertices
    #print('ninit',n)
    v = []              #List of Vertex class elements
    tr = []             #Triangle list
    n_tr = 0            #Number of triangles
    bd_v = []          #i# list of boundary vertices
    #fixed_v = []        #i# list of boundary vertices which are fixed
    n_obtuse = 0        #i# number of obtuse triangles after analysis
    curr_bd_c = []      #i# temporary list of boundary positions to assess free boundaries    
    E_array = []
    M_array = []
    dE_array = []
    S_array = []
    Conv_array = []
    fresh = 1           #If 1, recalculate before taking a step
    dataDF = pd.DataFrame(columns=DFcol)
        
  # Read file     
    # From export_conf -> "i,  bd,  x,  y,  z"
    fil = open(fileName, "r")
    lines = fil.readlines()
    
    if oldFormat:
        initData = 3
        n_line = 1
        print('\n\t **!!** No params read, make sure to change by hand cH,H0, etc)\n')
    else:
        #read_specific_fixed = False
        initData = 5
        n_line = 3    #params = np.fromstring(lines[0], sep='\t')
        param_values = np.fromstring(lines[1], sep='\t')
        #for p in param_values:
            # Ordered as: cH, H0, cW, W0, cL, l0
        cH = param_values[0]
        H0 = param_values[1]
        cW = param_values[2]
        W0 = param_values[3]
        cL = param_values[4]
        l_0 = param_values[5]   
        print('l0',l_0)
    
    n = int(lines[n_line])
    stop_at = initData+n
    read_specific_fixed = True if 'fix' in np.fromstring(lines[initData-1],sep='\t') else False 
    for i, l in enumerate(lines[initData:stop_at]):
        info = np.fromstring(l,sep='\t')        
        v.append(Vertex())
        v[i].c = info[2:5]
        v[i].bd = info[1]        
        if read_specific_fixed:       # Previous .config versions did not include 'fix' column. bd_mode was either all fix or none 
            v[i].fixed = info[5]
      
    n_tr = int(lines[stop_at+1])    
    #tr = np.array(lines[stop])
    for l in lines[stop_at+2:]:  
        tt = np.fromstring(l,sep='\t',dtype=int)
        tr.append(tt.tolist())
    tr = np.array(tr)
    tr_obtuse=[0]*n_tr
    
  # Double-check if len(tr) matches n_tr read, same as len(v). 
    if len(v) != n:
       print("Only {} vertices read instead of {} in the file").format(len(v),n)
       return 1
    if len(tr) != n_tr:
       print("Only {} triangles read instead of {} in the file").format(len(tr),n_tr)
       return 1
   
    fixed_v = np.flatnonzero([v[i].fixed for i in range(n)]).tolist()
  # Setup neighborhood   
    for i in range(n):
        constr_nb(i)
    assign_bd()
    
  # To wrap up 
    if oldFormat:
        l_0 = av_edge_len()
    max_step_size = max_f*l_0

    store_bd_coords()
    print('Surface reading and setup complete')
    if show3d:
        show()
    analysis(True)    
    return 0

def read_dataDF(fileName):
    ''' Read from a file, run information, into a pandas DataFrame,
    with columns: 
    {'E':E_array,'dE':dE_array,'M':M_array,'S':S_array,'Conv':Conv_array}.
    Return dataframe'''
    thisDF = pd.read_cvs(fileName,sep='\t')
    return thisDF

    
# =============================================================================
# Core operation functions
# =============================================================================
#constr_nb(v,):
#Constructs the neighbourhood of vertex i, the neighbourhood is given by all
#other vertices connected through a single triangle (or equivallently edge)
#Result v_nb[i] (neigbourhood of vertex i) is ordered such that v_nb[i][j] and
#v_nb[i][j+1] are also neighbours for all j.
#Works for hexagonal triangulation given by set_tr()
# =============================================================================
def constr_nb(i):
    nb_rows, nb_columns = np.where(tr==i) #triangles attached to vi (neighbourhood)
    #less than 3 triangles is a boundary vertex
    if np.size(nb_rows) < 3:
        constr_nb_bd(i)
        return 0
    v[i].nb = np.zeros(np.size(nb_rows), dtype=int) #neighboring vertices of vi, ordered        
    for j in range(np.size(nb_rows)):   #i# for every triangle that i is part of
        #Setup the first two indices by taking the first triangle containing vi
        if j==0:
            v[i].nb[0] = tr[nb_rows[0],(nb_columns[0]+1)%3]  #i# take "first" neighbor of the triangle where i appears first
        elif j==1:
            v[i].nb[1] = tr[nb_rows[0],(nb_columns[0]+2)%3]  #i# take the other neighbor of that same triangle
            #Delete the triangles already used
            nb_rows = np.delete(nb_rows,0)
            nb_columns = np.delete(nb_columns,0)
        else:
            #Determine the next adjacent triangle
            other_tr = np.where(tr==v[i].nb[j-1])[0] #Triangles containing last neighbour
            #If there is no adjacent triangle found, vertex i must be on the boundary
            if np.intersect1d(other_tr,nb_rows).size == 0:
                constr_nb_bd(i)
                return 0
            row  = np.intersect1d(other_tr,nb_rows)  #i# take the adjacent triangle to "first" edge 
            #Find the next vertex in the adjacent triangle
            for k in np.arange(3):
                if tr[row,k%3]!=i and tr[row,k%3]!=v[i].nb[j-1]:
                    v[i].nb[j] = tr[row,k%3]
                    break
            index = np.where(nb_rows==row)
            nb_rows = np.delete(nb_rows,index)
            nb_columns = np.delete(nb_columns,index)
    #There should be one triangle left, if not -> on boundary
    if nb_rows.size == 0:
        constr_nb_bd(i)
        return 0
    last_tr = tr[nb_rows[0],:]
    #If this last triangle does not connect the beginning and end of the nb
    #of vertex i -> on boundary
    if (v[i].nb[0] not in last_tr) or (v[i].nb[-1] not in last_tr):
        constr_nb_bd(i)
        return 0
    return 0

def constr_nb_bd(i): #constr_nb for boundary vertices
    #For boundary vertices, v_nb is not ordered. This is not needed since the
    #curvatures are not computed, the neigbhourhood is only needed for the 
    #Hooke energy  ________ Now it is important that they are, since the bulk vertex
    #energy depends on it. 
    
    #print('aaaaaaaa')
    if not i in bd_v:
        bd_v.append(i)
            
    nb_tr_rows, nb_tr_columns = np.where(tr==i) #triangles attached to vi (neighbourhood)    (recall tr=array(n_tr,3))    
    neighborhood = np.unique((tr[nb_tr_rows]).flatten())
    neighbors = neighborhood[~np.in1d(neighborhood,i)]     
    
    v[i].nb = np.array([],dtype=int)
    nb_tri_dict = {}
    done_tri = []
    
 # Get starting point, to then have an ordered list   
    first_neigh = -1
    for j in neighbors:
        j_nb_rows, j_nb_cols = np.where(tr==j)      # triangles where j is
        shared_tri_idxs = np.intersect1d(nb_tr_rows, j_nb_rows)   #triangles where both j and i are
        nb_tri_dict[j] = shared_tri_idxs
     
    for j in nb_tri_dict:     
        shared_tri_idxs = nb_tri_dict[j]
        if len(shared_tri_idxs)==1:    # Start from a boundary triangle
            shared_tri_idx =shared_tri_idxs[0]
            shared_tri = tr[shared_tri_idx]
            pos_j = np.where(shared_tri==j)[0]
            pos_i = np.where(shared_tri==i)[0]
            if pos_j == (pos_i+1)%3:     #check boundary single-edge neighbor is "to the right" to start in counter-clockwise order
                first_neigh = j
                second = shared_tri[(pos_j+1)%3]
                v[i].nb = np.append(v[i].nb, first_neigh)
                v[i].nb = np.append(v[i].nb, second)
                neighbors = np.delete(neighbors, np.where(neighbors==first_neigh))  #added, remove from list
                neighbors = np.delete(neighbors, np.where(neighbors==second))  #added, remove from list
                nb_tr_rows = np.delete(nb_tr_rows,np.where(nb_tr_rows==shared_tri_idx))   #maybe remove and do in len()-1
                done_tri.append(shared_tri_idx)
    
    if first_neigh ==-1:
        print("Something is off finding the starting point boundary neighborhood around i={}. CHECK".format(i))
        #return 1
    
    for Ts in nb_tr_rows:
    #while len(neighbors)!=0:
        j = v[i].nb[-1]
        #print('j is{}'.format(j))
        #print(len(neighbors))
        tri_indx = nb_tri_dict[j][~np.in1d(nb_tri_dict[j],done_tri[-1])][0]
        pos_j = np.where(tr[tri_indx]==j)[0][0]
        #print('posj {}'.format(pos_j))
        #print(tr[tri_indx])
        k = tr[tri_indx][(pos_j+1)%3]
        #print('k is {}'.format(k))
        if k==i:
            print("Something is wrong with the ordering of triangle {0} with respect to i={1}".format(tri_indx,i))
            return 1
        if k not in v[i].nb:
            v[i].nb = np.append(v[i].nb,k)
            neighbors =  np.delete(neighbors, np.where(neighbors==k)) 
            nb_tr_rows = np.delete(nb_tr_rows,np.where(nb_tr_rows==tri_indx)) 
        done_tri.append(tri_indx)
        
    if len(neighbors)!=0 or len(nb_tr_rows)!=0:        
        print("Something is wrong with the ordering bd vertex i={0}".format(i))
        return 1
        
    return 0
           

def refine():
    '''Split each triangle in four by adding three new vertices at each mid-section.
    Update triangle list and reconstruct boundary at each vertex. 
    Restarts fresh=1 at the end of refinement. '''
    global v,tr,n_tr,n,l_0, fresh, bd_v, fixed_v, max_step_size, tr_obtuse
    
    new_tr = np.zeros((4*n_tr,3),dtype=int)
    print('Refining...')
    for i in range(n_tr):
        #Create new vertices where necessary for this triangle:
        index = np.array([-1,-1,-1],dtype=int)      
        for j in range(3):
            v1 = tr[i,j]
            v2 = tr[i,(j+1)%3]
            coords = (v[v1].c + v[v2].c)/2

            for k in range(n):
                if np.array_equal(coords,v[k].c):
                    index[j] = k
                    break
                     
            if index[j] == -1:
                v.append(Vertex())
                n+=1
                index[j]=n-1
                v[-1].c = coords                
                if v1 in fixed_v and v2 in fixed_v:   #for partial fix
                    fixed_v.append(len(v)-1)
                    #v[-1].fixed = 1               #partial fix
                    
        #Update the new triangle list
        for j in range(3):
            new_tr[4*i+j] = [tr[i,j],index[j],index[(j+2)%3]]
        new_tr[4*i+3] = [index[0],index[1],index[2]]
    tr = new_tr    
    n_tr *= 4
    tr_obtuse=[0]*n_tr
    l_0 *= 1/2
    max_step_size = max_f*l_0
    for i in range(n):
        constr_nb(i)
       # if bd_mode in ['F','f']:
         #   v[i].fixed = v[i].bd #Fixes boundary vertices   
    assign_bd()
    
    fresh = 1
    print('Refinement complete!')
    return 0

def refine_mult(steps=2):
    ''' Call refine() "steps" number of steps. '''
    for s in range(steps):
        refine()
    return 0


def calc_full_gradient():
    ''' Called when the grad_mode is full analytical. Because the gradient of 
    the full energy with respect to some vertex i coordinates depend on the 
    gradient of its neighboring vertices, first we need to calculate the local
    values at every point of the mesh, since these are needed for the full 
    energy gradient. For this, we use calc_vertex_energy on ALL vertices, which
    would update the v.dE of all vertices to the greedy gradient. We then add the 
    contributions of the neighboring vertices. 
    At the moment, only call this in recalculate, when grad_mode = 'fa'.'''

       
    #start_time = time.perf_counter()
    
    for i in range(n):
        calc_vertex_energy_grad(i)    #calculates local fields and grad_i E_i
    
    #mid_time = time.perf_counter()
    
    for i in range(n):
        calc_neighborhood_energy(i)
    
    #end_time = time.perf_counter()
    #print("Running time local: {0:.4}s \n\t running time neighborhood: {0:.4}".format(mid_time-start_time,end_time-mid_time))
    fullGrad = getArray('dE')
    
    return fullGrad


############################## START of NEW 
###############################################################################

def START():
    return 0

def itIsFixed():
    ''' For fixed boundaries, to not change the coordinates of bd vertices'''
    global v, n
    arr =[[v[i].fixed]*3 for i in range(n)]
    return np.array(arr).flatten()

def getArray(attribute = 'c'):
    global v, n
    arr = [getattr(v[i],attribute) for i in range(n)]    
    return (np.array(arr).flatten())

def moveTo(posArray):
    coords = np.reshape(posArray,(n,3))
    for i in range(n):
        v[i].c = coords[i]
    
def norm(some1DArray):
    ''' Returns the norm of a vector, given as an argument either as a 1D array, or 
    list '''
    return np.linalg.norm(some1DArray)

def angle(coord2, coord1, coord3):
    ''' Returns the angle at vertex 2 between two edge vectors,
    between vertices 1,2,3'''
    e12 = coord1 - coord2
    e32 = coord3 - coord2    
    cos_at_2 = np.dot(e12, e32)/norm(e12)/norm(e32)
    return np.arccos(cos_at_2)

def cotang(angle):
    return 1/np.tan(angle)
     
def get_hooke(pos_i, neighborsCoordList):
    '''
    Calculates hooke contribution for a vertex at coordinates pos_i, connected
    to vertices at coordinates given by the list neighborsCoordList. 
    Returns summed value
    '''
    L = 0 
    for pos_j in neighborsCoordList:
        e = pos_i- pos_j
        L += (norm(e)-l_0)**2
    return L/2
    
def bd_area(i, pos_i, neighborsCoordList):
    '''
    Calculates the vertex area for a boundary vertex i, located at pos_i, having
    as neighbors the vertices numbered in neihborsCoordList. Since the triangle 
    neighborhood is not "closed", it first checks that the (i,j,k) vertices do 
    form a triangle. Then adds up the are contribution from that triangle; 
    either the Voronoi area or a fraction of the triangle area in the case of 
    an obtuse triangle. 
    Returns total area for that vertex 
    '''
    
    global v, tr_obtuse
    
    v_nb = v[i].nb
    n_nb = len(neighborsCoordList)
    if len(v_nb) != n_nb: 
        print("\n\nERROR: something wrong with neihborhood list of ",i)
        
    A = 0.
    
    #FINISH
    for nj in range(n_nb):
        j = v_nb[nj]
        nk = (nj+1)%n_nb
        k = v_nb[nk]    

        t = tr_index([i,j,k])
        if t<0: continue
    
        pos_j = neighborsCoordList[nj]
        pos_k = neighborsCoordList[nk]
        
    
    ###  -- Angles at vertex i, k, j   -- ###
        angle_at_i = angle(pos_j, pos_i, pos_k)
        angle_at_k = angle(pos_i, pos_k, pos_j)
        angle_at_j = angle(pos_k, pos_j, pos_i)
      
    ### -- Total triangle area -- ###
        e_jk = pos_j - pos_k
        e_ji = pos_j - pos_i
        base = norm(e_jk)
        height = norm(e_ji-np.dot(e_ji,e_jk)*e_jk/base/base)
        area_triangle = base*height/2.
        
        
    ### -- Contribution to vertex area -- ###    
    # Area for obtuse triangles by splitting triangle in four regions    
        if angle_at_i >= np.pi/2:
            area_i = area_triangle/2
            tr_obtuse[t] = 1            
        elif angle_at_k >= np.pi/2 or np.pi/2-angle_at_k >= angle_at_i:
            area_i = area_triangle/4
            tr_obtuse[t] = 1
     # Voronoi area for non-obtuse triangles        
        else:
            r_ik = norm(pos_i - pos_k)
            r_ji = norm(e_ji)
            area_i = cotang(angle_at_j)*r_ik**2/8 + cotang(angle_at_k)*r_ji**2/8     
            tr_obtuse[t] = 0
            
        A += area_i        
        
    return A

    
   


def calc_vertex_energy(i, pos_i, neighborsCoordList, update=False):
    global v
  
 #--------------------------
 ### ALL vertices ###    

  # Add Hooke energy and area to all vertices
    A = 0.
    L = get_hooke(pos_i, neighborsCoordList)
    Ei = cL * L
    
    v[i].L = L
    v[i].E_L = Ei
    
 #--------------------------
 ### Boundary vertices ###
 
  # If vertex is at the boundary, add only hooke energy and calculate area
    if v[i].bd:#True:#v[i].bd:
        if update:
            v[i].A = bd_area(i, pos_i, neighborsCoordList)   #Checked
        # Any other boundary energy could be added here, such as a line tension
        return Ei

    else:    
     #--------------------------
     ### Bulk vertices ###
        
      # Initialize vertex fields     
        S = np.zeros(3)
        angDeficit = 2*np.pi   
        n_nb = len(neighborsCoordList)
        v_nb = v[i].nb
        if len(v_nb) != n_nb: 
            print("\n\nERROR: something wrong with neihborhood list of ",i)
      
       #__________  NEIGHBOR LOOP  ______________# 
      # Sum contributions of every neighboring triangle  
        for nj, pos_j in enumerate(neighborsCoordList):
            j = v_nb[nj]
            nk = (nj+1)%n_nb
            k = v_nb[nk]    
    
            t = tr_index([i,j,k])
            if t<0: 
                print("\n\nERROR: something wrong with neihborhood of BULK ",i)
            
            pos_k = neighborsCoordList[nk]
            
        ###  -- Angles at vertex i, k, j   -- ###
            angle_at_i = angle(pos_i, pos_j, pos_k)
            angle_at_k = angle(pos_k, pos_i, pos_j)
            angle_at_j = angle(pos_j, pos_k, pos_i)
            cot_j = cotang(angle_at_j)
            cot_k = cotang(angle_at_k)
            
         ###  -- Gaussian curvature contribution  --  ###
            angDeficit -= angle_at_i    
            
         ###  -- Mean curvature contribution  --  ###
            e_ik = pos_i - pos_k
            e_ij = pos_i - pos_j
            S += cot_j * e_ik + cot_k * (e_ij)
            
        ### -- Total triangle area -- ###
            e_jk = pos_j - pos_k
            e_ji = pos_j - pos_i
            base = norm(e_jk)
            height = norm(e_ji-np.dot(e_ji,e_jk)*e_jk/base/base)
            area_triangle = base*height/2.
    
        ### -- Contribution to vertex area -- ###    
        # Area for obtuse triangles by splitting triangle in four regions    
            if angle_at_i >= np.pi/2:
                area_ij = area_triangle/2
                tr_obtuse[t] = 1            
            elif angle_at_k >= np.pi/2 or angle_at_j >= np.pi/2:
                area_ij = area_triangle/4
                tr_obtuse[t] = 1
         # Voronoi area for non-obtuse triangles        
            else:
                r_ik = norm(pos_i - pos_k)
                r_ji = norm(e_ji)
                area_ij = (cot_j * r_ik**2 + cot_k * r_ji**2)/8     
                tr_obtuse[t] = 0
            A += area_ij
                
        #__________ END of  NEIGHBOR LOOP  ______________# 
       
      ###  -- Gaussian curvature  --  ###
        K = angDeficit/A   
        
    
       ###  -- Mean curvature    --  ###
        curvatureVector = S / (4*A) 
        dotProduct = 0.
        for nj, pos_j in enumerate(neighborsCoordList):        
            j = v_nb[nj]
            nk = (nj+1)%n_nb
            k = v_nb[nk]  
            pos_k = neighborsCoordList[nk]            
            someNormalVector = np.cross(pos_j - pos_i, pos_k - pos_i)
            dotProduct = np.dot(someNormalVector, S)
            if abs(dotProduct) > 1e-6:
                break
        #someNormalVector = np.cross(pos_j - pos_i, pos_k - pos_i)
        #sign = np.sign(np.dot(someNormalVector, S))
        sign = np.sign(dotProduct)
        H = sign * norm(curvatureVector)     
        
       ###  -- Warp    --  ###
        W2 = H**2 - K
        W = 0.
        if W2 > 0:
            W = np.sqrt(W2)
               
      # Get energy
        EH = cH * A * (H - H0)**2       
        EW = cW * A * (W - W0)**2       
        Ei += EH + EW    # Hooke contribution was added at the beginning of the function
    
        
      # Update Vertex object values
        if update:
            v[i].A = A
            v[i].Hnorm = curvatureVector   
            v[i].norm = sign*curvatureVector/norm(curvatureVector)
            v[i].H = H
            v[i].H2 = H**2  
            v[i].K = K
            v[i].W = W
            v[i].W2 = W2   
            v[i].E_H = EH
            v[i].E_W = EW    
            v[i].E = Ei
        
        return Ei

ZTracker = 0
def total_energy(coordinates1DArray):
    '''
    Return the total energy as a sum over the energy of all vertices. 
    The energy per vertex is then called by calculating the energy for a given
    central vertex in position Ri, and having neighbors at positions Rjs
    Rj is a (3,Nb) array, where Nb is the numbers of mesh neighbors to i. 
    
    '''
    global v,ZTracker 
    
    coordinatesArray = np.reshape(coordinates1DArray,(n,3))
    
    E = 0.
    for i, pos_i in enumerate(coordinatesArray):
        #print(i)
        #print(v[i])
        neighbors = v[i].nb 
        pos_js = [coordinatesArray[j] for j in neighbors]
        Ei = calc_vertex_energy(i, pos_i, pos_js)
        E += Ei
    ZTracker = E
    return E

def safe_division(x, y):
    """
    Computes safe division x/y for small positive values x and y
    """
    return np.exp(np.log(x) - np.log(y)) if y != 0 else 1e16



def g_AG(steps,df=calc_full_gradient, la_0=1e-6, convTol=CONVTOL, interrupt=False):
    ''' 
    g(steps, lmin=None, modifier=1, interrupt=False, jolt_when_stuck=False)
    
    Like g in Surface Evolver. Displace all vertices "at once" according to
    the dE previously calculated. Repeat for steos. If this is not initial step,
    check whether the energy is lowered this way and if not, whether to continue.
    If cont, continue storing this. Otherwise, undo change in coordinates and 
    recalculate'''
    global v, E_array, fresh, dE_array, Conv_array, dx, vc 
    cont = ''#'y'
    
    start_time = time.perf_counter()
      
    if fresh:
        print('Using the g algorithm in grad_mode={0}'.format(grad_mode))
        print('Using boundary mode = {0}'.format(bd_mode))
        recalculate()
        E_array.append(total_energy_Vimplementation())
        dE_array.append(total_grad())
        M_array.append(0)
        S_array.append(0)
        Conv_array.append(0)
        fresh = 0
        
    cont = 'y'
    if interrupt:
        cont='n'
    
    x0 = getArray('c')        
    x_old = x0.copy()
    grad_old = getArray('dE')#yur#df(x0)
    x = x0 - la_0 * grad_old  # we want to find a point close to the initial one.
    la_old = 1
    th = 1e9
    
    moveTo(x)
    grad = df()
    #USING ONLY FA GRADMODE #recalculate()
    
    for t in range(steps):
        #print("** Run ",t)        
            
        E_preMove = total_energy_Vimplementation()
   
        
        norm_x = norm(x - x_old)
        norm_grad = norm(grad - grad_old)
        # compute the stepsize
        la = min(np.sqrt(1 + th) * la_old,  0.5 * safe_division(norm_x, norm_grad))
        th = la / la_old
        x_old = x.copy()
        x -= la * grad *(1-itIsFixed()) 
        la_old = la
        grad_old = grad
        
        moveTo(x)
        grad = df()
        
        Enew = total_energy_Vimplementation()
        
        E_array.append(Enew)
        dE_array.append(total_grad())
        M_array.append(1)
        S_array.append(la)
        
        
        if Enew > E_preMove:
            print("** Run ",t) 
            print('Warning: Energy increase. Continue?')
            #cont = input('(y/n): ')
            if cont not in ['y','Y','yes','Yes']:
                return 0
            
   # Check convergence
        dEs = np.reshape(grad, (n,3))       
        norms_dE = [np.linalg.norm(dEi) for dEi in dEs]
        displace = np.mean(np.array(norms_dE)*la)
        if displace < CONVTOL:
            Conv_array.append(displace)
            return "It is converged"  
        Conv_array.append(displace)

    end_time = time.perf_counter()
    print("Execution time for {} steps of adaptative gradient descent is: ".format(steps) + str(round(end_time-start_time,2)))
    return 0    
        
def END():
    return 0    



############################## END of NEW 

############################################################
###############################################################################
######################## ALL PREVIOUS FUNCTIONS ##############################




    

def recalculate():
    ''' Call functions to calculate vertex gradients according to initial 
    gradient mode. These functions update Vertex objects values.
    Called in g() and gs_opt()'''
    
    if grad_mode in ['fa','FA']:
        tot_grad = calc_full_gradient()
    
    if grad_mode in ['a','A']:
        for i in range(n):
            calc_vertex_energy_grad(i)
        if bd_mode in ['A','a']:     #i#TODO Shouldn't the average mode be available for any grad_mode?
            for i in bd_v:
                calc_bd_nb_grad(i)
          
    elif grad_mode in ['n','N']:
        for i in range(n):
           calc_num_grad(i)
                
    elif grad_mode in ['f','F']:
        for i in range(n):
            calc_num_grad_full(i)
    elif grad_mode in ['fc','FC']:
        for i in range(n):
            calc_num_grad_full_center(i)
        #chk#for i in range(n):
         #chk#   calc_vertex_energy_num(i)
            
                
                
def recalc_energy():
    '''Recalculates the energy for all the vertices, calling functions 
    for the numerical method, as these do not recalculate the vertices gradient.
    Return 0'''
    for i in range(n):
        calc_vertex_energy_num(i)       

    return 0

            

def new_energy(lstep):
    ''' Calculate the new energy values using recalc_energy(), after taking a 
    step of length "lstep", in the direction of the negative energy gradient.
    Function part of the line minimization linmin(), returns the total energy.
    '''
    #E0 = 0
    Enow = 0
    for i in range(n):       
        v[i].c -= v[i].dE*lstep*(1-v[i].fixed) #i# if fixed bd, don't move
        v[i].dE_prev = v[i].dE
        
    recalc_energy()
    
    for i in range(n):
        Enow += v[i].E
        if (v[i].dE != v[i].dE_prev).all():         #tmp
            print( "ENERGY GRAD CHANGED IN NEW_ENERGY() FOR V{}".format(i)  ) #tmp
        v[i].c += v[i].dE_prev*lstep*(1-v[i].fixed) #i# if fixed bd, don't move               
        #v[i].dE = v[i].dE_prev
    
    recalc_energy()  #TODO check if this recalculation is necessary
        
    return Enow    


def linmin_opt(upperBound=1.):
    '''Use scipy.optimize.fminbound() as line minimizing function to find
    optimal step size in the direction of the energy gradient
    
    Returns (size_step, energy there)'''
    
    start_time = time.perf_counter()
    
    out,val,err,num = opt.fminbound(new_energy, 0., upperBound,disp=1,xtol=1e-8,full_output=1)
        
    end_time = time.perf_counter()
    print("Running time is: " + str(round(end_time-start_time,2)))    
    
    Enow = total_energy()
    if val > Enow:
        print("Energy at linmin_opt minimum is higher than current one, try linmin")
        return 0., Enow
    
    return out,val

    
#chk#rm#tmpCheck = [[],[]]            
def linmin():    
    ''' Line minimization of the energy on the direction of the negative 
    energy gradient, using "bracketing" algorithm with bounds given in terms
    of the parameter "step_size" defined above. Returns step size to lowest
    energy if found, the large bound if energy constantly decreases or zero
    if energy increases for the low bound.
    Returns (size_step, energy there)'''
    
    start_time = time.perf_counter()

    start, stop = (min_step_size, max_step_size)
    
    E0 = total_energy()
    Estart = new_energy(start)    
    
    # The following while loop for finding a starting point where indeed the E goes down. 
    #print(total_energy())
    #print('Estart for {}'.format(start), Estart)
    while(Estart > E0):  
        #print('Start step {0} in linmin not lowering E, doubling'.format(start))
        start = start*2        
        if start > 5e-2:
            break
        Estart = new_energy(start)
        #print('Estart for {}'.format(start), Estart)
        
    #chk#rm#tmpCheck[0].append(0)
    #chk#rm#tmpCheck[1].append(E0)
    #chk#rm#tmpCheck[0].append(start)
    #chk#rm#tmpCheck[1].append(Estart)       
    
    if Estart > E0:
        print('Start steps {0}-{1}, dont not lower the energy'.format(min_step_size,start))
        return 0.,Estart   #no step done
    
    energies = [E0, Estart]
    steps = [0.,start]
    end = start
    
    
    while((energies[-1] <= energies[-2]) & (end < stop) ):
        end *=2
        energies.append(new_energy(end))
        steps.append(end)
   
    #chk#rm#tmpCheck[0].append(end)
    #chk#rm#tmpCheck[1].append(new_energy(end))    
    
    if (end > stop) & (energies[-1] < energies[-2]): #chk if second is necessary
        print('it stopped by stop; using max step size')
        end_time = time.perf_counter()
        print("Running time is: " + str(round(end_time-start_time,2)))    
        return end, new_energy(end)

     
    start = steps[-3]
    lo = steps[-2]
    stop = steps[-1]
    
    right = lo + (stop-lo)/2
    left = start + (lo-start)/2
        
    loE = energies[-2]    

    while((abs(left-start)>1e-12)  & (abs(right-stop)>1e-12)):        
        leftE = new_energy(left)
        rightE = new_energy(right)
        
        #chk#rm#tmpCheck[0].append(lo)
        #chk#rm#tmpCheck[1].append(loE)
        
        if leftE < loE:
            stop = lo
            lo = left            
        elif rightE < loE:
            start = lo
            lo = right
        else:
            start = left
            stop = right        
        right = lo + (stop-lo)/2
        left = start + (lo-start)/2        
        loE = new_energy(lo) 
 
       
    #chk#rm#tmpCheck[0].append(lo)
    #chk#rm#tmpCheck[1].append(loE)
        
    
    end_time = time.perf_counter()
    print("Running time is: " + str(round(end_time-start_time,2)))
    return lo, loE#mid


def g(steps, lmin=None, modifier=1, interrupt=False, jolt_when_stuck=False):
    ''' 
    g(steps, lmin=None, modifier=1, interrupt=False, jolt_when_stuck=False)
    
    Like g in Surface Evolver. Displace all vertices "at once" according to
    the dE previously calculated. Repeat for steos. If this is not initial step,
    check whether the energy is lowered this way and if not, whether to continue.
    If cont, continue storing this. Otherwise, undo change in coordinates and 
    recalculate'''
    global v, E_array, fresh, dE_array, Conv_array, dx
    cont = ''#'y'
    
    start_time = time.perf_counter()
    
    for t in range(steps):
        print("** Run ",t)
        if fresh:
            print('Using the g algorithm in grad_mode={0}'.format(grad_mode))
            recalculate()
            E=0
            for i in range(n):
                E += v[i].E
            E_array.append(E)
            dE_array.append(total_grad())
            M_array.append(0)
            S_array.append(0)
            Conv_array.append(0)
            fresh = 0
            
        E_preMove = total_energy()

        if lmin:
            modifier=1
            step, Enew = lmin()
            if step==0 and lmin==linmin_opt:
                print('\t>>No min found with scipy.optimize, trying with linmin at run {}'.format(len(S_array)))
                step, Enew = linmin()
            if step==0 and jolt_when_stuck:
                print('Linmin is stuck, jolting by max {}'.format(noise*av_edge_len()))
                add_noise(noise)
                step, Enew = lmin()   
                if step==0 and lmin==linmin_opt:
                    print('\t>>No min found with scipy.optimize, trying with linmin at run {}'.format(len(S_array)))
                    step,Enew = linmin()            
        else:
            step = step_size*modifier
            Enew = new_energy(step)
            if Enew > E_preMove:
                print('\t Step size {0:.6} is too big, find one with linmin()'.format(step))
                step, Enew = linmin()
                if step==0 and jolt_when_stuck:
                    print('Linmin is stuck, jolting by max {}'.format(noise*av_edge_len()))
                    add_noise(noise)
                    step, Enew = linmin()   
        
        
    # If even after jolting and trying the two linmin fxs, there is no step_size, leave
        if step ==0.:
            print("No step_size, or Start step not found, trying from {0:.3}".format(min_step_size))
            print("Not all steps were done, {} were missing".format(steps-t))
            if interrupt:
                return 1.
                          
        
        
    # Check if the energy does decrease (it can be an issue for linmin_opt() output)
        if Enew > E_preMove:
            print('Warning: Energy increase. Continue?')
            cont = input('(y/n): ')
            if cont not in ['y','Y','yes','Yes']:
                return 0
            
    # If the energy is lowered, or indeed continue, make the move             
        for i in range(n):
            v[i].dE_prev = v[i].dE
            v[i].c -= v[i].dE*step*modifier*(1-v[i].fixed) #i# if fixed bd, don't move
            
    # Get new system
        recalculate() 
        
    # Save info
        #chkcont=''
        E_array.append(total_energy())
        dE_array.append(total_grad())
        M_array.append(modifier)
        S_array.append(step)        
        
# Check convergence
    #dEs = [np.array([v[i].dE_x]) for i in range(n)]np.reshape(grad, (n,3))       
    norms_dE = [np.linalg.norm(v[i].dE) for i in range(n)]
    displace = np.mean(np.array(norms_dE)*step)
    if displace < CONVTOL:
        Conv_array.append(displace)
        return "It is converged"  
    Conv_array.append(displace)
        
    end_time = time.perf_counter()
    print("Running time for {} steps is: ".format(steps) + str(round(end_time-start_time,2)))
    return 0    
        

def nb_energy_num(i):   
    ''' Numerical calculation of the energy of a vertex i plus the energy of its
    neighboring vertices. Return float energy.'''
    calc_vertex_energy_num(i)
    E = v[i].E
    for j in v[i].nb:  
        calc_vertex_energy_num(j)
        E += v[j].E
    return E

    
def functions(r,k):
    x = r/R    #i# Implemented so far only for rotationally symmetric shapes
    if k==0:
        return 1
    if k==1:
        return x
    if k==2:
        return 1-x
    if k==3:
        return 1-4*(x-0.5)**2
    if k==4:
        return 4*(x-0.5)**2
    
def total_energy_Vimplementation():
    ''' Sum over the energy at each vertex. The actual calculation of the energy
    is done upon call of another function.  Return E'''
    E = 0
    for i in range(n):
        E+=v[i].E
    return E

def total_grad():
    ''' Return norm of 3n-dimensional energy gradient vector. Sum over each 
    component of each vertex and return square root of this. Used to look at 
    the energy minimisation convergence.'''
    
    dE2 = 0
    for i in range(n):
        dE2+= v[i].dE[0]**2+v[i].dE[1]**2+v[i].dE[2]**2
    return np.sqrt(dE2)



# =============================================================================
# Vertex energy functions
# =============================================================================
    
#calc_hooke(i):
#Calc Hooke energy and gradient at a single vertex
#L = cL/2*sum_[vj in nb[vi]] (l-l_0)^2; l = |vi-vj|
#grad_L = cL*sum(vi-vj)(1-l_0/l) 
# =============================================================================

def calc_hooke(i):
    #ii#if v[i].bd:
        #ii#return 0
    v[i].L = 0                  #Hooke energy
    v[i].dL = np.zeros(3)       #Hooke gradient
    for j in v[i].nb:
        e = v[i].c - v[j].c     #Vector from vj to vi
        l = np.linalg.norm(e)   #Length of e
        v[i].L  += ((l-l_0)**2) / 2
        v[i].dL += e*(1-l_0/l)
    #if v[i].bd:
     #   print('Boundary element '+str(i)+', with energy '+str(v[i].L))
    return 0 

#calc_localValues(v,i):
#Calc values of A, H2 and K
# =============================================================================
def calc_localValues(i):
    ''' Calculate each energy local field at vertex i, with ANALYTICAL exprs.        
    The calculations are done per triangle neighborhood of vertex i .     
    Only called from calc_vertex_energy_num() for NON bd atm
    
    With triangle (i,j,k) in the loop of neighbors to be (i,j,j+1), such that (previously):
        e0 = e_ij;  e1 = e_il; e3 = e_ik; etc...
    '''
    
    global v, tr_obtuse
    v_nb = v[i].nb
    
    v[i].A = 0
         
    
    ##v[i].dA = np.zeros(3)
    S = np.zeros(3)
    n_nb = np.size(v_nb)
    deficit = 2*np.pi
    for nj in range(n_nb):
        j = v_nb[nj]
        k = v_nb[(nj+1)%n_nb]
        
        #rm#print('(i,j,k) = ({0},{1},{2})'.format(i,j,k))
        
   # Triangle is i, j, j+1 
   # Triangle is i, j, j+1 
        t = tr_index([i,j,k])
      
      ###  -- Angle at vertex i   -- ### 
        e_ji = v[j].c - v[i].c
        r_ji= np.linalg.norm(e_ji)
        e_ki  =  v[k].c - v[i].c
        r_ki = np.linalg.norm(e_ki)
        cos_i = np.dot(e_ji, e_ki)/r_ji/r_ki
        angle_at_i = np.arccos(cos_i)
        
      ###  -- Angle at vertex i   -- ###
        e_ik = v[i].c - v[k].c
        r_ik = r_ki
        e_jk = v[j].c - v[k].c
        r_jk = np.linalg.norm(e_jk)       
        cos_k = np.dot(e_jk, e_ik)/r_jk/r_ik
        angle_at_k = np.arccos(cos_k)
        cot_k = 1/np.tan(angle_at_k)
                
     ###  -- Angle at vertex j from other two angles    --  ###
        angle_at_j = np.pi - angle_at_i - angle_at_k
        cot_j = 1/np.tan(angle_at_j)
        
     ###  -- Gaussian curvature contribution  --  ###
        deficit -= angle_at_i
        
     ###  -- Mean curvature contribution  --  ###
        S += cot_j * (e_ik) + cot_k*(-e_ji)
        
     ###  -- Mean curvature contribution  --  ###          
        base = r_jk       
        height = r_ki*np.sin(angle_at_k)       
        area_triangle = base*height/2.
        
                
    # Area for obtuse triangles given by fixed ratio.    
        #chk#                print("With neighbor {0}, the angle_at_i is {1} ({2}o)".format(j,angle_at_i,angle_at_i*180/np.pi))                  
        if angle_at_i >= np.pi/2:            
            area_ij = area_triangle/2
            tr_obtuse[t] = 1 
            #chk#            print("Obtuse triangle at i, go half {}".format(area_ij))
        elif angle_at_k >= np.pi/2 or angle_at_j >= np.pi/2:            
            area_ij = area_triangle/4
            tr_obtuse[t] = 1
            #chk#            print("Obtuse triangle at neighbor, go quart {}".format(area_ij))
            
     # Voronoi area for non-obtuse triangles        
        else:            
            area_ij = (cot_j * r_ik**2 + cot_k * r_ji**2)/8
            tr_obtuse[t] = 0
            #chk#           print("All good with cot {}".format(area_ij))
        v[i].A += area_ij
                 
        ############## END OF NEIGHBOR LOOP  ####################
        
  ###  -- Gaussian curvature  --  ###        
    v[i].K = deficit/v[i].A

   ###  -- Mean curvature    --  ### 
    #TODO rm###print('For {0} -> S = {1}'.format(i,S))
    v[i].Hnorm = S/(4*v[i].A)
    ###print(v[i].Hnorm*4*v[i].A)
    normal = np.cross(v[v_nb[0]].c - v[i].c,v[v_nb[1]].c - v[i].c) #not really the normal
    sign = np.sign(np.dot(normal,S))
    ###print('SIGN {}'.format(sign))
    v[i].H = sign*np.linalg.norm(v[i].Hnorm)
    ###print(v[i].H)
    ###print(v[i].H*v[i].H)
    v[i].H2 = v[i].H**2
    #v[i].H2 = (1/16/(v[i].A**2))*np.dot(S,S)
    ###print(v[i].H2)
    if abs(v[i].H2 - v[i].H*v[i].H) > 1e-12:
        print(' In {2} H(norm) = {0}  differs from H(Sdot) = {1}'.format(abs(v[i].H),np.sqrt(v[i].H2),i))
    
    ##print('\n\n')
    return 0   

def calc_bd_area(i):
    ''' Calculate the Voronoi area of boundary points with an incomplete
    triangle neighborhood, taking into account possible obstuse triangles
    This function assumes that the vertex i is indeed a boundary point, it does
    not check.
    The calculation of v[bd].dA is not necessary since the analytical derivative 
    terms for the curvature dE_W and dE_H (and the curvature fields themselves) 
    can't be calculated here, unless an approximation is taken such as 
    assigning the average of the neighboring bulk terms. 
    
    Note that, although the third vertex is labeled 'k', it is not necessarily a 
    reflection of the orientation contained in the Vertex.nb list, whose order
    was set in constr_nb(). This is not an issue, since the area is just a scalar
    and therefore not sensitive to this orientation. 
    '''
        
    global v, tr_obtuse
    
    v[i].A = 0       
     
    nb_row_triangle, nb_vertex = np.where(tr==i) #triangles attached to vi (neighbourhood)    
    for tix, t in enumerate(nb_row_triangle):            
        
        triangle = tr[t]
        vj = triangle[(nb_vertex[tix]+1)%3]
        vk = triangle[(nb_vertex[tix]+2)%3]
        
        e_ji = v[vj].c - v[i].c
        e_ki = v[vk].c - v[i].c
        rji = np.linalg.norm(e_ji)
        rki = np.linalg.norm(e_ki)        
        cos_i = np.dot(e_ji, e_ki)/rji/rki
        angle_at_i = np.arccos(cos_i)
        
        e_jk = v[vj].c - v[vk].c
        e_ik = v[i].c - v[vk].c
        rjk = np.linalg.norm(e_jk)
        rik = np.linalg.norm(e_ik)        
        cos_b = np.dot(e_jk, e_ik)/rjk/rik
        angle_at_k = np.arccos(cos_b)
        
        angle_at_j = np.pi-angle_at_i-angle_at_k
        
        base = np.linalg.norm(e_jk)
        height = np.linalg.norm(e_ji-np.dot(e_ji,e_jk)*e_jk/base/base)
        area_triangle = base*height/2.
        
    # Area for obtuse triangles by splitting triangle in four regions    
        if angle_at_i >= np.pi/2:
            area_i = area_triangle/2
            tr_obtuse[t] = 1            
        elif angle_at_k >= np.pi/2 or np.pi/2-angle_at_k >= angle_at_i:
            area_i = area_triangle/4
            tr_obtuse[t] = 1
     # Voronoi area for non-obtuse triangles        
        else:
            cot_k = 1/np.tan(angle_at_k)
            cot_j = 1/np.tan(angle_at_j)
            area_i = cot_j*np.dot(e_ik,e_ik)/8 + cot_k*np.dot(e_ji,e_ji)/8     
            tr_obtuse[t] = 0
        v[i].A += area_i        
    #chk# print("The area of bd vertex {0} is A={1:.3}".format(i,v[i].A)) 
        
    return 0


def calc_localValues_grad(i):
    ''' Calculate each energy local field at vertex i, with ANALYTICAL exprs.
    It also calculates the gradient of each field.
    
    The calculations are done per neighboring triangle, defined by the order of
    the neighbor list of i.     
    The area calculation was corrected (before if ang_alpha was obtuse, 
    this was not considered and cotangent formula was wrongly used). Moreover, 
    all analytical expressions are now per-triangle, hence also corrected. 
    Only called from calc_vertex_energy_grad() for NON bd atm'''
    
    global v, tr_obtuse 
    v_nb = v[i].nb
    n_nb = np.size(v_nb)
    
    v[i].A = 0    
    v[i].dA = np.zeros(3)
    S = np.zeros(3)
    DS = np.zeros((3,3))
    deficit = 2*np.pi
    d_ang_i_sum = 0
    
    for nj in range(n_nb):
        j = v_nb[nj]
        k = v_nb[(nj+1)%n_nb]    #i# if n_nb take back n_nb0
        
        # Triangle is i, j, j+1 
        t = tr_index([i,j,k])
      
      ###  -- Angle at vertex i   -- ###
        e_ji = v[j].c - v[i].c
        r_ji= np.linalg.norm(e_ji)
        e_ki  =  v[k].c - v[i].c
        r_ki = np.linalg.norm(e_ki)
        dot_at_i = np.dot(e_ji, e_ki)
        cos_i = dot_at_i/r_ji/r_ki
        angle_at_i = np.arccos(cos_i)
     # -> GRADIENT
        preFactor = r_ji*r_ki*np.sin(angle_at_i)
        term2 = e_ki/r_ki/r_ki + e_ji/r_ji/r_ji
        d_ang_i = (e_ki + e_ji - dot_at_i*term2)/preFactor       
        
     ###  -- Angle at vertex k   -- ###
        e_ik = v[i].c - v[k].c
        r_ik = np.linalg.norm(e_ik)#r_ki
        e_jk = v[j].c - v[k].c
        r_jk = np.linalg.norm(e_jk)       
        dot_at_k = np.dot(e_jk, e_ik)
        cos_k = dot_at_k/r_jk/r_ik
        angle_at_k = np.arccos(cos_k)
        cot_k = 1/np.tan(angle_at_k)
      # -> GRADIENT
        sin_k = np.sin(angle_at_k)
        preFactor =  r_jk*r_ik*sin_k
        term2 = e_ik/r_ik/r_ik
        d_ang_k = -(e_jk - dot_at_k*term2)/preFactor#(e_jk - dot_at_k*term2)/preFactor   #TODO correction
                
     ###  -- Angle at vertex j from other two angles    --  ###
        angle_at_j = np.pi - angle_at_i - angle_at_k
        cot_j = 1/np.tan(angle_at_j)      
        sin_j = np.sin(angle_at_j)
      # -> GRADIENT
        d_ang_j = -d_ang_k -d_ang_i

     ###  -- Gaussian curvature contribution  --  ###
        deficit -= angle_at_i
      # -> GRADIENT
        d_ang_i_sum += d_ang_i
        
     ###  -- Mean curvature contribution  --  ###
        S += cot_j * e_ik + cot_k * (-e_ji)
      # -> GRADIENT        
        DkEij = np.tensordot(d_ang_k, -e_ji, 0)/sin_k/sin_k  #e_ij according to analyt
        DjEik = np.tensordot(d_ang_j, e_ik, 0)/sin_j/sin_j
        DS += (cot_k + cot_j)*np.identity(3) - DkEij - DjEik       
      
          
     ###  -- Area contribution obtuse --  ###
        base = r_jk        
        height = r_ki*np.sin(angle_at_k)
        area_triangle = base*height/2.        
                
    # Area for obtuse triangles given by fixed fraction. 
        #chk#print('Angles ({0}, {1}) -> i={2}, k={3}'.format(i,j,angle_at_i*180/np.pi,angle_at_k*180/np.pi))
        #chk#print("With neighbor {0}, the angle_at_i is {1} ({2}o)".format(j,angle_at_i,angle_at_i*180/np.pi))                  
        if angle_at_i >= np.pi/2:            
            area_ij = area_triangle/2            
            tr_obtuse[t] = 1
            #chk#print("Obtuse triangle at i, go half {}".format(area_ij))
            
          # -> GRADIENT
            dA1 = sin_k * e_ik / r_ik
            dA2 = r_ik*cos_k * d_ang_k
            v[i].dA += r_jk * (dA1 + dA2)/4
            
        elif angle_at_k >= np.pi/2 or angle_at_j >= np.pi/2:            
            area_ij = area_triangle/4
            #chk#print("Obtuse triangle at neighbor, go quart {}".format(area_ij))
     # Voronoi area for non-obtuse triangles 
          # -> GRADIENT
            dA1 = sin_k * e_ik / r_ik
            dA2 = r_ik*cos_k * d_ang_k
            v[i].dA += r_jk * (dA1 + dA2)/8
            tr_obtuse[t] = 1            
        else:            
            area_ij = (cot_j * r_ik**2 + cot_k * r_ji**2)/8
            #chk#print("All good with cot {}".format(area_ij))
          # -> GRADIENT
            dA1 = 2*(-e_ji * cot_k + e_ik * cot_j)      #e_ij according to analyti
            dA2 = r_ji**2 * d_ang_k/sin_k/sin_k + r_ik**2 * d_ang_j/sin_j/sin_j
            v[i].dA += (dA1 - dA2)/8  
            tr_obtuse[t] = 0
            
        v[i].A += area_ij
    
        ############## END OF NEIGHBOR LOOP  ####################
        
  ###  -- Gaussian curvature  --  ###
    v[i].K = deficit/v[i].A   
    v[i].dK = -(v[i].K*v[i].dA + d_ang_i_sum)/v[i].A  
    
  
   ###  -- Mean curvature    --  ###
    v[i].Hnorm = S/4/v[i].A
    normal = np.cross(v[v_nb[0]].c - v[i].c,v[v_nb[1]].c - v[i].c) #not really the normal
    sign = np.sign(np.dot(normal,S))
    v[i].H = sign*np.linalg.norm(v[i].Hnorm)         
    if v[i].H ==0:
        v[i].dH = 0.
        v[i].dH2 = 0.
    else:
        Smag = np.linalg.norm(S)
        dH1 = np.tensordot(DS,S,1)/Smag
        dH2 = -Smag * v[i].dA / v[i].A
        #guarp#v[i].dH = (dH1 + dH2)/(4*v[i].A)
        v[i].dH = sign*(dH1 + dH2)/(4*v[i].A)   #guarp
        #TODOrm#v[i].dH2 = 2*abs(v[i].H)*v[i].dH  
        v[i].dH2 = 2*v[i].H*v[i].dH  #ask on sign of H in gradient        
    v[i].H2 = v[i].H**2
    return 0
     
    

def calc_neighborhood_grad(i):
    ''' Calculate the gradient grad_i of the energy of the neighborhood vertices
     using the ANALYTICAL expres. of the gradient \grad_i E_j. 
        
    The calculations are done per neighboring triangle, defined by the order of
    the neighbor list of i, and hence also the area calculation is the corrected
    one. 
    When calculating the analytical full gradient, this has to be called after 
    calculating the local fields for ALL vertices
    WILL be called from calc_vertex_energy_grad()(full gradient -> TODO how) for NON bd atm'''
    
    global v  , tmpcnt
    v_nb = v[i].nb
    n_nb = np.size(v_nb)
  
    tmpcnt=[]    
    
  ###  -- Clear neighbor variables   -- ###    
    for j in v_nb:
        v[j].dA_i = np.zeros(3)   
        v[j].dH_i = np.zeros(3)
        v[j].dK_i = np.zeros(3)
        v[j].DS_i = np.zeros((3,3))
   
  ###  -- Sum over "temporary" variables cleared above   -- ###   
    for nj in range(n_nb):
        j = v_nb[nj]
        k = v_nb[(nj+1)%n_nb]    #i# if n_nb take back n_nb0   
        
        t = tr_index([i,j,k])
        if t<0: continue
        #print(t)
        
        tmpcnt.append(t)
        
            
     ###  -- Angle at vertex i   -- ###
        e_ji = v[j].c - v[i].c
        r_ji= np.linalg.norm(e_ji)
        e_ki  =  v[k].c - v[i].c
        r_ki = np.linalg.norm(e_ki)
        dot_at_i = np.dot(e_ji, e_ki)
        cos_i = dot_at_i/r_ji/r_ki
        angle_at_i = np.arccos(cos_i)
     # -> GRADIENT
        sin_i  = np.sin(angle_at_i)
        preFactor = r_ji*r_ki*sin_i
        term2 = e_ki/r_ki/r_ki + e_ji/r_ji/r_ji
        d_ang_i = (e_ki + e_ji - dot_at_i*term2)/preFactor 
        
     ###  -- Angle at vertex k   -- ###
        e_ik = v[i].c - v[k].c
        r_ik = r_ki
        e_jk = v[j].c - v[k].c
        r_jk = np.linalg.norm(e_jk)       
        dot_at_k = np.dot(e_jk, e_ik)
        cos_k = dot_at_k/r_jk/r_ik
        angle_at_k = np.arccos(cos_k)
        cot_k = 1/np.tan(angle_at_k)
      # -> GRADIENT
        sin_k = np.sin(angle_at_k)
        preFactor =  r_jk*r_ik*sin_k
        term2 = e_ik/r_ik/r_ik
        d_ang_k = -(e_jk - dot_at_k*term2)/preFactor #(e_jk - dot_at_k*term2)/preFactor  #TODO correction
                
     ###  -- Angle at vertex j from other two angles    --  ###
        angle_at_j = np.pi - angle_at_i - angle_at_k
        cot_j = 1/np.tan(angle_at_j)      
        sin_j = np.sin(angle_at_j)
      # -> GRADIENT
        d_ang_j = -d_ang_k -d_ang_i

     ###  -- Gaussian curvature contribution  --  ###       
      # -> GRADIENT, first sum grad_i theta_j. Outside loop sum the area term. 
        v[j].dK_i += d_ang_j
        v[k].dK_i += d_ang_k
                
     ###  -- Mean curvature contribution  --  ###       
        DkEji = np.tensordot(d_ang_k, e_ji, 0)/sin_k/sin_k  
        DiEjk = np.tensordot(d_ang_i, e_jk, 0)/sin_i/sin_i
        v[j].DS_i += -cot_k * np.identity(3) - DkEji - DiEjk
        
        DjEki = np.tensordot(d_ang_j, e_ki, 0)/sin_j/sin_j  
        DiEkj = -DiEjk
        v[k].DS_i += -cot_j * np.identity(3) - DjEki - DiEkj   
        
        
    # Area for obtuse triangles given by fixed fraction.     
        dA1 = sin_k * e_ik / r_ik
        dA2 = r_ik*cos_k * d_ang_k            
        
        if angle_at_i >= np.pi/2:                           
            v[j].dA_i += r_jk * (dA1 + dA2)/8
            v[k].dA_i += r_jk * (dA1 + dA2)/8
            
        elif angle_at_k >= np.pi/2:
            v[j].dA_i += r_jk * (dA1 + dA2)/8
            v[k].dA_i += r_jk * (dA1 + dA2)/4         
            
        elif angle_at_j >= np.pi/2:            
            v[j].dA_i += r_jk * (dA1 + dA2)/4
            v[k].dA_i += r_jk * (dA1 + dA2)/8 
         
        else:                       
            dA1 = 2*(-e_ji * cot_k)      #e_ij according to analyti
            dA2 = r_ji**2 * d_ang_k/sin_k/sin_k + r_jk**2 * d_ang_i/sin_i/sin_i
            v[j].dA_i += (dA1 - dA2)/8  
            
            dA1 = 2*(e_ik * cot_j)      
            dA2 = r_ki**2 * d_ang_j/sin_j/sin_j + r_jk**2 * d_ang_i/sin_i/sin_i
            v[k].dA_i += (dA1 - dA2)/8  
        
    ############## END OF SUM LOOP  ####################
    global allCNT
    if len(tmpcnt) < len(np.where(tr==i)[0]):
        print("COUNT IS OFF for {}".format(i))
        print(tmpcnt)
        print(np.where(tr==i)[0])
        print('\n')
        allCNT+=1
        
  ###  -- Final terms   -- ###           
    for j in v_nb:          
      ###  -- Gaussian curvature  --  ###
        v[j].dK_i += v[j].K* v[j].dA_i
        v[j].dK_i /= -v[j].A
        
       ###  -- Mean curvature    --  ###
        Svec = 4 * v[j].A * v[j].Hnorm
        Smag = np.linalg.norm(Svec)
        sign = np.sign(v[j].H)
        if Smag ==0:
            v[j].dH_i = np.zeros(3)            
        else:            
            dH1 = np.tensordot(v[j].DS_i,Svec,1)/Smag
            dH2 = -Smag * v[j].dA_i / v[j].A
            v[j].dH_i = sign*(dH1 + dH2)/(4*v[j].A)                
    
    return 0        
        
     
        
def calc_vertex_energy_grad(i):
    ''' Sum at a local level, all energy contributions. The local contributions
    are calculated within calc_localValues_grad(i), which is called only in this fx. 
    The hooke term is calculated before with calc_hooke(i), which also assings 
    the gradient dL, including the boundaries. 
    Called atm in recalculate() for 'A' grad_mode and in analysis() 
    in general.
    '''
    
    global v
    calc_hooke(i)
    v[i].E_L = cL*v[i].L    
    v[i].dE_L  = cL*v[i].dL    
    
    if not v[i].bd:         #ii# checks if boundary 
        calc_localValues_grad(i)
            
        v[i].W2 = v[i].H2 - v[i].K
        ##v[i].W2 =  v[i].K #guarp
        if v[i].W2 < 0:     #i# warp is always positive. If numerically negative, make zero
            v[i].W2 = 0
        v[i].dW2 = v[i].dH2 - v[i].dK
        ##v[i].dW2 = v[i].dK  #guarp
        v[i].W = np.sqrt(v[i].W2)


        if v[i].W == 0:
            v[i].dW = 0
        else:            
            v[i].dW = (2 * v[i].H * v[i].dH - v[i].dK)/ (2*v[i].W)      
            ##v[i].dW =  v[i].dK/ (2*v[i].W)      #guarp
        
        v[i].E_H = v[i].A*cH*(v[i].H-H0)**2
        v[i].E_W = v[i].A*cW*(v[i].W-W0)**2
        
        v[i].dE_H  =  cH * 2*v[i].A * (v[i].H - H0) * v[i].dH #v[i].A * cH * (v[i].dH2-2*H0*v[i].dH)
        v[i].dE_H +=  cH * v[i].dA * (v[i].H - H0)**2
               
        v[i].dE_W  = cW * 2*v[i].A *(v[i].W - W0) * v[i].dW #v[i].A*cW*(v[i].dW2-2*W0*v[i].dW)
        v[i].dE_W += cW * (v[i].W - W0)**2 * v[i].dA #v[i].dA*cW*(v[i].W-W0)**2
        
        #TODOrm#dW1 = cW * 2*v[j].A *(v[j].W - W0) * v[j].dW_i         
        #TODOrm#dW2 = cW * (v[j].W - W0)**2 * v[j].dA_i
        #TODOrm#v[i].dE_js += dW1+dW2 
        
    else:
        calc_bd_area(i)
        
    
    v[i].E   = v[i].E_H + v[i].E_W + v[i].E_L
    v[i].dE_i  = v[i].dE_H + v[i].dE_W + v[i].dE_L
    v[i].dE    = v[i].dE_i
    return 0


def calc_neighborhood_energy(i):
    ''' Sum at a local level, all energy contributions. The local contributions
    are calculated within calc_localValues_grad(i), which is called only in this fx. 
    The hooke term is calculated before with calc_hooke(i), which also assings 
    the gradient dL, including the boundaries. 
    Called atm in recalculate() for 'A' grad_mode and in analysis() 
    in general.
    '''
    
    global v
    
    v[i].dE_js = np.zeros(3)
    
    v_nb = v[i].nb
 
  #In full gradient, since stretching is an edge energy, including the neighborhood gradient    
    #is equivalent to multiplying the previous contribution times two (or adding another one)
    v[i].dE_js  += cL*v[i].dL  
    
    
    # If i is a boundary, but the neighbors are not, the grad E_j still has to be summed up. 
    #if not v[i].bd:         #ii# checks if boundary 
    calc_neighborhood_grad(i)   ##what if i is bd?-> OK by adding check on whether a triplet forms a triangle
        
    for j in v_nb:
        v[j].dW_i = np.zeros(3)
        
    # if j is a boundary vertex, its bending energy should be zero and therefore 
        # also the grad_i E_j energy. 
        
        if v[j].bd: continue
    
     ###  -- Mean curvature contribution  --  ###       
        dH1 = cH * 2*v[j].A *(v[j].H - H0) * v[j].dH_i         
        dH2 = cH * (v[j].H-H0)**2 * v[j].dA_i
        v[i].dE_js += dH1+dH2
        
     ###  -- Deviatoric curvature contribution  --  ###       \
     
        if not v[j].W == 0:                        
            v[j].dW_i = (2*v[j].H * v[j].dH_i - v[j].dK_i) / (2*v[j].W)  #Check about gradH2
            ##v[j].dW_i =   v[j].dK_i / (2*v[j].W)  #guarp
        dW1 = cW * 2*v[j].A *(v[j].W - W0) * v[j].dW_i         
        dW2 = cW * (v[j].W - W0)**2 * v[j].dA_i
        v[i].dE_js += dW1+dW2        
                
     ###  -- Update full gradient at i  --###           
    v[i].dE = v[i].dE_i + v[i].dE_js
            
    return 0

#Only calculates energy, not gradient, which is determined numerically
def calc_vertex_energy_num(i):    #i#assigns new energy values to that i-vertex object
    ''' Sum at a local level, all energy contributions. The local contributions
    are calculated within calc_localValues(i), which is called only in this fx. 
    The hooke term is calculated before with calc_hooke(i) for all vertices,
    including the boundaries. 
    Called atm in calc_num_grad to calculate energy before and after displacement 
    in general.
    '''
    global v
    calc_hooke(i) 
    v[i].E_L = cL*v[i].L 
    
    if not v[i].bd:
        calc_localValues(i)
        v[i].W2 = v[i].H2 - v[i].K
        ##v[i].W2 =  v[i].K   #guarp
        if v[i].W2 < 0:
            v[i].W2 = 0
        v[i].W = np.sqrt(v[i].W2)
        v[i].E_H = v[i].A*cH*(v[i].H-H0)**2
        v[i].E_W = v[i].A*cW*(v[i].W-W0)**2       
    else:
        calc_bd_area(i)

    v[i].E   = v[i].E_H + v[i].E_W + v[i].E_L
    return 0

def calc_bd_nb_grad(i):
    '''  Calculate the energy grad terms dE_H and dE_W of vertex i as averages
    over these of their neighboring non-boundary vertices. Add these to the 
    current value of dE for that vertex. 
    ** If used somewhere else, corroborate that the hooke gradient was computed
    somewhere else already
    
    Currently only called in recalculate() when bd_mode is 'A' '''
   # n_nb = len(v[i].nb)
    non_bd_nb = 0
    #for j in range(n_nb):
    for j in v[i].nb:
        if not v[j].bd:   #i# if not a boundary vertex
            v[i].dE_H += v[j].dE_H
            v[i].dE_W += v[j].dE_W
            non_bd_nb += 1
    if non_bd_nb != 0:
        v[i].dE_H /= non_bd_nb
        v[i].dE_W /= non_bd_nb
        v[i].dE += v[i].dE_H + v[i].dE_W    # dE_L computed before in calc_vertex_energy_grad(i)
        
def calc_num_grad(i, dh = dx):
    '''Calculate energy numerically with calc_vertex_evergy_num(i). 
    Assign gradient of total energy locally by numerical approximation of the
    gradient in the direction of coordinate c as: 
    dE_c = [E(c+dh)-E(c)]/dh, where by default dh=dx, set as an input parameter, and 
    E refers to the energy of sigle vertex i. 
    ONLY assigns dE, but NOT dE_H, dE_W, and dE_L
    ATM only called in recalculate(), when grad_mode='N'
    '''

    E_coord = np.zeros(3)
        
    #i# 'virtually' displace a dh in each direction and calculate energy change
    for j in range(3):  #Foreach direction x,y,z
        v[i].c[j] += dh
        calc_vertex_energy_num(i)
        E_coord[j] = v[i].E
        v[i].c[j] -= dh
    
    calc_vertex_energy_num(i)
    E_0 =  v[i].E               
    v[i].dE = (E_coord-E_0)/dh
    return 0
    
def calc_num_grad_full(i, dh=dx, shift=0):
    ''' Calculate the energy numerically with calc_vertex_num(i).
    Assign gradient of total energy locally by numerical approximation of the
    gradient in the direction of coordinate c as: 
    dE_c = [E(c+dh)-E(c)]/dh, where dh by default is dx, set as an input parameter,  and 
    E refers to the energy of the vertex i plus the energy of its neighboring.  
    ONLY assigns dE, but NOT dE_H, dE_W, and dE_L
    ATM called in gs_opt(), and analysis() and recalculate(), when grad_mode='F'
    '''
    E_coord = np.zeros(3)
    #E_shift = np.zeros(3)
    
    for k in range(3):  #For each direction x,y,z
        v[i].c[k] += dh
        E_coord[k] = nb_energy_num(i)
        v[i].c[k] -= dh
    
    if shift:
        dh = dh/2
        v[i].c += dh
        E_0 = nb_energy_num(i)
        v[i].c -= dh
        restore = nb_energy_num(i)  #this is just to recalculate the energy at initial coords
    else:    
        E_0 = nb_energy_num(i)      
               
    v[i].dE = (E_coord-E_0)/dh
    return 0

def calc_num_grad_full_center(i, dh=dx):
    ''' Calculate the energy gradient with the symmetric difference by taking 
    the difference between the energies of the point displaced +dh and -dh, divided
    by 2dh. Calculate and update the energy values by the numerical integration 
    of the energy. This function is only called by the gs() algorithm'''
    E_coord = np.zeros(3)
    
    for k in range(3):  #For each direction x,y,z
        v[i].c[k] += dh
        E_coord[k] = nb_energy_num(i)        
        
        v[i].c[k] -= 2*dh
        E_coord[k] -= nb_energy_num(i)
        v[i].c[k] += dh        
    
    calc_vertex_energy_num(i)         #i# A value for the energy there still has to be computed
    for j in v[i].nb:
        calc_vertex_energy_num(j)
        
    v[i].dE = E_coord/2/dh
    return 0



######################## DATA ##############################

def store_bd_coords():
    ''' Take the positions of the bd points in 'bd_v' and store them in the 
    three-element list cointaining the x,y,z coords, in order to easily plot them'''
    global curr_bd_c
    xs = np.array([])
    ys = np.array([])
    zs = np.array([])
    
    for i in bd_v:
        xs = np.append(xs,v[i].c[0])
        ys = np.append(ys,v[i].c[1])
        zs = np.append(zs,v[i].c[2])
    curr_bd_c = [xs, ys, zs]
    return 0    

def export_conf(fileName):
    ''' Save to a file, the current vertex and triangle information'''
    params_heading = "cH\tH0\tcW\tW0\tcS\tl0\n"
    params_values = "{0}\t{1}\t{2}\t{3}\t{4}\t{5}\n".format(cH,H0,cW,W0,cL,l_0)
    heading = "N_vertices\n{0}\n".format(n)    
    columns_a = "i\tbd\tx\ty\tz\tfix\tE\tE_H\tE_W\t"
    columns_b = "dEx\tdEy\tdEz\tdEHx\tdEHy\tdEHz\tH\tH2\tA\tW\tL\t"   
    columns_c = "Nx\tNy\tNz\n"
    with open(fileName,'w') as f:
        f.write(params_heading)
        f.write(params_values)
        f.write(heading)
        f.write(columns_a+columns_b+columns_c)
        for i in range(n):
            vx = v[i]
            args_a = [i,vx.bd,vx.c[0],vx.c[1],vx.c[2],vx.fixed,vx.E,vx.E_H,vx.E_W]
            la = '{0}\t{1}\t{2}\t{3}\t{4}\t{5}\t{6}\t{7}\t{8}'.format(*args_a)
            args_b = [vx.dE[0],vx.dE[1],vx.dE[2],vx.dE_H[0],vx.dE_H[1],vx.dE_H[2],vx.H,vx.H2,vx.A,vx.W,vx.L]
            lb = '{0}\t{1}\t{2}\t{3}\t{4}\t{5}\t{6}\t{7}\t{8}\t{9}\t{10}\t'.format(*args_b)
            args_c = [vx.norm[0],vx.norm[1],vx.norm[2]]
            lc = '{0}\t{1}\t{2}\n'.format(*args_c)
            f.write(la+lb+lc)
        f.write("N_triangles\n{0}\n".format(n_tr))
        for t in tr:
            f.write("{0}\t{1}\t{2}\n".format(t[0],t[1],t[2]))
            
def dDic(i):
    dicNew = {'bd':v[i].bd,'A':v[i].A, 'dA':v[i].dA, 'K':v[i].K,'dK':v[i].dK, 'H':v[i].H,'dH':v[i].dH,'W':v[i].W,'dW':v[i].dW}
    return dicNew
        
    
def analysis(update=False):
    ''' Calculate local energy terms and integrate each contribution. Print info.
    It does calculate the gradient at that time step but only prints the norm. 
    ATM  it is only called at go()'''
    global dataDF,grad_mode, n_obtuse
    
    av_mean, av_gauss, av_warp, tot_area, n_obtuse = 0,0,0,0,0
    E_tot, E_mean, E_warp, E_hooke, dE_norm = 0,0,0,0,0

    
    if grad_mode in ['fa','FA']:   #This calls two loops, so leave it out of i-loop
        tot_grad = calc_full_gradient()

    for i in range(n):
        if grad_mode in ['a','A']:
            calc_vertex_energy_grad(i)        
        elif grad_mode in ['n','N']:
            calc_vertex_energy_num(i)
            calc_num_grad(i)
        elif grad_mode in ['f','F']:
            calc_vertex_energy_num(i)
            calc_num_grad_full(i)
        elif grad_mode in ['fc','FC']:
            calc_vertex_energy_num(i)
            calc_num_grad_full_center(i)
            
        av_mean += v[i].H*v[i].A
        av_gauss += v[i].K*v[i].A
        av_warp += v[i].W*v[i].A
        tot_area += v[i].A            
        E_tot += v[i].E
        E_mean += v[i].E_H
 
        E_warp += v[i].E_W
        E_hooke += v[i].E_L
    av_mean /= tot_area
    av_gauss /= tot_area
    av_warp /= tot_area
    dE_norm = total_grad()
    n_obtuse = np.sum(tr_obtuse)

    print('Using the following, cH {}, H0 {}, cW {}, W0 {}, cL {}'.format(cH,H0,cW, W0,cL))
    print('Scipy.optimized method set at', lmin_method)
    print('Number of vertices: ' + str(n) + ', of which ' + str(len(bd_v)) + ' on boundary.')    
    print('Number of triangles: ' + str(n_tr)+ ' (' + str(np.sum(tr_obtuse)) + ' obtuse)')
    print('Total area: ' + str(round(tot_area,3)))
    print('---Average curvature values---')
    print('Mean: ' + str(round(av_mean,3)))
    print('Gauss: ' + str(round(av_gauss,3)))
    print('Warp: ' + str(round(av_warp,3)))
    print('---Energy components---')
    print('Mean: ' + str(round(E_mean,3)))
    print('Warp: ' + str(round(E_warp,3)))
    print('Hooke: ' + str(round(E_hooke,3)))
    print('Total: ' + str(round(E_tot,3)))
    print('Gradient norm: ' + str(round(dE_norm,5)))
    
    # DF
    if update:
        dic = {'conf':start_conf}
        dic['dE'] = dE_norm
        dic['E'] = E_tot
        dic['E_H'] = E_mean
        dic['E_W'] = E_warp
        dic['E_L'] = E_hooke
        dic['A'] = tot_area
        dic['avH'] = av_mean
        dic['avK'] = av_gauss
        dic['avW'] = av_warp
        dic['Nm'] = n
        dic['Nbd'] = len(bd_v)
        dic['Macumm'] = np.sum(np.array(M_array))
        dataDF = dataDF.append(dic, ignore_index=True)
        print("DF updated")

def plot_vsRuns(array=E_array, key='E'):
    ''' Plot attribute in [array] per minimization iteration, given by relative
    step in the M_array. [key] is a label to assign plot label and, in case it
    is the size step, set a log scale. 
    Arrays: S_array ['S'] -> size step, E_array ['E'] -> total energy, 
    dE_array ['dE'] -> total gradient, Conv_array ['C'] -> convergence tracer. 
    while M_array contains 0 at begin step and the modifier (default=1) at rest.
    '''     
    global M_array
    nameDic = {'S':'Relative steps', 'E':'Total Surface Energy', 'dE':'Total Gradient','C':'Average vertex displacement'}    
    
    steps = np.array(M_array)
    cum_steps = np.cumsum(steps)
    
    plt.figure()
    plt.plot(cum_steps, array, '.')
    plt.ylabel(nameDic[key])
    plt.xlabel('Relative steps')
    if key=='S' or key=='C':
        plt.yscale('log')


def saveFiles(tag,outdir, offset=0):
    num = len(M_array)-1+offset    
    export_conf(outdir+'Confs/conf'+tag+'_{}.config'.format(num))
    plot_vsRuns(E_array, 'E')
    plt.savefig(outdir+'E_runs'+tag+'.png')
    plot_vsRuns(dE_array,'dE')
    plt.savefig(outdir+'dE_runs'+tag+'.png')
    plot_vsRuns(S_array, 'S')
    plt.savefig(outdir+'S_runs'+tag+'.png')
    plot_vsRuns(Conv_array, 'C')
    plt.savefig(outdir+'C_runs'+tag+'.png')
    runsDF = pd.DataFrame({'E':E_array,'dE':dE_array,'M':M_array,'S':S_array, 'Conv':Conv_array})
    runsDF.to_csv('runsDF'+tag+'.dat',sep='\t')         

############################################################
###############################################################################
######################## PLOTTING FUNCTIONS ##############################


    
def show(show_init_bd = False,show_bd=True):
    showFig.clf()    #i#plt.close()
    vc=[]
    for i in range(n):
        vc.append(v[i].c)
    vc = np.array(vc)
    tg = tri.Triangulation(vc[:,0],vc[:,1],triangles=tr)
    fig1 = plt.figure("show")   #i#plt.figure()
    plt.axes(projection='3d')
    ax1 = fig1.gca()#projection='3d')
    ax1.plot_trisurf(tg, Z=vc[:,2])
    
    if start_conf in ['O','o']:
        ax1.set_zlim3d(0,2*R)
    elif start_conf in ['L', 'l']:
        ax1.set_zlim3d(-n_seg*h_seg/2,n_seg*h_seg/2)
        ax1.set_box_aspect((R,R,n_seg*h_seg))
    elif start_conf in ['S', 's']:
        ax1.set_zlim3d(0.75,1.25)
        ax1.set_box_aspect((1, n_y/n_x, 0.5))
    elif start_conf in ['H', 'h']:
        ax1.set_zlim3d(-0.1,wh+hh)
        #ax1.set_box_aspect((2*rh, 2*rh, wh+hh))
    else:
        ax1.set_zlim3d(0,2)
        ax1.set_box_aspect((R,R,h_cyl))
        
    ax1.set_xlabel('x',fontsize=14)
    ax1.set_ylabel('y',fontsize=14)
    ax1.set_zlabel('z',fontsize=14)
    
    #ax1.set_xlim((-1.2,1.2))
    #ax1.set_ylim((-1.2,1.2))
    #ax1.set_zlim((-0.01,2.2))
    ax1.set_axis_off()
    
  # CHECK boundary elements
    if show_bd:
        for vertex in bd_v:    
            p = ax1.scatter(v[vertex].c[0],v[vertex].c[1],v[vertex].c[2],c='r', s=50)
    if show_init_bd:
        ax1.scatter(curr_bd_c[0],curr_bd_c[1],curr_bd_c[2],c='g',s=40)
     
    return 0

def plot_vertex(i, nb=False):
    off = 1.001
    if nb:
        newFig = plt.figure()
        ax1 = newFig.gca(projection='3d')
        n_nb = len(v[i].nb)
        trs = []
        vcs = [v[i].c]
        for j in range(n_nb):
            color = str(0.8-0.7*j/n_nb)
            nei = v[i].nb[j]
            if tr_index([i,nei,v[i].nb[(j+1)%n_nb]])>=0:
                trs.append([0,j+1,(j+1)%n_nb+1])
            vcs.append(v[nei].c)
            ax1.scatter(v[nei].c[0],v[nei].c[1],v[nei].c[2],c=color,s=50)
            ax1.text(v[nei].c[0]*off,v[nei].c[1]*off,v[nei].c[2]*off, str(nei), color=color)
        vcs = np.array(vcs)
        tg = tri.Triangulation(vcs[:,0],vcs[:,1],triangles=trs)
        ax1.plot_trisurf(tg, Z=vcs[:,2])            
    else:
        ax1 = showFig.gca(projection='3d')
    
    ax1.scatter(v[i].c[0],v[i].c[1],v[i].c[2],c='k',s=150)
    
    
def plot_attribute(attr): 
    localFig.clf()
    fig1 = plt.figure("local")   #i#plt.figure()
    ax1 = fig1.gca(projection='3d')
    vc=[]
    vcolor = []
    tcolor = []
    for i in range(n):
        vx = v[i]
        vc.append(vx.c)
        vcolor.append(getattr(vx,attr))
        #ax1.text(vx.c[0],vx.c[1],vx.c[2], str(getattr(vx,attr)))
    for t in tr:
        color = getattr(v[t[0]],attr)
        color += getattr(v[t[1]],attr)
        color += getattr(v[t[2]],attr)
        tcolor.append(color/3)
    vc = np.array(vc)
    tg = tri.Triangulation(vc[:,0],vc[:,1],triangles=tr)

    ax1.plot_trisurf(tg, Z=vc[:,2])
    p = ax1.scatter(vc[:,0],vc[:,1],vc[:,2],c=vcolor)    
        
    if start_conf in ['O','o']:
        ax1.set_zlim3d(0,2*R)
    elif start_conf in ['L', 'l']:
        ax1.set_zlim3d(-n_seg*h_seg/2,n_seg*h_seg/2)
        ax1.set_box_aspect((R,R,n_seg*h_seg))
    elif start_conf in ['S', 's']:
        ax1.set_zlim3d(0.75,1.25)
        ax1.set_box_aspect((1, n_y/n_x, 0.5))
    elif start_conf in ['H', 'h']:
        ax1.set_zlim3d(-0.1,wh+hh)
        #ax1.set_box_aspect((2*rh, 2*rh, wh+hh))
    else:
        ax1.set_zlim3d(0,h_cyl)
        ax1.set_box_aspect((R,R,h_cyl))
        
    ax1.set_xlabel('x',fontsize=14)
    ax1.set_ylabel('y',fontsize=14)
    ax1.set_zlabel('z',fontsize=14)
    
    localFig.colorbar(p,ax=ax1)

    return 0    
            
############################################################
###############################################################################
######################## RUN FUNCTIONS ##############################   


def runRoutine(gSteps, nRep, tag, offset):#g_steps=[50,250,500,2000], nRepeats=[10,20,30,20]):
    
  # Save initial fast evolution 
    for r in range(nRep):
        num = len(M_array)-1+offset
        plot_attribute('H')
        localFig.savefig(outDir+'LocalH/LocalH'+tag+'_{}.png'.format(num))

        itisDone = g_AG(gSteps)
        
        if itisDone:
            num = len(M_array)-1+offset
            print("\n**Convergence found @",num,'\n')
            analysis(True)
            show(True)
            saveFiles(tag,outDir,offset)
            plot_attribute('H')
            localFig.savefig(outDir+'LocalH/LocalH'+tag+'_{}_CONV.png'.format(num))
            dataDF.to_csv(outDir+'dataDF'+tag+'.dat', sep='\t')
            return 'it is done'
        
        print("Currently @",num)
        analysis(True)
        show(True)    
        saveFiles(tag,outDir,offset)             
        if r%4 == 0:
            dataDF.to_csv(outDir+'dataDF'+tag+'.dat', sep='\t')
    return 0
  
 

def runThis(routines=[[50,10],[250,20],[500,30],[1000,40]],
            ref=False, offset=0):    
    start_time = time.perf_counter()    
    tag = '_cH{1}_Ho{2}_cW{3}_Wo{4}_cS{0}'.format(cL,cH,H0,cW,W0)
    if not os.path.exists(outDir+'LocalH/'):
        os.makedirs(outDir+'LocalH/')
    if not os.path.exists(outDir+'Confs/'):
        os.makedirs(outDir+'Confs/')
    
    export_conf(outDir+'conf'+tag+'_INIT{}.config'.format(offset))
    
    totalRuns = np.sum(np.array([r[0]*r[1] for r in routines]))
    print('Running in total {} iterations. Stopping at avg net displacement {}'.format(totalRuns, CONVTOL))
    
    for gSteps, rReps in routines:
        itisDone = runRoutine(gSteps, rReps, tag, offset)
        if itisDone:
            break  

    if ref:
        start_time_refinement = time.perf_counter()
        print("Time for shape pre refinement was {0:4}s".format(start_time_refinement-start_time))
        refine()
        start_extra_steps = time.perf_counter()
        tag = tag+'_n{}'.format(n)
        export_conf(outDir+'conf'+tag+'_INIT.config')
        for r in range(1):
            g(20,linmin_opt,interrupt=True,jolt_when_stuck=True)
            analysis(True)
            show(True)
            saveFiles(tag,outDir,offset)
        end_time_refined = time.perf_counter()
        print ('Extra refinement step done')
        print('Refinement took {0:2}s'.format(start_extra_steps-start_time_refinement))
        print('Extra steps after refinement took {0:.4}s'.format(end_time_refined-start_extra_steps))
    
        
    end_time = time.perf_counter()    
    print("TOTAL run time: {0:.4}s".format(end_time-start_time))    
    
    
        
full_time_start = time.time()
def endTime():
    full_time_end = time.time()
    dt = full_time_end-full_time_start
    stime = "{0} h: {1} min: {2:.3} s".format(int(np.floor(dt/3600)), int(np.floor((dt/60))%60), dt%60)
    print("\n\n TOTAL program TIME was ",stime)
atexit.register(endTime)


summary()
print('\nNow running initial surface setup:\n')
showFig = plt.figure("show",figsize=(9,12))#i#
localFig = plt.figure("local",figsize=(9,9))#i#
reset()
analysis(True) 
runThis(routines=[[50,10],[250,20],[500,30],[1000,40]],offset=0)
