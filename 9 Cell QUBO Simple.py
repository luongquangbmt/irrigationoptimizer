#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import dimod
from itertools import product
from collections.abc import Iterable
from dwave.system import LeapHybridDQMSampler
get_ipython().system('pip install pyqubo')
from pyqubo import Binary
import neal


class Prescription():
    def __init__(self, water, nutrients):
        
        if isinstance(water,Iterable):
            assert (len(water)==len(nutrients))
        
        self.water = water
        self.nutrients = nutrients
    
    def __len__(self):
        return len(self.water)
    
    def get_yield(self, w_range=12, n_range=12, w_peak=6,n_peak=8):
        a,b,c,d,e,f = 0.413,1.949,1.352,-2.472,1.218,-2.033
        # function maximum if concave (down)
        maxn = (2*f*b/e - c)/(e - 4*f*d/e)
        maxw = (2*d*c - e*b) / (e**2 - 4*d*f)
        # normalize water and nutrients
        w_min = w_peak - w_range*maxw
        n_min = n_peak - n_range*maxn
    
        w = (self.water - w_min) / w_range
        n = (self.nutrients - n_min) / n_range
    
        return (a + b*n + c*w + d*(n)**2 + e*n*w + f*(w)**2)

    


# In[2]:


# available water and nutrients
#RW Changed water and nutrient totals from 24 and 32 to 54 and 72 respectively (9/4) * 24 = 54 and (9/4) * 32 = 72
Wtotal = 54
Ntotal = 72

#RW changed the number of gridcells to 9 from 4 (from 2x2 to 3x3)
ngridcells = 9

gridcells = np.arange(ngridcells)

# possible values of water, as integers
water = np.array([4,5,6,7])
w_peak=6
w_range=12 # sets approx scaling of integer labels corresponding to 0 to 1

# possible values of nutrients, as integers
nutrients = np.array([5,6,7,8,9,10])
n_peak=8
n_range=12 # "0" to 1

# these values map to these yields, with optimal at w=2, n=2
#field_yield = get_yield(*np.array([i for i in product(water,nutrients)]).T,
#                        w_range=w_range,
#                        w_peak=w_peak,
#                        n_peak=n_peak,
#                        n_range=n_range).reshape(-1,len(nutrients))

prescription = Prescription(*np.array(list(product(water, nutrients))).T)
# always try highest yield first
bias = np.sort(prescription.get_yield())[::-1]


# In[7]:


# different yields for different grid cells

#Original eta values for 4 cells
#eta = [1.0*100, 0.7*100, 0.5*100, 0.9*100]

#RW (7/15/2021) added additional (random) eta values for each new gridcell 
eta = [1.0*100, 0.7*100, 0.5*100, 0.9*100, 0.8*100, 0.95*100, 0.9*100, 0.7*100, 0.6*100]
#eta = [100.,100.,100.,100.]
#eta = [100.,1.]
#eta = [100]*4/


# number of grid cells in field
ngridcells = len(eta)

# ZONE definitions for fixed irrigation
#waterzones = {'a':[0,1,2,3,4], 'b':[5,6,7,8,9,10,11],
#              'c':[12,13,14,15,16,17,18,19,20,21,22,23,24],'d':[25,26,27,28,29,30]}

#Original waterzones
#waterzones = {'a':[0,2], 'b':[1,3]}

#RW 7/15/2021 changed waterzones from original 2x2 to 3x3
waterzones = {'a':[0,3,6], 'b':[1,4,7], 'c':[2,5,8]}

try:
    assert (np.sort(np.array([v1 for v in waterzones.values() for v1 in v ]).flatten()) == np.arange(ngridcells)).all()
except:
    print('ERROR. Check that all grid cells have been assigned to a water zone.')


# CONSTRAINTS    
LL1, LL2 = 1e6, 70
onehotconstraint, zonelinking = True, True
LL3, LL4 = 100,100
waterconstraint, nutrientconstraint = True, True





# In[18]:


H = 0
Wused=0
Nused=0
field_yield = prescription.get_yield()

f = 0

for g in range(ngridcells):
    
    # only one prescription (nutrient+water) allowed per grid cell
    index = np.array([Binary('{}_{}_{}'.format(g,w,n)) 
                      for w,n in zip(prescription.water,prescription.nutrients)])
    H -= (eta[g]*prescription.get_yield()*index).sum()
    
    # one-hot restriction
    onehot = index.sum()
    # water used
    Wused+=(prescription.water*index).sum()
    # nutrients used
    Nused+=(prescription.nutrients*index).sum()
    if onehotconstraint:
        H += LL1*(onehot - 1)**2
    
#zone-linking constraint
# penalize if water j != water j' when looking at cells in the same zone

#Original 2x2 waterlinking constraint
"""
for zone, gridcells in waterzones.items():
    # there are no linking constraints needed
    if len(gridcells)==1:
        continue
    # first grid cell. Comparing all others to this first grid cell.
    g = gridcells[0]
    for ww in water:
        # these are allowed
        allowed = np.array([Binary('{}_{}_{}'.format(g,ww,n))*Binary('{}_{}_{}'.format(g1,ww,n1))
                   for g1 in gridcells[1:] for n in prescription.nutrients 
                   for n1 in prescription.nutrients])
        H -= LL2*allowed.sum()
        
"""

#RW 7/15/2021 3x3 waterlinking constraint
for zone, gridcells in waterzones.items():
    # there are no linking constraints needed
    print(gridcells)
    if len(gridcells)==1:
        continue
    # first grid cell. Comparing all others to this first grid cell.
    g = gridcells[0]
    #print(g)
    #print(gridcells[1:])
    #print(gridcells[1])
    #print(gridcells[2:])
    #print(gridcells[2])
    
    for ww in water:
        print(ww)
        # these are allowed
        allowed = np.array([Binary('{}_{}_{}'.format(g,ww,n))*Binary('{}_{}_{}'.format(g1,ww,n1))
            #*Binary('{}_{}_{}'.format(g2,ww,n2))
                   for g1 in gridcells[1:] for n in prescription.nutrients 
                   for n1 in prescription.nutrients])
                    #for g2 in gridcells[2:] 
                       
                   #for n2 in prescription.nutrients])
        print(allowed)
        H -= LL2*allowed.sum() 


# In[10]:



# water constraint (possible values: 0 to 15)
slackVar = 8*Binary('I8w')+4*Binary('I4w')+2*Binary('I2w') + 1*Binary('I1w')
if waterconstraint:
  H+=LL3*(Wused + slackVar - Wtotal)**2

# nutrient constraint (possible values: 0 to 15)
slackVar = 8*Binary('I8n')+4*Binary('I4n')+2*Binary('I2n') + 1*Binary('I1n')
if nutrientconstraint:
  H+=LL4*(Nused + slackVar - Ntotal)**2


# In[11]:


#H


# In[12]:


model = H.compile()
#model


# In[13]:


bqm = model.to_bqm()
#bqm = model.to_dimod_bqm()


# In[14]:


sa = neal.SimulatedAnnealingSampler()
sampleset = sa.sample(bqm, num_reads=1000)


# In[15]:


best_sample = sampleset.first
#print(best_sample)


# In[19]:


residual = {'w':0, 'n':0}
total_yield = 0
waterused = 0
nutrientsused = 0
print('Peak yield is at w={},n={}'.format(w_peak,n_peak))
for key,value in best_sample.sample.items():
    #print(key)
    #print(value)
   
    if value==1:
        if (key[0]=='I'):
            kind = key[-1]
            residual[kind] = residual[kind] + int(key[1:-1])
        else:
            #if ("_" in key): #RW 7/16/2021 added this if statement because I found that the model was returning multiplication
                #of 2 numbers (ex. 9*262). This is added so that only prescriptions are processed
                f,w,n = [int(i) for i in key.split('_')]
                total_yield += eta[f]*Prescription(w,n).get_yield()
                print('Grid cell {:} used {:} Water and {:} Nutrients (Yield {:.3f})'.format(
                    f,w,n, eta[f]*Prescription(w,n).get_yield()))
                waterused +=w
                nutrientsused +=n
print('Water used: {:3d} (out of available {})'.format(waterused,Wtotal))
print('Nutri used: {:3d} (out of available {})'.format(nutrientsused,Ntotal))
print('\nUnused resources (residuals from inequality):')
print(residual)
print('\nWater zone restrictions:')
print(waterzones)
print('\nTotal Yield {:.3f}'.format(total_yield))


# In[ ]:





# In[ ]:




