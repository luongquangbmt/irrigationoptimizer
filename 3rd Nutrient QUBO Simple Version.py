#!/usr/bin/env python
# coding: utf-8

# In[2]:


import numpy as np
import dimod
from itertools import product
from collections.abc import Iterable
from dwave.system import LeapHybridDQMSampler
get_ipython().system('pip install pyqubo')
from pyqubo import Binary
import neal


# In[3]:


#RW 7/14 added phosphorous to the prescription (added things like p_min, p, and p_range)
class Prescription():
    def __init__(self, water, nutrients, phosphorous):
        
        if isinstance(water,Iterable):
            assert (len(water)==len(nutrients))
        
        self.water = water
        self.nutrients = nutrients
        self.phosphorous = phosphorous
    
    def __len__(self):
        return len(self.water)
    
    def get_yield(self, w_range=12, n_range=12, p_range=12, w_peak=6, n_peak=8, p_peak = 8):
        a,b,c,d,e,f = 0.413,1.949,1.352,-2.472,1.218,-2.033
        # function maximum if concave (down)
        maxn = (2*f*b/e - c)/(e - 4*f*d/e)
        maxw = (2*d*c - e*b) / (e**2 - 4*d*f)
        maxp = (2*b*f - c*e)/(e - 4*f*d/e)
        # normalize water and nutrients
        w_min = w_peak - w_range*maxw
        n_min = n_peak - n_range*maxn
        p_min = p_peak - p_range*maxp
    
        w = (self.water - w_min) / w_range
        n = (self.nutrients - n_min) / n_range
        p = (self.phosphorous - p_min) / p_range
    
        return (a + b*n + c*w + d*(n)**2 + e*n*w + f*(w)**2 + p**2 + p*f*n)

    


# In[29]:


# available water and nutrients
Wtotal = 24
Ntotal = 32
#RW 7/14/2021 Added this value for the new phosphorous nutrient
Ptotal = 32

ngridcells = 4

gridcells = np.arange(ngridcells)

# possible values of water, as integers
water = np.array([4,5,6,7])
w_peak=6
w_range=12 # sets approx scaling of integer labels corresponding to 0 to 1

# possible values of nutrients, as integers
nutrients = np.array([5,6,7,8,9,10])
n_peak=8
n_range=12 # "0" to 1

#RW 7/14/2021 added possible values of phosphorous nutrient
phosphorous = np.array([5,6,7,8,9,10])
p_peak=8
p_range=12

# these values map to these yields, with optimal at w=2, n=2
#field_yield = get_yield(*np.array([i for i in product(water,nutrients)]).T,
#                        w_range=w_range,
#                        w_peak=w_peak,
#                        n_peak=n_peak,
#                        n_range=n_range).reshape(-1,len(nutrients))

#RW 7/14/2021 added phosohprous to the function below
prescription = Prescription(*np.array(list(product(water, nutrients, phosphorous))).T)
# always try highest yield first
bias = np.sort(prescription.get_yield())[::-1]


# In[30]:


# different yields for different grid cells
eta = [1.0*100, 0.7*100, 0.5*100, 0.9*100]
#eta = [100.,100.,100.,100.]
#eta = [100.,1.]
#eta = [100]*4/


# number of grid cells in field
ngridcells = len(eta)

# ZONE definitions for fixed irrigation
#waterzones = {'a':[0,1,2,3,4], 'b':[5,6,7,8,9,10,11],
#              'c':[12,13,14,15,16,17,18,19,20,21,22,23,24],'d':[25,26,27,28,29,30]}

waterzones = {'a':[0,2], 'b':[1,3]}

try:
    assert (np.sort(np.array([v1 for v in waterzones.values() for v1 in v ]).flatten()) == np.arange(ngridcells)).all()
except:
    print('ERROR. Check that all grid cells have been assigned to a water zone.')


# CONSTRAINTS    
LL1, LL2 = 2000, 30
onehotconstraint, zonelinking = True, True

#RW 7/14/2021 added LL5 and phosphorousconstraint
LL3, LL4, LL5 = 100,100,100
waterconstraint, nutrientconstraint, phosphorousconstraint = True, True, True





# In[31]:


H = 0
Wused=0
Nused=0

#RW 7/14/2021 added PUsed which represents phosphorous used
Pused=0
field_yield = prescription.get_yield()

f = 0

for g in range(ngridcells):
    
    # only one prescription (nutrient+water) allowed per grid cell
    #RW 7/14/2021 added prescription.phosphorous and p
    index = np.array([Binary('{}_{}_{}_{}'.format(g,w,n,p)) 
                      for w,n,p in zip(prescription.water,prescription.nutrients,prescription.phosphorous)])
    print(index)
    H -= (eta[g]*prescription.get_yield()*index).sum()
    
    # one-hot restriction
    onehot = index.sum()
    # water used
    Wused+=(prescription.water*index).sum()
    # nutrients used
    Nused+=(prescription.nutrients*index).sum()
    #RW 7/14/2021 added phosphorous used
    Pused+=(prescription.phosphorous*index).sum() 
    
    if onehotconstraint:
        H += LL1*(onehot - 1)**2


# In[32]:



#zone-linking constraint
# penalize if water j != water j' when looking at cells in the same zone

for zone, gridcells in waterzones.items():
print(zone)
print(gridcells)
# there are no linking constraints needed
if len(gridcells)==1:
    continue
# first grid cell. Comparing all others to this first grid cell.
g = gridcells[0]
for ww in water:
    print(ww)
    # these are allowed
    #RW 7/14/2021 added p, p1, and the for statements involving p and p1. Also added a 4th bracket to
    #account for phosphorous
    allowed = np.array([Binary('{}_{}_{}_{}'.format(g,ww,n,p))*Binary('{}_{}_{}_{}'.format(g1,ww,n1,p1))
               for g1 in gridcells[1:]
               for n,p in zip(prescription.nutrients,prescription.phosphorous)
               for n1,p1 in zip(prescription.nutrients,prescription.phosphorous)])
                #for g1 in gridcells[1:] 
                   #for n in prescription.nutrients 
                       #for n1 in prescription.nutrients 
                          # for p in prescription.phosphorous
                               #for p1 in prescription.phosphorous])
    
    print(allowed)    
    H -= LL2*allowed.sum()


# In[33]:


# water constraint (possible values: 0 to 15)
slackVar = 8*Binary('I8w')+4*Binary('I4w')+2*Binary('I2w') + 1*Binary('I1w')
if waterconstraint:
    H+=LL3*(Wused + slackVar - Wtotal)**2

# nutrient constraint (possible values: 0 to 15)
slackVar = 8*Binary('I8n')+4*Binary('I4n')+2*Binary('I2n') + 1*Binary('I1n')
if nutrientconstraint:
    H+=LL4*(Nused + slackVar - Ntotal)**2
    
#RW 7/14/2021 added phosphorous constraint
slackVar = 8*Binary('I8n')+4*Binary('I4n')+2*Binary('I2n') + 1*Binary('I1n')
if phosphorousconstraint:
    H+=LL5*(Pused + slackVar - Ptotal)**2


# In[ ]:


H


# In[34]:


model = H.compile()
bqm = model.to_bqm()
#bqm = model.to_dimod_bqm()

sa = neal.SimulatedAnnealingSampler()
sampleset = sa.sample(bqm, num_reads=1000)


# In[35]:


best_sample = sampleset.first


# In[36]:


residual = {'w':0, 'n':0, 'p':0}
total_yield = 0
waterused = 0
nutrientsused = 0
#RW 7/14/2021 added phosphorous used
phosphorousused = 0

#RW 7/14/2021 added p_peak to the format
print('Peak yield is at w={},n={}, p={}'.format(w_peak,n_peak,p_peak))
for key,value in best_sample.sample.items():
    #print(key)
    if value==1:
        if (key[0]=='I'):
            kind = key[-1]
            residual[kind] = residual[kind] + int(key[1:-1])
        #RW 7/14/2021 added p variable, phosphorous, and phosphorous used to the format
        else:
            f,w,n,p = [int(i) for i in key.split('_')]
            total_yield += eta[f]*Prescription(w,n,p).get_yield()
            print('Grid cell {:} used {:} Water, {:} Nutrients, and {:} Phosphorous (Yield {:.3f})'.format(
                f,w,n,p ,eta[f]*Prescription(w,n,p).get_yield()))
            waterused +=w
            nutrientsused +=n
            phosphorousused+=p
print('Water used: {:3d} (out of available {})'.format(waterused,Wtotal))
print('Nutri used: {:3d} (out of available {})'.format(nutrientsused,Ntotal))
#RW 7/14/2021 added phosphorous used into formatting
print('Phosphorous used: {:3d} (out of available {})'.format(phosphorousused,Ptotal))
print('\nUnused resources (residuals from inequality):')
print(residual)
print('\nWater zone restrictions:')
print(waterzones)
print('\nTotal Yield {:.3f}'.format(total_yield))


# In[ ]:




