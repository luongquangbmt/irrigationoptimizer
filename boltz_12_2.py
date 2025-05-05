#!/usr/bin/env python
# coding: utf-8

# ### Import Library

# In[1]:


import numpy as np
from itertools import product,combinations
from collections.abc import Iterable
import functools
import qubovert as qv
from dwave.system import LeapHybridDQMSampler
from neal import SimulatedAnnealingSampler
import time
import dimod


# ### Outer Function

# In[2]:


def get_yield(water, nutrients, w_peak=6, n_peak=8, w_range=12, n_range=12):
    a,b,c,d,e,f = 0.413,1.949,1.352,-2.472,1.218,-2.033
    # function maximum if concave (down)
    maxn = (2*f*b/e - c)/(e - 4*f*d/e)
    maxw = (2*d*c - e*b) / (e**2 - 4*d*f)
    # normalize water and nutrients
    #   results in yield peak at w_peak and n_peak
    w_min = w_peak - w_range*maxw
    n_min = n_peak - n_range*maxn
    
    w = (water - w_min) / w_range
    n = (nutrients - n_min) / n_range
    
    return (a + b*n + c*w + d*(n)**2 + e*n*w + f*(w)**2)


# ### Classes

# #### Super Classes QM containt DQM and BQM:

# In[3]:


class QM:
    def __init__(self, primary,secondary,weight):
            # psize: number of primary (top-level) variables, 
            #       (each has to select from dsize variables)
            self.weight = weight
            self.primary = [vv for vv in primary.values()][0]
            self.secondary = secondary
            self.labels = [k for k in secondary.keys()]


# In[15]:


class DQM(QM):
    def __init__(self, primary, secondary,weight):
        super().__init__(primary, secondary,weight)
        dvalues = np.array([i for i in product(*self.secondary.values())]).T
        self.secondary = {self.labels[i]:dvalues[i][:] for i in range(len(self.labels))}
        
        # psize: number of primary (top-level) variables, 
        #       (each has to select from dsize variables)
        psize = len(self.primary)
        self.psize = psize
        
        # dsize: number of discretized variables
        dsize = len(dvalues.T)
        self.dsize = dsize
        
        # keep track of size of QUBO we are solving
        self.qsize = psize*dsize
        
        # keep track of size of QUBO we are solving
        self.qsize = psize*dsize
        # These are the initial base linear biases based on function, func
        self.base_linear_biases = {}
        for i,p in enumerate(self.primary):
            self.base_linear_biases[p] = weight[i]*get_yield(*dvalues)

        self.constraint = {}
        self.nslacks = 0
        self.slack_labels = []
            
    def add_constraint(self, lhs, rhs, ll=1.0, kind='less_equal',
                       slack_variables=None, normalize = True):
        # max_slack of zero is equivalent to an equality constraint
        """slackVar: list of possible values of slack variables
                These may be limited based on the constraints and the
                possible combinations of variables.
        """
        
        if kind == 'equal':
            max_slack = 0
            slack_variables = np.array([])
            label = '{}_{}_{}'.format(lhs,kind,rhs)
        elif kind == 'many_to_one':
            # map many primary variables ('grids') to the same value of lhs
            max_slack = 0
            slack_variables = np.array([])
            label = '{}_{}'.format(lhs,kind)
        else:
            # this assumes that lhs is Positive
            max_slack = rhs
            label = '{}_{}_{}'.format(lhs,kind,rhs)
            
            if slack_variables is None:
                slack_variables = np.array([2**k for k in range(int(np.floor(np.log2(max_slack)))+1)])

        #print('{}. Adding {} slack variables'.format(lhs, len(slack_variables)))
        
        v = np.tile(self.secondary[lhs], self.psize)
        vnew = np.concatenate((v, np.zeros(self.nslacks),slack_variables))
        if kind in ['equal','less_equal']:
            # set diagonal
            result = (-2*rhs+1)*vnew*np.eye(len(vnew))
    
            # set off-diagonal elements
            result += 2*(1-np.eye(len(vnew)))*vnew.reshape(-1,len(vnew)).T * vnew
    
            # add constant (rhs**2)
            result += result+rhs**2
        elif kind in ['many_to_one']:
            # create vector for all primary variables
            primary_all = np.concatenate((np.repeat(self.primary, self.dsize),np.zeros(self.nslacks)))

            # find where lhs matches other lhs values
            matched = np.sqrt(vnew.reshape(-1,len(vnew)).T * vnew) % 1 == 0
            result = np.zeros_like(matched).astype(int)
            for _, pp in rhs.items():
                ind = np.isin(primary_all,pp)
                penalty_ = (~matched).astype(int)
        # remove penalties for grids not included in many_to_one zone
                penalty_[~ind,:] = 0
                penalty_[:,~ind] = 0
                result += penalty_
    
        # upper triangle
        result = np.triu(result)
    
        # zero out slack rows/columns from previous constraints
        result[len(v):len(v)+self.nslacks,:] = 0
        result[:,len(v):len(v)+self.nslacks] = 0
        
        if normalize:
            result = np.triu(self._normalize_array(result))
        
        self.constraint[label] = {
            'penalty': result,
            'lambda':ll,
            'slack_variables':slack_variables
        }
        self.slack_labels += ['{}_S{}'.format(label,ss) for ss in slack_variables]

        self.nslacks += len(slack_variables)
        self.qsize += len(slack_variables)
    
#     def solve_dqm(self,maximum=True,sampler=LeapHybridDQMSampler):
#         self.build_dqm(maximum=maximum)
#         chosen_sampler = sampler()
#         sampleset = chosen_sampler.sample_dqm(self.dqm)
#         self.sampleset = sampleset
    
    def _normalize_array(self, a, amin=0, amax=1):
        return a/(a.max() - a.min())
    
    def _pad_zeroes(self, x, tsize):
        """ Pad array with zeroes so that it matches tsize.
        This is needed when adding slack variables.
        """
        # x is array
        # tsize is target size
        result = np.zeros((tsize,tsize))
        result[:x.shape[0],:x.shape[1]] = x
        return result
    
    def summarize(self):
        print('Model: Dicrete Quadratic Model')
        # number of variables in each field
        print('Number of primary variables:', self.psize)
        print('Number of secondary options:', self.dsize)
        print('Number of constraints:', len(self.constraint))
        print('Number of slack variables:', self.nslacks)
        # total number of variables
        print('Number of variables:', self.dqm.num_cases())
        print('Number of variable interactions:',self.dqm.num_case_interactions())
        print('Number of DQM variable interactions:', self.dqm.num_variable_interactions())
        print('Number of DQM variables:',self.dqm.num_variables())
    
    def build_qm(self, maximum=True):
        dqm = dimod.DiscreteQuadraticModel()
        if maximum:
            factor = -1.
        else:
            factor = 1.
        # Build penalty terms as fully-connected array
        penalty = 0
        for _,constraint in self.constraint.items():
            try:
                penalty+=constraint['penalty']*constraint['lambda']
            except:
                # add zeroes if needed
                penalty = self._pad_zeroes(penalty, constraint['penalty'].shape[0])
                penalty+=constraint['penalty']*constraint['lambda']

        # cycle through primary variables, adding linear terms
        nvariables = self.dsize
        for i,p in enumerate(self.primary):
            dqm.add_variable(nvariables, label=str(p))
            # add base linear terms
            biases = factor*self.base_linear_biases[p]
            #print(biases)
            # add linear terms from constraint
            biases -= factor*np.diag(
                    penalty[i*nvariables:(i+1)*nvariables,i*nvariables:(i+1)*nvariables])
            #print(biases)
            dqm.set_linear(str(p),biases)
        # add slacks, linear terms
        slacks = np.diag(penalty[nvariables*self.psize:,nvariables*self.psize:])
        for s,ss in zip(slacks, self.slack_labels):
            nvariables = 1
            dqm.add_variable(nvariables, label=str(ss))
            bias = -factor*s
            dqm.set_linear(str(ss), [bias])
        
        combos = [i for i in product(self.primary, self.primary) if i[0]!=i[1]]

        nvariables = self.dsize
        for f1,f2 in combos:
            biases = -penalty[f1*nvariables:(f1+1)*nvariables,f2*nvariables:(f2+1)*nvariables]
            dqm.set_quadratic(str(f1),str(f2), biases)
        slacks = np.arange(nvariables*self.psize, 
                           nvariables*self.psize+self.nslacks+1)
        for s,ss in zip(slacks, self.slack_labels):  
            for f1 in self.primary:
                bias = -factor*penalty[s:s+1,f1*nvariables:(f1+1)*nvariables]
                dqm.set_quadratic(str(f1),
                              '{}'.format(ss), 
                              bias)
        self.dqm = dqm


# In[16]:


class BQM(QM):
    def __init__(self, primary, secondary,weight):
        super().__init__(primary, secondary,weight)
        self.bqm = 0
        self.water = self.secondary['water']
        self.nutrients = self.secondary['nutrients']
        self.var = [[[qv.boolean_var('x_{}_{}_{}'.format(i,n,w)) for w in range(len(self.water))]for n in range(len(self.nutrients))] for i in range(len(self.primary))]
        self.num_constraints = 0
        ### Objective Function in Qubo
        for i in range(len(self.primary)):
            for n in range(len(self.nutrients)):
                for w in range(len(self.water)):
                    self.bqm -= weight[i]*self.var[i][n][w]*get_yield(self.water[w], self.nutrients[n])
                    
        ### Total Water & Total Nutrients Constraint
        self.TotalWaterConstraint = 0
        self.TotalNutrientConstraint = 0
        
        ### Water:
        for c in range(len(self.primary)):
            for n in range(len(self.nutrients)):
                for w in range(len(self.water)):
                    self.TotalWaterConstraint += self.water[w]*self.var[c][n][w]
        ### Nutrient
        for c in range(len(self.primary)):
            for w in range(len(self.water)):
                for n in range(len(self.nutrients)):
                    self.TotalNutrientConstraint += self.nutrients[n]*self.var[c][n][w]
        ### Cell Unique Constraints:
        self.CellUniqueConstraints = [ 0 for i in range(len(self.primary))]
        for c in range(len(self.primary)):
            for n in range(len(self.nutrients)):
                for w in range(len(self.water)):
                      self.CellUniqueConstraints[c] += self.var[c][n][w]
             
    ##################################################
    def add_constraint(self, lhs, rhs, ll, kind,
                        slack_variables=None, normalize = True):
            if kind == 'equal':
                self.bqm.add_constraint_eq_zero(lhs - rhs, lam = ll)
                self.num_constraints += 1
            elif kind == 'less_equal':
                self.bqm.add_constraint_le_zero(lhs - rhs, lam = ll)
                self.num_constraints += 1
            else:
                print('Can not define this constraint in BQM, please other models')
        
    ##################################################
    def summarize(self):
        print('Model: Binary Quadratic Model')
        # number of variables in each field
        print('Number of primary variables:', len(self.primary))
        print('Number of secondary options:', len(self.nutrients)*len(self.water))
        print('Number of constraints:', self.num_constraints)
        print('Number of total variable:', self.bqm.num_binary_variables)
        print('Number of QUBO variable:', self.qm.num_binary_variables)
    ##################################################
    def build_qm(self, maximum=True):
        self.qm = self.bqm.to_qubo()
        
class Solver:
    """
    build sampler to solve the quadratic problem:
    """
    def __init__(self,qm):
        #self.qm = qm
        if isinstance(qm,BQM):
            chosen_sampler = SimulatedAnnealingSampler()
            sampleset = chosen_sampler.sample_qubo(qm.qm.Q,num_reads=8000)
            #self.sampleset = sampleset
            self.sample = sampleset.first
        elif isinstance(qm,DQM):
            chosen_sampler = LeapHybridDQMSampler()
            sampleset = chosen_sampler.sample_dqm(qm)
            #self.sampleset = sampleset
            self.sample = sampleset.first
