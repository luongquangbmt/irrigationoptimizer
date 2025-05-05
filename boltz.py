import numpy as np
from itertools import product
from collections.abc import Iterable
import functools
import pandas as pd

import dimod

from dwave.system import LeapHybridDQMSampler


def convert_slacks_to_binary(slacks, delta_s=1):

    max_slacks = slacks.max()/delta_s

    num_slack_variables = int(np.floor(np.log2(max_slacks)) + 1)
    slack_variables = [int(delta_s*2**k) for k in range(num_slack_variables-1)]

    # last one takes residual
    res = delta_s* (max_slacks+1 - 2**(num_slack_variables-1))
    if res > 0:
        slack_variables.append(res)

    return slack_variables

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

class QUBO():
    """
    Example use:
        qubomodel = QUBO(primary_variable, secondary_variables,
               weight=weights,func=get_yield)
        qubomodel.add_constraint('water',50, kind='less_equal',ll=10)
        qubomodel.build_dqm()
        [ or directly qubomodel.solve_dqm()]
               
        primary_variable is a dictionary of finest division of grid
        secondary_variables is a dictionary of additional variables (e.g., water, nutrients)
        func is the function used to calculate a value based on chosen secondary_variables
            Here, this is the yield based on water and nutrients
        weights is an array which allows for differences in yield (SIZE: len(primary_variable))
        ll is Lagrange parameter for constraint
    """
    def __init__(self, primary, secondary, 
                 func=get_yield, weight=None,**kwargs):

        # psize: number of primary (top-level) variables, 
        #       (each has to select from dsize variables)
        self.weight = weight
        self.primary = [vv for vv in primary.values()][0]
        psize = len(self.primary)
        self.psize = psize
        dvalues = np.array([i for i in product(*secondary.values())]).T
        labels = [k for k in secondary.keys()]
        self.secondary = {labels[i]:dvalues[i][:] for i in range(len(labels))}
        
        
        self.dqm_slack = False
        
        # dsize: number of discretized variables
        dsize = len(dvalues.T)
        self.dsize = dsize
        
        # keep track of size of QUBO we are solving
        self.qsize = psize*dsize
        
        # These are the initial base linear biases based on function, func
        self.base_linear_biases = {}
        for i,p in enumerate(self.primary):
            #print(p)
            self.base_linear_biases[p] = weight[i]*func(*dvalues)
        
        self.constraint = {}
        self.nslacks = 0
        self.slack_labels = []
        self.sampleset = None
        
    def add_constraint(self, lhs, rhs, ll=1.0, kind='less_equal',
                       slack_variables=None, dqm_slack=True, normalize = True):
        # max_slack of zero is equivalent to an equality constraint
        
    
        """slackVar: list of possible values of slack variables
                These may be limited based on the constraints and the
                possible combinations of variables.
        """
        self.dqm_slack = dqm_slack
        
        if dqm_slack:
            slacks_factor = 1
        else:
            slacks_factor = 2
        
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
            # finds all unique integers
                num_slack_variables = int(np.floor(np.log2(max_slack)) + 1)
                slack_variables = [2**k for k in range(num_slack_variables-1)]
                # last one takes residual
                slack_variables.append(max_slack+1 - 2**(num_slack_variables-1))
                
        v = np.tile(self.secondary[lhs], self.psize)

        if dqm_slack and (len(slack_variables)> 0):
            print('{}. Adding {} slack variables as single DQM variable.'.format(lhs, len(slack_variables)))
            vnew = np.concatenate((v, np.zeros(slacks_factor*self.nslacks),
                        np.array(slack_variables)))
        else:
            print('{}. Adding {} independent slack variables'.format(lhs, len(slack_variables)))
        
        #print(v)
            # needs to be self.nslacks*2 since each is independent. could be 'on' or 'off'
            vnew = np.concatenate((v, np.zeros(slacks_factor*self.nslacks),
                        np.stack([np.zeros(len(slack_variables)),slack_variables]).T.flatten()))
                        
#        slack_variables))
        #print(vnew)
        if kind in ['equal','less_equal']:
            # add (lhs-rhs)**2 as a penalty
            # assuming all slack_variables are allowed
            # set diagonal
            #print(vnew,rhs)
            # sum(lhs) + slack_variables = rhs
            #print(len(vnew))
            result = (-2*rhs*vnew + vnew**2)*np.eye(len(vnew))
            #print(result)
            # set off-diagonal elements
            result += 2*(1-np.eye(len(vnew)))*vnew.reshape(-1,len(vnew)).T * vnew
            #print(result)
            # add constant (rhs**2)
            result += rhs**2
            #print(rhs, result)
            normalize_offset = True
            #print(lhs, slack_variables)
           
            
        elif kind in ['many_to_one']:
            # create vector for all primary variables
            primary_all = np.concatenate((np.repeat(self.primary, self.dsize),np.zeros(slacks_factor*self.nslacks)))

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
            normalize_offset = False
    
        # upper triangle
        result = np.triu(result)
    
        # zero out slack rows/columns from previous constraints
        #print(len(v), slacks_factor, self.nslacks)
        result[len(v):len(v)+slacks_factor*self.nslacks,:] = 0
        result[:,len(v):len(v)+slacks_factor*self.nslacks] = 0
        
        if normalize:
            result = np.triu(self._normalize_array(result, offset=normalize_offset))
        
        self.constraint[label] = {
            'penalty': result,
            'lambda':ll,
            'slack_variables':slack_variables
        }
        if dqm_slack and (len(slack_variables)>0):
            self.slack_labels += ['{}_Slack{}'.format(label, len(slack_variables))]
        else:
            self.slack_labels += ['{}_S{}'.format(label,ss) for ss in slack_variables]

        self.nslacks += len(slack_variables)
        self.qsize += slacks_factor*len(slack_variables)
    
    
    #############
    #############
    
    def solve_dqm(self,maximum=True,sampler=LeapHybridDQMSampler, *args, **kwargs):
        self.build_dqm(maximum=maximum)
        chosen_sampler = sampler()
        sampleset = chosen_sampler.sample_dqm(self.dqm, *args, **kwargs)
        self.sampleset = sampleset
    
    def _normalize_array(self, a, offset=True):
        amin = a[a!=0].min()
        amax = a[a!=0].max()
        if amin==amax:
            amin = a.min()
            amax = a.max()

        if not offset:
            result = a / (amax - amin)
        else:
            result = (a - amin)/(amax - amin)
            result[a==0] = 0
        return result
    
    def _pad_zeroes(self, x, tsize):
        """ Pad array with zeroes so that it matches tsize.
        This is needed when adding slack variables.
        """
        # x is array
        # tsize is target size
        result = np.zeros((tsize,tsize))
        result[:x.shape[0],:x.shape[1]] = x
        return result
    
    def summarize(self, printonly=False):
        # number of variables in each field
        if printonly:
            print('Number of primary variables:', self.psize)
            print('Number of secondary options:', self.dsize)
            print('Number of constraints:', len(self.constraint))
            print('Number of slack variables:', self.nslacks)
            # total number of variables
            print('Number of variables:', self.dqm.num_cases())
            print('Number of variable interactions:',self.dqm.num_case_interactions())
            print('Number of DQM variable interactions:', self.dqm.num_variable_interactions())
            print('Number of DQM variables:',self.dqm.num_variables())
            return
        else:
            df = pd.DataFrame({'n_cells':self.psize,
                           'n_prescriptions':self.dsize,
                           'n_constraints':len(self.constraint),
                           'n_slacks':self.nslacks,
                           'n_dqm_cases':self.dqm.num_cases(),
                           'n_dqm_variables':self.dqm.num_variables(),
                           'n_dqm_case_interactions':self.dqm.num_case_interactions(),
                           'n_dqm_variable_interactions':self.dqm.num_variable_interactions()},index=[0])
        return df
            
    def build_dqm(self, maximum=True):
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
                # add zeroes if needed (if constraint['penalty'] is larger than penalty)
                penalty = self._pad_zeroes(penalty, constraint['penalty'].shape[0])
                penalty+=constraint['penalty']*constraint['lambda']

        # cycle through primary variables, adding linear terms
        nvariables = self.dsize
        for i,p in enumerate(self.primary):
            dqm.add_variable(nvariables, label=str(p))
            # add base linear terms
            biases = factor*self.base_linear_biases[p]
            #print('simple yield biases:', biases)
            # add linear terms from constraint
            if not isinstance(penalty,int):
                biases -= factor*np.diag(
                        penalty[i*nvariables:(i+1)*nvariables,i*nvariables:(i+1)*nvariables])
            #print('post-constraint linear biases:', biases)
            dqm.set_linear(str(p),biases)
        # add slacks, linear terms
        if self.nslacks > 0:
            slacks = np.diag(penalty[nvariables*self.psize:,nvariables*self.psize:])
            #print(slacks)
            #print('building', slacks.reshape(-1,2), slacks.shape, self.slack_labels)
            if self.dqm_slack:
                iix = 0
                for ss in self.slack_labels:
                    # number of variables is saved in the label name for this variable
                    nvariables = int(str(ss).split('Slack')[-1])
                    dqm.add_variable(nvariables, label=str(ss))
                    s = slacks[iix:iix+nvariables]
                    bias = -1.0*factor*s
                    #print(ss,bias)
                    dqm.set_linear(str(ss), bias)
                    iix += nvariables
            else:
                for s,ss in zip(slacks.reshape(-1,2), self.slack_labels):
                # note that slack has two options a 'zero' or a 'one'
                    nvariables = 2
                    dqm.add_variable(nvariables, label=str(ss))
                    #print(ss, bias)
                    bias = -1.0*factor*s
                    #print(ss,bias)
                    dqm.set_linear(str(ss), bias)
        
        combos = [i for i in product(self.primary, self.primary) if i[0]!=i[1]]

        nvariables = self.dsize
        for f1,f2 in combos:
            if not isinstance(penalty,int):
                biases = -1.0*factor*penalty[f1*nvariables:(f1+1)*nvariables,f2*nvariables:(f2+1)*nvariables]
                dqm.set_quadratic(str(f1),str(f2), biases)
                
        # add slacks, quadratic terms
        if self.nslacks > 0:
#            slacks = np.arange(nvariables*self.psize, 
#                           nvariables*self.psize+self.nslacks+1)
            offset = nvariables*self.psize
            if self.dqm_slack:
                iix = 0
                for ss in self.slack_labels:
                    nn = int(str(ss).split('Slack')[-1])
                    for f1 in self.primary:
                        bias = -1.0*factor*penalty[f1*nvariables:(f1+1)*nvariables,
                                                        offset+iix:offset+iix+nn]
                        #print('slack bias', ss, bias)
                        dqm.set_quadratic(str(f1),
                                  '{}'.format(ss), 
                                  bias)
                    iix += nn
            else:    
                for s,ss in zip(np.arange(self.nslacks), self.slack_labels):  
                    for f1 in self.primary:
                        #print(offset+s*2, offset+(s+1)*2)
                        bias = -1.0*factor*penalty[f1*nvariables:(f1+1)*nvariables,
                                                        offset+s*2:offset+(s+1)*2]
                        
                        dqm.set_quadratic(str(f1),
                                  '{}'.format(ss), 
                                  bias)
                    for s2,ss2 in zip(np.arange(self.nslacks), self.slack_labels):
                        bias = -1.0*factor*penalty[offset+s2*2:offset+(s2+1)*2,
                                                    offset+s*2:offset+(s+1)*2]
                        if (ss!=ss2) & np.any(bias):
                            dqm.set_quadratic('{}'.format(ss2),
                                    '{}'.format(ss),
                                    bias)
                            #print('slack bias', ss,ss2, bias)
        self.dqm = dqm