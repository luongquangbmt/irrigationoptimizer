from boltz import BQM, DQM, Solver
from boltzKB import QUBO
import numpy as np
import matplotlib.pyplot as plt
import time

# number of farms
ngridcells = 4

# fraction of optimal water available
wr_f = 0.8
# fraction of optimal nutrients available
nr_f = 0.8

# yield efficiency for each grid cell,
#     based on previous years' average yield maps
weights = [0.5, 1.0, 0.5, 0.8]

gridcells = np.arange(ngridcells)

# possible values of water, as integers
water = np.array([3,6,9])
w_peak = 6

# possible values of nutrients, as integers
nutrients = np.array([4,8,12])
n_peak = 8

# available water and nutrients
# This is set to a certain percentage of optimal values
Wtotal = np.ceil(w_peak*ngridcells * wr_f)
Ntotal = np.ceil(n_peak*ngridcells * nr_f)


# Variables
primary_variable = {'gridcells':gridcells}
secondary_variables = {'water':water, 'nutrients':nutrients}


# Defining Models
bqm = BQM(primary_variable, secondary_variables,weight=weights)
dqm = DQM(primary_variable, secondary_variables,weight=weights)
#qubo = QUBO((primary_variable, secondary_variables,weight=weights)

# Prepare Constraint

### DQM
dqm_constraints = {
    'water1':{
        'lhs':'water',
        'rhs':Wtotal,
        'kind':'less_equal',
        'll': 10
    },
    'nutrients1':{
        'lhs':'nutrients',
        'rhs':Ntotal,
        'kind':'less_equal',
        'll':10
    },
     'water2':{
         'lhs':'water',
         'rhs':{'a':[0,1],'b':[2,3]},
         'kind':'many_to_one',
         'll':1.0
     }
}
### BQM
bqm_constraints = {
    'Total Water Constraint':{
        'lhs':bqm.TotalWaterConstraint,
        'rhs':Wtotal,
        'kind':'less_equal',
        'll': 10
    },
    'Total Nutrients Constraint':{
        'lhs':bqm.TotalNutrientConstraint,
        'rhs':Ntotal,
        'kind':'less_equal',
        'll':10
    }
}

for c in range(len(bqm.CellUniqueConstraints)):
    bqm_constraints['Cell {} constraint'.format(c)] = {
        'lhs':bqm.CellUniqueConstraints[c],
        'rhs':1,
        'kind':'equal',
        'll':10
    }
# Adding constraint:
### DQM
for _, constraint in dqm_constraints.items():
    dqm.add_constraint(constraint['lhs'],
                             constraint['rhs'],
                             kind=constraint['kind'], 
                             ll=constraint['ll'])
### BQM
for _, constraint in bqm_constraints.items():
    bqm.add_constraint(constraint['lhs'],
                             constraint['rhs'],
                             kind=constraint['kind'], 
                             ll=constraint['ll'])
### Build QUBO and Summerize
dqm.build_qm()
dqm.summarize()
print('\n')
bqm.build_qm()
bqm.summarize()


### Timing

list_of_num_grid = np.arange(4,50)
#print(list_of_num_grid)
list_of_run_time = np.zeros(list_of_num_grid.shape)

for g in list_of_num_grid:
    # Variables
    grid = np.arange(g)
    weights = np.ones(g); 
    primary_variable_x = {}
    primary_variable_x['cell ' + str(g)] = {'gridcells': grid}; 
    secondary_variables_x = {'water':water, 'nutrients':nutrients}
    tmp_bqm = BQM(primary_variable_x['cell ' + str(g)], secondary_variables_x,weight=weights)
    ### BQM Constraint
    bqm_constraints = {
        'Total Water Constraint':{
            'lhs':tmp_bqm.TotalWaterConstraint,
            'rhs':Wtotal,
            'kind':'less_equal',
            'll': 10
        },
        'Total Nutrients Constraint':{
            'lhs':tmp_bqm.TotalNutrientConstraint,
            'rhs':Ntotal,
            'kind':'less_equal',
            'll':10
        }
    }

    for c in range(len(tmp_bqm.CellUniqueConstraints)):
        bqm_constraints['Cell {} constraint'.format(c)] = {
            'lhs':tmp_bqm.CellUniqueConstraints[c],
            'rhs':1,
            'kind':'equal',
            'll':10
        }
    ### Adding Constraint
    for _, constraint in bqm_constraints.items():
        tmp_bqm.add_constraint(constraint['lhs'],
                                 constraint['rhs'],
                                 kind=constraint['kind'], 
                                 ll=constraint['ll'])
    ### Build-Solve and Timing
    tmp_bqm.build_qm()
    #tmp_bqm.summarize()
    start = time.time()
    sampleset = Solver(tmp_bqm.qm)
    #qubo_solution = sampleset.first
    #SAS_solution = tmp_bqm.bqm.convert_solution(qubo_solution)
    list_of_run_time[g-4] = time.time() - start
    print('{} farm sampling in: {}'.format(g,list_of_run_time[g-4]))
plt.plot(list_of_num_grid,list_of_run_time)