#!/usr/bin/env python
# coding: utf-8

# # JAX-Fluids: Cylinderflow demo
# 
# In this demo we show how to simulate a subsonic viscous cylinderflow at the Reynolds numbers 40 and 200. It is known that the phenomenon of periodic vortex shedding first appears for Reynolds numbers around 40-50 (Linnick, Mark N., and Hermann F. Fasel. "A high-order immersed interface method for simulating unsteady incompressible flows on irregular domains." Journal of Computational Physics 204.1 (2005)).
# 
# Frist we import all necessary packages

# In[13]:


import json
import matplotlib.pyplot as plt
from matplotlib import colors
import numpy as np

from jaxfluids import InputReader, Initializer, SimulationManager
from jaxfluids.post_process import load_data


# In[ ]:





# In[14]:



import jax.numpy as jnp
# print(modify_alpha(format, alpha=.5))
# create_json(modify_alpha(format, .6))



def modify_alpha(case_file_path, alpha: float):

    # jsonFile = open(case_setup, "r") # Open the JSON file for reading
    # setup = json.load(jsonFile) # Read the JSON into the buffer
    # jsonFile.close() # Close the JSON file

    jsonfile = open(case_file_path, "r")
    setup = json.load(jsonfile)
    jsonfile.close()

    levelset_string = "lambda x,y: - 0.5 + jnp.sqrt(" + str(alpha) + "*x**2 + y**2)"
    setup["initial_condition"]["levelset"] = levelset_string

    jsonfile = open(case_file_path, "w+")
    jsonfile.write(json.dumps(setup))
    jsonfile.close()

def compute_flow_quantity(setup, data_dict, alpha):
    # jsonFile = open(case_setup, "r") # Open the JSON file for reading
    # setup = json.load(jsonFile) # Read the JSON into the buffer
    # jsonFile.close() # Close the JSON file

    cells_per_unit_dist_x = setup["domain"]["x"]["cells"]/(setup["domain"]["x"]["range"][1] - setup["domain"]["x"]["range"][0])
    cells_per_unit_dist_y = setup["domain"]["y"]["cells"]/(setup["domain"]["y"]["range"][1] - setup["domain"]["y"]["range"][0])
    x_coor = int(cells_per_unit_dist_x * 1/jnp.sqrt(alpha)) 
    x_coor = int(cells_per_unit_dist_x * (0-setup["domain"]["x"]["range"][0]) - x_coor) 
    
    major_axis = 1.0 #fixed
    y_coor_lim = int(cells_per_unit_dist_y * major_axis) 
    
    y_start = int(cells_per_unit_dist_y * (0-setup["domain"]["y"]["range"][0]) - y_coor_lim)
    y_end = int(cells_per_unit_dist_y * (0-setup["domain"]["y"]["range"][0]) + y_coor_lim)
    
    print(f'x_coor:{x_coor}')
    print(f'y_start:{y_start}')
    print(f'y_end:{y_end}')

    drag = jnp.sum(data_dict["velocity"][-1, 0, x_coor, y_start:y_end, 0]**2, axis=0)

    return drag

#compute_flow_quantity(json.load(open("cylinderflow(1).json")), data_dict=None, alpha=0.5)
# alphas = np.linspace(0,1,2)[1:]    

val_tuple = []
num = 100
alphas = np.linspace(2, 4, num=num+2)
alphas = np.round(alphas[1:-1], 3)

for i,alpha in enumerate(alphas):
    #jsonFile = open(case_setup, "r")
    

    f = open ("current_vals.txt", "a")
    f.write(f"{i+1}) alpha: {alpha}")
    

    modify_alpha(case_file_path="cylinderflow.json", alpha=alpha)

    print(f"Iterating for alpha={alpha}")
    case_setup = json.load(open("cylinderflow.json"))
    input_reader = InputReader(case_setup, "numerical_setup.json")
    initializer  = Initializer(input_reader)
    sim_manager  = SimulationManager(input_reader)

    # RUN SIMULATION
    buffer_dictionary = initializer.initialization()
    sim_manager.simulate(buffer_dictionary)

    if i!=0:
        path = f"./results/cylinderflow-{i}/domain"
    else:
        path = f"./results/cylinderflow/domain"

    quantities = ['mask_real', 'velocity']
    cell_centers, cell_sizes, times, data_dict = load_data(path, quantities, start=-1)
    #print(data_dict)
    

    drag = compute_flow_quantity(setup=case_setup, data_dict=data_dict, alpha=alpha)

    f.write(f" Drag: {drag}\n")
    f.close()

    print(f" Drag:{drag}\n")
    val_tuple.append((alpha, drag))



print(val_tuple)

exit(1)


