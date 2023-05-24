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
    
    y_start = cells_per_unit_dist_y * (0-setup["domain"]["y"]["range"][0]) - y_coor_lim
    y_end = cells_per_unit_dist_y * (0-setup["domain"]["y"]["range"][0]) + y_coor_lim
    
    print(f'x_coor:{x_coor}')
    print(f'y_start:{y_start}')
    print(f'y_end:{y_end}')

    drag = jnp.sum(data_dict["velocity"][-1, 0, x_coor, y_start:y_end, 0]**2, axis=0)

    return drag

#compute_flow_quantity(json.load(open("cylinderflow(1).json")), data_dict=None, alpha=0.5)
# alphas = np.linspace(0,1,2)[1:]    

# #exit(1)
val_tuple = []

num = 10
alphas = np.linspace(0, 1, num=num+2)
alphas = np.round(alphas[1:-1], 3)

for alpha in alphas:
    #jsonFile = open(case_setup, "r")
    
    modify_alpha(case_file_path="cylinderflow(1).json", alpha=alpha)

    print(f"Iterating for alpha={alpha}")
    case_setup = json.load(open("cylinderflow.json"))
    input_reader = InputReader(case_setup, "numerical_setup.json")
    initializer  = Initializer(input_reader)
    sim_manager  = SimulationManager(input_reader)

    # RUN SIMULATION
    buffer_dictionary = initializer.initialization()
    sim_manager.simulate(buffer_dictionary)

    path = "./results/cylinderflow-1/domain"

    quantities = ['mask_real', 'velocity']
    cell_centers, cell_sizes, times, data_dict = load_data(path, quantities, start=-1)
    #print(data_dict)
    

    drag = compute_flow_quantity(case_setup=case_setup, data_dict=data_dict, alpha=alpha)

    print(f"Drag:{drag}")
    val_tuple.append((alpha, drag))



print(val_tuple)

exit(1)



# In[15]:


#alpha = np.linspace(0,1,2)[1:][0]

num = 10
alphas = np.linspace(0, 1, num=num+2)
alphas = np.round(alphas[1:-1], 3)

for i, alpha in enumerate(alphas):
    path = f"./results1/cylinderflow-{i+1}/domain"
    #print(path)
    case_setup = json.load(open("cylinderflow copy.json"))
    quantities = ['mask_real', 'velocity']
    cell_centers, cell_sizes, times, data_dict = load_data(path, quantities, start=-1)
    #print(data_dict)

    modify_alpha(case_file_path="cylinderflow copy.json", alpha=alpha)
    drag = compute_flow_quantity(setup=case_setup, data_dict=data_dict, alpha=alpha)

    print(f"Drag:{drag}")


# In[2]:


from jax.lib import xla_bridge
import jax
import os
#os.environ['CUDA_VISIBLE_DEVICES'] = ''
#jax.config.update('jax_platform_name', 'cpu')
print(xla_bridge.get_backend().platform)


# The present case setup file *cylinder.json* specifies a cylinder with a diamenter of 1.0, an inlet velocity of 0.1 and a dynamic viscosity of 0.0025. This results in a Reynolds number of 40.

# In[3]:


# SETUP SIMULATION
case_setup = json.load(open("cylinderflow(1).json"))
input_reader = InputReader(case_setup, "numerical_setup.json")
initializer  = Initializer(input_reader)
sim_manager  = SimulationManager(input_reader)

# RUN SIMULATION
buffer_dictionary = initializer.initialization()
sim_manager.simulate(buffer_dictionary)


# Now we decrease the dynamic viscosity to 0.0005 increasing the Reynolds number to 200 and simulate.

# In[4]:


# SETUP SIMULATION
case_setup = json.load(open("cylinderflow(1).json"))
case_setup["material_properties"]["dynamic_viscosity"] = 0.0005
input_reader = InputReader(case_setup, "numerical_setup.json")
initializer  = Initializer(input_reader)
sim_manager  = SimulationManager(input_reader)

# RUN SIMULATION
buffer_dictionary = initializer.initialization()
sim_manager.simulate(buffer_dictionary)


# ## Visualize the flow field
# We plot the vorticity contours at the final time snapshot of both Reynolds numbers.

# In[26]:


# LOAD AND PLOT DATA
quantities = ["vorticity", "mask_real"]

path = "./results/cylinderflow-1/domain"
cell_centers, cell_sizes, times, data_dict = load_data(path, quantities, start=-1)
mask = data_dict["mask_real"][-1,:,:,0]
vorticity_40 = data_dict["vorticity"][-1,-1,:,:,0] * mask

path = "./results/cylinderflow-2/domain"
cell_centers, cell_sizes, times, data_dict = load_data(path, quantities, start=-1)
vorticity_200 = data_dict["vorticity"][-1,-1,:,:,0] * mask

X,Y = np.meshgrid(cell_centers[0],cell_centers[1],indexing="ij")

norm = colors.CenteredNorm(vcenter=0.0, halfrange=0.3)

fig, ax = plt.subplots(2,1)
fig.set_size_inches([10,10])
ax[0].pcolormesh(X,Y,vorticity_40,cmap="seismic",norm=norm)
ax[1].pcolormesh(X,Y,vorticity_200,cmap="seismic",norm=norm)
ax[0].set_aspect("equal")
ax[1].set_aspect("equal")
plt.show()


# In[6]:


quantities = ["vorticity", "mask_real"]

path = "./results/cylinderflow-1/domain"
cell_centers, cell_sizes, times, data_dict = load_data(path, quantities, start=-1)
mask = data_dict["mask_real"][-1,:,:,0]
vorticity_40 = data_dict["vorticity"][-1,-1,:,:,0] * mask

path = "./results/cylinderflow-2/domain"
cell_centers, cell_sizes, times, data_dict = load_data(path, quantities, start=-1)
vorticity_200 = data_dict["vorticity"][-1,-1,:,:,0] * mask

X,Y = np.meshgrid(cell_centers[0],cell_centers[1],indexing="ij")

norm = colors.CenteredNorm(vcenter=0.0, halfrange=0.3)

fig, ax = plt.subplots(2,1)
fig.set_size_inches([10,10])
ax[0].pcolormesh(X,Y,vorticity_40,cmap="seismic",norm=norm)
ax[1].pcolormesh(X,Y,vorticity_200,cmap="seismic",norm=norm)
ax[0].set_aspect("equal")
ax[1].set_aspect("equal")
plt.show()


# In[7]:


import haiku as hk
import jax.numpy as jnp
import jax
def net_fn(x_in):
    """Multi-layer perceptron """
    x = jnp.transpose(x_in[:, :, :, 0])
    mlp = hk.Sequential([
        hk.Linear(32), jax.nn.relu,
        hk.Linear(32), jax.nn.relu,
        hk.Linear(1), 
    ])
    x_out = jnp.exp(mlp(x))
    x_out = jnp.expand_dims(jnp.transpose(x_out), axis=-1)
    return x_out
net = hk.without_apply_rng(hk.transform(net_fn))


# In[ ]:




