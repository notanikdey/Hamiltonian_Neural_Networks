import autograd 
import autograd.numpy as np 

import scipy.integrate
solve_ivp = scipy.integrate.solve_ivp 

import matplotlib.pyplot as plt 



def hamiltonian(coords):
    q , p = np.split(coords, 2)
    #for an ideal pendulum system, the hamiltonian is given by H = 2mgl(1-cos(q)) + (l^2*p^2)/2m
    H = 3*(1-np.cos(q)) + p**2 #Acc to the code on the paper's repo
    return H 

def dynamics(t, coords):
    dcoords = autograd.grad(hamiltonian)(coords)
    dHdq, dHdp = np.split(dcoords, 2)
    S = np.concatenate([dHdp,-dHdq]) #(dq/dt, dp/dt) = (dH/dp, -dH/dq )
    return S 

def trajectory(t_span=[0,3], timescale=10, radius=None, y0=None, noise_std=0.1, **kwargs):
    #creating the different time steps for the trajectories 
    t_eval = np.linspace(t_span[0], t_span[1], int(timescale*(t_span[1]-t_span[0])))

    #getting intial state 
    if y0 is None:
        y0 = np.random.rand(2)*2-1 #samples between -1 and 1
    if radius is None:
        radius = np.random.rand() + 1.3 #sampling a range of radii
    y0 = y0/np.linalg.norm(y0) * radius 
    # In Hamiltonian systems where:
    #     H(q, p) = q^2 + p^2
    # the total energy of a state (q, p) is proportional to the squared distance
    # from the origin in phase space.
    # So when we normalize y0 (which is [q, p]) and rescale it to a fixed radius:
    # we are projecting the point onto a circle (or energy shell) of fixed total energy.
    # This ensures that all sampled initial conditions lie on a constant-energy
    # shell, which is useful when generating trajectories with controlled and
    # uniformly distributed energy levels for training or testing HamiltonianNNs

    spring_ivp = solve_ivp(fun=dynamics, t_span=t_span, y0=y0, t_eval=t_eval, rtol=1e-10, **kwargs)
    q , p = spring_ivp['y'][0], spring_ivp['y'][1]
    dydt = [dynamics(None, y) for y in spring_ivp['y'].T]
    dydt = np.stack(dydt).T 
    dqdt, dpdt = np.split(dydt,2) 

    #adding noise to the values of q and p 
    q += np.random.randn(*q.shape)*noise_std
    p += np.random.randn(*p.shape)*noise_std

    return q, p, dqdt, dpdt, t_eval 


def get_dataset(seed, samples=50, test_split=0.5, **kwargs):
    data = {'meta':locals()} 

    #randomly sampling inputs 
    np.random.seed(seed)
    xs, dxs = [], []
    for s in range(samples):
        x, y, dx, dy, t = trajectory(**kwargs)
        xs.append(np.stack([x,y]).T) #transpose is important to get the right dimensions 
        dxs.append(np.stack([dx, dy]).T)
    
    data['x'] = np.concatenate(xs)
    data['dx'] = np.concatenate(dxs).squeeze() #acc to the repo code 

    #making the train test split 
    split_ix = int(len(data['x'])*test_split)
    split_data = {}
    for k in ['x', 'dx']:
        split_data['train_'+k], split_data['test_'+k] = data[k][:split_ix], data[k][split_ix:]
    data = split_data
    return data 

def get_field(xmin=-1.2, xmax=1.2, ymin=-1.2, ymax=1.2, gridsize=20):
    field = {'meta':locals()}

    #meshgrid to get vector field 
    b, a = np.meshgrid(np.linspace(xmin, xmax, gridsize), np.linspace(ymin, ymax, gridsize))
    ys = np.stack([b.flatten(), a.flatten()])

    #get vector directions 
    dydt = [dynamics(None, y) for y in ys.T]
    dydt = np.stack(dydt).T

    field['x'] = ys.T
    field['dx'] = dydt.T

    return field 