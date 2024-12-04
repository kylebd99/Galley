import time
import numpy as np
import torch
torch.set_num_threads(1)

def mat_chain(A, B, C):
    return A @ B @ C

def mat_chain_back(A, B, C):
    return C @ B @ A

N = 2000
densities = [.1, .01, .001, .0001, .00001, .000001]
forward_times = []
backward_times =[]
sum_times = []
for d in densities:
    A = torch.from_numpy(np.astype(np.random.rand(N,N) < .5, float)).to_sparse_csc()
    B = torch.from_numpy(np.astype(np.random.rand(N,N) < .5, float)).to_sparse_csc()
    C = torch.from_numpy(np.astype(np.random.rand(N,N) < d, float)).to_sparse_csc()

    scripted_mat_chain = torch.jit.script(mat_chain, example_inputs=[(A, B, C)])
    scripted_mat_chain_back = torch.jit.script(mat_chain_back, example_inputs=[(A, B, C)])

    n_reps = 7
    avg_time_forward = 0
    avg_time_backward = 0
    avg_time_sum = 0
    for i in range(1, n_reps+1):
        if i < 3:
            avg_time_forward = 0
            avg_time_backward = 0
            avg_time_sum = 0
        start = time.time()
        E1 = scripted_mat_chain(A, B, C)
        end = time.time()
        avg_time_forward = avg_time_forward + end-start
        
        start = time.time()
        E2 = scripted_mat_chain_back(A, B, C)
        end = time.time()
        avg_time_backward = avg_time_backward + end-start

        start = time.time()
        E2 = torch.sparse.sum(scripted_mat_chain(A, B, C).to_sparse_coo())
        end = time.time()
        avg_time_sum = avg_time_sum + end-start

    avg_time_forward = avg_time_forward / (n_reps - 2) 
    avg_time_backward = avg_time_backward / (n_reps - 2) 
    avg_time_sum = avg_time_sum / (n_reps - 2) 
    forward_times.append(avg_time_forward)
    backward_times.append(avg_time_backward)
    sum_times.append(avg_time_sum)
    print("Density: ", d)
    print("Forward Time: ", avg_time_forward)
    print("Backward Time: ", avg_time_backward)
    print("Sum Time: ", avg_time_sum)
print(forward_times)
print(backward_times)
print(sum_times)