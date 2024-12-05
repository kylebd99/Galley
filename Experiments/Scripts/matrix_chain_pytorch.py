import time
import numpy as np
import csv
import torch
num_cores = 14
torch.set_num_threads(num_cores)

def mat_chain(A, B, C):
    return A @ B @ C

def mat_chain_back(A, B, C):
    return C @ B @ A

def mat_elementwise(A, B, C):
    return A * B * C

N = 2000
A_density = .5
B_density = .5
densities = [.1, .01, .001, .0001, .00001, .000001]
forward_times = []
backward_times =[]
sum_times = []
elementwise_times = []
for d in densities:
    A = torch.from_numpy(np.astype(np.random.rand(N,N) < A_density, float)).to_sparse_csc()
    B = torch.from_numpy(np.astype(np.random.rand(N,N) < B_density, float)).to_sparse_csc()
    A_dense = torch.from_numpy(np.astype(np.random.rand(N,N), float))
    B_dense = torch.from_numpy(np.astype(np.random.rand(N,N), float))
    C = torch.from_numpy(np.astype(np.random.rand(N,N) < d, float)).to_sparse_csc()

    scripted_mat_chain = torch.jit.script(mat_chain, example_inputs=[(A, B, C)])
    scripted_mat_chain_back = torch.jit.script(mat_chain_back, example_inputs=[(A, B, C)])
    scripted_mat_elementwise = torch.jit.script(mat_chain_back, example_inputs=[(A_dense, B_dense, C)])

    n_reps = 7
    avg_time_forward = 0
    avg_time_backward = 0
    avg_time_sum = 0
    avg_time_elementwise = 0
    for i in range(n_reps):
        if i == 2:
            avg_time_forward = 0
            avg_time_backward = 0
            avg_time_sum = 0
            avg_time_elementwise = 0
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

        start = time.time()
        E3 = scripted_mat_elementwise(A_dense, B_dense, C)
        end = time.time()
        avg_time_elementwise = avg_time_elementwise + end-start

    avg_time_forward = avg_time_forward / (n_reps - 2) 
    avg_time_backward = avg_time_backward / (n_reps - 2) 
    avg_time_sum = avg_time_sum / (n_reps - 2) 
    avg_time_elementwise = avg_time_elementwise / (n_reps - 2) 
    forward_times.append(avg_time_forward)
    backward_times.append(avg_time_backward)
    sum_times.append(avg_time_sum)
    elementwise_times.append(avg_time_elementwise)
    print("Density: ", d)
    print("Forward Time: ", avg_time_forward)
    print("Backward Time: ", avg_time_backward)
    print("Sum Time: ", avg_time_sum)
    print("Elementwise Time: ", avg_time_elementwise)

method = "PyTorch (Serial)"
if num_cores > 1:
    method = "PyTorch (Parallel)"

data = [("Method", "Algorithm", "Sparsity", "Runtime")]
for i in range(len(densities)):
    data.append((method, "ABC", densities[i], forward_times[i]))
    data.append((method, "CBA", densities[i], backward_times[i]))
    data.append((method, "SUM(ABC)", densities[i], sum_times[i]))
    data.append((method, "A*B*C", densities[i], elementwise_times[i]))

filename = "Experiments/Results/mat_exps_pytorch_serial.csv"
if num_cores > 1:
    filename = "Experiments/Results/mat_exps_pytorch_parallel.csv"

# writing to csv file
with open(filename, 'w') as csvfile:
    # creating a csv writer object
    csvwriter = csv.writer(csvfile)
    # writing the data rows
    csvwriter.writerows(data)

print(forward_times)
print(backward_times)
print(sum_times)
print(elementwise_times)