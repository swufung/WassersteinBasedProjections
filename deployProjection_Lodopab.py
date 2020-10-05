import numpy as np
import torch

import os
import pickle
import utils

from torch.utils.data import Dataset, TensorDataset, DataLoader
import matplotlib.pyplot as plt
import time

# ======================================================================================================================
# Define proj_M_L2O Operator
# ======================================================================================================================
def proj_M_L2O(u_gen0, list_J, his_wass, gamma_constant, alpha, step_type, device):
    # u_gen          = initial image(s)
    # list_J         = list of pre-trained weights
    # his_wass       = list of Wasserstein distances/Average distances
    # gamma_constant = scalar multiplying gamma_k sequence
    # alpha          = constant for step size in (0,1)

    # require grad for computing nablaJ
    u_gen = u_gen0.detach().clone()
    u_gen.requires_grad_(True)
    n_samples = u_gen.shape[0]

    start_time = time.time()

    # perform relaxed projections
    for gen_iter in range(1, num_gen):

        Jout = list_J[gen_iter](u_gen)  # n_samples x 1

        # take derivative w.r.t only inputs
        nablaJ = torch.autograd.grad(outputs=Jout, inputs=u_gen,
                                     grad_outputs=torch.ones(Jout.size()).to(device), only_inputs=True)[0].detach()

        gamma_gen_iter = gamma_constant / gen_iter
        anchor = gamma_gen_iter * u_gen0

        # step size, combines general and individual step size
        if step_type == 'Mean':
            lam = alpha * (his_wass[gen_iter - 1])
        elif step_type == 'Combined':
            lam = alpha * (his_wass[gen_iter - 1] + torch.max(Jout.detach(), torch.zeros(Jout.size(), device=device)))
        elif step_type == 'Individual':
            lam = alpha * (torch.max(Jout.detach(), torch.zeros(Jout.size(), device=device)))

        # lambda * gradient term; reshape to multiply by lam, then reshape back
        lam_gradient_term = (lam * (nablaJ.reshape(n_samples, -1))).reshape(u_gen.shape)
        u_gen = anchor + (1 - gamma_gen_iter) * (u_gen - lam_gradient_term)

        end_time = time.time()

    # print(f'proj_M_L2O time: {end_time - start_time}')

    u_gen = u_gen.detach()

    return u_gen







# ======================================================================================================================
# Load Validation Set
# ======================================================================================================================
CTLodopab_folder = './CTLodopab/experimentsLodopab/'

dataset_path = CTLodopab_folder + \
               'lodopab___validation_size-2000___tv_lam-0.0005__july-25-2020.pkl'

experiment_path = './CTLodopab/experimentsLodopab/' \
                  'lodopabCT_CNN_2020-09-30-22-06-29_lr1e-04_genFreq200_nTrain20000_batchSize32_optimADAM_etaFreq20_gamma_const1e-01_etaTol1e-05_noise_level1e-02_alpha5e-01_gp_lam50_distMan_lam1e-01_lr_decay_iter50doAugment0_architectureHuber_stepTypeMean_tvlam001'

# proj_M_L2O constants (based on experiment_path)
gamma_constant = 1e-1
alpha          = 5e-1
step_type      = 'Mean'
image_scaling = 'unit'
data_range = {"unit": 1.0, "symmetric": 2.0}
device         = 'cpu'

from adversarial_networks import discriminator_net_Huber as discriminator_net; architecture = 'Huber' # USING FC at the end

with open(dataset_path, 'rb') as f:
    load_dataset = pickle.load(f)

b_data = load_dataset['y']
u_true = load_dataset['x_true']
u_gen0 = load_dataset['tv']
# gen_zero_dataset_fbp = load_dataset['fbp']

# swap dimensions so that u_train is n_samples x n_channels x n_feature1 x n_feature2
b_data = np.transpose(b_data, [0, 3, 1, 2])
u_true = np.transpose(u_true, [0, 3, 1, 2])
u_gen0 = np.transpose(u_gen0, [0, 3, 1, 2])

print("u_true.shape = ", u_true.shape)
print("u_gen0.shape = ", u_gen0.shape)
print("b_data.shape = ", b_data.shape)

# Renormalize everything to the unit interval [0,1]
for idx in range(u_gen0.shape[0]):
    u_gen0[idx] = utils.scale_to_interval(u_gen0[idx], image_scaling)

# ======================================================================================================================
# Choose image(s) to be projected
# ======================================================================================================================

ind_val1 = 999
# ind_val1 = 1000
# ind_val1 = 1001
n_images = ind_val1+1

# ind_val1  = 0
# n_images = 2000
b_data = torch.FloatTensor(b_data[ind_val1:n_images,:,:,:])       ##################################################################
u_true = torch.FloatTensor(u_true[ind_val1:n_images,:,:,:])       ##################################################################
u_gen0 = torch.FloatTensor(u_gen0[ind_val1:n_images,:,:,:])       ##################################################################

# gen_zero_dataset_fbp = gen_zero_dataset_fbp[0:n_images,:,:,:].reshape(n_images,128,128,1) ##################################################################

print("u_true.shape = ", u_true.shape)
print("gen_zero_dataset.shape = ", u_gen0.shape)
print("b_data.shape = ", b_data.shape)


print()
print(f'u_true.min(): {u_true.min()}')
print(f'u_true.max(): {u_true.max()}')
print(f'u_gen0.min(): {u_gen0.min()}')
print(f'u_gen0.max(): {u_gen0.max()}')
print()

# ======================================================================================================================
# Load Pretrained Weights
# ======================================================================================================================

# models_path = os.path.join(experiment_path, './checkpoints/')
models_path = experiment_path + '/checkpoints/'

list_files = [name for name in os.listdir(models_path)]
list_files.sort()
# print('list_files:', list_files)
# num_gen = len(list_files)  # so the number of files in checkpoints should be the number of generators
num_gen = 20 # pick a spot to stop
# num_gen = 15 # 23 for FCCNN architecture
list_J = []

# Loading the J's
for idx, filename in enumerate(list_files):
    assert(filename[0:5] == 'step_')
    # print(f'filename: {filename}')
    if idx < 10:
        load_J = torch.load(models_path + 'step_0' + str(idx) + '.pth', map_location='cpu')
    else:
        load_J = torch.load(models_path + 'step_' + str(idx) + '.pth', map_location='cpu')

    netD = discriminator_net()
    netD.load_state_dict(load_J['state_dict'])
    netD.eval()
    list_J.append(netD)

# Load the last J for his_wass
if num_gen < 10:
    his_wass = torch.load(models_path + 'step_0' + str(num_gen) + '.pth', map_location='cpu')['his_wass']
else:
    his_wass = torch.load(models_path + 'step_' + str(num_gen) + '.pth', map_location='cpu')['his_wass']






# ======================================================================================================================
# Load ODL Operators
# ======================================================================================================================
import odl
# extremely important that the correct ODL GitHub functions are available
from odl.contrib import torch as odl_torch
n_mesh = 128
space = odl.uniform_discr([-64, -64], [64, 64], [n_mesh, n_mesh],
                          dtype='float32')

n_angles   = 30
geometry = odl.tomo.parallel_beam_geometry(space, num_angles=n_angles)
fwd_op = odl.tomo.RayTransform(space, geometry)

opnorm = odl.power_method_opnorm(fwd_op)
fwd_op = (1 / opnorm) * fwd_op  # normalize fwd operator to be scale-invariant

L = odl.power_method_opnorm(fwd_op.adjoint*fwd_op)

print('fwd_op.adjoint*fwd_op = ', fwd_op.adjoint*fwd_op)
print('L = ', L)


# ======================================================================================================================
# Relaxed Projected Gradient Descent
# min𝑢∈ℝ2𝐻(𝑢)+𝛿(𝑢)=min𝑢∈𝐻(𝑢),
#
# where
#
# 𝐻(𝑢):= 1/2 ‖𝐴𝑢−𝑏‖_2^2.
#
# The iterative method we use to solve this problem is a relaxed version of projected gradient. For  𝜆∈(0,1)  and  𝛼∈(0,2/Lip(𝐻)) , we use
#
# u_{k+1} = (1 - kappa) + kappa ⋅ 𝑃(𝑢𝑘− xi ∇𝐻(𝑢𝑘))
# ======================================================================================================================
# computing only one projection:
SSIM0, PSNR0 = utils.compute_avg_SSIM_PSNR(u_true, u_gen0, n_mesh,
                                                     data_range[image_scaling])

u1 = proj_M_L2O(u_gen0, list_J, his_wass, gamma_constant, alpha, step_type, device)
relerr1 = torch.norm(u1 - u_true)/torch.norm(u_true)
SSIM1, PSNR1 = utils.compute_avg_SSIM_PSNR(u_true, u1, n_mesh,
                                                     data_range[image_scaling])

print(
        'relerr1: %.3e   PSNR1: %.3e   SSIM1: %.3e'
        % (relerr1, PSNR1, SSIM1))

u2 = proj_M_L2O(u1, list_J, his_wass, gamma_constant, alpha, step_type, device)
relerr2 = torch.norm(u2 - u_true)/torch.norm(u_true)
SSIM2, PSNR2 = utils.compute_avg_SSIM_PSNR(u_true, u2, n_mesh,
                                                     data_range[image_scaling])
print(
        'relerr2: %.3e   PSNR2: %.3e   SSIM2: %.3e'
        % (relerr2, PSNR2, SSIM2))


uk              = u_gen0.clone()
yk              = torch.zeros(uk.shape)
kappa           = 1e-1
n_samples       = uk.shape[0]
max_iters       = 10
xi              = 5e-1 / L

print('\n\n' + experiment_path)
print('n_images = ', n_images)

print('\n\n ----------- Relaxed Projected Gradient Descent ----------- \n')
print('kappa: %.2e   max_iters: %.d   xi: %.2e   alpha: %.2e   num_gen: %d'
        % (kappa, max_iters, xi, alpha, num_gen))

for k in range(max_iters):

    start_time = time.time()
    Auk             = odl_torch.OperatorFunction.apply(fwd_op, uk)
    grad_H_uk       = odl_torch.OperatorFunction.apply(fwd_op.adjoint, Auk - b_data)

    yk = uk - xi * grad_H_uk

#     # 𝑃𝜆(𝑢):=(1−𝜆)𝑢 + lam * P(u - lam ( grad) .
    uk = (1-kappa) * uk + kappa * proj_M_L2O(yk, list_J, his_wass, gamma_constant, alpha, step_type, device)
    # uk = yk

    relerr = torch.norm(uk - u_true)/torch.norm(u_true)
    SSIM, PSNR = utils.compute_avg_SSIM_PSNR(u_true, uk, n_mesh,
                                                     data_range[image_scaling])

    plt.figure()
    suptitle_str = 'k = ' + str(k+1)
    plt.suptitle(suptitle_str)
    plt.subplot(2,2,1)
    plt.imshow(u_gen0[0,0,:,:], clim=(0,1))
    plt.title('u0' + ', siim:{:.2f}'.format(SSIM0) +  ', psnr:{:.2f}'.format(PSNR0))
    plt.colorbar()
    plt.subplot(2, 2, 2)
    plt.imshow(u_true[0, 0, :, :], clim=(0,1))
    plt.title('u_true')
    plt.colorbar()
    plt.subplot(2, 2, 3)
    plt.imshow(u1[0, 0, :, :], clim=(0,1))
    plt.title('PM(u0)' + ', siim:{:.2f}'.format(SSIM1) +  ', psnr:{:.2f}'.format(PSNR1))
    plt.colorbar()
    plt.subplot(2, 2, 4)
    plt.imshow(uk[0,0,:,:], clim=(0,1))
    uk_title_str = 'u' + str(k+1) + ', siim:{:.2f}'.format(SSIM) +  ', psnr:{:.2f}'.format(PSNR)
    plt.title(uk_title_str)
    plt.colorbar()
    plt.show()

    end_time = time.time()
    t_iter = end_time - start_time

    # compute Relative Error, SSIM and PSNR
    print(
        '%d   relerr: %.3e   PSNR: %.3e   SSIM: %.3e   time: %.3f'
        % (k+1, relerr, PSNR, SSIM, t_iter))



# create and save gray image
cmap = 'gray'
fig = plt.figure()
plt.imshow(np.rot90(uk[0,0,:,:].detach().cpu().numpy()),cmap=cmap)
plt.axis('off')
save_loc = './saved_images/adv_proj_lodopab_ind' + str(ind_val1) + '.pdf'
plt.savefig(save_loc,bbox_inches='tight')
plt.show()
# plt.close(fig)





