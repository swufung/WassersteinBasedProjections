import numpy as np
import torch

from torch.utils.data import Dataset, TensorDataset, DataLoader
import matplotlib.pyplot as plt
import torch.optim         as optim
import time
import os
import pickle
import utils

from adversarial_networks import discriminator_net_Huber as discriminator_net; architecture = 'Huber' 


n_train         = 10000  # number of training samples (10000 max)
n_val           = 1000  # number of validation (1000 max)
batch_size      = 16 # 100 for dcgan, 75 for custom
batch_size_val  = 100 # 100 for dcgan, 100 for custom
max_epochs   = 100000  # max epochs
total_steps  = 51 # max number of generator updates

gamma_constant = 1e-1 # scalar that multiplies sequence gamma_k

# switch once, to 1e-5 after 10 gen updates
lr_decay_iter = 50
lr_decay_rate = 1e-1

learning_rate = 1e-5  # initial learning rate
gen_freq = 200  # cuda0: 500 # number of epochs before updating generator (gen_freq = generator frequency)

device = 'cuda:0'  # choose device
optim_string = "ADAM"  # optimizers: "ADAM" or "SGD"

step_type = 'Mean' # 'Combined', 'Individual', or 'Mean'
noise_level = 5e-3 # noise level to add to current approx. distribution


# distance to manifold terms
dist_man_lam = 1e-1 # penalty parameter on distance to manifold
do_manifold_distance = True

# augment data with rotatons about x and y axis
do_augment = False
# number of training samples with augmentation included
n_train_augmented = n_train

mu = 0.5 # relaxation parameter
gp_lam = 20  # gradient penalty strength

eta_tol = 1e-5 # update generator if |1 - eta| < eta_tol or |eta - etaOld| < eta_tol
eta_freq = 20 # compute eta every eta_freq epochs
print_freq = 1 # print every epoch


n_mesh = 128  # mesh size of image
n_features = n_mesh ** 2  # total number of features
n_angles = 30  # number of angles
do_grad_pen = True
weight_decay = 1e-4  # weight decay for ADAM or SGD
image_scaling = 'unit'  # how to scale the image. Possible options: 'unit', 'symmetric', or 'none'
data_range = {"unit": 1.0, "symmetric": 2.0}
gen0_type = 'tv'  # options are 'tv' or 'fbp'

# ------------------------------------------------------------------------------------------------------------------
# set up device
# ------------------------------------------------------------------------------------------------------------------
device = torch.device(device if torch.cuda.is_available() else "cpu")
print(device)

# ------------------------------------------------------------------------------------------------------------------
# load the data
# ------------------------------------------------------------------------------------------------------------------
data_path = './CTEllipse/experimentsEllipse/ellipses__train_size-10000_tv_lam-0.001__beams-30__noise-0.025__july-16-2020.pkl'

with open(data_path, 'rb') as f:
    data_state = pickle.load(f)
    data_obs_full_train = data_state['y']
    u_true_full_train = data_state['x_true']
    u_gen_full_train = data_state[gen0_type]

val_data_path = './CTEllipse/experimentsEllipse/ellipses__val_size-1000_tv_lam-0.001__beams-30__noise-0.025__july-16-2020.pkl'

with open(val_data_path, 'rb') as f:
    data_state = pickle.load(f)
    data_obs_full_val = data_state['y']
    u_true_full_val = data_state['x_true']
    u_gen_full_val = data_state[gen0_type]

# swap dimensions so that u_train is n_samples x n_channels x n_feature1 x n_feature2
u_true_full_train = np.transpose(u_true_full_train, [0, 3, 1, 2])
u_gen_full_train = np.transpose(u_gen_full_train, [0, 3, 1, 2])
u_true_full_val = np.transpose(u_true_full_val, [0, 3, 1, 2])
u_gen_full_val = np.transpose(u_gen_full_val, [0, 3, 1, 2])

# # Renormalize everything to the unit interval [0,1]
# for idx in range(u_gen_full_train.shape[0]):
#     u_gen_full_train[idx] = utils.scale_to_interval(u_gen_full_train[idx], image_scaling)
#
# for idx in range(u_gen_full_val.shape[0]):
#     u_gen_full_val[idx] = utils.scale_to_interval(u_gen_full_val[idx], image_scaling)

# ------------------------------------------------------------------------------------------------------------------
# Create Dataset
# ------------------------------------------------------------------------------------------------------------------
# transform to torch tensors
u_true_full_train = torch.FloatTensor(u_true_full_train)[0:n_train,:,:,:]
u_gen_full_train = torch.FloatTensor(u_gen_full_train)[0:n_train,:,:,:]
u_true_full_val = torch.FloatTensor(u_true_full_val)[0:n_val,:,:,:]
u_gen_full_val = torch.FloatTensor(u_gen_full_val)[0:n_val,:,:,:]

# clamp to [0,1]
u_gen_full_train    = torch.clamp(u_gen_full_train,0,1)
u_gen_full_val      = torch.clamp(u_gen_full_val,0,1)


ind_val = 0  # index for validation plot image

# initial approx. distirbution for anchoring (Halpern)
u0_gen_full_train = u_gen_full_train.clone()
u0_gen_full_val = u_gen_full_val.clone()

# Create training datasets
dataset = TensorDataset(u_true_full_train, u_gen_full_train, u0_gen_full_train)
dataset_val = TensorDataset(u_true_full_val, u_gen_full_val, u0_gen_full_val)
data_loader = DataLoader(dataset=dataset, batch_size=batch_size, shuffle=False)
data_loader_val = DataLoader(dataset=dataset_val, batch_size=batch_size_val, shuffle=False)

# Info about the dataset
print()
print(f'u_true_full_train.min(): {u_true_full_train.min()}')
print(f'u_true_full_train.max(): {u_true_full_train.max()}')
print()

print(f'u_gen_full_train.min(): {u_gen_full_train.min()}')
print(f'u_gen_full_train.max(): {u_gen_full_train.max()}')
print()

print(f'u_true_full_val.min(): {u_true_full_val.min()}')
print(f'u_true_full_val.max(): {u_true_full_val.max()}')
print()

print(f'u_gen_full_val.min(): {u_gen_full_val.min()}')
print(f'u_gen_full_val.max(): {u_gen_full_val.max()}')
print()

print("u_true_full_train.shape = ", u_true_full_train.shape)
print("u_true_full_val.shape = ", u_true_full_val.shape)
print("u_gen_full_train.shape = ", u_gen_full_train.shape)
print("u_gen_full_val.shape = ", u_gen_full_val.shape)
print('u0_gen_full_train.shape = ', u0_gen_full_train.shape)
print('u0_gen_full_val.shape = ', u0_gen_full_val.shape)

print("data_obs_full_train.shape = ", data_obs_full_train.shape)
print("data_obs_full_val.shape = ", data_obs_full_val.shape)
print('\n\n')

# ----------------------------------------------------------------------------------------------------------------------
# Create neural network model
# ----------------------------------------------------------------------------------------------------------------------
# Discriminator
netD = discriminator_net()
netD.to(device)
pytorch_total_params = sum(p.numel() for p in netD.parameters() if p.requires_grad)
print(f'Number of trainable parameters: {pytorch_total_params}')

# ------------------------------------------------------------------------------------------------------------------
# Set up optimizer
# ------------------------------------------------------------------------------------------------------------------
if optim_string == "ADAM":
    optimizer = optim.Adam(netD.parameters(), lr=learning_rate, weight_decay=weight_decay)
else:
    optimizer = optim.SGD(netD.parameters(), lr=learning_rate, weight_decay=weight_decay)


gen_iter_array = []  # create array for recording epochs of gen_freq updates

# ------------------------------------------------------------------------------------------------------------------
# set up save path
# ------------------------------------------------------------------------------------------------------------------

# Save initial flow image
if not os.path.exists(os.path.dirname('CTEllipse/experimentsEllipse/')):
    print("created experimentsEllipse/")
    print("os.getcwd()", os.getcwd())
    os.makedirs(os.path.dirname('CTEllipse/experimentsEllipse/'))
else:
    print("experimentsEllipse already exists")

save_path = './CTEllipse/experimentsEllipse/' + 'ellipseCT_'  + utils.get_time_string()  + \
             '_lr{:.0e}'.format(learning_rate) +\
            '_genFreq{:d}'.format(gen_freq) + '_nTrain{:d}'.format(n_train) + '_batchSize{:d}'.format(batch_size) + \
            '_optim' + optim_string + '_etaFreq{:d}'.format(eta_freq) + '_gamma_const{:.0e}'.format(gamma_constant) +\
            '_etaTol{:.0e}'.format(eta_tol) + '_noise_level{:.0e}'.format(noise_level) +\
            '_mu{:.0e}'.format(mu) + '_gp_lam{:d}'.format(gp_lam) +\
            '_distMan_lam{:.0e}'.format(dist_man_lam) + '_lr_decay_iter{:d}'.format(lr_decay_iter) + \
            'doAugment{:d}'.format(do_augment) + '_architecture' + architecture +\
            '_stepType' + step_type + '_tvlam001/'

# max and min values for plotting
vmin = min(u0_gen_full_train[0, :].view(-1))
vmax = max(u0_gen_full_train[0, :].view(-1))

vmin_val = min(u_true_full_val[ind_val, :].view(-1))
vmax_val = max(u_true_full_val[ind_val, :].view(-1))

# vmin_val = min(u_true_full_val.reshape(n_val, -1)[ind_val, :])
# vmax_val = max(u_true_full_val.reshape(n_val, -1)[ind_val, :])

print('os.path.exists(os.path.dirname(save_path)) = ', os.path.exists(os.path.dirname(save_path)))
print('os.path.dirname(save_path) = ', os.path.dirname(save_path))

# ------------------------------------------------------------------------------------------------------------------
# save initial flow plot/checkpoint
# ------------------------------------------------------------------------------------------------------------------

if not os.path.exists(os.path.dirname(save_path)):
    print("save path = ", save_path)
    os.makedirs(os.path.dirname(save_path))
    plt.savefig(save_path + '/flow_00.png')
else:
    print("CTEllipse/experimentsEllipse/" + " already exists")
    plt.savefig(save_path + '/flow_00.png')

# Save Initial Discriminator Checkpoint path
checkpt_path = save_path + 'checkpoints/'
if not os.path.exists(os.path.dirname(checkpt_path)):
    print("created", checkpt_path)
    os.makedirs(os.path.dirname(checkpt_path))
else:
    print(checkpt_path + "already exists")
state = {
    'learning_rate': learning_rate,
    'state_dict': netD.state_dict(),
}
save_checkpt_str = checkpt_path + 'step_00.pth'
torch.save(state, save_checkpt_str)

utils.plot_and_save_images(u_true_full_train[0, :].cpu(),
                                       u0_gen_full_train[0, :].cpu(),
                                       u_gen_full_train[0, :].cpu(),
                                       u_true_full_val[ind_val, :].cpu(),
                                       u0_gen_full_val[ind_val, :].cpu(),
                                       u_gen_full_val[ind_val, :].cpu(),
                                       save_path, 0, n_mesh, vmin, vmax, vmin_val, vmax_val)

# initialize histories
his_MSE = []
his_MSE_val = []
his_wass = []
his_dist_man = []
his_wass_val = []
his_dist_man_val = []
his_eta = []
his_J_gen = []
his_J_real = []
his_ssim = []
his_psnr = []

# ----------------------------------------------------------------------------------------------------------------------
# Train
# ----------------------------------------------------------------------------------------------------------------------
print("max_epochs = ", max_epochs, ", batch_size = ", batch_size, "batch_size_val = ", batch_size_val,
      "device = ", device, ", gen_freq = ", gen_freq,
      ", n_train = ", n_train, ", n_val = ", n_val,
      ', eta_tol = ', eta_tol, ', gp lam = ', gp_lam, ', dist_man_lam = ', dist_man_lam,
      ', gen0_type = ', gen0_type, ', mu', mu)

gen_iter = 1
epoch_update = 0
running_MSELoss_val = (u_true_full_val - u_gen_full_val).pow(2).sum().detach() / n_val
running_MSELoss = (u_true_full_train - u_gen_full_train).pow(2).sum().detach() / n_train


ssim_val, psnr_val = utils.compute_avg_SSIM_PSNR(u_true_full_val, u_gen_full_val, n_mesh,
                                                     data_range[image_scaling])

with open(save_path + 'val_logging.csv', 'w') as f:
    f.write('mse_val,psnr_val,ssim_val\n')
    f.write(f'{running_MSELoss_val},{psnr_val},{ssim_val}\n')

wass_dist_val = 0.0
dist_man_val = 0.0
dist_man_batch_sum_val = 0.0
eta_old = torch.FloatTensor([0.0]);
eta = torch.FloatTensor([eta_tol])  # initialize etas s.t. their diff >= eta_tol


for epoch in range(1, max_epochs + 1):  # loop over the dataset multiple times

    if gen_iter < total_steps:
        start_time = time.time()
        # running_MSELoss = 0.0
        J_real = 0.0
        J_gen = 0.0
        wass_dist = 0.0
        dist_man = 0.0 # distance to manifold
        dist_man_batch_sum = 0.0  # used to accumulate |dist to true manifold| per batch
        grad_pen_list = []
        max_pixel = -np.inf
        min_pixel = np.inf
        for i, data in enumerate(data_loader, 0):
            # get the inputs; data is a list of [inputs, labels]
            u_real, u_gen = data[0].to(device), data[1].to(device)

            # augment data
            # flip(2) = flip about x-axis, flip(3) = flip about y axis
            if do_augment:
                u_gen = torch.cat((u_gen, u_gen.flip(2), u_gen.flip(3)),0)
                u_real = torch.cat((u_real, u_real.flip(2), u_real.flip(3)), 0)
                n_train_augmented = 3*n_train # augmented n_train
            # u_gen = torch.cat((u_gen, u_gen.flip(2), u_gen.flip(3), u_gen.transpose(2, 3).flip(2),
            #                    u_gen.flip([3, 2]), u_gen.transpose(2, 3).flip(3)), 0)
            # print("u_gen[0,0] = %.8e" % u_gen[0, 0])
            # print('u_gen_augmented.shape = ', u_gen.shape)

            # zero the parameter gradients
            optimizer.zero_grad()

            # note max errReal - errFake = - (min errFake - errReal)
            # real data

            disc_real = netD(u_real)
            err_real = torch.mean(disc_real)

            # add noise and clamp u_gen
            u_gen = torch.clamp(u_gen + noise_level * (torch.randn(u_gen.shape, device=device)), min=0, max=1)

            # fake data
            disc_fake = netD(u_gen) # add noise_level% noise
            err_fake = torch.mean(disc_fake)

            # note that we use definition of wass_dist in adversarial regularizers
            err = err_real - err_fake

            # gradient penalty
            if do_grad_pen:
                # assert (batch_size == u_real.size()[0]), f"def_batch: {batch_size} | u_real.size()[0]: {u_real.size()[0]}"
                # min_batch_size = np.min([u_real.shape[0], u_gen.shape[0]]) # pick smaller batch size
                # grad_pen = utils.grad_penalty(u_real[0:min_batch_size,:,:,:], u_gen[0:min_batch_size,:,:,:], netD, device)

                # do gradient penalty without augmentation
                grad_pen = utils.grad_penalty(u_real[0:batch_size, :, :, :], u_gen[0:batch_size, :, :, :], netD, device)
                # grad_pen = utils.grad_penalty(u_real, u_gen, netD, device)
                grad_pen_list.append(grad_pen.item())
                grad_pen_lam = grad_pen * gp_lam

                err = err + grad_pen_lam

            if do_manifold_distance:
                err = err + dist_man_lam * torch.mean((disc_real ** 2))
                # dist_man += torch.mean(torch.abs(disc_real)).detach().cpu() ###### CHECK THIS
                dist_man_batch_sum += (torch.abs(disc_real)).detach().cpu().sum()
                # dist_man = 0.01 * torch.mean(torch.abs(disc_real)).detach().numpy() + 0.99 * dist_man

            err.backward()

            # accumulate losses (for printing)
            wass_dist += disc_fake.sum().detach().cpu() - disc_real.sum().detach().cpu()
            J_real += disc_real.sum().detach().cpu()
            J_gen += disc_fake.sum().detach().cpu()

            optimizer.step()

            del disc_real, disc_fake, u_real, u_gen, grad_pen, grad_pen_lam

        # compute wasserstein dist
        wass_dist /= n_train_augmented
        # dist_man /= n_train
        dist_man = 0.999 * dist_man + 0.001 * (dist_man_batch_sum / n_train_augmented)
        J_real /= n_train_augmented  # mean of J(u_real)
        J_gen /= n_train_augmented  # mean of J(u_fake)

        # --------------------------------------------------------------------------------------------------------------
        # compute eta every eta_freq epochs
        # --------------------------------------------------------------------------------------------------------------
        if epoch % eta_freq == 0:
            eta_old = eta
            eta = 0.0

            for i, data in enumerate(data_loader, 0):
                u_gen = data[1].to(device)

                u_gen.requires_grad_(True)
                Jout = netD(u_gen)  # n_samples x 1

                # take deriv w.r.t only inputs
                nablaJ = torch.autograd.grad(outputs=Jout, inputs=u_gen,
                                             grad_outputs=torch.ones(Jout.size()).to(device), only_inputs=True)[
                    0].detach()

                eta += (torch.norm(nablaJ.reshape(nablaJ.size(0), -1), p=2, dim=1) ** 2).sum()

                del nablaJ, Jout, u_gen

            eta = (eta / n_train).detach().cpu()

        # --------------------------------------------------------------------------------------------------------------
        # update u_gen
        # --------------------------------------------------------------------------------------------------------------
        # conditions for updating generator (first gen_iter use twice as many epochs)
        # condition 1: first gen_iter and we reached 2*gen_freq epochs
        # condition 2: performed gen_freq epochs since last update
        # condition 3: two consecutive etas are eta_tol away from each other
        # condition 4: eta is eta_tol away from close to 1
        if (gen_iter == 1 and (epoch - epoch_update) % (2 * gen_freq) == 0) \
                or ( (epoch - epoch_update) % gen_freq == 0 and gen_iter > 1 ) \
                or abs(eta - eta_old) < eta_tol \
                or abs(1 - eta) < eta_tol:

            if (gen_iter == 1 and (epoch - epoch_update) % (2 * gen_freq) == 0): print('first gen update')
            if ((epoch - epoch_update) % gen_freq == 0 and gen_iter > 1): print('reason for updating u_gen: ',
                                                                                gen_freq, 'epochs passed')
            if abs(eta - eta_old) < eta_tol: print('reason for updating u_gen: ', 'abs(eta-eta_old) = ',
                                                   abs(eta - eta_old), " < eta_tol = ", eta_tol)
            if abs(1 - eta) < eta_tol: print('reason for updating u_gen: ', 'abs(1 - eta) = ', abs(1 - eta),
                                                 ' < eta_tol = ', eta_tol)

            epoch_update = epoch  # saves epoch where update happened

            # update wasserstein and mse loss histories
            his_MSE.append(running_MSELoss.item())
            his_eta.append(eta.item());
            his_wass.append(wass_dist.item())  # convert tensor to float with .item()
            his_dist_man.append(dist_man.item()) # history of distance to manifold
            running_MSELoss_val = 0.0
            J_gen_val = 0.0  # for computing wass distance on validation set

            # add current epoch to gen_iter_array
            gen_iter_array.append(epoch - 1)

            # update generator
            print(" ----- updating generator -----")
            # ------------------------------------------------
            # update u_gen training
            # ------------------------------------------------
            J_real_val = 0.0
            for i, data in enumerate(data_loader, 0):
                u_real, u_gen, u_gen0 = data[0].to(device), data[1].to(device), data[2].to(device)

                u_gen.requires_grad_(True)
                Jout = netD(u_gen)  # n_samples x 1

                disc_real = netD(u_real).detach()
                # err_real = torch.mean(disc_real).detach()

                # take derivative w.r.t only inputs
                nablaJ = torch.autograd.grad(outputs=Jout, inputs=u_gen,
                                             grad_outputs=torch.ones(Jout.size(),device=device), only_inputs=True)[
                    0].detach()

                gamma_gen_iter = gamma_constant / gen_iter
                anchor = gamma_gen_iter * u_gen0

                # step size, combines general and individual step size
                if  step_type == 'Mean':
                    lam = mu * (his_wass[gen_iter - 1])
                elif step_type == 'Combined':
                    lam = mu * (his_wass[gen_iter - 1] + torch.max(Jout.detach(), torch.zeros(Jout.size(), device=device)))
                elif step_type == 'Individual':
                    lam = mu * (torch.max(Jout.detach(), torch.zeros(Jout.size(), device=device)))

                # lambda * gradient term; reshape to multiply by lam, then reshape back
                lam_gradient_term = (lam * (nablaJ.reshape(batch_size, -1))).reshape(u_gen.shape)

                u_gen = anchor + (1 - gamma_gen_iter) * (u_gen - lam_gradient_term )

                # clip
                u_gen = torch.clamp(u_gen, min=0, max=1)

                # update u_gen_full for data_set
                u_gen_full_train[i * batch_size:((i + 1) * batch_size), :] = u_gen.detach().cpu()
                del nablaJ, u_gen, u_real, u_gen0

            # ------------------------------------------------
            # update u_gen validation
            # ------------------------------------------------

            J_real_val = 0.0
            for i, data in enumerate(data_loader_val, 0):
                # u_real_val, u_gen_val = data[0].to(device), data[1].to(device)
                u_real_val, u_gen_val, u_gen0_val = data[0].to(device), data[1].to(device), data[2].to(device)

                u_gen_val.requires_grad_(True)
                Jout_val = netD(u_gen_val)  # n_samples x 1

                disc_real_val = netD(u_real_val).detach()
                # err_real_val = torch.mean(disc_real_val).detach()

                nablaJ_val = torch.autograd.grad(outputs=Jout_val, inputs=u_gen_val,
                                                 grad_outputs=torch.ones(Jout_val.size(), device=device), only_inputs=True)[
                    0].detach()

                gamma_gen_iter = gamma_constant / gen_iter
                anchor = gamma_gen_iter * u_gen0_val

                # step size, combines general and individual step size
                if  step_type == 'Mean':
                    lam = mu * (his_wass[gen_iter - 1])
                elif step_type == 'Combined':
                    lam = mu * (his_wass[gen_iter - 1] + torch.max(Jout_val.detach(), torch.zeros(Jout_val.size(), device=device)))
                elif step_type == 'Individual':
                    lam = mu * (torch.max(Jout_val.detach(), torch.zeros(Jout_val.size(), device=device)))

                # lambda * gradient term; reshape to multiply by lam, then reshape back
                lam_gradient_term = (lam * (nablaJ_val.reshape(batch_size_val, -1))).reshape(u_gen_val.shape)

                # u_gen_val = anchor + (1 - gamma_gen_iter) * (u_gen_val - lam * nablaJ_val)
                u_gen_val = anchor + (1 - gamma_gen_iter) * (u_gen_val - lam_gradient_term)

                # clip
                u_gen_val = torch.clamp(u_gen_val, min=0, max=1)

                J_gen_val += Jout_val.sum().detach().cpu()
                J_real_val += disc_real_val.sum().detach().cpu()
                dist_man_batch_sum_val += torch.abs(disc_real_val).sum().detach().cpu()
                running_MSELoss_val += (u_real_val - u_gen_val).pow(2).sum().detach().cpu()

                # update u_gen_full for data_set
                u_gen_full_val[i * batch_size_val:((i + 1) * batch_size_val), :] = u_gen_val.detach().cpu()

                del nablaJ_val, u_real_val, u_gen_val, Jout_val

            # ------------------------------------------------
            # update convergence histories
            # ------------------------------------------------
            # compute wass distance on validation dataset
            wass_dist_val = (J_gen_val.sum().detach() - J_real_val.sum().detach()).cpu() / n_val
            dist_man_val = 0.999 * dist_man_val + 0.001 * dist_man_batch_sum_val / n_val
            his_wass_val.append(wass_dist_val.item())
            his_dist_man_val.append(dist_man_val)
            running_MSELoss_val = running_MSELoss_val / n_val
            his_MSE_val.append(running_MSELoss_val.item())
            running_MSELoss = (u_true_full_train - u_gen_full_train).pow(2).sum().detach().cpu() / n_train

            # compute ssim and psnr
            ssim_val, psnr_val = utils.compute_avg_SSIM_PSNR(u_true_full_val, u_gen_full_val, n_mesh,
                                                                 data_range[image_scaling])

            with open(save_path + 'val_logging.csv', 'a') as f:
                f.write(f'{running_MSELoss_val},{psnr_val},{ssim_val}\n')

            his_ssim.append(ssim_val.item())
            his_psnr.append(psnr_val.item())

            # ------------------------
            #  update dataloader
            # ------------------------
            dataset = TensorDataset(u_true_full_train, u_gen_full_train, u0_gen_full_train)
            dataset_val = TensorDataset(u_true_full_val, u_gen_full_val, u0_gen_full_val)
            data_loader = DataLoader(dataset=dataset, batch_size=batch_size, shuffle=False)
            data_loader_val = DataLoader(dataset=dataset_val, batch_size=batch_size_val, shuffle=False)

            # --------------------------
            # save histories and weights
            # --------------------------
            state = {
                'epoch': epoch,
                'gen_iter_array': gen_iter_array,
                'gen_freq': gen_freq,
                'his_wass': his_wass,
                'his_wass_val': his_wass_val,
                'his_MSE': his_MSE,
                'his_MSE_val': his_MSE_val,
                'his_eta': his_eta,
                'his_J_gen': his_J_gen,
                'his_J_real': his_J_real,
                'eta': eta,
                'learning_rate': learning_rate,
                'state_dict': netD.state_dict(),
            }
            checkpt_path = save_path + 'checkpoints/'
            if not os.path.exists(os.path.dirname(checkpt_path)):
                print("created", checkpt_path)
                os.makedirs(os.path.dirname(checkpt_path))
            else:
                print(checkpt_path + "already exists")

            if gen_iter < 10:
                save_checkpt_str = checkpt_path + 'step_0' + str(gen_iter) + '.pth'
            else:
                save_checkpt_str = checkpt_path + 'step_' + str(gen_iter) + '.pth'
            torch.save(state, save_checkpt_str)

            # ----------------------------------------------------------------------------------------------------------
            # create and save plots
            # ----------------------------------------------------------------------------------------------------------
            # plot images
            utils.plot_and_save_images(u_true_full_train[0, :].cpu(),
                                                   u0_gen_full_train[0, :].cpu(),
                                                   u_gen_full_train[0, :].cpu(),
                                                   u_true_full_val[ind_val, :].cpu(),
                                                   u0_gen_full_val[ind_val, :].cpu(),
                                                   u_gen_full_val[ind_val, :].cpu(),
                                                   save_path, gen_iter, n_mesh, vmin, vmax, vmin_val, vmax_val)

            # plot convergence histories
            utils.plot_and_save_histories(his_wass,
                                                      his_ssim,
                                                      his_psnr,
                                                      his_MSE_val,
                                                      his_eta,
                                                      gen_iter,
                                                      save_path)

            # ----------------------------------------------------------------------------------------------------------
            # recompute eta (otherwise it will continue updating the generator with same etas from before)
            # ----------------------------------------------------------------------------------------------------------
            eta_old = eta
            eta = 0.0

            for i, data in enumerate(data_loader, 0):
                u_gen = data[1].to(device)
                u_gen.requires_grad_(True)
                Jout = netD(u_gen)  # n_samples x 1

                # take deriv w.r.t only inputs
                nablaJ = torch.autograd.grad(outputs=Jout, inputs=u_gen,
                                             grad_outputs=torch.ones(Jout.size()).to(device), only_inputs=True)[
                    0].detach()
                # sum up all norms. Note here that nablaJ is n_samples x dim
                # so we want to take the norms along dimension 1

                # print("nablaJ.reshape(nablaJ.size(0), -1).shape = ", nablaJ.reshape(nablaJ.size(0), -1).shape)
                eta += (torch.norm(nablaJ.reshape(nablaJ.size(0), -1), p=2, dim=1) ** 2).sum()

                del nablaJ, Jout, u_gen

            eta = (eta / n_train).detach().cpu()

            # ----------------------------------------------------------------------------------------------------------
            # update outer iteration number
            # ----------------------------------------------------------------------------------------------------------
            gen_iter += 1

            # ----------------------------------------------------------------------------------------------------------
            # decay learning rate
            # ----------------------------------------------------------------------------------------------------------
            if gen_iter == lr_decay_iter:
                optimizer.param_groups[0]['lr'] *= lr_decay_rate
                print('DECAYING LEARNING RATE')

        # running_MSELoss = running_MSELoss / n_samples

        # compute time per epoch
        end_time = time.time()
        epoch_time = end_time - start_time

        # --------------------------------------------------------------------------------------------------------------
        # output training stats
        # --------------------------------------------------------------------------------------------------------------
        if epoch % print_freq == 0:
            print(
                '[%d: %d/%d]   W1-Dist: %.3e   W1-DistVal: %.3e   dist_man: %.3e   dist_man_val: %.3e   MSE: %.3e   MSEVal: %.3e   ssim: %.3e   psnr:%.3e   '
                'grad_pen_avg:%.3e   eta %.3e   time: %.2e'
                % (
                    gen_iter, epoch, max_epochs, wass_dist, wass_dist_val, dist_man, dist_man_val, running_MSELoss,
                    running_MSELoss_val, ssim_val, psnr_val, np.mean(grad_pen_list),
                    eta, epoch_time))
