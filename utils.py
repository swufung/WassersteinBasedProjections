import numpy as np
import odl
import torch
import datetime

from skimage.metrics import structural_similarity as ssim
from skimage.metrics import peak_signal_noise_ratio as psnr

import matplotlib.pyplot as plt

def random_shapes(interior=False):
    if interior:
        x_0 = np.random.rand() - 0.5
        y_0 = np.random.rand() - 0.5
    else:
        x_0 = 2 * np.random.rand() - 1.0
        y_0 = 2 * np.random.rand() - 1.0

    return ((np.random.rand() - 0.5) * np.random.exponential(0.4),
            np.random.exponential() * 0.2, np.random.exponential() * 0.2,
            x_0, y_0,
            np.random.rand() * 2 * np.pi)

def random_phantom(spc, n_ellipse=50, interior=False, form='ellipse'):
    n = np.random.poisson(n_ellipse)
    shapes = [random_shapes(interior=interior) for _ in range(n)]
    if form == 'ellipse':
        return odl.phantom.ellipsoid_phantom(spc, shapes)
    if form == 'rectangle':
        return odl.phantom.cuboid_phantom(spc, shapes)
    else:
        raise Exception('unknown form')

def scale_to_interval(im, type):
    if type == 'unit':
        im_min = np.min(im)
        im_max = np.max(im)
        out = (im - im_min) / (im_max - im_min + 1e-8)
    elif type == 'symmetric':
        im_min = np.min(im)
        im_max = np.max(im)
        out = 2 * (im - im_min) / (im_max - im_min + 1e-8) - 1
    elif type == 'none':
        out = im
    else:
        raise ValueError(f"Bad type. Got type: {type}")

    return out


def generate_data_CNN(n_samples, fwd_operator, fbp_operator, space, noise_level, scale, validation=False):  # output (nFeat1*nFeat2 x nSamples)
    """Generate a set of random data."""
    n_generate = 1 if validation else n_samples

    y_arr = np.empty((n_generate, fwd_operator.range.shape[0], fwd_operator.range.shape[1], 1), dtype='float32')
    x_true_arr = np.empty((n_generate, space.shape[0], space.shape[1], 1), dtype='float32')
    fbp_arr = np.empty((n_generate, space.shape[0], space.shape[1], 1), dtype='float32')

    for i in range(n_generate):
        if validation:
            phantom = odl.phantom.shepp_logan(space, True)
            phantom = scale_to_interval(phantom, scale)
        else:
            phantom = random_phantom(space)
            phantom = scale_to_interval(phantom, scale)
        data = fwd_operator(phantom)
        noisy_data = data + odl.phantom.white_noise(fwd_operator.range) * np.mean(np.abs(data)) * noise_level

        fbp_arr[i, ..., 0] = fbp_operator(noisy_data)
        x_true_arr[i, ..., 0] = phantom
        y_arr[i, ..., 0] = noisy_data

    x_true_arr = np.transpose(x_true_arr, (0, 3, 1, 2))
    y_arr = np.transpose(y_arr, (0, 3, 1, 2))
    fbp_arr = np.transpose(fbp_arr, (0, 3, 1, 2))

    return y_arr, x_true_arr, fbp_arr


def grad_penalty(u_real, u_gen, netD, device):
    bs = u_real.size()[0]
    alpha = torch.rand(bs, 1, 1, 1).to(device)
    interpolated = alpha * u_real + (1 - alpha) * u_gen
    interpolated.requires_grad_(True)
    netD_interp = netD(interpolated)

    netD_interp_grad = torch.autograd.grad(outputs=netD_interp, inputs=interpolated,
                                           grad_outputs=torch.ones(netD_interp.size()).to(device),
                                           create_graph=True, retain_graph=True, only_inputs=True)[0]

    netD_interp_grad_vec = netD_interp_grad.view(bs, -1)

    # grad_penalty = ((torch.norm(netD_interp_grad_vec, p=2, dim=-1) - 1) ** 2).mean()
    grad_penalty = torch.relu(torch.norm(netD_interp_grad_vec, p=2, dim=-1) - 1).mean()

    return grad_penalty


def get_time_string():
    """
    Just get the current time in a string
    """
    ## New format, but messes with legacy
    # out = str(datetime.datetime.now()).replace(':', '-').replace(' ', '-')[:-7]
    # out = out[:10] + '__' + out[11:]  # separate year-month-day from hour-minute-seconds

    # Old format
    out = str(datetime.datetime.now()).replace(':', '-').replace(' ', '-')[:-7]

    return out

def compute_avg_SSIM_PSNR(u_true, u_gen, n_mesh, data_range):
    # assumes images are size n_samples x n_features**2 and are detached

    n_samples = u_true.shape[0]
    u_true = u_true.reshape(n_samples, n_mesh, n_mesh).cpu().numpy()
    u_gen  = u_gen.reshape(n_samples, n_mesh, n_mesh).cpu().numpy()

    ssim_val = 0
    psnr_val = 0

    for j in range(n_samples):
        ssim_val = ssim_val + ssim(u_true[j,:,:], u_gen[j,:,:], data_range=data_range)
        psnr_val = psnr_val + psnr(u_true[j,:,:], u_gen[j,:,:], data_range=data_range)

    return ssim_val/n_samples, psnr_val/n_samples

def compute_avg_SSIM_PSNR_numpy(u_true, u_gen, n_mesh, data_range):
    # assumes images are size n_samples x n_features**2 and are detached

    n_samples = u_true.shape[0]
    u_true = u_true.reshape(n_samples, n_mesh, n_mesh)
    u_gen  = u_gen.reshape(n_samples, n_mesh, n_mesh)

    ssim_val = 0
    psnr_val = 0

    for j in range(n_samples):
        ssim_val = ssim_val + ssim(u_true[j,:,:], u_gen[j,:,:], data_range=data_range)
        psnr_val = psnr_val + psnr(u_true[j,:,:], u_gen[j,:,:], data_range=data_range)

    return ssim_val/n_samples, psnr_val/n_samples


def plot_and_save_images(u_true_train, u_gen0_train, u_gen_train, u_true_val, u_gen0_val, u_gen_val,
                        save_path, gen_iter, n_mesh, vmin, vmax, vmin_val, vmax_val):

    # assumes images are already in cpu and detached

    plt.figure()
    title_str = 'Flow Step ' + str(gen_iter)
    plt.suptitle(title_str)
    plt.subplot(2, 3, 1)
    im = plt.imshow(np.rot90(u_true_train.reshape(n_mesh, n_mesh).detach()), vmin=vmin, vmax=vmax);
    plt.title("u_true train")  # plt.colorbar();
    plt.subplot(2, 3, 2)
    im = plt.imshow(np.rot90(u_gen0_train.reshape(n_mesh, n_mesh).detach()), vmin=vmin, vmax=vmax);
    plt.title("u_gen train 0") # plt.colorbar();
    plt.subplot(2, 3, 3)
    im = plt.imshow(np.rot90(u_gen_train.reshape(n_mesh, n_mesh).detach()), vmin=vmin, vmax=vmax);
    plt.title("u_gen train")  # plt.colorbar();

    plt.subplot(2, 3, 4)
    im = plt.imshow(np.rot90(u_true_val.reshape(n_mesh, n_mesh).detach()), vmin=vmin_val,
                    vmax=vmax_val);
    plt.title("u_true val")  # plt.colorbar();
    plt.subplot(2, 3, 5)
    im = plt.imshow(np.rot90(u_gen0_val.reshape(n_mesh, n_mesh).detach()), vmin=vmin_val,
                    vmax=vmax_val);
    plt.title("u_gen val 0")  # plt.colorbar();
    plt.subplot(2, 3, 6)
    im = plt.imshow(np.rot90(u_gen_val.reshape(n_mesh, n_mesh).detach()), vmin=vmin_val,
                    vmax=vmax_val);
    plt.title("u_gen val")

    if gen_iter < 10:
        plt.savefig(save_path + '/flow_0' + str(gen_iter) + '.png')
    else:
        plt.savefig(save_path + '/flow_' + str(gen_iter) + '.png')
    # plt.show()
    plt.close()


def plot_and_save_histories(his_wass, his_ssim, his_psnr, his_MSE_val, his_eta, gen_iter,save_path):

    # plot Wasserstein history
    plt.figure(figsize=(10, 10))
    title_str = 'wass_distance ' + str(gen_iter)
    plt.semilogy(np.linspace(1, gen_iter, gen_iter), his_wass[0:gen_iter], c='k')
    plt.scatter(np.linspace(1, gen_iter, gen_iter), his_wass[0:gen_iter], c='r')
    plt.title(title_str)
    plt.savefig(save_path + '/wass_dist_' + str(gen_iter) + '.png')
    plt.close()

    # plot ssim history
    plt.figure(figsize=(10, 10))
    title_str = 'ssim ' + str(gen_iter)
    plt.semilogy(np.linspace(1, gen_iter, gen_iter), his_ssim[0:gen_iter], c='k')
    plt.scatter(np.linspace(1, gen_iter, gen_iter), his_ssim[0:gen_iter], c='r')
    plt.title(title_str)
    plt.savefig(save_path + '/ssim_' + str(gen_iter) + '.png')
    plt.close()

    # plot psnr history
    plt.figure(figsize=(10, 10))
    title_str = 'psnr ' + str(gen_iter)
    plt.semilogy(np.linspace(1, gen_iter, gen_iter), his_psnr[0:gen_iter], c='k')
    plt.scatter(np.linspace(1, gen_iter, gen_iter), his_psnr[0:gen_iter], c='r')
    plt.title(title_str)
    plt.savefig(save_path + '/psnr_' + str(gen_iter) + '.png')
    plt.close()

    # plot MSE Loss history
    plt.figure(figsize=(10, 10))
    title_str = 'MSE (red val)' + str(gen_iter)
    plt.semilogy(np.linspace(1, gen_iter, gen_iter), his_MSE_val[0:gen_iter], c='k')
    plt.scatter(np.linspace(1, gen_iter, gen_iter), his_MSE_val[0:gen_iter], c='r')
    plt.title(title_str)
    plt.savefig(save_path + '/his_mse_' + str(gen_iter) + '.png')
    plt.close()

    # plot eta history
    plt.figure(figsize=(10, 10))
    title_str = 'eta' + str(gen_iter)
    plt.semilogy(np.linspace(1, gen_iter, gen_iter), his_eta[0:gen_iter])
    plt.scatter(np.linspace(1, gen_iter, gen_iter), his_eta[0:gen_iter], c='r')
    plt.title(title_str)
    plt.savefig(save_path + '/eta_his_' + str(gen_iter) + '.png')
    plt.close()
