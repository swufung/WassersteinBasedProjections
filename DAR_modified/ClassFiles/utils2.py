import numpy as np
import matplotlib.pyplot as plt
import matplotlib.pyplot as plt
from matplotlib.ticker import FormatStrFormatter
import os

def plot_pic(x_true, fbp, guess, plot_pic_savepath, plot_title):
    fig, ax = plt.subplots(nrows=1, ncols=3)
    ax[0].imshow(x_true[0].squeeze(-1))
    ax[0].set_title('x_true')
    ax[1].imshow(fbp[0].squeeze(-1))
    ax[1].set_title('fbp')
    ax[2].imshow(guess[0].squeeze(-1))
    ax[2].set_title('guess')

    plt.suptitle(plot_title)

    plt.savefig(plot_pic_savepath)

    # plt.show()

def plot_pic_custom(x_true, gen_zero, guess, plot_pic_savepath, plot_title):
    fig, ax = plt.subplots(nrows=1, ncols=3)
    ax[0].imshow(x_true[0].squeeze(-1))
    ax[0].set_title('x_true')
    ax[1].imshow(gen_zero[0].squeeze(-1))
    ax[1].set_title('gen_zero')
    ax[2].imshow(guess[0].squeeze(-1))
    ax[2].set_title('guess')

    plt.suptitle(plot_title)

    plt.savefig(plot_pic_savepath)

    # plt.show()

def plot_metrics(l2_arr, psnr_arr, ssim_arr, step_arr, plot_metric_savepath, plot_title):
    # Plot l2, psnr, and ssim
    fig, ax = plt.subplots(nrows=1, ncols=3, figsize=(18, 6))
    plt.suptitle(plot_title)

    ax[0].semilogy(step_arr, l2_arr, c='k')
    ax[0].scatter(step_arr, l2_arr, c='r')
    ax[0].set_title('l2')
    ax[0].tick_params(axis='both', which='major')
    ax[0].yaxis.set_major_formatter(FormatStrFormatter('%.2f'))
    # ax[0].yaxis.set_major_formatter(mtick.FormatStrFormatter('%.2e'))
    # ax[0].set_aspect('equal')
    # ax[0].set(adjustable='box-forced', aspect='equal')

    ax[1].semilogy(step_arr, psnr_arr, c='k')
    ax[1].scatter(step_arr, psnr_arr, c='r')
    ax[1].set_title('psnr')
    ax[1].tick_params(axis='both', which='major')
    ax[1].yaxis.set_major_formatter(FormatStrFormatter('%.2f'))
    # ax[1].yaxis.set_major_formatter(mtick.FormatStrFormatter('%.2e'))
    # ax[1].set_aspect('equal')
    # ax[1].set(adjustable='box-forced', aspect='equal')

    ax[2].semilogy(step_arr, ssim_arr, c='k')
    ax[2].scatter(step_arr, ssim_arr, c='r')
    ax[2].set_title('ssim')
    ax[2].tick_params(axis='both', which='major')
    ax[2].yaxis.set_major_formatter(FormatStrFormatter('%.2f'))
    # ax[2].yaxis.set_major_formatter(mtick.FormatStrFormatter('%.2e'))
    # ax[2].set_aspect('equal')
    # ax[2].set(adjustable='box-forced', aspect='equal')

    # plot_metric_savepath = os.path.join(self.path, 'plot_metrics_logmin2.png')
    plt.savefig(plot_metric_savepath)

    # plt.show()

def write_to_csv_logmin2(csv_file, step, qualities):
    if not os.path.exists(csv_file):
        with open(csv_file, 'w') as f:
            f.write('step,l2,psnr,ssim\n')
    with open(csv_file, 'a') as f:
        f.write(f'{step},{qualities[0]},{qualities[1]},{qualities[2]}\n')

def write_to_csv_logopt2(csv_file, mu, step_size, total_steps, qualities):
    if not os.path.exists(csv_file):
        with open(csv_file, 'w') as f:
            f.write('mu,step_size,total_steps,l2,psnr,ssim\n')
    with open(csv_file, 'a') as f:
        f.write(f'{mu},{step_size},{total_steps},{qualities[0]},{qualities[1]},{qualities[2]}\n')