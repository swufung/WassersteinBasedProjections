from ClassFiles.Framework import AdversarialRegulariser
from ClassFiles.networks import ConvNetClassifier
from ClassFiles.forward_models import CT

from ClassFiles.data_pips import lodopab

DATA_PATH = './Datasets/Lodopab/'
SAVES_PATH = './Savefolder/DeepAdversarialRegulariser/'

step_s = 0.01
batch_s = 32
def_mu = 3
def_lr = 0.0001
mini_start = 'Mini'  # None

outer_k_max = 20
inner_k_max = 200

def_lmb = 20

class Experiment1(AdversarialRegulariser):
    experiment_name = 'Lodopab__0p025_individualnoise-0p015__tv_lam_0p0005_start__April-06-2021' \
                      + '__step_s-' + str(step_s) + '__mu-' + str(def_mu) + '__outer_k_max-' + str(outer_k_max) \
                      + '_inner_k_max-' + str(inner_k_max) + '__lr-' + str(def_lr) + '__batch-size-' + str(batch_s) \
                      + '__lmb-' + str(def_lmb) + '__mini_start-' + str(mini_start)
    noise_level = 0.015

    # relation between L2 error and regulariser
    # 0 corresponds to pure L2 loss, infty to pure adversarial loss
    mu_default = def_mu

    learning_rate = def_lr
    step_size = step_s
    total_steps_default = 10
    starting_point = mini_start
    batch_size = batch_s

    lmb = def_lmb

    def get_network(self, size, colors):
        return ConvNetClassifier(size=size, colors=colors)

    def unreg_mini(self, y, fbp):
        return self.update_pic(10, 1, y, fbp, 0)

    def get_Data_pip(self, data_path):
        return lodopab(data_path)

    def get_model(self, size):
        return CT(size=size)

experiment = Experiment1(DATA_PATH, SAVES_PATH)
CTLodopab_folder = './CTLodopab/'
train_path = CTLodopab_folder + 'lodopab_TRAIN_tv___tv_lam-0.0005__individualnoise-0.015__20k__march-28-2021.pkl'
val_path = CTLodopab_folder + 'lodopab_VAL_tv___tv_lam-0.0005__individualnoise-0.015__2k__march-28-2021.pkl'
experiment.load_custom_dataset(train_path, val_path)
experiment.find_good_lambda(32)

# Start training
for k in range(outer_k_max):
    print(f'\nouter_k: {k}\n')
    experiment.train(inner_k_max)


# Try a bunch of gradient descent parameters
for total_steps in [100, 50, 25, 10]:
    for step_s in [1e-2]:
        for mu in [3, 2, 1, 0.5]:
            print(f'total_steps: {total_steps}, step_size: {step_s}, mu: {mu}')
            qualities = experiment.log_optimization_2(None, total_steps, step_s, mu)
            print(f'l2: {qualities[0]}, psnr: {qualities[1]}, ssim: {qualities[2]}')
            print()
