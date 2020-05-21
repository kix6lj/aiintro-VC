from model import Generator, Generatorf0
from model import Discriminator, Discriminatorf0
from torch.autograd import Variable
from torchvision.utils import save_image
import torch
import torch.nn.functional as F
import numpy as np
import os
from os.path import join, basename, dirname, split
import time
import datetime
from data_loader import to_categorical
import librosa
from utils import *
from tqdm import tqdm

speakers = ['ANG', 'DIS', 'FEA', 'HAP', 'NEU', 'SAD']

spk2idx = dict(zip(speakers, range(len(speakers))))

class Solver(object):
    """Solver for training and testing StarGAN."""

    def __init__(self, train_loader, test_loader, config):
        """Initialize configurations."""

        # Data loader.
        self.train_loader = train_loader
        self.test_loader = test_loader
        self.sampling_rate = config.sampling_rate
        
        # Model configurations.
        self.num_speakers = config.num_speakers
        self.num_scales = config.num_scales
        self.lambda_cls = config.lambda_cls
        self.lambda_rec = config.lambda_rec
        self.lambda_gp = config.lambda_gp
        
        # Training configurations.
        self.batch_size = config.batch_size
        self.num_iters = config.num_iters
        self.num_iters_decay = config.num_iters_decay
        self.g_lr = config.g_lr
        self.gf0_lr = config.gf0_lr
        self.d_lr = config.d_lr
        self.df0_lr = config.df0_lr
        self.n_critic = config.n_critic
        self.beta1 = config.beta1
        self.beta2 = config.beta2
        self.resume_iters = config.resume_iters

        # Test configurations.
        self.test_iters = config.test_iters

        # Miscellaneous.
        self.use_tensorboard = config.use_tensorboard
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.device = 'cpu'
        # Directories.
        self.log_dir = config.log_dir
        self.sample_dir = config.sample_dir
        self.model_save_dir = config.model_save_dir

        # Step size.
        self.log_step = config.log_step
        self.sample_step = config.sample_step
        self.model_save_step = config.model_save_step
        self.lr_update_step = config.lr_update_step

        # Build the model and tensorboard.
        self.build_model()
        if self.use_tensorboard:
            self.build_tensorboard()

    def build_model(self):
        """Create a generator and a discriminator."""
        self.G = Generator(num_speakers=self.num_speakers)
        self.D = Discriminator(num_speakers=self.num_speakers)
        self.Gf0 = Generatorf0(num_speakers=self.num_speakers, scale=self.num_scales)
        self.Df0 = Discriminatorf0(num_speakers=self.num_speakers, scale=self.num_scales)
        self.g_optimizer = torch.optim.Adam(self.G.parameters(), self.g_lr, [self.beta1, self.beta2])
        self.d_optimizer = torch.optim.Adam(self.D.parameters(), self.d_lr, [self.beta1, self.beta2])
        self.gf0_optimizer = torch.optim.Adam(self.Gf0.parameters(), self.gf0_lr, [self.beta1, self.beta2])
        self.df0_optimizer = torch.optim.Adam(self.Df0.parameters(), self.df0_lr, [self.beta1, self.beta2])
        self.print_network(self.G, 'G')
        self.print_network(self.D, 'D')
            
        self.G.to(self.device)
        self.D.to(self.device)
        self.Gf0.to(self.device)
        self.Df0.to(self.device)
        
    def print_network(self, model, name):
        """Print out the network information."""
        num_params = 0
        for p in model.parameters():
            num_params += p.numel()
        print(model)
        print(name)
        print("The number of parameters: {}".format(num_params))

    def restore_model(self, resume_iters):
        """Restore the trained generator and discriminator."""
        print('Loading the trained models from step {}...'.format(resume_iters))
        G_path = os.path.join(self.model_save_dir, '{}-G.ckpt'.format(resume_iters))
        D_path = os.path.join(self.model_save_dir, '{}-D.ckpt'.format(resume_iters))
        Gf0_path = os.path.join(self.model_save_dir, '{}-Gf0.ckpt'.format(resume_iters))
        Df0_path = os.path.join(self.model_save_dir, '{}-Df0.ckpt'.format(resume_iters))
        self.G.load_state_dict(torch.load(G_path, map_location=lambda storage, loc: storage))
        self.D.load_state_dict(torch.load(D_path, map_location=lambda storage, loc: storage))
        self.Gf0.load_state_dict(torch.load(Gf0_path, map_location=lambda storage, loc: storage))
        self.Df0.load_state_dict(torch.load(Df0_path, map_location=lambda storage, loc: storage))
        
    def build_tensorboard(self):
        """Build a tensorboard logger."""
        from logger import Logger
        self.logger = Logger(self.log_dir)

    def update_lr(self, g_lr, d_lr, gf0_lr, df0_lr):
        """Decay learning rates of the generator and discriminator."""
        for param_group in self.g_optimizer.param_groups:
            param_group['lr'] = g_lr
        for param_group in self.d_optimizer.param_groups:
            param_group['lr'] = d_lr
        for param_group in self.gf0_optimizer.param_groups:
            param_group['lr'] = gf0_lr
        for param_group in self.df0_optimizer.param_groups:
            param_group['lr'] = df0_lr

    def reset_grad(self):
        """Reset the gradient buffers."""
        self.g_optimizer.zero_grad()
        self.d_optimizer.zero_grad()
        self.gf0_optimizer.zero_grad()
        self.df0_optimizer.zero_grad()
        
    def denorm(self, x):
        """Convert the range from [-1, 1] to [0, 1]."""
        out = (x + 1) / 2
        return out.clamp_(0, 1)

    def gradient_penalty(self, y, x):
        """Compute gradient penalty: (L2_norm(dy/dx) - 1)**2."""
        weight = torch.ones(y.size()).to(self.device)
        dydx = torch.autograd.grad(outputs=y,
                                   inputs=x,
                                   grad_outputs=weight,
                                   retain_graph=True,
                                   create_graph=True,
                                   only_inputs=True)[0]

        dydx = dydx.view(dydx.size(0), -1)
        dydx_l2norm = torch.sqrt(torch.sum(dydx**2, dim=1))
        return torch.mean((dydx_l2norm-1)**2)

    def label2onehot(self, labels, dim):
        """Convert label indices to one-hot vectors."""
        batch_size = labels.size(0)
        out = torch.zeros(batch_size, dim)
        out[np.arange(batch_size), labels.long()] = 1
        return out

    def sample_spk_c(self, size):
        spk_c = np.random.randint(0, self.num_speakers, size=size)
        spk_c_cat = to_categorical(spk_c, self.num_speakers)
        return torch.LongTensor(spk_c), torch.FloatTensor(spk_c_cat)

    def classification_loss(self, logit, target):
        """Compute softmax cross entropy loss."""
        return F.cross_entropy(logit, target)

    def load_wav(self, wavfile, sr=16000):
        wav, _ = librosa.load(wavfile, sr=sr, mono=True)
        return wav_padding(wav, sr=16000, frame_period=5, multiple = 4)  # TODO
    
    def convert(self, f0, timeaxis, sp, ap, to_class):
        '''
            Give f0, timeaxis, sp, ap (get these arugments directly by world_decompose in utils)
            Reture converted f0, timeaxis, coded_sp, ap (put these arguments into world_speech_synthesis in utils)
        '''
        target_stats = np.load(os.path.join('./data/mc/train', '{}_stats.npz'.format(to_class)))
        log_f0s_mean_trg, log_f0s_std_trg = target_stats['log_f0s_mean'], target_stats['log_f0s_std']
        mcep_mean_trg, mcep_std_trg = target_stats['coded_sps_mean'], target_stats['coded_sps_std']
        mcep_mean_trg, mcep_std_trg = np.expand_dims(mcep_mean_trg, axis=-1), np.expand_dims(mcep_std_trg, axis=-1)
        # get target categorical
        spkidx = spk2idx[to_class]
        spk_cat = np.zeros(6)
        spk_cat[spkidx] = 1
    
        # coded_sp
        coded_sp = world_encode_spectral_envelop(sp = sp, fs = self.sampling_rate, dim = 36)
        coded_sp_transposed = coded_sp.T # (time, dim) to (dim, time)
        mcep_mean_org, mcep_std_org = np.mean(coded_sp_transposed, axis=-1), np.std(coded_sp_transposed, axis=-1)
        mcep_mean_org, mcep_std_org = np.expand_dims(mcep_mean_org, axis=-1), np.expand_dims(mcep_std_org, axis=-1)
        coded_sp_norm = (coded_sp_transposed - mcep_mean_org) / mcep_std_org
        
        # Wavelet lf0
        uv, cont_lf0_lpf = get_cont_lf0(f0)
        logf0s_mean_org, logf0s_std_org = np.mean(cont_lf0_lpf), np.std(cont_lf0_lpf)
        cont_lf0_lpf_norm = (cont_lf0_lpf - logf0s_mean_org) / logf0s_std_org
        Wavelet_lf0, scales = get_lf0_cwt(cont_lf0_lpf_norm)
        
        Wavelet_lf0_norm, mean, std = norm_scale(Wavelet_lf0) 
        lf0_cwt_norm = Wavelet_lf0_norm.T 
        
        # additional process for the model
        input_sp = torch.FloatTensor(coded_sp_norm)
        input_lf0 = torch.FloatTensor(lf0_cwt_norm)
        input_cat = torch.FloatTensor(spk_cat)
        input_sp = input_sp.to(self.device)
        input_lf0 = input_lf0.to(self.device)
        input_cat = input_cat.to(self.device)
        
        input_sp.unsqueeze_(0)
        input_sp.unsqueeze_(0)
        input_lf0.unsqueeze_(0)
        input_cat.unsqueeze_(0)
        
        # feed into model
        coded_sp_converted_norm = self.G(input_sp, input_cat).data.cpu().numpy()
        lf0 = self.Gf0(input_lf0, input_cat).data.cpu().numpy()
        coded_sp_converted_norm, lf0 = np.squeeze(coded_sp_converted_norm, axis=(0, 1)), np.squeeze(lf0, axis=(0))
        
        ########## Recover ###############
        coded_sp_converted = coded_sp_converted_norm * mcep_std_trg + mcep_mean_trg #mceps

        lf0_cwt_denormalize = denormalize(lf0.T, mean, std)
        lf0_rec = inverse_cwt(lf0_cwt_denormalize, scales)
        lf0_converted = lf0_rec * log_f0s_std_trg + log_f0s_mean_trg
        f0_converted = np.squeeze(uv) * np.exp(lf0_converted)
        f0_converted = np.ascontiguousarray(f0_converted)
        
        coded_sp_converted = coded_sp_converted.T
        coded_sp_converted = np.ascontiguousarray(coded_sp_converted)
        
        return f0_converted, timeaxis, coded_sp_converted, ap
    
    def train(self):
        """Train StarGAN."""
        # Set data loader.
        train_loader = self.train_loader

        data_iter = iter(train_loader)

        # Read a batch of testdata (currently no test)
        '''
        test_wavfiles = self.test_loader.get_batch_test_data(batch_size=4)
        test_wavs = [self.load_wav(wavfile) for wavfile in test_wavfiles]
        '''
        # Determine whether do copysynthesize when first do training-time conversion test.
        cpsyn_flag = [True, False][0]
        # f0, timeaxis, sp, ap = world_decompose(wav = wav, fs = sampling_rate, frame_period = frame_period)

        # Learning rate cache for decaying.
        g_lr = self.g_lr
        d_lr = self.d_lr
        gf0_lr = self.gf0_lr
        df0_lr = self.df0_lr
        
        # Start training from scratch or resume training.
        start_iters = 0
        if self.resume_iters:
            print("resuming step %d ..."% self.resume_iters)
            start_iters = self.resume_iters
            self.restore_model(self.resume_iters)

        # Start training.
        print('Start training...')
        start_time = time.time()
        for i in range(start_iters, self.num_iters):

            # =================================================================================== #
            #                             1. Preprocess input data                                #
            # =================================================================================== #

            # Fetch labels.
            try:
                mc_real, cwt_lf0_real, spk_label_org, spk_c_org = next(data_iter)
            except:
                data_iter = iter(train_loader)
                mc_real, spk_label_org, spk_c_org = next(data_iter)

            mc_real.unsqueeze_(1) # (B, D, T) -> (B, 1, D, T) for conv2d

            # Generate target domain labels randomly.
            # spk_label_trg: int,   spk_c_trg:one-hot representation 
            spk_label_trg, spk_c_trg = self.sample_spk_c(mc_real.size(0)) 

            mc_real = mc_real.to(self.device)                         # Input mc.
            cwt_lf0_real = cwt_lf0_real.to(self.device)                         # Input cwt_lf0.
            spk_label_org = spk_label_org.to(self.device)             # Original spk labels.
            spk_c_org = spk_c_org.to(self.device)                     # Original spk acc conditioning.
            spk_label_trg = spk_label_trg.to(self.device)             # Target spk labels for classification loss for G.
            spk_c_trg = spk_c_trg.to(self.device)                     # Target spk conditioning.

            # =================================================================================== #
            #                             2. Train the discriminator                              #
            # =================================================================================== #

            # Compute loss with real mc feats.
            out_src, out_cls_spks = self.D(mc_real)
            d_loss_real = - torch.mean(out_src)
            d_loss_cls_spks = self.classification_loss(out_cls_spks, spk_label_org)

            # Compute loss with fake mc feats.
            mc_fake = self.G(mc_real, spk_c_trg)
            out_src, out_cls_spks = self.D(mc_fake.detach())
            d_loss_fake = torch.mean(out_src)

            # Compute loss for gradient penalty.
            alpha = torch.rand(mc_real.size(0), 1, 1, 1).to(self.device)
            x_hat = (alpha * mc_real.data + (1 - alpha) * mc_fake.data).requires_grad_(True)
            out_src, _ = self.D(x_hat)
            d_loss_gp = self.gradient_penalty(out_src, x_hat)

            # Backward and optimize.
            d_loss = d_loss_real + d_loss_fake + self.lambda_cls * d_loss_cls_spks + self.lambda_gp * d_loss_gp
            self.reset_grad()
            d_loss.backward()
            self.d_optimizer.step()

            # Logging.
            loss = {}
            loss['D/loss_real'] = d_loss_real.item()
            loss['D/loss_fake'] = d_loss_fake.item()
            loss['D/loss_cls_spks'] = d_loss_cls_spks.item()
            loss['D/loss_gp'] = d_loss_gp.item()
            
            #============================================#
            #                  Train Df0                 #
            #============================================#
            
            out_src, out_cls_spks = self.Df0(cwt_lf0_real)
            df0_loss_real = - torch.mean(out_src)
            df0_loss_cls_spks = self.classification_loss(out_cls_spks, spk_label_org)
            

            # Compute loss with fake mc feats.
            cwt_lf0_fake = self.Gf0(cwt_lf0_real, spk_c_trg)
            out_src, out_cls_spks = self.Df0(cwt_lf0_fake.detach())
            df0_loss_fake = torch.mean(out_src)

            # Compute loss for gradient penalty.
            alpha = torch.rand(cwt_lf0_real.size(0), 1, 1).to(self.device)
            x_hat = (alpha * cwt_lf0_real.data + (1 - alpha) * cwt_lf0_fake.data).requires_grad_(True)
            out_src, _ = self.Df0(x_hat)
            df0_loss_gp = self.gradient_penalty(out_src, x_hat)

            # Backward and optimize.
            df0_loss = df0_loss_real + df0_loss_fake + self.lambda_cls * df0_loss_cls_spks + self.lambda_gp * df0_loss_gp
            self.reset_grad()
            df0_loss.backward()
            self.df0_optimizer.step()

            # Logging.
            loss['Df0/loss_real'] = df0_loss_real.item()
            loss['Df0/loss_fake'] = df0_loss_fake.item()
            loss['Df0/loss_cls_spks'] = df0_loss_cls_spks.item()
            loss['Df0/loss_gp'] = df0_loss_gp.item()
            
            # =================================================================================== #
            #                               3. Train the generator                                #
            # =================================================================================== #
            
            if (i+1) % self.n_critic == 0:
                # Original-to-target domain.
                mc_fake = self.G(mc_real, spk_c_trg)
                out_src, out_cls_spks = self.D(mc_fake)
                g_loss_fake = - torch.mean(out_src)
                g_loss_cls_spks = self.classification_loss(out_cls_spks, spk_label_trg)

                # Target-to-original domain.
                mc_reconst = self.G(mc_fake, spk_c_org)
                g_loss_rec = torch.mean(torch.abs(mc_real - mc_reconst))

                # Backward and optimize.
                g_loss = g_loss_fake + self.lambda_rec * g_loss_rec + self.lambda_cls * g_loss_cls_spks
                self.reset_grad()
                g_loss.backward()
                self.g_optimizer.step()

                # Logging.
                loss['G/loss_fake'] = g_loss_fake.item()
                loss['G/loss_rec'] = g_loss_rec.item()
                loss['G/loss_cls_spks'] = g_loss_cls_spks.item()
                
                #===========================================#
                #               Train Gf0                   #
                #===========================================#
                # Original-to-target domain.
                cwt_lf0_fake = self.Gf0(cwt_lf0_real, spk_c_trg)
                out_src, out_cls_spks = self.Df0(cwt_lf0_fake)
                gf0_loss_fake = - torch.mean(out_src)
                gf0_loss_cls_spks = self.classification_loss(out_cls_spks, spk_label_trg)

                # Target-to-original domain.
                cwt_lf0_reconst = self.Gf0(cwt_lf0_fake, spk_c_org)
                gf0_loss_rec = torch.mean(torch.abs(cwt_lf0_real - cwt_lf0_reconst))

                # Backward and optimize.
                gf0_loss = gf0_loss_fake + self.lambda_rec * gf0_loss_rec + self.lambda_cls * gf0_loss_cls_spks
                self.reset_grad()
                gf0_loss.backward()
                self.gf0_optimizer.step()

                # Logging.
                loss['Gf0/loss_fake'] = gf0_loss_fake.item()
                loss['Gf0/loss_rec'] = gf0_loss_rec.item()
                loss['Gf0/loss_cls_spks'] = gf0_loss_cls_spks.item()

            # =================================================================================== #
            #                                 4. Miscellaneous                                    #
            # =================================================================================== #

            # Print out training information.
            if (i+1) % self.log_step == 0:
                et = time.time() - start_time
                et = str(datetime.timedelta(seconds=et))[:-7]
                log = "Elapsed [{}], Iteration [{}/{}]".format(et, i+1, self.num_iters)
                for tag, value in loss.items():
                    log += ", {}: {:.4f}".format(tag, value)
                print(log)

                if self.use_tensorboard:
                    for tag, value in loss.items():
                        self.logger.scalar_summary(tag, value, i+1)
            
            if (i+1) % self.sample_step == 0:
                sampling_rate=16000
                num_mcep=36
                frame_period=5
                test_wavfiles = ['./cremad/AudioWAV/NEU/1001_DFA_NEU_XX.wav',
                                  './cremad/AudioWAV/ANG/1001_DFA_ANG_XX.wav',
                                  './cremad/AudioWAV/DIS/1001_DFA_DIS_XX.wav']
                test_wavs = [self.load_wav(wav_files) for wav_files in test_wavfiles]
                
                with torch.no_grad():
                    for idx, wav in tqdm(enumerate(test_wavs)):
                        wav_name = basename(test_wavfiles[idx])
                        # print(wav_name)
                        f0, timeaxis, sp, ap = world_decompose(wav=wav, fs=sampling_rate, frame_period=frame_period)
                        c_f0, c_timaxis, c_coded_sp, c_ap = self.convert(f0, timeaxis, sp, ap, 'ANG')
                        wav_transformed = world_speech_synthesis(f0=c_f0, coded_sp=c_coded_sp, 
                                                                ap=c_ap, fs=sampling_rate, frame_period=frame_period)
                        
                        librosa.output.write_wav(
                            join(self.sample_dir, str(i+1)+'-'+wav_name.split('.')[0]+'-vcto-{}'.format('ANG')+'.wav'), wav_transformed, sampling_rate)
                        if cpsyn_flag:
                            wav_cpsyn = world_speech_synthesis(f0=f0, coded_sp=c_coded_sp, 
                                                        ap=ap, fs=sampling_rate, frame_period=frame_period)
                            librosa.output.write_wav(join(self.sample_dir, 'cpsyn-'+wav_name), wav_cpsyn, sampling_rate)
                    cpsyn_flag = False
                    
            # Save model checkpoints.
            if (i+1) % self.model_save_step == 0:
                G_path = os.path.join(self.model_save_dir, '{}-G.ckpt'.format(i+1))
                D_path = os.path.join(self.model_save_dir, '{}-D.ckpt'.format(i+1))
                Gf0_path = os.path.join(self.model_save_dir, '{}-Gf0.ckpt'.format(i+1))
                Df0_path = os.path.join(self.model_save_dir, '{}-Df0.ckpt'.format(i+1))
                torch.save(self.G.state_dict(), G_path)
                torch.save(self.D.state_dict(), D_path)
                torch.save(self.Gf0.state_dict(), Gf0_path)
                torch.save(self.Df0.state_dict(), Df0_path)
                print('Saved model checkpoints into {}...'.format(self.model_save_dir))

            # Decay learning rates.
            if (i+1) % self.lr_update_step == 0 and (i+1) > (self.num_iters - self.num_iters_decay):
                g_lr -= (self.g_lr / float(self.num_iters_decay))
                d_lr -= (self.d_lr / float(self.num_iters_decay))
                gf0_lr -= (self.gf0_lr / float(self.num_iters_decay))
                df0_lr -= (self.df0_lr / float(self.num_iters_decay))
                self.update_lr(g_lr, d_lr, gf0_lr, df0_lr)
                print ('Decayed learning rates, g_lr: {}, d_lr: {}.'.format(g_lr, d_lr))


