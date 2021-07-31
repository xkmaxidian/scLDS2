#
import os
import time
import dateutil.tz
import datetime
import argparse
import importlib
import tensorflow as tf
import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics.cluster import normalized_mutual_info_score, adjusted_rand_score
from sklearn.cluster import SpectralClustering
import metric
import util
import pandas
import csv

tf.set_random_seed(0)


class clusGAN(object):
    def __init__(self, g_net, d_net, enc_net, x_sampler, z_sampler, data, model, sampler,
                 num_classes, n_cat,  batch_size, beta_1, beta_2,beta_3):
        self.model = model
        self.data = data
        self.sampler = sampler
        self.g_net = g_net
        self.d_net = d_net
        self.enc_net = enc_net
        self.x_sampler = x_sampler
        self.z_sampler = z_sampler
        self.num_classes = num_classes
        self.n_cat = n_cat
        self.batch_size = batch_size
        scale = 10.0
        self.beta_3 = beta_3
        self.beta_2 = beta_2
        self.beta_1 = beta_1

        self.x_dim = self.d_net.x_dim
        self.z_dim = self.g_net.z_dim



        self.x = tf.placeholder(tf.float32, [None, self.x_dim], name='x')
        self.z = tf.placeholder(tf.float32, [None, self.z_dim], name='z')

        # self.z_gen = self.z[:,0:self.dim_gen]
        # self.z_hot = self.z[:,self.dim_gen:]

        self.x_ = self.g_net(self.z)

        self.z_enc = self.enc_net(self.x_, reuse=False)
        self.z_infer = self.enc_net(self.x)


        self.d = self.d_net(self.x, reuse=False)

        bz = self.z_sampler(batch_size, self.z_dim, self.sampler, self.num_classes, self.n_cat)

        # self.sess.run(tf.global_variables_initializer())
        sess = tf.InteractiveSession()
        sess.run(tf.initialize_all_variables())
        self.z_enc_np = sess.run(self.z_enc, feed_dict = {self.z: bz})

        self.d_ = self.d_net(self.x_)
        # print(self.z_enc)
        self.g_loss = tf.reduce_mean(self.d_) + \
                      self.beta_1 * tf.reduce_mean(tf.square(self.z - self.z_enc)) +\
                      self.beta_2 * tf.reduce_mean(tf.norm(self.z_enc, ord=1)) +\
                      self.beta_3 * tf.reduce_mean(tf.convert_to_tensor(np.linalg.norm(self.z_enc_np, ord='nuc')))

        self.d_loss = tf.reduce_mean(self.d) - tf.reduce_mean(self.d_)
        epsilon = tf.random_uniform([], 0.0, 1.0)
        x_hat = epsilon * self.x + (1 - epsilon) * self.x_
        d_hat = self.d_net(x_hat)
        ddx = tf.gradients(d_hat, x_hat)[0]
        ddx = tf.sqrt(tf.reduce_sum(tf.square(ddx), axis=1))
        ddx = tf.reduce_mean(tf.square(ddx - 1.0) * scale)
        self.d_loss = self.d_loss + ddx
        self.d_adam = tf.train.AdamOptimizer(learning_rate=1e-4, beta1=0.5, beta2=0.9) \
                .minimize(self.d_loss, var_list=self.d_net.vars)
        self.g_adam = tf.train.AdamOptimizer(learning_rate=1e-4, beta1=0.5, beta2=0.9) \
                .minimize(self.g_loss, var_list=[self.g_net.vars, self.enc_net.vars])
        self.recon_loss = tf.reduce_mean(tf.abs(self.x - self.x_), 1)
        self.compute_grad = tf.gradients(self.recon_loss, self.z)
        self.saver = tf.train.Saver()
        run_config = tf.ConfigProto()
        run_config.gpu_options.per_process_gpu_memory_fraction = 1.0
        run_config.gpu_options.allow_growth = True
        self.sess = tf.Session(config=run_config)
#
    def train(self, num_batches=1000):
        L2_LOSS=[]
        now = datetime.datetime.now(dateutil.tz.tzlocal())
        timestamp = now.strftime('%Y_%m_%d_%H_%M_%S')
        batch_size = self.batch_size

        self.sess.run(tf.global_variables_initializer())
        start_time = time.time()
        print(
        'Training {} on {}, sampler = {}, z = {} dimension, beta_n = {}, beta_c = {}'.
            format(self.model, self.data, self.sampler, self.z_dim, self.beta_1, self.beta_2))
        for t in range(0, num_batches):
            d_iters = 5
            for _ in range(0, d_iters):
                bx = self.x_sampler.train(batch_size)
                bz = self.z_sampler(batch_size, self.z_dim, self.sampler, self.num_classes, self.n_cat)
                self.sess.run(self.d_adam, feed_dict={self.x: bx, self.z: bz})
            bz = self.z_sampler(batch_size, self.z_dim, self.sampler, self.num_classes, self.n_cat)
            self.sess.run(self.g_adam, feed_dict={self.z: bz})
            if (t+1) % 100 == 0:
                bx = self.x_sampler.train(batch_size)
                bz = self.z_sampler(batch_size, self.z_dim, self.sampler, self.num_classes, self.n_cat)

                d_loss = self.sess.run(
                    self.d_loss, feed_dict={self.x: bx, self.z: bz}
                )
                g_loss = self.sess.run(
                    self.g_loss, feed_dict={self.z: bz}
                )
                print('Iter [%8d] Time [%5.4f] d_loss [%.4f] g_loss [%.4f]' %
                      (t+1, time.time() - start_time, d_loss, g_loss))
                print(self.sess.run(tf.reduce_mean(self.d_), feed_dict={self.x: bx, self.z: bz}))
                print(self.sess.run(tf.reduce_mean(tf.square(self.z - self.z_enc)),feed_dict={self.z: bz}))
                print(self.sess.run(tf.reduce_mean(tf.norm(self.z_enc, ord=1)), feed_dict={self.z: bz}))
                print(self.sess.run(tf.reduce_mean(tf.convert_to_tensor(np.linalg.norm(self.z_enc_np, ord='nuc'))), feed_dict={self.z: bz}))
            L2_LOSS.append(self.sess.run(tf.reduce_mean(tf.square(self.z - self.z_enc)), feed_dict={self.z: bz}))
        self.recon_enc(timestamp, val=True)
        self.save(timestamp)

    def save(self, timestamp):

        checkpoint_dir = 'checkpoint_dir/{}/{}_{}_{}_z{}_beta1{}_beta2{}_beta3{}'.format(self.data, timestamp, self.model, self.sampler,
                                                                             self.z_dim, self.beta_1,self.beta_2,
                                                                             self.beta_3)

        if not os.path.exists(checkpoint_dir):
            os.makedirs(checkpoint_dir)

        self.saver.save(self.sess, os.path.join(checkpoint_dir, 'model.ckpt'))

    def load(self, pre_trained = False, timestamp = ''):

        if pre_trained == True:
            print('Loading Pre-trained Model...')
            checkpoint_dir = 'pre_trained_models/{}/{}_{}_z{}_beta1{}_beta2{}_beta3{}'.format(self.data, self.model, self.sampler,
                                                                            self.z_dim, self.beta_1,self.beta_2,
                                                                             self.beta_3)
        else:
            if timestamp == '':
                print('Best Timestamp not provided. Abort !')
                checkpoint_dir = ''
            else:
                checkpoint_dir = 'checkpoint_dir/{}/{}_{}_{}_z{}_beta1{}_beta2{}_beta3{}'.format(self.data, timestamp, self.model, self.sampler,
                                                                                     self.z_dim, self.beta_1,self.beta_2,
                                                                             self.beta_3)
        self.saver.restore(self.sess, os.path.join(checkpoint_dir, 'model.ckpt'))
        print('Restored model weights.')

    def _gen_samples(self, num_samples):

        batch_size = self.batch_size
        bz = self.z_sampler(batch_size, self.z_dim, self.sampler, self.num_classes, self.n_cat)
        fake_samples = self.sess.run(self.x_, feed_dict = {self.z : bz})

        for t in range(num_samples // batch_size):
            bz = self.z_sampler(batch_size, self.z_dim, self.sampler, self.num_classes, self.n_cat)
            samp = self.sess.run(self.x_, feed_dict = {self.z : bz})
            fake_samples = np.vstack((fake_samples, samp))

        print(' Generated {} samples .'.format(fake_samples.shape[0]))
    def recon_enc(self, timestamp, val = True):

        if val:
            data_recon, label_recon = self.x_sampler.validation()
        else:
            data_recon, label_recon = self.x_sampler.test()

        num_pts_to_plot = data_recon.shape[0]
        recon_batch_size = self.batch_size
        latent = np.zeros(shape=(num_pts_to_plot, self.z_dim))

        print('Data Shape = {}, Labels Shape = {}'.format(data_recon.shape, label_recon.shape))
        for b in range(int(np.ceil(num_pts_to_plot*1.0 / recon_batch_size))):

            if (b+1)*recon_batch_size > num_pts_to_plot:
               pt_indx = np.arange(b*recon_batch_size, num_pts_to_plot)
            else:
               pt_indx = np.arange(b*recon_batch_size, (b+1)*recon_batch_size)
            xtrue = data_recon[pt_indx, :]

            zhats_gen= self.sess.run(self.z_infer, feed_dict={self.x : xtrue})

            latent[pt_indx, :] = zhats_gen
        self._eval_cluster(latent, label_recon, timestamp, val)
        return latent


    def _eval_cluster(self, latent_rep, labels_true, timestamp, val):

        km = KMeans(n_clusters=14, random_state=0).fit(latent_rep)
        labels_pred = km.labels_
        purity = metric.compute_purity(labels_pred, labels_true)
        ari = adjusted_rand_score(labels_true, labels_pred)
        nmi = normalized_mutual_info_score(labels_true, labels_pred)
        # acc = metric.compute_acc(labels_true, labels_pred)
        acc , acc_r , ari_r, nmi_r= metric.err_rate(labels_true, labels_pred)
        if val:
            data_split = 'Validation'
        else:
            data_split = 'Test'
            np.savetxt("label_True.txt", labels_true)
            np.savetxt("label_Predicted.txt", labels_pred)

        print('Data = {}, Model = {}, sampler = {}, z_dim = {}, beta_1 = {}, beta_2 = {} , beta_3 = {}'
              .format(self.data, self.model, self.sampler, self.z_dim, self.beta_1, self.beta_2, self.beta_3))
        print(' #Points = {}, K = {}, Purity = {},  NMI = {}, ARI = {}, acc={}'
              .format(latent_rep.shape[0], self.num_classes, purity, nmi, ari, acc))

        with open('logs/Res_{}_{}.txt'.format(self.data, self.model), 'a+') as f:
                f.write('{}, {} : K = {}, z_dim = {}, beta_1 = {}, beta_2 = {} , beta_3 = {}, sampler = {}, Purity = {}, NMI = {}, ARI = {}\n'
                        .format(timestamp, data_split, self.num_classes, self.z_dim, self.beta_1, self.beta_2, self.beta_3,
                                self.sampler, purity, nmi, ari))
                f.flush()


if __name__ == '__main__':
    parser = argparse.ArgumentParser('')
    parser.add_argument('--data', type=str, default='10x_73k')
    parser.add_argument('--model', type=str, default='clus_wgan')
    parser.add_argument('--sampler', type=str, default='mix_gauss')
    parser.add_argument('--K', type=int, default=14)
    parser.add_argument('--dz', type=int, default=49)
    parser.add_argument('--bs', type=int, default=64)
    parser.add_argument('--beta1', type=float, default=10)
    parser.add_argument('--beta2', type=float, default=0.0001)
    parser.add_argument('--beta3', type=float, default=0.001)
    parser.add_argument('--timestamp', type=str, default='')
    parser.add_argument('--train', type=str, default='False')
    args = parser.parse_args()
    data = importlib.import_module(args.data)
    model = importlib.import_module(args.data + '.' + args.model)

    num_classes = args.K

    n_cat = 1
    batch_size = args.bs
    beta_1 = args.beta1
    beta_2 = args.beta2
    beta_3 = args.beta3
    timestamp = args.timestamp

    z_dim = args.dz
    d_net = model.Discriminator()
    g_net = model.Generator(z_dim=z_dim)
    enc_net = model.Encoder(z_dim=z_dim)
    xs = data.DataSampler()
    zs = util.sample_Z

    cl_gan = clusGAN(g_net, d_net, enc_net, xs, zs, args.data, args.model, args.sampler,
                     num_classes, n_cat, batch_size, beta_1,beta_2, beta_3)

    if args.train == 'True':
        cl_gan.train()
    else:

        print('Attempting to Restore Model ...')
        if timestamp == '':
            cl_gan.load(pre_trained=True)
            timestamp = 'pre-trained'
        else:
            cl_gan.load(pre_trained=False, timestamp = timestamp)

        latent = cl_gan.recon_enc(timestamp, val=False)
        # cl_gan._gen_samples(3600)

