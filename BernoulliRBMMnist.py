# %%
# https://rubikscode.net/2018/10/22/implementing-restricted-boltzmann-machine-with-python-and-tensorflow/
import matplotlib.pyplot as plt
import numpy as np
import random
# import tensorflow as tf
import matplotlib.image as mpimg
import imageio
import tqdm
from datasets import load_dataset, Dataset
from IPython.display import Image, display
# import tensorflow_datasets as tfds
np.random.seed(42)
dataset = load_dataset('mnist', split='train', streaming=0)
ds = dataset.shuffle().train_test_split(test_size=0.5)
train_data, val_data = ds['train'], ds['test']
train_data

# %%
# ds = ds.shuffle(1024).batch(32).prefetch(tf.data.experimental.AUTOTUNE)


# %%
class RBM:
    def __init__(
        self,
        input: Dataset,
        n_hidden=6,
        W=None,
        hbias=None,
        vbias=None,
        lr=1,
        epochs=1000,
        k=10,
        batch_size=32,
    ) -> None:
        self.n_hidden = n_hidden
        self.lr = lr
        self.epochs = epochs
        self.k = k
        self.batch_size = batch_size
        # Insert bias units of 1 into the first column.
        # data = np.insert(data, 0, 1, axis = 1)
        self.input = np.array(input).reshape((len(input), -1)) / 255
        # self.input = self.input[:20000]
        self.n_full_data, self.n_visible = self.input.shape
        if W is None:
            # Xavier Glorot and Yoshua Bengio
            init_W = np.asarray(
                np.random.uniform(low=-0.1 * np.sqrt(6. / (n_hidden + self.n_visible)),
                                  high=0.1 * np.sqrt(6. / (n_hidden + self.n_visible)),
                                  size=(self.n_visible, n_hidden)))
            self.W = init_W

        if hbias is None:
            # Initial hidden biases of 0 are usually fine.
            self.hbias = np.random.randn(n_hidden)

        if vbias is None:
            self.vbias = np.random.randn(self.n_visible)

        self.collector_Ws = []
        self.collector_hs = []
        self.collector_vs = []
        self.collector_minibatch = []
        self.collector_reconstruction = np.zeros((3, self.epochs + 1, self.n_visible))

    def sigmoid(self, x):
        # print(x.shape)
        return (1 / (1 + np.exp(-x)))

    def ph_given_v(self, v):
        return self.sigmoid((v @ self.W) + self.hbias)

    def pv_given_h(self, h):
        return self.sigmoid((h @ self.W.T) + self.vbias)

    def contrastive_divergence(self, batch, lr=0.1, k=1, input=None):

        pv_0 = batch
        num_examples = batch.shape[0]
        ph_0, h_0 = self.sample_h_given_v(pv_0)
        h_k = h_0
        for step in range(k):
            # For the last update of the hidden units, it is silly to use stochastic binary states because nothing depends on which state is chosen.
            # So use the probability itself to avoid unnecessary sampling noise.
            # When using CDn, only the final update of the hidden units should use the probability
            pv_k, v_k = self.sample_v_given_h(h_k)
            ph_k, h_k = self.sample_h_given_v(v_k)

        # Expectation of (pi x pj) data where pi is the visible unit probability and pj the hidden unit probability (Could also be binary hence E[pi, hj]).
        # Outer product to get each h and v node combination multiplied
        pvh_data = (pv_0.T @ ph_0)
        pvh_recs = (pv_k.T @ ph_k)

        # Division to get expected joint probability of all notes. In Hinton's words: average, per-case gradient computed on a mini-batch
        joint_p_vh = (pvh_data - pvh_recs) / num_examples
        self.W += lr * (joint_p_vh)
        self.vbias += lr * ((pv_0 - pv_k).sum(axis=0) / num_examples)
        self.hbias += lr * ((ph_0 - ph_k).sum(axis=0) / num_examples)

        self.collector_Ws.extend((lr * (joint_p_vh)).flatten())
        self.collector_vs.extend((lr * ((pv_0 - pv_k).sum(axis=0) / num_examples)).flatten())
        self.collector_hs.extend((lr * ((ph_0 - ph_k).sum(axis=0) / num_examples)).flatten())
        self.collector_minibatch.append(ph_k)

    def compute_reconstruction_cross_entropy(self, batch):
        ph = self.ph_given_v(batch)
        pv = self.pv_given_h(ph)

        binary_cross_entropy = -np.mean(np.sum(batch * np.log(pv) + (1 - batch) * np.log(1 - pv)))

        return binary_cross_entropy

    def reconstruct(self, v):
        ph, h = self.sample_h_given_v(v)
        pv, v = self.sample_v_given_h(h)
        reconstructed = pv
        return reconstructed

    def sample_h_given_v(self, v):
        # It is very important to make these hidden states binary, rather than using the probabilities themselves.
        h_prob = self.ph_given_v(v)
        h_sampled = np.random.binomial(
            size=h_prob.shape,  # discrete: binomial
            n=1,
            p=h_prob)
        return [h_prob, h_sampled]

    def sample_v_given_h(self, h):
        # It is common to use the probability, pi, instead of sampling a binary value.
        # This is not nearly as problematic as using probabilities for the data-driven hidden states
        # and it reduces sampling noise thus allowing faster learning.
        # There is some evidence that it leads to slightly worse density models.
        v_prob = self.pv_given_h(h)
        v_sampled = np.random.binomial(
            size=v_prob.shape,  # discrete: binomial
            n=1,
            p=v_prob)

        return [v_prob, v_sampled]

    def run_train_loop(self, lr=None, k=None):
        num_batches = self.n_full_data // self.batch_size
        epochs = self.epochs
        total_runs = num_batches * self.epochs

        i = 0
        print(f"Amount of total runs is {total_runs}")
        progress_bar = tqdm.tqdm(total=total_runs)
        selected1 = self.input[random.randint(0, len(self.input) - 1)]
        selected2 = self.input[random.randint(0, len(self.input) - 1)]
        selected3 = self.input[random.randint(0, len(self.input) - 1)]
        self.collector_reconstruction[0] = selected1
        self.collector_reconstruction[1] = selected2
        self.collector_reconstruction[2] = selected3
        for epoch in range(epochs):
            for batch in range(num_batches):
                start = batch * self.batch_size
                end = start + self.batch_size
                batch_subset = self.input[start:min([len(self.input), end])]
                self.contrastive_divergence(batch_subset, lr=lr or self.lr, k=k or self.k)
                cost = self.compute_reconstruction_cross_entropy(batch_subset)
                i += 1
                progress_bar.update(1)
                if ((i % 10) == 0):
                    progress_bar.set_description(f"Epoch {epoch:2d}: Cost is {cost:.2f}")
            reconstructed1 = self.reconstruct(selected1)
            reconstructed2 = self.reconstruct(selected2)
            reconstructed3 = self.reconstruct(selected3)
            self.collector_reconstruction[0, epoch + 1] = reconstructed1
            self.collector_reconstruction[1, epoch + 1] = reconstructed2
            self.collector_reconstruction[2, epoch + 1] = reconstructed3

    def daydream(self, num_samples=10):
        # Create a matrix, where each row is to be a sample of of the visible units
        # (with an extra bias unit), initialized to all ones.
        samples = np.ones((num_samples, self.n_visible))

        # Take the first sample from a uniform distribution.
        samples[0] = np.random.rand(self.n_visible)

        v = samples[0, :]
        for i in range(1, num_samples):
            v0 = samples[i - 1, :]
            ph_0, h_0 = self.sample_h_given_v(v0)
            pv, vk = self.sample_v_given_h(h_0)
            samples[i, :] = vk

        return samples.reshape((-1, 28, 28, 1))

    def plot_histograms(self):
        fig, axes = plt.subplots(3, 1, figsize=(15, 10))
        ax = axes[0]
        ax.hist(self.collector_Ws, bins=25)
        ax.set_title('Weights')
        ax = axes[1]
        ax.hist(self.collector_hs, bins=25)
        ax.set_title('Hidden Bias')
        ax = axes[2]
        ax.hist(self.collector_vs, bins=25)
        ax.set_title('Visible Bias')
        fig.tight_layout()
        plt.show()

    def plot_minibatch_probs(self, index=-1):
        fig, axes = plt.subplots(1, 1, figsize=(15, 10))
        data = self.collector_minibatch[index].T
        y, x = data.shape
        axes.imshow(data, cmap='gray', vmin=0, vmax=1, origin='lower')
        axes.set_xlabel('Data Point')
        axes.set_ylabel('Hidden Unit Activation')
        axes.set_xticks(range(x))
        axes.set_yticks(range(y))
        axes.set_xticklabels(range(1, x + 1))
        axes.set_yticklabels(range(1, y + 1))
        fig.tight_layout()
        plt.show()

    def plot_reconstruction(self, sample=0, index=-1):
        fig, axes = plt.subplots(1, 2, figsize=(15, 10))
        orig = self.collector_reconstruction[sample, 0]
        recon = self.collector_reconstruction[sample, index]
        axes[0].imshow(orig.reshape((28, 28)), cmap='gray', vmin=0, vmax=1)
        axes[1].imshow(recon.reshape((28, 28)), cmap='gray', vmin=0, vmax=1)
        fig.tight_layout()
        plt.show()

    def plot_all_reconstructions(self, sample=0):
        epochs = self.collector_reconstruction.shape[1]
        fig, axes = plt.subplots(1, epochs, figsize=(15, 10))
        for index in range(epochs):
            data = self.collector_reconstruction[sample, index]
            axes[index].imshow(data.reshape((28, 28)), cmap='gray', vmin=0, vmax=1)
            fig.tight_layout()
        plt.show()

    def plot_daydream(self, num_samples=10, filename='./dream.gif'):
        dream = (self.daydream(num_samples) * 255).astype(int)
        with imageio.get_writer(filename, mode='I') as writer:
            for data in dream:
                writer.append_data(data)

        return Image(filename)


rbm = RBM(input=train_data['image'], epochs=15, n_hidden=30, batch_size=16)
# train
rbm.run_train_loop(0.1, 10)
# %%
rbm.plot_reconstruction()
# %%
rbm.plot_all_reconstructions(0)
# %%
rbm.plot_all_reconstructions(1)
# %%
rbm.plot_all_reconstructions(2)
# %%
rbm.plot_daydream(30)
# %%
rbm.plot_minibatch_probs()
# %%
# rbm.plot_histograms()
# %%
