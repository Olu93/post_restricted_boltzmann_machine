# %%
# https://rubikscode.net/2018/10/22/implementing-restricted-boltzmann-machine-with-python-and-tensorflow/
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
import os
import tensorflow_datasets as tfds

ds = tfds.load('mnist', split='train', shuffle_files=True)

# %%
np.random.seed(42)
ds


# %%
# https://gist.github.com/yusugomori/4428308
# https://towardsdatascience.com/restricted-boltzmann-machine-as-a-recommendation-system-for-movie-review-part-2-9a6cab91d85b
# https://medium.com/machine-learning-researcher/boltzmann-machine-c2ce76d94da5
# https://github.com/echen/restricted-boltzmann-machines/blob/master/rbm.py
# https://datascience.stackexchange.com/questions/30186/understanding-contrastive-divergence
# https://www.edureka.co/blog/restricted-boltzmann-machine-tutorial/#training
class RBM:
    def __init__(
        self,
        input,
        n_visible=5,
        n_hidden=6,
        W=None,
        hbias=None,
        vbias=None,
        lr=1,
        epochs=1000,
        k=10,
    ) -> None:
        self.n_visible = n_visible
        self.n_hidden = n_hidden
        self.num_examples = input.shape[0]
        self.lr = lr
        self.epochs = epochs
        self.k = k
        # Insert bias units of 1 into the first column.
        # data = np.insert(data, 0, 1, axis = 1)
        self.input = input
        if W is None:
            # Xavier Glorot and Yoshua Bengio
            init_W = np.asarray(
                np.random.uniform(low=-0.1 * np.sqrt(6. / (n_hidden + n_visible)),
                                  high=0.1 * np.sqrt(6. / (n_hidden + n_visible)),
                                  size=(n_visible, n_hidden)))
            self.W = init_W

        if hbias is None:
            # Initial hidden biases of 0 are usually fine. 
            self.hbias = np.random.randn(n_hidden)

        if vbias is None:
            self.vbias = np.random.randn(n_visible)

        self.collector_Ws = []
        self.collector_hs = []
        self.collector_vs = []
        self.collector_minibatch = []

    def sigmoid(self, x):
        # print(x.shape)
        return (1 / (1 + np.exp(-x)))

    def ph_given_v(self, v):
        return self.sigmoid((v @ self.W) + self.hbias)

    def pv_given_h(self, h):
        return self.sigmoid((h @ self.W.T) + self.vbias)

    def contrastive_divergence(self, lr=0.1, k=1, input=None):
        if input is not None:
            self.input = input
        pv_0 = self.input
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
        joint_p_vh = (pvh_data - pvh_recs) / self.num_examples
        self.W += lr * (joint_p_vh)
        self.vbias += lr * ((pv_0 - pv_k).sum(axis=0) / self.num_examples)
        self.hbias += lr * ((ph_0 - ph_k).sum(axis=0) / self.num_examples)

        self.collector_Ws.extend((lr * (joint_p_vh)).flatten())
        self.collector_vs.extend((lr * ((pv_0 - pv_k).sum(axis=0) / self.num_examples)).flatten())
        self.collector_hs.extend((lr * ((ph_0 - ph_k).sum(axis=0) / self.num_examples)).flatten())
        self.collector_minibatch.append(ph_k)

    def compute_reconstruction_cross_entropy(self):
        ph = self.ph_given_v(self.input)
        pv = self.pv_given_h(ph)

        binary_cross_entropy = -np.mean(np.sum(self.input * np.log(pv) + (1 - self.input) * np.log(1 - pv)))

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

    def run_train_loop(self, epochs=None, lr=None, k=None):
        for epoch in range(epochs or self.epochs):
            self.contrastive_divergence(lr=lr or self.lr, k=k or self.k)
            cost = self.compute_reconstruction_cross_entropy()
            if (epoch % (epochs // 10)) == 0:
                print(f'Training epoch {epoch}, cost is {cost}')

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

        return samples

    def plot_histograms(self):
        fig, axes = plt.subplots(3,1, figsize=(15,10))
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

    def plot_minibatch_probs(self, index = -1):
        fig, axes = plt.subplots(1,1, figsize=(15,10))
        data = self.collector_minibatch[index].T
        y, x = data.shape
        axes.imshow(data, cmap='gray', vmin=0, vmax=1, origin='lower')
        axes.set_xlabel('Data Point')
        axes.set_ylabel('Hidden Unit Activation')
        axes.set_xticks(range(x))
        axes.set_yticks(range(y))
        axes.set_xticklabels(range(1,x+1))
        axes.set_yticklabels(range(1,y+1))
        fig.tight_layout()
        plt.show()


def test_rbm(lr=1, k=5, epochs=2000):
    data = np.array([[1, 1, 1, 0, 0, 0], [1, 0, 1, 0, 0, 0], [1, 1, 1, 0, 0, 0], [0, 0, 1, 1, 1, 0], [0, 0, 1, 1, 0, 0],
                     [0, 0, 1, 1, 1, 0], [0, 0, 1, 0, 1, 0], [1, 0, 1, 0, 0, 0], [0, 0, 1, 1, 0, 0]])

    # construct RBM
    rbm = RBM(input=data, n_visible=6, n_hidden=4)

    # train
    rbm.run_train_loop(epochs, lr, k)

    # test

    print("Data:")
    print(data)

    return rbm






rbm = test_rbm(lr=1, epochs=2000, k=20)
# %%
v = np.array([[0, 0, 0, 1, 1, 0], [1, 1, 0, 0, 0, 0], [1, 1, 0, 0, 0, 1], [0, 0, 1, 0, 1, 0]])
print("Test:")
print(v)
print("Reconstruction:")
print(rbm.reconstruct(v).round(decimals=2))

# %%
rbm.plot_minibatch_probs(3)
# %%
rbm.plot_minibatch_probs(10)
# %%
rbm.plot_minibatch_probs(50)
# %%
rbm.plot_minibatch_probs()
