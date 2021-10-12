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
    def __init__(self, input, n_visible=5, n_hidden=6, W=None, hbias=None, vbias=None) -> None:
        self.n_visible = n_visible
        self.n_hidden = n_hidden
        self.input = input
        if W is None:
            a = 1. / n_visible
            init_W = np.random.uniform(low=-a, high=a, size=(n_visible, n_hidden))
            self.W = init_W

        if hbias is None:
            self.hbias = np.random.randn(n_hidden)

        if vbias is None:
            self.vbias = np.random.randn(n_visible)

        self.sigma = np.eye(n_hidden) * 1

    def sigmoid(self, x):
        # print(x.shape)
        return (1 / (1 + np.exp(-x)))

    def forward(self, v):
        # print("Forward")
        return self.sigmoid((v @ self.W) + self.hbias)

    def backward(self, h):
        # print("Backward")
        # print(h.shape)
        # print(self.W.shape)
        return self.sigmoid((h @ self.W.T) + self.vbias)

    def contrastive_divergence(self, lr=0.1, k=1, input=None):
        if input is not None:
            self.input = input
        v_0 = self.input
        h_, h_0 = self.sample_h_given_v(self.input)
        for step in range(k):
            v_, v_k = self.sample_v_given_h(h_0)
            h_, h_k = self.sample_h_given_v(v_k)

        print(self.input.shape)
        print(h_k.shape)

        self.W += lr * (v_0 @ h_0 - v_k @ h_k).sum(axis=0)  # Division through feature count?
        self.vbias += lr * (v_0 - v_k).sum(axis=0)
        self.hbias += lr * (h_0 - h_k).sum(axis=0)

    def compute_reconstruction_loss(self):
        h_layer = self.forward(self.input)
        v_layer = self.backward(h_layer)

        binary_cross_entropy = -np.mean(np.sum(self.input * np.log(v_layer) + (1 - self.input) * np.log(1 - v_layer)))

        return binary_cross_entropy

    def reconstruct(self):
        reconstructed = self.backward(self.forward(self.input))
        return reconstructed

    def pass_through(self, h):
        v_new, v_sample = self.sample_v_given_h(h)
        h_new, h_sample = self.sample_h_given_v(v_sample)
        return v_new, v_sample, h_new, h_sample

    # def pass_through2(self, v):
    #     h_new, h_sample = self.sample_h_given_v(v)
    #     v_new, v_sample = self.sample_v_given_h(h)
    #     return v_new, v_sample, h_new, h_sample

    def sample_h_given_v(self, v):
        h_prob = self.forward(v)
        # randoms = np.random.randn(*new_h.shape)
        # sampled_h = new_h + randoms
        h_sampled = (np.random.uniform(size=h_prob.shape) > h_prob) * 1.0
        return [h_prob, h_sampled]

    def sample_v_given_h(self, h):
        v_prob = self.backward(h)
        v_sampled = np.random.binomial(
            size=v_prob.shape,  # discrete: binomial
            n=1,
            p=v_prob)

        return [v_prob, v_sampled]


def test_rbm(learning_rate=0.1, k=1, training_epochs=1000):
    data = np.array([[1, 1, 1, 0, 0, 0], [1, 0, 1, 0, 0, 0], [1, 1, 1, 0, 0, 0], [0, 0, 1, 1, 1, 0], [0, 0, 1, 1, 0, 0],
                     [0, 0, 1, 1, 1, 0]])

    # construct RBM
    rbm = RBM(input=data, n_visible=6, n_hidden=2)

    # train
    for epoch in range(training_epochs):
        rbm.contrastive_divergence(lr=learning_rate, k=k)
        cost = rbm.get_reconstruction_cross_entropy()
        print(f'Training epoch {epoch}, cost is {cost}')

    # test
    v = np.array([[0, 0, 0, 1, 1, 0], [1, 1, 0, 0, 0, 0]])

    print(rbm.reconstruct(v))


test_rbm()
# %%