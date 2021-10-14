# %%
import matplotlib.pyplot as plt
import numpy as np


def init_parameters(n_visible, n_hidden):
    weights = np.asarray(
        np.random.uniform(low=-0.1 * np.sqrt(6. / (n_hidden + n_visible)),
                          high=0.1 * np.sqrt(6. / (n_hidden + n_visible)),
                          size=(n_visible, n_hidden)))
    hbias = np.random.randn(n_hidden)
    vbias = np.random.randn(n_visible)
    return weights, vbias, hbias


def sigmoid(x):
    return (1 / (1 + np.exp(-x)))


def ph_given_v(v, W, hbias):
    return sigmoid((v @ W) + hbias)


def pv_given_h(h, W, vbias):
    return sigmoid((h @ W.T) + vbias)


def sample_h_given_v(v, W, hbias):
    # It is very important to make these hidden states binary, rather than using the probabilities themselves.
    h_prob = ph_given_v(v, W, hbias)
    h_sampled = np.random.binomial(
        size=h_prob.shape,  # discrete: binomial
        n=1,
        p=h_prob)
    return [h_prob, h_sampled]


def sample_v_given_h(h, W, vbias):
    # It is common to use the probability, pi, instead of sampling a binary value.
    # This is not nearly as problematic as using probabilities for the data-driven hidden states
    # and it reduces sampling noise thus allowing faster learning.
    # There is some evidence that it leads to slightly worse density models.
    v_prob = pv_given_h(h, W, vbias)
    v_sampled = np.random.binomial(
        size=v_prob.shape,  # discrete: binomial
        n=1,
        p=v_prob)

    return [v_prob, v_sampled]


def contrastive_divergence(data, W, vbias, hbias, lr=0.1, k=1):
    num_examples = data.shape[0]
    pv_0 = data
    ph_0, h_0 = sample_h_given_v(pv_0, W, hbias)
    h_k = h_0
    for _ in range(k):
        # For the last update of the hidden units, it is silly to use stochastic binary states because nothing depends on which state is chosen.
        # So use the probability itself to avoid unnecessary sampling noise.
        # When using CDn, only the final update of the hidden units should use the probability
        pv_k, v_k = sample_v_given_h(h_k, W, vbias)
        ph_k, h_k = sample_h_given_v(v_k, W, hbias)

    # Expectation of (pi x pj) data where pi is the visible unit probability and pj the hidden unit probability (Could also be binary hence E[pi, hj]).
    # Outer product to get each h and v node combination multiplied
    pvh_data = (pv_0.T @ ph_0)
    pvh_recs = (pv_k.T @ ph_k)

    # Division to get expected joint probability of all notes. In Hinton's words: average, per-case gradient computed on a mini-batch
    joint_p_vh = (pvh_data - pvh_recs) / num_examples
    W += lr * (joint_p_vh)
    vbias += lr * ((pv_0 - pv_k).sum(axis=0) / num_examples)
    hbias += lr * ((ph_0 - ph_k).sum(axis=0) / num_examples)

    return W, vbias, hbias


def compute_reconstruction_cross_entropy(v0, W, vbias, hbias):
    ph = ph_given_v(v0, W, hbias)
    pv = pv_given_h(ph, W, vbias)

    binary_cross_entropy = -np.mean(np.sum(v0 * np.log(pv) + (1 - v0) * np.log(1 - pv)))

    return binary_cross_entropy


def reconstruct(v, W, vbias, hbias):
    ph, h = sample_h_given_v(v, W, hbias)
    pv, v = sample_v_given_h(h, W, vbias)
    reconstructed = pv
    return reconstructed


def run_train_loop(data, W, vbias, hbias, epochs=1000, lr=1, k=10):
    for epoch in range(epochs):
        W, vbias, hbias = contrastive_divergence(data, W, vbias, hbias, lr, k)
        cost = compute_reconstruction_cross_entropy(data, W, vbias, hbias)
        if (epoch % (epochs // 10)) == 0:
            print(f'Training epoch {epoch}, cost is {cost}')
    return W, vbias, hbias


def daydream(n_visible, W, vbias, hbias, num_samples=10):
    # Create a matrix, where each row is to be a sample of of the visible units
    # (with an extra bias unit), initialized to all ones.
    samples = np.ones((num_samples, n_visible))

    # Take the first sample from a uniform distribution.
    samples[0] = np.random.rand(n_visible)

    v = samples[0, :]
    for i in range(1, num_samples):
        v0 = samples[i - 1, :]
        ph_0, h_0 = sample_h_given_v(v0, W, hbias)
        pv, vk = sample_v_given_h(h_0, W, vbias)
        samples[i, :] = vk

    return samples


data = np.array([[1, 1, 1, 0, 0, 0], [1, 0, 1, 0, 0, 0], [1, 1, 1, 0, 0, 0], [0, 0, 1, 1, 1, 0], [0, 0, 1, 1, 0, 0],
                 [0, 0, 1, 1, 1, 0], [0, 0, 1, 0, 1, 0], [1, 0, 1, 0, 0, 0], [0, 0, 1, 1, 0, 0]])

n_visible = data.shape[1]
n_hidden = 2
num_examples = data.shape[0]
lr = 1
epochs = 2000
k = 2
weights, vbias, hbias = init_parameters(n_visible, n_hidden)

trained_weights, trained_vbias, trained_hbias = run_train_loop(data, weights, vbias, hbias, epochs=epochs, lr=lr, k=k)
# %%
v = np.array([[0, 0, 0, 1, 1, 0], [1, 1, 0, 0, 0, 0], [1, 1, 0, 0, 0, 1], [0, 0, 1, 0, 1, 0]])
print("Test:")
print(v)
print("Reconstruction:")
print(reconstruct(v, trained_weights, trained_vbias, trained_hbias).round(decimals=2))
# %%