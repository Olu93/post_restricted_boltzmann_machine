# %% [markdown]
# # Bernoulli RBM
# This notebook will show my implementation of the BernoulliRBM. I devided everything up into dedicated functions
# and I will try to reference Hinton's [A Practical Guide to Training Restricted Boltzmann Machines](https://www.cs.toronto.edu/~hinton/absps/guideTR.pdf).
# I was also inspired by code and blogs from following ressources:
# - [Amir Ali's comprehensive article](https://medium.com/machine-learning-researcher/boltzmann-machine-c2ce76d94da5)
# - [Luke Sun's very helpful Blog article](https://towardsdatascience.com/restricted-boltzmann-machine-as-a-recommendation-system-for-movie-review-part-2-9a6cab91d85b)
# - [Yusugomori's implementation](https://gist.github.com/yusugomori/4428308)
# - [Echen's implementation](https://github.com/echen/restricted-boltzmann-machines/blob/master/rbm.py)
# 
# For everyone lacking the basic understanding of RBM's I advice strongly to checkout Amir Ali's blog
# and for the implementation Luke Sun's article. The code examples help to understand how to properly structure the code.
# Both are fairly different because there are a lot of slightly different implementations due to the design choices that one can make according to Hinton.
#
# %% [markdown]
# ## The dataset
# I am starting with the data I want to train on. It's quite simple and just 6 binary values per data point.
# What's important here, is to notice that the 3rd and the 6th value for each data point is 1 and 0 respectively.
# Why is that important? Because that means that the reconstruction should always reconstruct samples that show the same trait.
# View it as some "binary-code" which we corrupt in the test data and that needs to be reconstructed again.
# Another example would be an image with a whole which we impaint upon reconstruction.
# %%
import matplotlib.pyplot as plt
import numpy as np

data = np.array([[1, 1, 1, 0, 0, 0], [1, 0, 1, 0, 0, 0], [1, 1, 1, 0, 0, 0], [0, 0, 1, 1, 1, 0], [0, 0, 1, 1, 0, 0],
                 [0, 0, 1, 1, 1, 0], [0, 0, 1, 0, 1, 0], [1, 0, 1, 0, 0, 0], [0, 0, 1, 1, 0, 0]])
data
# %% [markdown]
# ## The hyperparameters
# Next I define some structural parameters of the RBM like the number of visible and hidden nodes.
# I also define hyper parameters learning rate, number of training epochs and sampling rounds *k*.
# k is important for the gibbs-sampling.
# %%
n_visible = data.shape[1]
n_hidden = 2
num_examples = data.shape[0]
lr = 1
epochs = 2000
k = 2


# %% [markdown]
# ## Initializing the parameters of the model
# I wrote a function to initialize the weight matrix W, the visible bias vector v and the hidden bias vector h.
# %%
def init_parameters(n_visible, n_hidden):
    weights = np.asarray(
        np.random.uniform(low=-0.1 * np.sqrt(6. / (n_hidden + n_visible)),
                          high=0.1 * np.sqrt(6. / (n_hidden + n_visible)),
                          size=(n_visible, n_hidden)))
    hbias = np.random.randn(n_hidden)
    vbias = np.random.randn(n_visible)
    return weights, vbias, hbias


# %% [markdown]
# ## Base functions
# The sigmoid function is the non-linearity for each node.
# %%
def sigmoid(x):
    return (1 / (1 + np.exp(-x)))


# %% [markdown]
# We can use the sigmoid to now compute the probabilities
#
# [](./res/sample_ph)
#
# and
#
# [](./res/sample_pv)
#
# %%
def ph_given_v(v, W, hbias):
    return sigmoid((v @ W) + hbias)


def pv_given_h(h, W, vbias):
    return sigmoid((h @ W.T) + vbias)


# %% [markdown]
# Now it becomes necessary to use these probabilities to sample new versions of h as well as v.
# The simplest and most stable way is to use Bernoulli sampling. But Hinton also describes other possible sampling methods.
# An easy way is to generate uniformly distributed random numbers and every time they are lower than the respective element in the probability matrix you say it's a 1.
# Or... you just use a predefined numpy function which does exactly that for you.
#
# **Extra:** Maybe one additional point on why the hidden states need to be binary as well.
# There's no real reason other than it helps generalizing strongly and is very very stable - hence, easy to train and code.
# You can in theory sample other things like multiple binaries, real numbers or discrete numbers but that quickly becomes more complicated.
# If we were to pass the probabilities directly back into the network we don't add any probablistic component to our model.
#


# %%
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


# %% [markdown]
# ## How the contrastive divergence works
# I've long been thinking how to approach the next section which will talk about contrastive divergence.
# It's by far the most complicated concept.
#
# Having discussed some of the more fundamental functions, I will now take a more top-down approach to explaining the next few functions.
# Starting with the training loop. Essentially consisting of all the structural parameters.
# Per epoch we will run one contrastive divergence step. An epoch can be further devided into mini-batches, if necessary.
#


# %%
def run_train_loop(data, W, vbias, hbias, epochs=1000, lr=1, k=10):
    for epoch in range(epochs):
        W, vbias, hbias = contrastive_divergence(data, W, vbias, hbias, lr, k)
        cost = compute_reconstruction_cross_entropy(data, W, vbias, hbias)
        if (epoch % (epochs // 10)) == 0:
            print(f'Training epoch {epoch}, cost is {cost}')
    return W, vbias, hbias


# %% [markdown]
# ### Contrastive Divergences
# How does the contrastive step look like?
# It's quite simple: First you gibbs sample, then you train. Let's have a close look, though.
#


# %%
def contrastive_divergence(data, W, vbias, hbias, lr=0.1, k=1):
    pv_0, ph_0, pv_k, ph_k = gibb_sample(data, W, vbias, hbias, k)
    W, vbias, hbias = train_params(W, vbias, hbias, lr, pv_0, ph_0, pv_k, ph_k)
    return W, vbias, hbias


# %% [markdown]
# ### Gibbs sampling
# In order to approximate the gradient we need to have the expectation of the joint probability of the data ($<h,v>_{data}$)and one for the model ($<h,v>_{model}$).
# For the model we could start with a random v state of binary values ([1,0,1,0,1,0]) and run it through the network until nothing changes anymore.
# Which means the energy is quite low. OR, we use gibbs sampling. Gibb's sampling entails two key ideas.
# First, we start from a given high-dimensional example $x \in R^d$ of the data. Second, we use conditional probabilities to sample the next dimensions for the next data point using what we know about the example and the distribution as a whole.
#
# The second part is quite unimportant for our case.
# Why? Because every node in v is independent from all other v and the same holds for h.
# And each node represent one dimension in our example $x$. Hence, we don't need nasty conditionals to sample.
# We only require the probabilities that are given to us by a forward pass or backward pass through the model.
#
# That makes the *first* part quite important for this gibbs sampling approach.
# We choose a data point and pass it through the network in order to get probablities.
# We can use these probabilities with Bernoulli sampling as it is the most simple way to probablistically decide whether a dimension fires or not.
# And we can theoretically do this an infinite amount of times to get the $<h,v>_{model}$,
# but we do it just k-times to just get an approximation, which we call $<h_k,v_k>_{recon}$.
#
# This is what you see in the implementation. $p(v_0)$ is our data which we use to compute $p(h_0)$ but also sample $h_0$ which is a binary vector.
# Now for k times we use v_k and h_k to retrieve our $<h_k,v_k>_{recon}$.
# we return the initial and new values.
#
#
# **EXTRA:** Not sure if I've should have mentioned that the code uses a data matrix not single data points.
# Hence, my explanation might talk about individual data points but in the implementation I am doing things in bulk.


# %%
def gibb_sample(data, W, vbias, hbias, k):
    pv_0, v_0 = data, data
    ph_0, h_0 = sample_h_given_v(pv_0, W, hbias)
    h_k = h_0
    for _ in range(k):
        # For the last update of the hidden units, it is silly to use stochastic binary states because nothing depends on which state is chosen.
        # So use the probability itself to avoid unnecessary sampling noise.
        # When using CDn, only the final update of the hidden units should use the probability
        pv_k, v_k = sample_v_given_h(h_k, W, vbias)
        ph_k, h_k = sample_h_given_v(v_k, W, hbias)
    return pv_0, ph_0, pv_k, ph_k


# %% [markdown]
# ### Parameter update
# Uff, now onto the second complicated part. The training.
# You've probably seen two formulas:
# - [](./res/energy_function.png)
#
# and
#
# - [](./res/weight_update.png)
#
# To be honest, I don't entirely get how they come to this result but most examples online don't attempt to explain why.
# The first is quite reasonable but nowhere found in the implementation. But that's fine. We need the derivative.
#
# Hence, we have to compute $<v_i, h_j>$. Meaning we need the expectation of the joint probability of these values.
# We get the joint probability by computing the probability of a visible nodes activation $p(v=1)$ with the probability of a hidden nodes activation $p(h=1)$.
# However, as each hidden node depends on multiple visible nodes we have to sum these probabilities.
# This is done automatically by the network but this is why we often see p(H_j=1|v) in some algorithmic descriptions. (Or so I believe...)
#
# **EXTRA:** I know this is kinda confusing and it was for me, too. However, you can think of it differently.
# Assume you have 2 v-nodes and 2 h-nodes, then what we want is essentially a table like this
#
# |       | p(h1\|v) | p(h2\|v) |
# |-------|----------|----------|
# | **p(v1)** | p(v1,h1) | p(v1,h2) |
# | **p(v2)** | p(v2,h1) | p(v2,h2) |
#
# This table is equivalent with multiplying each dimension of v with each dimension of h.
# Great these are probabilites. How do we get the expectations?
# Well, this is kinda already the expectation value we need. It's quite complicated to explain why that is, though.
# The expectation formula would require us to look into each configuration [(0,0), (0,1), (1,0) and (1,1)] and compute their probabilities and sum them up in a fairly complicated way.
# However, I think because we are just interested in one configuration we can skip these complicated parts.
#
# Now back to business: With matrix computations we can greatly speed up the computation using the outer porduct of v and h.
# If we do things in bulk, we also have to devide buy the amount of data.
#
# The rules for the bias terms follow a similar pattern of reasoning.


# %%
def train_params(W, vbias, hbias, lr, pv_0, ph_0, pv_k, ph_k):
    num_examples = pv_0.shape[0]
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


# %% [markdown]
# ### How to compute the loss
# Lastly, it becomes important to also compute the loss itself to see how well the learning works.
# It's a pretty simple binary cross entropy loss.
#
# %%
def compute_reconstruction_cross_entropy(v0, W, vbias, hbias):
    ph = ph_given_v(v0, W, hbias)
    pv = pv_given_h(ph, W, vbias)

    binary_cross_entropy = -np.mean(np.sum(v0 * np.log(pv) + (1 - v0) * np.log(1 - pv)))

    return binary_cross_entropy


# %%[markdown]
# ## Running the code to train the model
# Putting everything together we only have to run our functions.
# %%
n_visible = data.shape[1]
n_hidden = 2
num_examples = data.shape[0]
lr = 1
epochs = 2000
k = 2
weights, vbias, hbias = init_parameters(n_visible, n_hidden)

trained_weights, trained_vbias, trained_hbias = run_train_loop(data, weights, vbias, hbias, epochs=epochs, lr=lr, k=k)

# %% [markdown]
# ## How to reconstruct values
# A simple reconstruction becomes trivial.

# %%
def reconstruct(v, W, vbias, hbias):
    ph, h = sample_h_given_v(v, W, hbias)
    pv, v = sample_v_given_h(h, W, vbias)
    reconstructed = pv
    return reconstructed


# %%[markdown]
# ## Testing the model
# Lastly, we want to test our model. 
# We just constuct some additional examples and run them through the reconstruction function.

# %%
v = np.array([[0, 0, 0, 1, 1, 0], [1, 1, 0, 0, 0, 0], [1, 1, 0, 0, 0, 1], [0, 0, 1, 0, 1, 0]])
print("Test:")
print(v)
print("Reconstruction:")
print(reconstruct(v, trained_weights, trained_vbias, trained_hbias).round(decimals=2))
# %%[markdown]
# ## Famous last words
# You should see how the 3 and 6 value are indeed reconstructed.
# For the remaining values we could in theory Bernoulli-sample them again to produce a new data point from the joint distribution that we learned.
# That's all there is to know. I also have variants of this computation, but they are implemented in an object oriented way.
# One of them computes the same RBM with the MNIST dataset.
