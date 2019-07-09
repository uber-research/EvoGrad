# EvoGrad

EvoGrad is a lightweight tool for differentiating through expectation,
built on top of PyTorch.

Tools that enable fast and flexible experimentation democratize and accelerate machine learning research.
However, one field that so far has not been greatly impacted by automatic differentiation tools is evolutionary computation
The reason is that most evolutionary algorithms are gradient-free:
they do not follow any explicit mathematical gradient (i.e., the mathematically optimal local direction of improvement), and instead proceed through a generate-and-test heuristic.
In other words, they create new variants, test them out, and keep the best.

Recent and exciting research in evolutionary algorithms for deep reinforcement learning, however, has highlighted how a specific class of evolutionary algorithms can benefit from auto-differentiation.
Work from OpenAI demonstrated that a form of Natural Evolution Strategies (NES) is massively scalable, and competitive with modern deep reinforcement learning algorithms.

EvoGrad enables fast prototyping of NES-like algorithms.
We believe there are many interesting algorithms yet to be discovered in this vein, and we hope this library will help to catalyze progress in the machine learning community.

## Examples
### Natural Evolution Strategies
As a first example, we’ll implement the simplified NES algorithm of [Salimans et al. (2017)](https://openai.com/blog/evolution-strategies/) in EvoGrad.
EvoGrad provides several probability distributions which may be used in the expectation function.
We will use a normal distribution because it is the most common choice in practice.

Let’s consider the problem of finding a fitness peak in a simple 1-D search space.

We can define our population distribution over this search space to be initially centered at 1.0, with a fixed variance of 0.05, with the following Python code:

```python
mu = torch.tensor([1.0], requires_grad=True)
p = Normal(mu, 0.05)
```

Next, let’s define a simple fitness function that rewards individuals for approaching the location 5.0 in the search space:

```python
def fitness(xs):
	return -(x - 5.0) ** 2
```

Each generation of evolution in NES takes samples from the population distribution and evaluates the fitness of each of those individual samples. Here we sample and evaluate 100 individuals from the distribution:

```python
sample = p.sample(n=100)
fitnesses = fitness(sample)
```

Optionally, we can apply a [whitening transformation](https://en.wikipedia.org/wiki/Whitening_transformation) to the fitnesses (a form of pre-processing that often increases NES performance), like this:

```python
fitnesses = (fitnesses - fitnesses.mean()) / fitnesses.std()
```

Now we can use these calculated fitness values to estimate the mean fitness over our population distribution:

```python
mean = expectation(fitnesses, sample, p=p)
```

Although we could have estimated the mean value directly with the snippet:
`mean = fitnesses.mean()`, what we gain by instead using the EvoGrad `expectation` function is the ability to backpropagate through mean.
We can then use the resulting auto-differentiated gradients to optimize the center of the 1D Gaussian population distribution (mu) through gradient descent (here, to increase the expected fitness value of the population):

```python
mean.backward()
with torch.no_grad():
	mu += alpha * mu.grad
	mu.grad.zero_()
```

### Maximizing Variance
As a more sophisticated example, rather than maximizing the mean fitness, we can maximize the variance of behaviors in the population.
While fitness is a measure of quality for a fixed task, in some situations we want to prepare for the unknown, and instead might want our population to contain a diversity of behaviors that can easily be adapted to solve a wide range of possible future tasks.

To do so, we need a quantification of behavior, which we can call a behavior characterization. Similarly to how you can evaluate an individual parameter vector drawn from the population distribution to establish its fitness (e.g. how far does this controller cause a robot to walk?), you could evaluate such a draw and return some quantification of its behavior (e.g., what position does a robot controlled by this neural network locomote to?).

For this example, let’s choose a simple but illustrative, 1D behavior characterization, namely, the product of two sine waves (one with much faster frequency than the other):

```python
def behavior(x):
	return 5 * torch.sin(0.2 * x) * torch.sin(20 * x)
```

Now, instead of estimating the mean fitness, we can calculate a statistic that reflects the diversity of sampled behaviors. The variance of a distribution is one metric of diversity, and one variant of evolvability ES measures and optimizes such variance of behaviors sampled from the population distribution:

```python
sample = p.sample(n=100)
behaviors = behavior(sample)
zscore = (behaviors - behaviors.mean()) / behaviors.std()
variance = expectation(zscore ** 2, sample, p=p)
```

### Maximizing Entropy
In the previous example, the gradient would be relatively straightforward to compute by hand.
However, sometimes we may need to maximize objectives whose derivatives would be much more challenging to derive.
In particular, this final example will seek to maximize the entropy of the distribution of behaviors (a variant of evolvability ES).

Note that for this example you'll also have to install `scipy` from pip.

To create a differentiable estimate of entropy, we first compute the pairwise distances between the different behaviors.
Next, we create a smooth probability distribution by fitting a [kernel density estimate](https://en.wikipedia.org/wiki/Kernel_density_estimation):

```python
dists = scipy.spatial.distance.squareform(scipy.spatial.distance.pdist(behaviors, "sqeuclidean"))
kernel = torch.tensor(scipy.exp(-dists / k_sigma ** 2), dtype=torch.float32)
p_x = expectation(kernel, sample, p=p, dim=1)
```

Then, we can use these probabilities to estimate the [entropy of the distribution](https://en.wikipedia.org/wiki/Entropy_(information_theory)), and run gradient descent on it as before:

```python
entropy = expectation(-torch.log(p_x), sample, p=p)
```

Full code for these examples can be found in the `demos` directory of
this repository.

## Installation
First, install EvoGrad's dependencies:
```
pip install -r requirements.txt
```

Then, either install EvoGrad from pip:
```
pip install evograd
```
Or from the source code in this repository:
```
git clone github.com/uber-research/EvoGrad
cd EvoGrad
pip install -e .
```
