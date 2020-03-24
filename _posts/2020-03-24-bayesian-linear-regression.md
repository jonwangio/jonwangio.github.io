---
title: Bayesian linear regression animated
subtitle: An intuitive introduction
layout: post
date: 2020-03-24T09:00:00-14:00
tags: [tutorial, data mining]
image: /img/bayes/bayes.png
---


# Bayesian linear regression animated

##### This is a tutorial for understanding the rationale of Bayesian statistics in linear regression.

##### The purpose of the tutorial is to show the mechanisms of _Bayesian statistics_ in an intuitive manner, mainly through general notations, graphics, and animations, without diving into the details of mathematical procedures.

-------------------
###### Credit to this work can be given as:
```
J. Wang, Bayesian linear regression animated, (2020), GitHub repository, https://github.com/wonjohn/Bayes_for_Regression
```

## Author's foreword
-------------------
_Bayesian epistemology_ introduces important constraints on top of rational degrees of belief and a rule of probabilistic inference--the principle of conditionalization, according to [William Talbott, 2008](https://plato.stanford.edu/entries/epistemology-bayesian/).

_Bayesian statistics_ forms a major branch in _statistics_. _Bayesian statistics_ relies on _Bayesian principle_ to reveal a beautiful epistemology scheme through probabilistic inference: one should rationally updates degrees of knowing or belief once new evidence is observed. Mathematically, it is denoted as:

_**P(S|E) = P(E|S)P(S)/P(E)**_

where, _**s**_ can be any arbitrary statement, and _**E**_ is observed evidence(s). Without observing any evidence, it is rational to stay with idealized belief denoted as the _prior_ belief _**P(s)**_. But if we have observed an evidence, there is something we can do to update our belief. One option is to utilize the measurement called the _likelihood_ function that quantifies how our _prior_ belief should manifest the evidence at hand. The _likelihood_ function _**P(E|S)**_ together with the _prior_ function _**P(S)**_ help to update our belief once there is more information from the reality. The updated belief is called the _posterior_ function of _**S**_, which is _**P(S|E)**_.

In this small snippet of tutorial, the principle of _Bayesian statistics_ is showcased through a prevalent prediction problem: _linear regression_.


## Recap of linear regression in frequentist view
-------------------
In the case of _linear regression_, without any consideration of probabilistic confidence, conventional linear regression only achieves point estimation of the model parameter through [_**least squares**_](https://en.wikipedia.org/wiki/Least_squares) method. The _least squares_ holds a frequentist view to exclusively rely on data observation and comes back with a _point estimation_ of the model parameters. The _least squares_ appears to be not a bad idea in the first place as we could be able to obtain exact model form and thus predictions.

Take the simplest case of _univariate linear regression_ problem for example, given the noisy observations are coming from an true underlying linear model plus some noise (Fig.1), frequentists attempt to recover this underlying model by starting with an assumption that the observations suffer from _Gaussian_ noise. It means that the noise follows a _Gaussian_ distribution around the underlying model. In short, the noise is symmetrical with respect to the true model and should add up to zero. So frequentists believe that the _least squares_ is a proper way as it searches the true model by minimizing the difference between its inference and the observations.

<p align="center"><img src="/img/bayes/0_data.png" width="450" heigth="390"></p>

_<p align="center">Fig.1 Linear regression problem setting.</p>_

Here in Fig.1 the true linear function is intentionally revealed as a line so that we can compare how _least squares_ help us to recover the true model from few observations. The true _univariate linear_ model I used here is:

_<p align="center">**M(x) = 3 + 2x**</p>_

However, this true underlying model is usually unknown in the form as:

_<p align="center">**M(x) = θ<sub>1</sub> + θ<sub>2</sub>x**</p>_

or in a vectorized form as:

_<p align="center">**M(X) = θ<sup>T</sup>X**</p>_

where _**X**_ is referred as designed vector or matrix in the form of _[1, x]<sup>T</sup>_ and _**θ**_ is _[θ<sub>1</sub>, θ<sub>2</sub>]<sup>T</sup>_. In reality, what we do know is nothing but a few observations _**Y**_ as _**(y<sub>1</sub>, y<sub>2</sub>, ... , y<sub>n</sub>)**_ but suffering from noises _**ε**_:

_<p align="center">**Y = θ<sup>T</sup>X + ε**</p>_

where the noises are preferably _Gaussian_ distributed noises around the true model. Then, can we really expect conventional methods such as the _least squares_ to find or approximate the true value of _[θ<sub>1</sub>, θ<sub>2</sub>]<sup>T</sup>_, which are _3_ and _2_ for the intercept and slope, respectively? We will stick to only three observations each time as shown in Fig.1, and intentionally create observations with _Gaussian_ noises to test the _least squares_ for several trials. Well, even with only three observations, the _least squares_ works fine to find linear models by minimizing their difference from the observations as shown in Fig.2 below.

<p align="center"><img src="/img/bayes/0_sampleOLS.gif" width="430" height="290"></p>

_<p align="center">Fig.2 Sample least squares solutions.</p>_

Each time, the _least squares_ fits a linear model to three randomly drawn observations. These fitted lines or models are more or less different from the true underlying model. The difference between the fitted and true model is understandable: due to the small number of the observations, the _Gaussian_ noise in the observations hardly manifest symmetry with respect to the true model, and sometimes all three observations fall on the same side of the true underlying model, thus fitting a model that minimizes its difference to a small number of observations almost always produce difference between the fitted model and the true model. Meantime, it is also worth noticing that most observations are close to the true model due to the _Gaussian_ noise in them, the fitted lines are more likely to be close to the true model. In short, there are few remarks from the application of _least squares_ to the solution of true underlying model:
- frequentist approach such as the _least squares_ is inherently uncertain and fits uncertain models, especially when observations are limited;
- frequentist relies heavily on the number samples and time of observation--with enough trials and observations, it is likely to be more certain about where is the true model;
- but frequentist fails to encode this process of how increasing observation times updates our confidence of finding the true model.

As long as we cannot reject other possible fitted models only except we could be 100% sure about the optimal one, it is so nice if we can quantify these uncertainties, isn't it? In reality, it is very important to know how certain is the model at hand as well as its prediction. In many practical cases, such as predicting housing values, stock values, pollution concentration, soil mineral distribution, etc., the confidence of our model performance help us to control the risk of making prediction and minimize economic loss. Frequentists of course are eliminated by its nature in making point estimation of the model parameters. The demand of quantifying uncertainty leads to natural transition from point estimation of the model parameter to a probabilistic perspective, and paves the way to the application of _Bayesian statistics_.


## Bayesian inference
-------------------
_Bayesian statistics_ attempt to explicitly impose credibility upon the underlying model. It does favor an optimal solution to the underlying model but does not reject other possibilities. So _Bayesian statistics_ seeks potential models along with confidence simultaneously. The credibility or probability of the underlying model is achieved through combining two important probabilities:
- the probability of all potential models encoding our knowledge or belief _prior_ to see any evidence;
- the probability of the evidence, once observed, given by any potential model.

In the case of linear regression as shown in Fig.1, _Bayesian statistics_ tries to figure out the probability of the unobserved linear model (linear parameters) through few point observations (points in this example).It applies _Bayesian principle_ _**P(S|E) = P(E|S)P(S)/P(E)**_ to the model parameters in _**M(x) = θ<sub>1</sub> + θ<sub>2</sub>x**_:

_<p align="center">**P(θ|D) = P(D|θ)P(θ)/P(D)**</p>_

where _**D**_, collection of noisy observations, becomes our evidence. In Fig.1, there are 3 observations available for us to find out the model parameters _**θ**_. We can denote observations points as collection of tuples _**{(x<sub>1</sub>, y<sub>1</sub>), (x<sub>2</sub>, y<sub>2</sub>), (x<sub>3</sub>, y<sub>3</sub>)}**_. _**P(θ|D)**_ is called the _posterior_ distribution of _**θ**_ as it is a distribution after we updating our knowledge by seeing data as evidence. The _posterior_ is determined by the terms on the right-hand side of the equation. _**P(D|θ)**_ is the _likelihood_ function that quantifying the probability of the observations produced by some model governed by parameters _**θ**_. _**P(θ)**_ is the _prior_ distribution of _**θ**_ encoding our knowledge of parameters before making any observations. How could it be possible to know anything about _**θ**_ before seeing any data? Well, in most practical cases, we do have some ideas: the relationships between precipitation and soil loss, traffic volume and road pollution, location and land price, etc...We more or less know the general range of _**θ**_, or its sign, at least. _**P(D)**_ is a normalization term that makes the right-hand side of the equation a true probabilistic distribution that integrated to 1.

If we stay simple enough in this tutorial, we can temporarily ignore the normalization term _**P(D)**_. Now,  in order to quantify the _posterior_ _**P(θ|D)**_, the problem reduces to specify the _likelihood_ _**P(D|θ)**_ and _prior_ _**P(θ)**_, which has been mentioned at the beginning of this section as important probabilities.


### Bayesian function specification: _likelihood_

For any single observed data point _**(x<sub>k</sub>, y<sub>k</sub>)**_, the _likelihood_ measures the probability of the model parameter _**θ**_ gives rise to this known data point. Thus, _given_ any possible _**θ**_, how likely it is to observe this particular point of tuple _**(x<sub>k</sub> , y<sub>k</sub>)**_? Referring above to the noisy observation from the linear model, by saying we have observations with noise _**ε**_ around the true model, it is most handy to impose a _Gaussian_ distribution over the noise around the true model. In short, the _likelihood_ of observing the tuple _**(x<sub>k</sub> , y<sub>k</sub>)**_ follows a _Gaussian_ distribution around the model specified by _**θ**_:

_<p align="center">**P(D|θ) = P(y<sub>k</sub>|x<sub>k</sub> , θ) ~ N(y<sub>k</sub> ; θ<sup>T</sup>X, ε)**</p>_

This _Gaussian_ form _likelihood_ can be easily implemented as a function in _python_ as:

```python
def likeli(theta1,theta2,obs_y,obs_x):  # It is a function of theta with known observations
    sigma = 1  # Standard deviation of the Gaussian likelihood
    func = (1/np.sqrt(6.28*sigma**2))*np.exp((obs_y-theta1-theta2*obs_x)**2/(-2*sigma**2))
    return func
```
where I chose a standard deviation of _**1**_ for this _Gaussian likelihood_ as I assume for the noise level. It shows my confidence interval that the observations should be in around the true linear model. This _likelihood_ is obviously a function wrt. _**θ**_ as the tuple _**(x<sub>k</sub> , y<sub>k</sub>)**_ is observed. More intuitively, if we observe one pair of _**(x<sub>k</sub> , y<sub>k</sub>)**_ as denoted red to the left of Fig.3, the above _likelihood_ is a function of _**θ**_ or _[θ<sub>1</sub> , θ<sub>2</sub>]<sup>T</sup>_, and can be plotted in a 2-dimensional space defined by θ<sub>1</sub> and θ<sub>2</sub> to the right of Fig.3. Here are two separate observations visualized, each of whose _likelihood_ function is plotted and appears roughly to be a line as a function of _**θ**_. It is only roughly a line in the space of _[θ<sub>1</sub> , θ<sub>2</sub>]<sup>T</sup>_, implying that there are infinite number of _[θ<sub>1</sub> , θ<sub>2</sub>]<sup>T</sup>_ options running all the way from positive to negative values to give rise to the observation. This is extremely reasonable as one point observation determines lines with either positive or negative interception and slope. Continue with the sample linear function above, if we can be able to make a couple of more noisy observations, we can obtain multiple _likelihood_ for each of the observation in the _[θ<sub>1</sub> , θ<sub>2</sub>]<sup>T</sup>_ space.

<p align="center"><img src="/img/bayes/1_likeli_2.gif" width="800" heigth="680"></p>

_<p align="center">Fig.3 Likelihood wrt. a single observation.</p>_

In this case of linear regression, isn't it getting clear that these line-shaped _likelihood_ functions are potentially intersected at some relatively fixed region? That is where we can combine these _likelihood_ function that the profile of _**θ**_ can be delineated. In what way to combine? As the _likelihood_ is a probability measurement, combining the _likelihood_ is simply a joint probability. Observing each data point as a _**(x<sub>k</sub> , y<sub>k</sub>)**_ tuple is considered to be an [_**iid**_](https://en.wikipedia.org/wiki/Independent_and_identically_distributed_random_variables) process, thus the joint _likelihood_ of any _**θ**_ gives rise to all the observations is a multiplication of all the individual _likelihood_:

_**P(D|θ) = ∏<sub>i</sub> P((x<sub>i</sub> , y<sub>i</sub>)|θ)**_

The animation (Fig.4) below shows how this joint probability is updated with each added observation. It is quite appealing that when the second _**(x<sub>k</sub> , y<sub>k</sub>)**_ tuple is observed, the joint _likelihood_ function already started take in shape and the inference of the model parameter can be made in a well delineated subspace within the _[θ<sub>1</sub> , θ<sub>2</sub>]<sup>T</sup>_ space. The joint _likelihood_ with an ellipse-shaped _Gaussian_ centers around _**[3, 2]**_ indicating a high confidence that it should be the model parameter. At the same time, the _likelihood_ does not reject other possibilities as it is still possible, with a certain noise level in the observations, that _[θ<sub>1</sub> , θ<sub>2</sub>]<sup>T</sup>_ can take some other values around _**[3, 2]**_. When the third point is observed, this _likelihood_ gives a more shrunk distribution representing the process of knowledge update wrt. _[θ<sub>1</sub> , θ<sub>2</sub>]<sup>T</sup>_.

<p align="center"><img src="/img/bayes/1_likeli1.gif" width="800" heigth="680"></p>

_<p align="center">Fig.4 Joint likelihood wrt. to observations.</p>_

At this point, it is no surprising that why the [_**Maximum Likelihood Estimation (MLE)**_](https://en.wikipedia.org/wiki/Maximum_likelihood_estimation) is so frequently adopted. For linear regression, especially simple as there is only few independent variable and enough observations, and of course without too much noise, the joint _likelihood_ function could already bring a desirable results. And it is then also quite safe to maximize the _likelihood_ to obtain a point estimation of the model parameter.

### Bayesian function specification: _prior_

Different from that _likelihood_ can be delineated from observed information, choosing _prior_ function for _Bayesian_ inference is tricky. It is called _prior_ distribution because it requires us to configure the distribution of the model parameters _prior_ to seeing any data, or based upon our _prior_ knowledge with regard to the parameters. Fortunately, in several situation, we DO have such prior knowledge when building a regression model. For instance, if someone is interested in the loss of a particular type of soil _**(y<sub>k</sub>)**_ due to rainfall _**(x<sub>k</sub>)**_ in a region, it is already handy to know that there should be a positive relationship between _**(y<sub>k</sub>)**_ and _**(x<sub>k</sub>)**_. It also means that we can more or less _constrain_ the _**θ**_ to be positive. Or, maybe someone has already done similar work in other places and brought some confident results, it is even possible to further _constrain_ the _**θ**_ to be a probabilistic distribution over these available results (values).

But here in this tutorial, although it is a simple linear regression example, we are running into a awkward situation: nothing is availabe except the observations to make inference about the model. This is where one has to rely on _improper_ or _non-informative_ _prior_ distribution for the model parameters. These keywords such as _improper_ and _non-informative_ indicate that the design of the _prior_ function is entirely arbitrary. One option is to make _**P(θ)=1**_, thus it is _non-informative_ and will have no effect over the _posterior_ when multplies with the _likelihood_. Another option is to make a general assumption that _**P(θ)**_ follows some well formed statistical distribution, such as _Gaussian_ distribution shown in Fig.5 below. This kind of specification can be _improper_ as it would potentially impose limited and unreasonable assumption that _**θ**_ is normally distributed around _**0**_.

<p align="center"><img src="/img/bayes/2_prior.png" width="380" heigth="380"></p>

_<p align="center">Fig.5 Non-informative prior distribution for model parameters.</p>_

This _improper Gaussian_ distribution within the _[θ<sub>1</sub> , θ<sub>2</sub>]<sup>T</sup>_ space means that if we draw points randomly from it, it is more likely to have _**θ**_ with values close to zero. Visually, each random point in the _[θ<sub>1</sub> , θ<sub>2</sub>]<sup>T</sup>_ space determines a random line in the space of _[x, y]<sup>T</sup>_ as shown below in Fig.6, but most lines are with interception and slope close to zero.  

<p align="center"><img src="/img/bayes/2_priorDraw.gif" width="800" heigth="680"></p>

_<p align="center">Fig.6 Randomly drawn prior functions for the model.</p>_

Since we know that the true model parameters, plus that we also know that the true parameters are well captured by the _likelihood_, this _improper prior_ appears to be way off the target. Is it possible to update this _prior_ to a meaningful state by using the _likelihood_?

### Bayesian posterior: combine the likelihood and prior

The effect of this _improper Gaussian prior_ combined with the _likelihood_ can be visualized as in Fig.7 below. The combination, again, follows the principle of joint statistical distribution, is achieved through multiplication. The resultant _posterior_ distribution of the model parameter _**θ**_ is compared with their _likelihood_ distribution solely determined by observed evidences.

In _Bayesian statistics_, this multiplication is normally referred as _update_ as mentioned at the beginning of this tutorial, where the _prior_ knowledge is _updated_ by the _likelihood_ brought by observations. Equivalently, I could also say that the _likelihood_ is being _shifted_ or _dragged_ by our _prior_ belief, because our _prior_ belief imposes a constraint even we have observed few evidence.

Although it seems like the _posterior_ distribution of _**θ**_ obtained by combining its _likelihood_ and _improper prior_ is acceptable in the first place, not too prominently, the _improperness_ of the _improper prior_ in this case is still highlighted as we already know that the _likelihood_ is perfectly centered around the true parameter values of _**[3, 2]**_, and now it is _shifted_ away. But we would never notice this in practice as we wouldn't know the true model parameters. One can see that the direction of the _shifting likelihood_ is towards the _prior_. But as it is a multiplication, the _shift_ is not quite intense. The multiplication combines the large value in both _prior_ and _likelihood_, thus the _shift_ is along the gentle gradient of the _likelihood_ while moving towards the _prior_. The major reason that the _posterior_ is not too far away shifted from the _likelihood_ is that the _improper prior_ is relatively "flat", whereas the distribution rendered by the _likelihood_ is strongly centered. So, the resultant multiplication would largely driven by the _likelihood_.

<p align="center"><img src="/img/bayes/3_post_likeli.gif" width="330" heigth="330"></p>

_<p align="center">Fig.7 Posterior distribution/function for model parameters.</p>_

Now, one may start to ask: what on earth is the point to use _prior_ functions that are improperly or non-informatively designed?! Unfortunately, there is no one-fits-all answer. It is the nature of statistic inference that we are forced to make some assumptions from scratch, just like we have to assume there is a linear relationship already before obtaining more data points than only a few of them. There are [few discussions](https://stats.stackexchange.com/questions/27813/what-is-the-point-of-non-informative-priors) going around for choosing _prior_ distributions scientifically. More formal research can be found in [**The Bayesian Choice: From Decision-Theoretic Foundations to Computational Implementation (Springer Texts in Statistics**](https://www.amazon.com/dp/0387715983/), as well as [**Moving beyond noninformative priors: why and how to choose weakly informative priors in Bayesian analyses**](https://onlinelibrary.wiley.com/doi/10.1111/oik.05985).

_Improper_ or _non-informative prior_ shouldn't be the reason to be pessimistic about _Bayesian principle_. In most cases, we are still doing [incremental science](https://www.statnews.com/2015/12/02/science-groundbreaking/), which means there is almost always some existing information we could leverage and to be encoded as _prior_ knowledge, like the example of soil loss prediction mentioned earlier.

Even without the context of domain knowledge, using _improper prior_ achieves some appealing results. A _prior_, along with the _likelihood_ builds a connection to the idea of [_**regularization**_](http://primo.ai/index.php?title=Regularization), which is a technique explicitly modifies how the model parameters are sought given any specified _loss function_. If the _likelihood_ is viewed as a _loss function_ specified by observations, then the _prior_ plays the role of a _regularizer_ to constrain the model parameters from being solely controlled by the _loss function_. In practice, there are few design options for the _regularizer_ for achieving different purposes: avoid overfitting (equivalent to _Gaussian prior function_), parameter selection (equivalent to _Laplace prior function_) and in combination. These _regularizer_ are called in different ways, for instance, as shown in Fig.8.

<p align="center"><img src="/img/bayes/4_regularization.png" width="600" heigth="510"></p>

_<p align="center">Fig.8 Regularization equivalence of prior functions (src: http://primo.ai/index.php?title=Regularization).</p>_

The **biggest difference** is probably that _Bayesian principle_ stays as a general statistical framework, whereas _regularization_ is more commonly adopted from the frequentist perspective that a particular solution to the model parameter is expected.

Apart from the _prior_ specification, it should NOT be a worse case where we have to heavily rely on observations. With the development of data acquirement approaches, we could be able to be confident about the model parameters with _likelihood_. The visualization of the _posterior_ below (Fig.9) shows how increasing observations may update our knowledge even from a poor _prior_. The multiplication in _**P(D|θ)P(θ)/P(D)**_ is explicitly two way: (1) the _prior_ is constraining or dragging, while (2) the _likelihood_ is updating and washing away the effect of constraining or dragging.

<p align="center"><img src="/img/bayes/4_postObs.gif" width="800" heigth="680"></p>

_<p align="center">Fig.9 Change of posterior distribution/function for model parameters along with increasing observations.</p>_

Holding the _posterior_ at hand, we now have a more concentrated distribution of the model parameters. The final distribution of the _**θ**_ in the _[θ<sub>1</sub> , θ<sub>2</sub>]<sup>T</sup>_ space, as shown in Fig.8 is confidently centering the point _[3 , 2]_, while still shows possibility of parameters such as _[2.5 , 2]_. This probabilistic perspective is nice as the solution of the parameter is optimal at _[3 , 2]_, we are not rejecting other possibilities without knowing how the qualities of the observations are disturbed by those random noises. We have few options to deal with this _posterior_. Similar to maximizing the _likelihood_, we can seek the maximum-a-posteriori (_**MAP**_) probability to achieve the optimal model parameters. We can also stay with the probability distribution to quantify confidence of prediction.

Since it is a probability distribution in the _posterior_, it means sampling is possible from this distribution for visualization. As shown in Fig.9, drawn from the distribution, many samples are very close to _[3 , 2]_ with few are bit far away from the center of the distribution. In the _[X , Y]_ space, the drawn samples seems to be wobbling around a potential "datum line". This "datum line" is in fact _**M(x) = 3 + 2x**_. Apparently, the sample lines are corresponding to the probability of drawn _[θ<sub>1</sub> , θ<sub>2</sub>]<sup>T</sup>_, thus we can encode this probability of _**θ**_ to the lines, namely the inferred linear model. It then forms a probability distribution of potential linear models in the _[X , Y]_ space and can be visualized as shadow to the right of Fig.10.

<p align="center"><img src="/img/bayes/5_postDraw.gif" width="800" heigth="680"></p>

_<p align="center">Fig.10 Model possibilities in terms of posterior distribution of model parameters.</p>_

The shadow defines an import confidence interval for making predictions: given any _**x<sub>new</sub>**_, what is the probability of _**y<sub>new</sub>**_? As the probability of giving rise to a _**y<sub>new</sub>**_ is the probability of the model, the distribution of _**y<sub>new</sub>**_ is now out-of-box given all the possible linear model. Mathematically, it is equivalent to weight all predictions made by each potential linear model by the probabilistic distribution of that model, as

_**P(D<sub>new</sub>|D) = ʃ P(D<sub>new</sub>|θ,D)P(θ|D)dθ**_

which is exactly how Fig.10 is plotted. The shadow is a combination of possible linear model weighted by their possibilities. The equation above also speaks the same idea: the _posterior_ considers all possible _**θ**_, which means the _posterior_ does NOT care about the exact _**θ**_! The integration plays a beautiful role to manifest such contradictory that all _**θ**_ are involved (considered), but then are integrated out (does NOT care)!


## Wrap-up
-------------------
So far we have been stay intuitively by using general notations (such as _**P(θ)**_), graphics and animations. In order to obtain exact measurement of the distribution regarding the _likelihood_, _prior_, and _posterior_, we can explicitly quantify the distributions by using hyper-parameters, for instance:
- _**P(θ) = P(θ|α) ~ N(θ ; 0, α<sup>-1</sup>I)**_ for the _prior_, where _**α**_ is the hyper-parameter controlling the shape of probabilistic distribution;
- _**P(D|θ) = P(y<sub>k</sub>|x<sub>k</sub> , θ, β) ~ N(y<sub>k</sub> ; θ<sup>T</sup>X, β<sup>-1</sup>I)**_, where _**β**_ is another hyper-parameter of precision (inverse variance) controlling the noise intensity as _**ε ~ N(ε ; 0, β<sup>-1</sup>I)**_;
- we can even have design function _**φ**_ for _**φ(x)=1+x**_.

If we can be able to specify the hyper-parameters, the _posterior_ is measurable, visually, the _width_ and _center_ of the shadow in Fig.10 is to be a function of hyper-parameters _**α**_ and _**β**_. These hyper-parameters can also be obtained automatically! The approach is called maximizing the _marginal likelihood_ through [_**Empirical Bayes method**_](https://en.wikipedia.org/wiki/Empirical_Bayes_method), which will not be covered in this tutorial.

After digesting the mechanisms of the _Bayesian statistics_ in linear regression, you are ready to leverage existing packages without bothering too much to write your own code. As long as you understand what does it mean by _likelihood_, _prior_, _posterior_ and the role of those hyper-parameters, you can simple do as follows:

```python
from bayesian_linear_regression_util import *
```
If you do wish to specify your own _Bayesian_ distributions for any parameters, say _**θ**_, you can leverage [`PyMC3`](https://docs.pymc.io/) as:

```python
import pymc3 as pm
with pm.Model() as model:
    μ = pm.Uniform('μ', lower=0, upper=300)
    σ = pm.HalfNormal('σ', sd=10)
    θ = pm.Normal('θ', mu=μ, sd=σ, observed=data['price'].values)
    trace = pm.sample(1000, tune=1000)
```


## Takeaways
-------------------
Few takeaways after digesting this tutorial:
- many distributions specified for _Bayesian_ linear regression are _Gaussian_ because of its nice property in conditioning, multiplication, etc.;
- maximum-likelihood-estimation (MLE) can be reliable if enough observations are made, but quite vulnerable to limited ones, which leads to a phenomenon of [_overfitting_](https://en.wikipedia.org/wiki/Overfitting) if someone is interested in the details;
- even the _improper_ or _non-informative_ _prior_ functions are confusing with limited domain knowledge, they play a useful role mathematically and build the connection to an important regression strategy: _regularization_;
- although specifying the _prior_ is sometimes referred as model selection, it is essentially specify the parameters of model, the assumption of the model as linear or non-linear is still arbitrary;
- for further interest: specifying the design function _**φ**_ for the design function _**x**_ can be related to kernel tricks and paves the way to [_**non-parametric modeling**_](https://en.wikipedia.org/wiki/Nonparametric_regression).
