---
title: Bayesian linear regression animated
subtitle: An intuitive introduction
layout: post
date: 2020-03-24T09:00:00-14:00
tags: [tutorial, data mining]
image: /img/bayes/bayes.png
---


# Bayesian linear regression animated

##### This is a tutorial for understanding the rationale of *Bayesian statistics* in linear regression.

##### The purpose of the tutorial is to show the mechanisms of *Bayesian statistics* in an intuitive manner, mainly through general notations, graphics, and animations, without diving into the details of mathematical procedures.

-------------------
###### Credit to this work can be given as:
```
J. Wang, Bayesian linear regression animated, (2020), GitHub repository,
https://github.com/wonjohn/Bayes_for_Regression
```

## Author's foreword
-------------------
*Bayesian epistemology* introduces important constraints on top of rational degrees of belief and a rule of probabilistic inference--the principle of conditionalization, according to [William Talbott, 2008](https://plato.stanford.edu/entries/epistemology-bayesian/).

*Bayesian statistics* forms a major branch in *statistics*. *Bayesian statistics* relies on *Bayesian principle* to reveal a beautiful epistemology scheme through probabilistic inference: one should rationally updates degrees of knowing or belief once new evidence is observed. Mathematically, it is denoted as:

***P(S│E) = P(E│S)P(S)/P(E)***

where, ***s*** can be any arbitrary statement, and ***E*** is observed evidence(s). Without observing any evidence, it is rational to stay with idealized belief denoted as the *prior* belief ***P(s)***. But if we have observed an evidence, there is something we can do to update our belief. One option is to utilize the measurement called the *likelihood* function that quantifies how our *prior* belief should manifest the evidence at hand. The *likelihood* function ***P(E│S)*** together with the *prior* function ***P(S)*** help to update our belief once there is more information from the reality. The updated belief is called the *posterior* function of ***S***, which is ***P(S│E)***.

In this small snippet of tutorial, the principle of *Bayesian statistics* is showcased through a prevalent prediction problem: *linear regression*.


## Recap of linear regression in frequentist view
-------------------
In the case of *linear regression*, without any consideration of probabilistic confidence, conventional linear regression only achieves point estimation of the model parameter through [***least squares***](https://en.wikipedia.org/wiki/Least_squares) method. The *least squares* holds a frequentist view to exclusively rely on data observation and comes back with a *point estimation* of the model parameters. The *least squares* appears to be not a bad idea in the first place as we could be able to obtain exact model form and thus predictions.

Take the simplest case of *univariate linear regression* problem for example, given the noisy observations are coming from an true underlying linear model plus some noise (Fig.1), frequentists attempt to recover this underlying model by starting with an assumption that the observations suffer from *Gaussian* noise. It means that the noise follows a *Gaussian* distribution around the underlying model. In short, the noise is symmetrical with respect to the true model and should add up to zero. So frequentists believe that the *least squares* is a proper way as it searches the true model by minimizing the difference between its inference and the observations.

<p align="center"><img src="/img/bayes/0_data.png" width="450" heigth="390"></p>

<p align="center">Fig.1 Linear regression problem setting.</p>

Here in Fig.1 the true linear function is intentionally revealed as a line so that we can compare how *least squares* help us to recover the true model from few observations. The true *univariate linear* model I used here is:

***M(x) = 3 + 2x***

However, this true underlying model is usually unknown in the form as:

***M(x) = θ<sub>1</sub> + θ<sub>2</sub>x***

or in a vectorized form as:

***M(X) = θ<sup>T</sup>X***

where ***X*** is referred as designed vector or matrix in the form of *[1, x]<sup>T</sup>* and ***θ*** is *[θ<sub>1</sub>, θ<sub>2</sub>]<sup>T</sup>*. In reality, what we do know is nothing but a few observations ***Y*** as ***(y<sub>1</sub>, y<sub>2</sub>, ... , y<sub>n</sub>)*** but suffering from noises ***ε***:

***Y = θ<sup>T</sup>X + ε***

where the noises are preferably *Gaussian* distributed noises around the true model. Then, can we really expect conventional methods such as the *least squares* to find or approximate the true value of *[θ<sub>1</sub>, θ<sub>2</sub>]<sup>T</sup>*, which are *3* and *2* for the intercept and slope, respectively? We will stick to only three observations each time as shown in Fig.1, and intentionally create observations with *Gaussian* noises to test the *least squares* for several trials. Well, even with only three observations, the *least squares* works fine to find linear models by minimizing their difference from the observations as shown in Fig.2 below.

<p align="center"><img src="/img/bayes/0_sampleOLS.gif" width="430" height="290"></p>

<p align="center">Fig.2 Sample least squares solutions.</p>

Each time, the *least squares* fits a linear model to three randomly drawn observations. These fitted lines or models are more or less different from the true underlying model. The difference between the fitted and true model is understandable: due to the small number of the observations, the *Gaussian* noise in the observations hardly manifest symmetry with respect to the true model, and sometimes all three observations fall on the same side of the true underlying model, thus fitting a model that minimizes its difference to a small number of observations almost always produce difference between the fitted model and the true model. Meantime, it is also worth noticing that most observations are close to the true model due to the *Gaussian* noise in them, the fitted lines are more likely to be close to the true model. In short, there are few remarks from the application of *least squares* to the solution of true underlying model:
- frequentist approach such as the *least squares* is inherently uncertain and fits uncertain models, especially when observations are limited;
- frequentist relies heavily on the number samples and time of observation--with enough trials and observations, it is likely to be more certain about where is the true model;
- but frequentist fails to encode this process of how increasing observation times updates our confidence of finding the true model.

As long as we cannot reject other possible fitted models only except we could be 100% sure about the optimal one, it is so nice if we can quantify these uncertainties, isn't it? In reality, it is very important to know how certain is the model at hand as well as its prediction. In many practical cases, such as predicting housing values, stock values, pollution concentration, soil mineral distribution, etc., the confidence of our model performance help us to control the risk of making prediction and minimize economic loss. Frequentists of course are eliminated by its nature in making point estimation of the model parameters. The demand of quantifying uncertainty leads to natural transition from point estimation of the model parameter to a probabilistic perspective, and paves the way to the application of *Bayesian statistics*.


## Bayesian inference
-------------------
*Bayesian statistics* attempt to explicitly impose credibility upon the underlying model. It does favor an optimal solution to the underlying model but does not reject other possibilities. So *Bayesian statistics* seeks potential models along with confidence simultaneously. The credibility or probability of the underlying model is achieved through combining two important probabilities:
- the probability of all potential models encoding our knowledge or belief *prior* to see any evidence;
- the probability of the evidence, once observed, given by any potential model.

In the case of linear regression as shown in Fig.1, *Bayesian statistics* tries to figure out the probability of the unobserved linear model (linear parameters) through few point observations (points in this example).It applies *Bayesian principle* ***P(S│E) = P(E│S)P(S)/P(E)*** to the model parameters in ***M(x) = θ<sub>1</sub> + θ<sub>2</sub>x***:

***P(θ│D) = P(D│θ)P(θ)/P(D)***

where ***D***, collection of noisy observations, becomes our evidence. In Fig.1, there are 3 observations available for us to find out the model parameters ***θ***. We can denote observations points as collection of tuples ***{(x<sub>1</sub>, y<sub>1</sub>), (x<sub>2</sub>, y<sub>2</sub>), (x<sub>3</sub>, y<sub>3</sub>)}***. ***P(θ│D)*** is called the *posterior* distribution of ***θ*** as it is a distribution after we updating our knowledge by seeing data as evidence. The *posterior* is determined by the terms on the right-hand side of the equation. ***P(D│θ)*** is the *likelihood* function that quantifying the probability of the observations produced by some model governed by parameters ***θ***. ***P(θ)*** is the *prior* distribution of ***θ*** encoding our knowledge of parameters before making any observations. How could it be possible to know anything about ***θ*** before seeing any data? Well, in most practical cases, we do have some ideas: the relationships between precipitation and soil loss, traffic volume and road pollution, location and land price, etc...We more or less know the general range of ***θ***, or its sign, at least. ***P(D)*** is a normalization term that makes the right-hand side of the equation a true probabilistic distribution that integrated to 1.

If we stay simple enough in this tutorial, we can temporarily ignore the normalization term ***P(D)***. Now,  in order to quantify the *posterior* ***P(θ│D)***, the problem reduces to specify the *likelihood* ***P(D│θ)*** and *prior* ***P(θ)***, which has been mentioned at the beginning of this section as important probabilities.


### Bayesian function specification: *likelihood*

For any single observed data point ***(x<sub>k</sub>, y<sub>k</sub>)***, the *likelihood* measures the probability of the model parameter ***θ*** gives rise to this known data point. Thus, *given* any possible ***θ***, how likely it is to observe this particular point of tuple ***(x<sub>k</sub> , y<sub>k</sub>)***? Referring above to the noisy observation from the linear model, by saying we have observations with noise ***ε*** around the true model, it is most handy to impose a *Gaussian* distribution over the noise around the true model. In short, the *likelihood* of observing the tuple ***(x<sub>k</sub> , y<sub>k</sub>)*** follows a *Gaussian* distribution around the model specified by ***θ***:

***P(D│θ) = P(y<sub>k</sub>│x<sub>k</sub> , θ) ~ N(y<sub>k</sub> ; θ<sup>T</sup>X, ε)***

This *Gaussian* form *likelihood* can be easily implemented as a function in *python* as:

```python
def likeli(theta1,theta2,obs_y,obs_x):  # It is a function of theta with known observations
    sigma = 1  # Standard deviation of the Gaussian likelihood
    func = (1/np.sqrt(6.28*sigma**2))*np.exp((obs_y-theta1-theta2*obs_x)**2/(-2*sigma**2))
    return func
```
where I chose a standard deviation of ***1*** for this *Gaussian likelihood* as I assume for the noise level. It shows my confidence interval that the observations should be in around the true linear model. This *likelihood* is obviously a function wrt. ***θ*** as the tuple ***(x<sub>k</sub> , y<sub>k</sub>)*** is observed. More intuitively, if we observe one pair of ***(x<sub>k</sub> , y<sub>k</sub>)*** as denoted red to the left of Fig.3, the above *likelihood* is a function of ***θ*** or *[θ<sub>1</sub> , θ<sub>2</sub>]<sup>T</sup>*, and can be plotted in a 2-dimensional space defined by θ<sub>1</sub> and θ<sub>2</sub> to the right of Fig.3. Here are two separate observations visualized, each of whose *likelihood* function is plotted and appears roughly to be a line as a function of ***θ***. It is only roughly a line in the space of *[θ<sub>1</sub> , θ<sub>2</sub>]<sup>T</sup>*, implying that there are infinite number of *[θ<sub>1</sub> , θ<sub>2</sub>]<sup>T</sup>* options running all the way from positive to negative values to give rise to the observation. This is extremely reasonable as one point observation determines lines with either positive or negative interception and slope. Continue with the sample linear function above, if we can be able to make a couple of more noisy observations, we can obtain multiple *likelihood* for each of the observation in the *[θ<sub>1</sub> , θ<sub>2</sub>]<sup>T</sup>* space.

<p align="center"><img src="/img/bayes/1_likeli_2.gif" width="800" heigth="680"></p>

<p align="center">Fig.3 Likelihood wrt. a single observation.</p>

In this case of linear regression, isn't it getting clear that these line-shaped *likelihood* functions are potentially intersected at some relatively fixed region? That is where we can combine these *likelihood* function that the profile of ***θ*** can be delineated. In what way to combine? As the *likelihood* is a probability measurement, combining the *likelihood* is simply a joint probability. Observing each data point as a ***(x<sub>k</sub> , y<sub>k</sub>)*** tuple is considered to be an [***iid***](https://en.wikipedia.org/wiki/Independent_and_identically_distributed_random_variables) process, thus the joint *likelihood* of any ***θ*** gives rise to all the observations is a multiplication of all the individual *likelihood*:

***P(D│θ) = ∏<sub>i</sub> P((x<sub>i</sub> , y<sub>i</sub>)│θ)***

The animation (Fig.4) below shows how this joint probability is updated with each added observation. It is quite appealing that when the second ***(x<sub>k</sub> , y<sub>k</sub>)*** tuple is observed, the joint *likelihood* function already started take in shape and the inference of the model parameter can be made in a well delineated subspace within the *[θ<sub>1</sub> , θ<sub>2</sub>]<sup>T</sup>* space. The joint *likelihood* with an ellipse-shaped *Gaussian* centers around ***[3, 2]*** indicating a high confidence that it should be the model parameter. At the same time, the *likelihood* does not reject other possibilities as it is still possible, with a certain noise level in the observations, that *[θ<sub>1</sub> , θ<sub>2</sub>]<sup>T</sup>* can take some other values around ***[3, 2]***. When the third point is observed, this *likelihood* gives a more shrunk distribution representing the process of knowledge update wrt. *[θ<sub>1</sub> , θ<sub>2</sub>]<sup>T</sup>*.

<p align="center"><img src="/img/bayes/1_likeli1.gif" width="800" heigth="680"></p>

<p align="center">Fig.4 Joint likelihood wrt. to observations.</p>

At this point, it is no surprising that why the [***Maximum Likelihood Estimation (MLE)***](https://en.wikipedia.org/wiki/Maximum_likelihood_estimation) is so frequently adopted. For linear regression, especially simple as there is only few independent variable and enough observations, and of course without too much noise, the joint *likelihood* function could already bring a desirable results. And it is then also quite safe to maximize the *likelihood* to obtain a point estimation of the model parameter.

### Bayesian function specification: *prior*

Different from that *likelihood* can be delineated from observed information, choosing *prior* function for *Bayesian* inference is tricky. It is called *prior* distribution because it requires us to configure the distribution of the model parameters *prior* to seeing any data, or based upon our *prior* knowledge with regard to the parameters. Fortunately, in several situation, we DO have such prior knowledge when building a regression model. For instance, if someone is interested in the loss of a particular type of soil ***(y<sub>k</sub>)*** due to rainfall ***(x<sub>k</sub>)*** in a region, it is already handy to know that there should be a positive relationship between ***(y<sub>k</sub>)*** and ***(x<sub>k</sub>)***. It also means that we can more or less *constrain* the ***θ*** to be positive. Or, maybe someone has already done similar work in other places and brought some confident results, it is even possible to further *constrain* the ***θ*** to be a probabilistic distribution over these available results (values).

But here in this tutorial, although it is a simple linear regression example, we are running into a awkward situation: nothing is availabe except the observations to make inference about the model. This is where one has to rely on *improper* or *non-informative* *prior* distribution for the model parameters. These keywords such as *improper* and *non-informative* indicate that the design of the *prior* function is entirely arbitrary. One option is to make ***P(θ)=1***, thus it is *non-informative* and will have no effect over the *posterior* when multplies with the *likelihood*. Another option is to make a general assumption that ***P(θ)*** follows some well formed statistical distribution, such as *Gaussian* distribution shown in Fig.5 below. This kind of specification can be *improper* as it would potentially impose limited and unreasonable assumption that ***θ*** is normally distributed around ***0***.

<p align="center"><img src="/img/bayes/2_prior.png" width="380" heigth="380"></p>

<p align="center">Fig.5 Non-informative prior distribution for model parameters.</p>

This *improper Gaussian* distribution within the *[θ<sub>1</sub> , θ<sub>2</sub>]<sup>T</sup>* space means that if we draw points randomly from it, it is more likely to have ***θ*** with values close to zero. Visually, each random point in the *[θ<sub>1</sub> , θ<sub>2</sub>]<sup>T</sup>* space determines a random line in the space of *[x, y]<sup>T</sup>* as shown below in Fig.6, but most lines are with interception and slope close to zero.  

<p align="center"><img src="/img/bayes/2_priorDraw.gif" width="800" heigth="680"></p>

<p align="center">Fig.6 Randomly drawn prior functions for the model.</p>

Since we know that the true model parameters, plus that we also know that the true parameters are well captured by the *likelihood*, this *improper prior* appears to be way off the target. Is it possible to update this *prior* to a meaningful state by using the *likelihood*?

### Bayesian posterior: combine the likelihood and prior

The effect of this *improper Gaussian prior* combined with the *likelihood* can be visualized as in Fig.7 below. The combination, again, follows the principle of joint statistical distribution, is achieved through multiplication. The resultant *posterior* distribution of the model parameter ***θ*** is compared with their *likelihood* distribution solely determined by observed evidences.

In *Bayesian statistics*, this multiplication is normally referred as *update* as mentioned at the beginning of this tutorial, where the *prior* knowledge is *updated* by the *likelihood* brought by observations. Equivalently, I could also say that the *likelihood* is being *shifted* or *dragged* by our *prior* belief, because our *prior* belief imposes a constraint even we have observed few evidence.

Although it seems like the *posterior* distribution of ***θ*** obtained by combining its *likelihood* and *improper prior* is acceptable in the first place, not too prominently, the *improperness* of the *improper prior* in this case is still highlighted as we already know that the *likelihood* is perfectly centered around the true parameter values of ***[3, 2]***, and now it is *shifted* away. But we would never notice this in practice as we wouldn't know the true model parameters. One can see that the direction of the *shifting likelihood* is towards the *prior*. But as it is a multiplication, the *shift* is not quite intense. The multiplication combines the large value in both *prior* and *likelihood*, thus the *shift* is along the gentle gradient of the *likelihood* while moving towards the *prior*. The major reason that the *posterior* is not too far away shifted from the *likelihood* is that the *improper prior* is relatively "flat", whereas the distribution rendered by the *likelihood* is strongly centered. So, the resultant multiplication would largely driven by the *likelihood*.

<p align="center"><img src="/img/bayes/3_post_likeli.gif" width="330" heigth="330"></p>

<p align="center">Fig.7 Posterior distribution/function for model parameters.</p>

Now, one may start to ask: what on earth is the point to use *prior* functions that are improperly or non-informatively designed?! Unfortunately, there is no one-fits-all answer. It is the nature of statistic inference that we are forced to make some assumptions from scratch, just like we have to assume there is a linear relationship already before obtaining more data points than only a few of them. There are [few discussions](https://stats.stackexchange.com/questions/27813/what-is-the-point-of-non-informative-priors) going around for choosing *prior* distributions scientifically. More formal research can be found in [**The Bayesian Choice: From Decision-Theoretic Foundations to Computational Implementation (Springer Texts in Statistics**](https://www.amazon.com/dp/0387715983/), as well as [**Moving beyond noninformative priors: why and how to choose weakly informative priors in Bayesian analyses**](https://onlinelibrary.wiley.com/doi/10.1111/oik.05985).

*Improper* or *non-informative prior* shouldn't be the reason to be pessimistic about *Bayesian principle*. In most cases, we are still doing [incremental science](https://www.statnews.com/2015/12/02/science-groundbreaking/), which means there is almost always some existing information we could leverage and to be encoded as *prior* knowledge, like the example of soil loss prediction mentioned earlier.

Even without the context of domain knowledge, using *improper prior* achieves some appealing results. A *prior*, along with the *likelihood* builds a connection to the idea of [***regularization***](http://primo.ai/index.php?title=Regularization), which is a technique explicitly modifies how the model parameters are sought given any specified *loss function*. If the *likelihood* is viewed as a *loss function* specified by observations, then the *prior* plays the role of a *regularizer* to constrain the model parameters from being solely controlled by the *loss function*. In practice, there are few design options for the *regularizer* for achieving different purposes: avoid overfitting (equivalent to *Gaussian prior function*), parameter selection (equivalent to *Laplace prior function*) and in combination. These *regularizer* are called in different ways, for instance, as shown in Fig.8.

<p align="center"><img src="/img/bayes/4_regularization.png" width="600" heigth="510"></p>

<p align="center">Fig.8 Regularization equivalence of prior functions (src: http://primo.ai/index.php?title=Regularization).</p>

The **biggest difference** is probably that *Bayesian principle* stays as a general statistical framework, whereas *regularization* is more commonly adopted from the frequentist perspective that a particular solution to the model parameter is expected.

Apart from the *prior* specification, it should NOT be a worse case where we have to heavily rely on observations. With the development of data acquirement approaches, we could be able to be confident about the model parameters with *likelihood*. The visualization of the *posterior* below (Fig.9) shows how increasing observations may update our knowledge even from a poor *prior*. The multiplication in ***P(D│θ)P(θ)/P(D)*** is explicitly two way: (1) the *prior* is constraining or dragging, while (2) the *likelihood* is updating and washing away the effect of constraining or dragging.

<p align="center"><img src="/img/bayes/4_postObs.gif" width="800" heigth="680"></p>

<p align="center">Fig.9 Change of posterior distribution/function for model parameters along with increasing observations.</p>

Holding the *posterior* at hand, we now have a more concentrated distribution of the model parameters. The final distribution of the ***θ*** in the *[θ<sub>1</sub> , θ<sub>2</sub>]<sup>T</sup>* space, as shown in Fig.8 is confidently centering the point *[3 , 2]*, while still shows possibility of parameters such as *[2.5 , 2]*. This probabilistic perspective is nice as the solution of the parameter is optimal at *[3 , 2]*, we are not rejecting other possibilities without knowing how the qualities of the observations are disturbed by those random noises. We have few options to deal with this *posterior*. Similar to maximizing the *likelihood*, we can seek the maximum-a-posteriori (***MAP***) probability to achieve the optimal model parameters. We can also stay with the probability distribution to quantify confidence of prediction.

Since it is a probability distribution in the *posterior*, it means sampling is possible from this distribution for visualization. As shown in Fig.9, drawn from the distribution, many samples are very close to *[3 , 2]* with few are bit far away from the center of the distribution. In the *[X , Y]* space, the drawn samples seems to be wobbling around a potential "datum line". This "datum line" is in fact ***M(x) = 3 + 2x***. Apparently, the sample lines are corresponding to the probability of drawn *[θ<sub>1</sub> , θ<sub>2</sub>]<sup>T</sup>*, thus we can encode this probability of ***θ*** to the lines, namely the inferred linear model. It then forms a probability distribution of potential linear models in the *[X , Y]* space and can be visualized as shadow to the right of Fig.10.

<p align="center"><img src="/img/bayes/5_postDraw.gif" width="800" heigth="680"></p>

<p align="center">Fig.10 Model possibilities in terms of posterior distribution of model parameters.</p>

The shadow defines an import confidence interval for making predictions: given any ***x<sub>new</sub>***, what is the probability of ***y<sub>new</sub>***? As the probability of giving rise to a ***y<sub>new</sub>*** is the probability of the model, the distribution of ***y<sub>new</sub>*** is now out-of-box given all the possible linear model. Mathematically, it is equivalent to weight all predictions made by each potential linear model by the probabilistic distribution of that model, as

***P(D<sub>new</sub>│D) = ʃ P(D<sub>new</sub>│θ,D)P(θ│D)dθ***

which is exactly how Fig.10 is plotted. The shadow is a combination of possible linear model weighted by their possibilities. The equation above also speaks the same idea: the *posterior* considers all possible ***θ***, which means the *posterior* does NOT care about the exact ***θ***! The integration plays a beautiful role to manifest such contradictory that all ***θ*** are involved (considered), but then are integrated out (does NOT care)!


## Wrap-up
-------------------
So far we have been stay intuitively by using general notations (such as ***P(θ)***), graphics and animations. In order to obtain exact measurement of the distribution regarding the *likelihood*, *prior*, and *posterior*, we can explicitly quantify the distributions by using hyper-parameters, for instance:
- ***P(θ) = P(θ│α) ~ N(θ ; 0, α<sup>-1</sup>I)*** for the *prior*, where ***α*** is the hyper-parameter controlling the shape of probabilistic distribution;
- ***P(D│θ) = P(y<sub>k</sub>│x<sub>k</sub> , θ, β) ~ N(y<sub>k</sub> ; θ<sup>T</sup>X, β<sup>-1</sup>I)***, where ***β*** is another hyper-parameter of precision (inverse variance) controlling the noise intensity as ***ε ~ N(ε ; 0, β<sup>-1</sup>I)***;
- we can even have design function ***φ*** for ***φ(x)=1+x***.

If we can be able to specify the hyper-parameters, the *posterior* is measurable, visually, the *width* and *center* of the shadow in Fig.10 is to be a function of hyper-parameters ***α*** and ***β***. These hyper-parameters can also be obtained automatically! The approach is called maximizing the *marginal likelihood* through [***Empirical Bayes method***](https://en.wikipedia.org/wiki/Empirical_Bayes_method), which will not be covered in this tutorial.

After digesting the mechanisms of the *Bayesian statistics* in linear regression, you are ready to leverage existing packages without bothering too much to write your own code. As long as you understand what does it mean by *likelihood*, *prior*, *posterior* and the role of those hyper-parameters, you can simple do as follows:

```python
from bayesian_linear_regression_util import *
```
If you do wish to specify your own *Bayesian* distributions for any parameters, say ***θ***, you can leverage [`PyMC3`](https://docs.pymc.io/) as:

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
- many distributions specified for *Bayesian* linear regression are *Gaussian* because of its nice property in conditioning, multiplication, etc.;
- maximum-likelihood-estimation (MLE) can be reliable if enough observations are made, but quite vulnerable to limited ones, which leads to a phenomenon of [*overfitting*](https://en.wikipedia.org/wiki/Overfitting) if someone is interested in the details;
- even the *improper* or *non-informative* *prior* functions are confusing with limited domain knowledge, they play a useful role mathematically and build the connection to an important regression strategy: *regularization*;
- although specifying the *prior* is sometimes referred as model selection, it is essentially specify the parameters of model, the assumption of the model as linear or non-linear is still arbitrary;
- for further interest: specifying the design function ***φ*** for the design function ***x*** can be related to kernel tricks and paves the way to [***non-parametric modeling***](https://en.wikipedia.org/wiki/Nonparametric_regression).
