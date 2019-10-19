# Maximum Likelihood

Lets say we start from bad modeling and calculate the probability of the classes, are those classes. Then multiply them and obtain total probability. Our goal is maximize this probability. This is called Maximum Likelihood.

From maximum Likelihood, we want to sum of instead of multiply all the classes probabilities to calculate total probability. To do this we use **log**

```
log(ab)= log(a) + log(b)
```

Since all the values between 0 and 1, after using log, those number will be converted into negative number. So, the solution is use negative **log**.

Now sum up all the values with negative **log** of the probabilities. This is call the **Cross entropy**. 

#### Note: Good model give low cross entropy and bad model give high entropy

# Cross Entropy

Cross entropy says?If I have a bunch of events and a bunch of probabilities, how likely is it that those events happen based on the probabilities?

If it is very likely -> low cross entropy

unlikely -> large cross entropy

It's simply says, the difference between two vectors.

## Example

Three door with gift and no gift

![1571201913906](https://github.com/Iamsdt/UdacityDeepLearningNanodegree/raw/master/img/1571201913906.png)

Now construct an equation

## Equation

![1571202865821](https://github.com/Iamsdt/UdacityDeepLearningNanodegree/raw/master/img/1571202865821.png)

 And calculate a math using this example

```
CE = -(ln(0.8) + ln(0.7) + ln(1-0.1))
CE = -(-.22 - .36 - .11)
CE = 0.68
```

## Implement in code

```python
import numpy as np

def cross_entropy(Y, P):
    y = np.float_(Y)
    p = np.float_(P)
    ce = y*np.log(p) + (1-y)*np.log(1-p)
    return -np.sum(ce)
```

## Multiclass Cross Entropy

![1571203421674](https://github.com/Iamsdt/UdacityDeepLearningNanodegree/raw/master/img/1571203421674.png)

Where, 

m = is the number of classes

p = probabilities

y = predictions

