# Multi class Classification Problem

On binary class classification problem, we use **Sigmoid** function, but now for we use **Softmax** function.

## Softmax

 The Softmax function is a function used for multiclass classification problem which returns probabilities of each class in a group of different classes, with the target class having the highest probability. The calculated probabilities are then helpful in determining the target class for the given inputs. 

 ![img](https://miro.medium.com/max/976/1*48FpDngytN34rvVlnw0ojA.jpeg) 

## Implement Softmax function

```python
import numpy as np

def softmax(x):
    ex = np.exp(x)
    s = np.sum(np.exp(x))
    return ex/s


arr = np.array([1, 2, 0, 2, 1, 4])
softmax(arr)
```

