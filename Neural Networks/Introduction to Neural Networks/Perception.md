# Perceptron

A Perceptron is an algorithm for supervised learning of binary classifiers. This algorithm enables neurons to learn and processes elements in the training set one at a time.

![img](https://www.simplilearn.com/ice9/free_resources_article_thumb/general-diagram-of-perceptron-for-supervised-learning.jpg)



# Perceptron at a Glance

Perceptron has the following characteristics:

- Perceptron is an algorithm for Supervised Learning of single layer binary linear classifier.
- Optimal weight coefficients are automatically learned.
- Weights are multiplied with the input features and decision is made if the neuron is fired or not.
- Activation function applies a step rule to check if the output of the weighting function is greater than zero.
- Linear decision boundary is drawn enabling the distinction between the two linearly separable classes +1 and -1.
- If the sum of the input signals exceeds a certain threshold, it outputs a signal; otherwise, there is no output.

Types of activation functions include the sign, step, and sigmoid functions.



### Introduced by Frank Rosenblatt in 1957



# Function

Perceptron is a function that maps its input “x,” which is multiplied with the learned weight coefficient; an output value ”f(x)”is generated.

![img](https://www.simplilearn.com/ice9/free_resources_article_thumb/mathematical-function-representation-of-perceptron.jpg)

In the equation given above:

“w” = vector of real-valued weights

“b” = bias (an element that adjusts the boundary away from origin without any dependence on the input value)

“x” = vector of input x values

![img](https://www.simplilearn.com/ice9/free_resources_article_thumb/value-of-vector-of-input-x-values-in-perceptron-function.jpg)

“m” = number of inputs to the Perceptron

The output can be represented as “1” or “0.”  It can also be represented as “1” or “-1” depending on which activation function is used.

Let us learn the inputs of a perceptron in the next section.



# Types of Logic gate

- AND
- NAND
- OR
- NOR
- NOT
- XOR
- XNOR



**AND**

If the two inputs are TRUE (+1), the output of Perceptron is positive, which amounts to TRUE.

**OR**

If  either of the two inputs are TRUE (+1), the output of Perceptron is positive, which amounts to TRUE.

**XOR**

A XOR gate, also called as Exclusive OR gate, has two inputs and one output.

![img](https://www.simplilearn.com/ice9/free_resources_article_thumb/symbolic-representation-for-xor-gate.jpg)

The gate returns a TRUE as the output if and ONLY if one of the input states is true.

XOR Truth Table

| **Input** | **Output** |      |
| --------- | ---------- | ---- |
| A         | B          |      |
| 0         | 0          | 0    |
| 0         | 1          | 1    |
| 1         | 0          | 1    |
| 1         | 1          | 0    |



# Perceptron Trick

For finding best line,

- we need to subtract for misclassified lines

- for correctly classified lines, use addition

  

Let's say we have an equation

3x<sub>1</sub> + 4x<sub>2</sub> - 10 = 0

and Poin: (4, 5)

```python
3 	4	-10
4	5	1
_________(-)
-1	-1	 -11
```

this line moves drastically towards the point, so to prevent this, we use learning rate

Say **Learning rate = 0.1**

before subtraction, multiply with learning rate

```java
3 		4		-10
4*0.1	5*0.1	1*0.1
______________________(-)
2.6		3.5		-10.1
```



# Perceptron Algorithm

Steps -

Start with some random weight and bias

For every misclassified point

- If the prediction is zero
  - that means positive point in negative area
  - change w1 + ax. where a is learning rate
  - change b to b + a

- If the prediction is 1 then
  - that means negative point in the positive area
  -  change w1 - ax. where a is learning rate
  - change b to b - a



# Tutorials

- [What is Perceptron](https://www.simplilearn.com/what-is-perceptron-tutorial)