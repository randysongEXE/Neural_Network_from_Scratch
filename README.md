# Neural_Network_from_Scratch

This project was created to explore how a neural network works, including its various components, layers, and functions. This neural network was written from scratch, and does not utilize the machine learning framworks Tensorflow or Pytorch. Instead, the libraries Numpy, Pandas, and Random were imported for matrices, data manipulation, and number generation support.

Overview
---------
In recent years, neural networks have become a deep area of interest and research due to their immense application potential. Similar to the human brain, neural networks are implemented through "neurons" that help them learn and model relationships, making it the backbone of machine learning. In essence, it is a series of algorithms that aim to recognize relationships in a set of data.

Components
---------
Generally, all neural networks contain some variation of these implementations:
 - Input layer: Where the network receives data. Each input neuron represents a single feature or attribute of the data.
 - Hidden layer(s): Where the network performs computations and transfers information from the input nodes to the output nodes. Some networks contain many hidden layers, which can be classified as "deep" neural networks.
 - Output layer: Where the final prediction is made or classification is made.
 - Weights and Biases: The parameters that the network adapts through learning. Weights control the degree to which two neurons are connected, which can become weaker or stronger. 
 - Activation Functions: This introduces non-linearity into the network. Without this function, a neural network would always behave like a single-layer network because the sum of linear functions is still equal to a linear function. Therefore, this would limit the complexity of functions that the network can learn.

Activation Function
----------
In general, there are two common types of activation functions: Sigmoid and ReLU. The sigmoid function is defined as f(x) = 1 / (1 + exp(-x)). It's an S-shaped curve that maps the input values into the range between 0 and 1. On the other hand, it is differentiable, which means we can find the slope of the sigmoid curve at any two points. However, it is important to note that there are some pros and cons to implementing the sigmoid function.
Pros:
- it is non-linear in nature, including combinations of itself
- retains a smooth gradient
- provides an analog activation

Cons:
- For extreme values of X, there is negligable change to the prediction, leading to the vanishing gradient problem
- computationally expensive and generally requires larger memory


