"""
Purpose:
This file introduces the basic building blocks of TensorFlow.
We want to see how tensors work and how TensorFlow performs common math operations on them.

What we are trying to achieve:
Understand the core tensor operations that later become the foundation of machine learning and deep learning.

Overall flow:
1. Create example tensors.
2. Perform basic arithmetic on them.
3. Show matrix multiplication.
4. Reshape a tensor into a new form.
5. Demonstrate eager execution with simple scalar addition.
"""

Theory:
TensorFlow works with tensors, which are arrays of numbers.
Basic operations like addition, multiplication, reshape, and matrix multiplication are the building blocks of neural network computation.

Viva:
Q: What is a tensor?
A: A multi-dimensional array of numbers.
Q: What is matrix multiplication used for?
A: It is a core operation in neural networks and linear algebra.
Q: What is eager execution?
A: Immediate computation of operations instead of building a delayed graph first.

Output:
The program prints two tensors and the results of addition, subtraction, multiplication, matrix multiplication, reshaping, and scalar addition.

import tensorflow as tf  # TensorFlow is a library for working with tensors and neural networks.

# Term: tensor means a multi-dimensional array of numbers used for computation.
# Create tensors.
# Why: tensors are the basic data objects in TensorFlow, like arrays with extra meaning.
a = tf.constant([[1, 2], [3, 4]])  # A 2x2 tensor with small example values.
b = tf.constant([[5, 6], [7, 8]])  # Another 2x2 tensor for operations.

# Term: constant means a value that does not change during computation.
print("Tensor A:")  # Label the output so we know what we are looking at.
print(a)  # Show the first tensor.

print("\nTensor B:")  # Blank line makes the output easier to read.
print(b)  # Show the second tensor.

# Term: element-wise means doing the operation separately on matching positions.
# Basic operations.
# Why: tensor math is the foundation of deep learning computations.
print("\nAddition:")  # Add corresponding elements from both tensors.
print(tf.add(a, b))  # Element-wise addition.

print("\nSubtraction:")  # Subtract corresponding elements.
print(tf.subtract(a, b))  # Element-wise subtraction.

print("\nElement-wise Multiplication:")  # Multiply matching elements.
print(tf.multiply(a, b))  # Element-wise multiplication.

# Term: matrix multiplication means combining rows and columns using dot products.
# Matrix multiplication.
# Why: matrix multiplication is the key operation behind many neural network layers.
print("\nMatrix Multiplication:")  # Different from element-wise multiplication.
print(tf.matmul(a, b))  # Compute the dot-product-based matrix result.

# Term: reshape means changing the shape of data without changing the values.
# Tensor manipulation.
# Why: reshaping changes the view of the data without changing the actual values.
print("\nReshaped Tensor:")  # Flatten the 2x2 tensor into a 1D vector.
print(tf.reshape(a, [4]))  # Reshape into a vector of length 4.

# Term: eager execution means TensorFlow computes operations immediately instead of building a delayed graph first.
# Eager execution.
# Why: in eager mode, TensorFlow computes results immediately, which is easier to debug and understand.
x = tf.constant(10)  # A single scalar tensor.
y = tf.constant(20)  # Another scalar tensor.

print("\nEager Execution:")  # Show that arithmetic works directly.
print("x + y =", x + y)  # TensorFlow evaluates the addition right away.
