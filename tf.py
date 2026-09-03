import tensorflow as tf

a = tf.constant([[1, 2], [3, 4]])
b = tf.constant([[5, 6], [7, 8]])

print("Tensor A:")
print(a)

print("\nTensor B:")
print(b)

print("\nAddition:")
print(tf.add(a, b))

print("\nSubtraction:")
print(tf.subtract(a, b))

print("\nElement-wise Multiplication:")
print(tf.multiply(a, b))

print("\nMatrix Multiplication:")
print(tf.matmul(a, b))

print("\nReshaped Tensor:")
print(tf.reshape(a, [4]))

x = tf.constant(10)
y = tf.constant(20)

print("\nEager Execution:")
print("x + y =", x + y)
