# Vanilla Minibatch Gradient Descent

while True:
    data_batch = sample_training_data(data,256)# sample 256 examples
    weights_grad = evaluate_gradient(loss_fun, data_batch, weights)
    weights += -step_size* weights_grad #perform parameter update