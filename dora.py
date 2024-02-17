import tensorflow as tf
import numpy as np

tf.random.set_seed(0)

class DoRALayer(tf.keras.layers.Layer):
    def __init__(self, d_in, d_out, rank=4, weight=None, bias=None, **kwargs):
        super(DoRALayer, self).__init__(**kwargs)
        
        self.d_in = d_in
        self.d_out = d_out
        
        if weight is not None:
            self.weight = self.add_weight(shape=(d_out, d_in),
                                          initializer=tf.constant_initializer(weight),
                                          trainable=False)
        else:
            self.weight = self.add_weight(shape=(d_out, d_in),
                                          initializer='random_normal',
                                          trainable=False)
        
        if bias is not None:
            self.bias = self.add_weight(shape=(d_out,),
                                        initializer=tf.constant_initializer(bias),
                                        trainable=False)
        else:
            self.bias = self.add_weight(shape=(d_out,),
                                        initializer='zeros',
                                        trainable=False)
        
        std_dev = 1 / tf.sqrt(tf.cast(rank, tf.float32))
        self.lora_A = self.add_weight(shape=(d_out, rank),
                                      initializer=tf.random_normal_initializer(stddev=std_dev),
                                      trainable=True)
        self.lora_B = self.add_weight(shape=(rank, d_in),
                                      initializer='zeros',
                                      trainable=True)
    
    def call(self, inputs):
        lora = tf.matmul(self.lora_A, self.lora_B)
        adapted = self.weight + lora
        column_norm = tf.norm(adapted, ord=2, axis=0, keepdims=True)
        norm_adapted = adapted / column_norm
        m = tf.norm(self.weight, ord=2, axis=0, keepdims=True)
        calc_weights = m * norm_adapted
        return tf.add(tf.matmul(inputs, tf.transpose(calc_weights)), self.bias)

class SimpleModel(tf.keras.Model):
    def __init__(self, input_dim, output_dim):
        super(SimpleModel, self).__init__()
        self.layer1 = tf.keras.layers.Dense(output_dim, input_dim=input_dim)
    
    def call(self, inputs):
        return self.layer1(inputs)

def generate_data(num_samples=100, input_dim=10):
    X = np.random.randn(num_samples, input_dim).astype(np.float32)
    y = np.sum(X, axis=1, keepdims=True)
    return X, y

def count_model_params(model):
    total_params = sum(np.prod(var.shape) for var in model.variables)
    trainable_params = sum(np.prod(var.shape) for var in model.trainable_variables)
    return total_params, trainable_params

def print_model_params(model):
    total_params, trainable_params = count_model_params(model)
    print(f"Total Parameters: {total_params}")
    print(f"Trainable Parameters: {trainable_params}")

def train(model, optimizer, data, epochs=5):
    #print_model_params(model)
    for epoch in range(epochs):
        epoch_loss_avg = tf.keras.metrics.Mean()
        for step, (x_batch, y_batch) in enumerate(data):
            with tf.GradientTape() as tape:
                logits = model(x_batch, training=True)
                loss_value = tf.reduce_mean(tf.keras.losses.MSE(y_batch, logits))
            grads = tape.gradient(loss_value, model.trainable_weights)
            optimizer.apply_gradients(zip(grads, model.trainable_weights))
            epoch_loss_avg.update_state(loss_value)
        
        print(f"Epoch {epoch + 1}/{epochs}, Loss: {epoch_loss_avg.result().numpy()}")


def replace_linear_with_dora(model, input_dim, output_dim):
    for layer in model.layers:
        if isinstance(layer, tf.keras.layers.Dense):
            weights, biases = layer.get_weights()
            new_layer = DoRALayer(input_dim, output_dim, weight=weights, bias=biases)
            model.layers[model.layers.index(layer)] = new_layer
    return model

if __name__ == "__main__":
    input_dim, output_dim = 10, 1
    model = SimpleModel(input_dim, output_dim)
    optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)

    X, y = generate_data(num_samples=1000, input_dim=input_dim)
    dataset = tf.data.Dataset.from_tensor_slices((X, y)).batch(64).shuffle(buffer_size=1000)

    train(model, optimizer, dataset, epochs=100)


    model = replace_linear_with_dora(model, input_dim, output_dim)

    print("Continuing training with DoRA layers...")
    train(model, optimizer, dataset, epochs=5)
