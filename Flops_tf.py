# After model.summary

import tensorflow as tf
from tensorflow.python.profiler.model_analyzer import profile
from tensorflow.python.profiler.option_builder import ProfileOptionBuilder

def get_flops(model):
  forward_pass = tf.function(model.call, input_signature=[tf.TensorSpec(shape=(1,) + model.input_shape[1:])])
  graph_info = profile(forward_pass.get_concrete_function().graph, options=ProfileOptionBuilder.float_operation())
  flops = graph_info.total_float_ops
  return flops

flops = get_flops(model)
macs = flops / 2
print(f"MACs: {macs:,}")
print(f"FLOPs: {flops:,}")
print(f"FLOPs: {flops / 1e9:.03} G")
print(f"MACs: {macs / 1e9:.03} G")


def keras_model_memory_usage_in_bytes(model,batch_size):
    """
    Return the estimated memory usage of a given Keras model in bytes.
    This includes the model weights and layers, but excludes the dataset.

    The model shapes are multipled by the batch size, but the weights are not.

    Args:
        model: A Keras model.
        batch_size: The batch size you intend to run the model with. If you
            have already specified the batch size in the model itself, then
            pass `1` as the argument here.
    Returns:
        An estimate of the Keras model's memory usage in bytes.

    """
    default_dtype = tf.keras.backend.floatx()
    shapes_mem_count = 0
    internal_model_mem_count = 0
    for layer in model.layers:
        if isinstance(layer, tf.keras.Model):
            internal_model_mem_count += keras_model_memory_usage_in_bytes(
                layer, batch_size=batch_size
            )
        single_layer_mem = tf.as_dtype(layer.dtype or default_dtype).size
        out_shape = layer.output_shape
        if isinstance(out_shape, list):
            out_shape = out_shape[0]
        for s in out_shape:
            if s is None:
                continue
            single_layer_mem *= s
        shapes_mem_count += single_layer_mem

    trainable_count = sum(
        [tf.keras.backend.count_params(p) for p in model.trainable_weights]
    )
    non_trainable_count = sum(
        [tf.keras.backend.count_params(p) for p in model.non_trainable_weights]
    )

    total_memory = (
        batch_size * shapes_mem_count
        + internal_model_mem_count
        + trainable_count
        + non_trainable_count
    )
    return total_memory
size_memory = keras_model_memory_usage_in_bytes(model,batch_size = 128)
print(size_memory)
print(f"Size_memory: {size_memory / 1e9:.03} G")

#####################################################
# Function to measure inference time
def measure_inference_time(model, input_data):
    start_time = time.time()
    _ = model(input_data)
    end_time = time.time()
    inference_time = (end_time - start_time) * 1000  # convert to milliseconds
    return inference_time


import time
input_shape = [1] + list(model.input_shape[1:])
input_data = tf.random.normal(input_shape)
inference_time = measure_inference_time(model, input_data)
print(f"Inference Time: {inference_time:.2f} ms")
