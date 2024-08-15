import tensorflow as tf

def adding_gaussian_noise_portion(inp, noise_level=0.1):
    last_column = inp[:, :, -1]
    std = tf.sqrt(tf.reduce_mean(tf.square(last_column)))
    noise = tf.random.normal(shape=tf.shape(last_column), mean=0.0, stddev=noise_level*std, dtype=last_column.dtype)
    noisy_last_column = last_column + noise
    inp_without_last_column = inp[:, :, :-1]
    inp_with_noise = tf.concat([inp_without_last_column, tf.expand_dims(noisy_last_column, -1)], axis=-1)
    return inp_with_noise


def add_noise_to_half_batches(inp, noise_level=0.1):
    stddevs = tf.math.reduce_std(inp[:, :, -1], axis=1)
    random_values = tf.random.uniform(tf.shape(stddevs))

    mask = random_values < 0.5 
    mask = tf.cast(mask, tf.float32)
    mask = tf.reshape(mask, [-1, 1])

    noise = tf.random.normal(tf.shape(inp[:,:,-1]), mean=0.0, stddev=noise_level)
    adjusted_noise = noise * mask * tf.expand_dims(stddevs, -1)
    inp_noisy_and_clean = inp[:, :, -1] + adjusted_noise

    # full_inp_noisy = inp[:, :, -1] + noise * tf.expand_dims(stddevs, -1)
    inp_with_noise_and_clean = tf.concat([inp[:,:,:-1], tf.expand_dims(inp_noisy_and_clean, -1)], axis=-1)
    # full_noise = tf.concat([inp[:,:,:-1], tf.expand_dims(full_inp_noisy, -1)], axis=-1)
    return inp_with_noise_and_clean
