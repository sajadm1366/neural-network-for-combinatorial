import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from set2seq import Set2seq
from dataset import data_gen
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--max_val', type=int, default=20)
parser.add_argument('--max_len', type=int, default=5)
parser.add_argument('--N', type=int, default=5000)

#################################################################
##
@tf.function
def train_step(model, x, y, vocab_size_trg):
    ''' One step of the training'''
    dec_input_y = tf.concat([vocab_size_trg * tf.ones([y.shape[0], 1], dtype=tf.int32), y[:, :-1]], axis=1)
    with tf.GradientTape() as tape:
        logits = model(x, dec_input_y, training=True)


        labels = tf.one_hot(y, depth=vocab_size_trg)
        loss_unweight = tf.nn.softmax_cross_entropy_with_logits(
            labels, logits, axis=-1)

        loss_weight = tf.reduce_mean(tf.reduce_sum(loss_unweight, axis=1))

    grads = tape.gradient(loss_weight, model.trainable_variables)
    model.optimizer.apply_gradients(zip(grads, model.trainable_variables))

    return loss_weight



############################################################################
##
def eval(model, max_val, max_len):
    for _ in range(20):
        x_test = np.random.randint(max_val, size=(1, max_len))
        y_test = np.sort(x_test)  # sort in ascending order
        y_pred = model(x_test, None,  training=False)
        print('print some examples')
        print(f'input: {x_test[0]}, output: {y_pred}')

##############################################################################
##
def train(max_val, max_len, N):
    ''' train on epochs'''
    data = data_gen(N=N, max_val=max_val, max_len=max_len)
    training_data = data.shuffle(buffer_size=500)
    training_data_batch = training_data.batch(64)
    vocab_size_trg = max_val + 1

    model = Set2seq(vocab_size_src=max_val, vocab_size_trg=vocab_size_trg, time_step=max_len)
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001))

    loss_all = []
    for batch in range(1500):
        losses = []
        for (x, y) in training_data_batch:
            loss_weight = train_step(model, x, y, vocab_size_trg)

            losses.append(loss_weight.numpy())
        loss_all.append(np.mean(losses))
        print(f"epoch: {batch}, loss_val: {np.mean(losses)}")
        # save the model parameters
        if batch % 25 == 0:
           model.save_weights("saved_model/set2seq_weights")

    # plt.plot(loss_all)
    # plt.grid()
    # plt.xlabel("epochs")
    # plt.xlabel("loss")
    # plt.show()
    #print some exmmples

    eval(model, max_val, max_len)
######################################################################
##
if __name__ == "__main__":
    args = parser.parse_args()
    max_val = args.max_val
    max_len = args.max_len
    N = args.N
    train(max_val, max_len, N)