import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers


class DeepSet(keras.Model):
    ''' this an implementation  fo deep set model: invariant to permutation of the input set'''
    def __init__(self, vocab_size, embed_size, num_hiddens, len_set):
        super(DeepSet, self).__init__()
        self.len_set = len_set
        self.embed_size = embed_size
        self.num_hiddens = num_hiddens
        self.embd = tf.keras.layers.Embedding(vocab_size, embed_size)
        self.phi_layer1 = keras.layers.Dense(num_hiddens, activation='relu', input_shape=(embed_size,))
        self.phi_layer2 = keras.layers.Dense(num_hiddens, activation='relu')
        self.phi_out = keras.layers.Dense(num_hiddens)

        self.ro_layer1 = keras.layers.Dense(num_hiddens, activation='relu', input_shape=(num_hiddens,))
        self.ro_layer2 = keras.layers.Dense(num_hiddens, activation='relu')
        self.ro_out = keras.layers.Dense(num_hiddens)

    def call(self, x):
        out_embd = self.embd(x)
        #         print(out_embd.shape)
        out = tf.reshape(out_embd, [-1, self.embed_size])
        out = self.phi_layer1(out)
        out = self.phi_layer2(out)
        out = self.phi_out(out)
        #         print(out.shape)

        out = tf.reshape(out, [-1, self.len_set, self.num_hiddens])
        out = tf.reduce_sum(out, axis=1)
        #         print(out.shape)
        out = self.ro_layer1(out)
        out = self.ro_layer2(out)
        return self.ro_out(out)




class Decoder(keras.Model):
    ''' Decoder model: takes context vector and outputs target langage'''

    def __init__(self, vocab_size, embed_size, num_hiddens,time_steps):
        super(Decoder, self).__init__()
        self.vocab_size = vocab_size
        self.embed_size = embed_size
        self.num_hiddens = num_hiddens
        self.time_steps = time_steps


        self.embed = layers.Embedding(input_dim=vocab_size, output_dim=embed_size)
        self.lstm = layers.LSTM(num_hiddens, return_sequences=True, return_state=True)
        self.dense_1 = tf.keras.layers.Dense(512, activation='relu')
        self.dense_2 = tf.keras.layers.Dense(vocab_size)

    def call(self, y, memory_carry_encoder):
        ''' Forward pass
            inputs:
                  y: target language  (Batch, length)
                  memory_carry_encoder: context vector  (2, hidden_dims)
            output:
                 output:  (Batch, length, target_vocab_size)
        '''

        y = self.embed(y)
        out_lstm, _, _ = self.lstm(inputs=y, initial_state=memory_carry_encoder)
        out_reshape = tf.reshape(out_lstm, [-1, self.num_hiddens])

        out_reshape = self.dense_1(out_reshape)
        out_reshape = self.dense_2(out_reshape)

        out = tf.reshape(out_reshape, [-1, self.time_steps, self.vocab_size])
        return out


class Set2seq(keras.Model):
    '''
    sequence to sequence model
    '''
    def __init__(self, vocab_size_src, vocab_size_trg, time_step):
        super(Set2seq, self).__init__()
        self.enc = DeepSet(vocab_size=vocab_size_src,  embed_size=128, num_hiddens=128, len_set=time_step)
        self.dec = Decoder(vocab_size=vocab_size_trg, embed_size=128, num_hiddens=128, time_steps=time_step)
        self.num_hiddens = 128

    def call(self, source, target, training):
        '''
        Forward pass
        inputs:
            source: source lanaguge
            target: target languge
        outputs:

        '''
        if training:
            final_memory = self.enc(source) # batch_size * hidden_size
            final_memory_carry = []
            final_memory_carry.append(final_memory)
            final_memory_carry.append(tf.zeros([source.shape[0], self.num_hiddens], tf.float32))
            return self.dec(target, final_memory_carry)
        else:

            final_memory = self.enc(source)  # batch_size * hidden_size
            final_memory_carry = []
            final_memory_carry.append(final_memory)
            final_memory_carry.append(tf.zeros([source.shape[0], self.num_hiddens], tf.float32))


            y_init = [[21]]
            word = []
            for i in range(5):
                y = tf.constant(y_init, dtype=tf.float32)
                y = self.dec.embed(y)
                out_lstm, *final_memory_carry = self.dec.lstm(inputs=y, initial_state=final_memory_carry)

                out = out_lstm[0]
                out = self.dec.dense_1(out)
                out = self.dec.dense_2(out)
                y_pred_index = np.argmax(tf.nn.softmax(out, axis=1)[0].numpy())
                y_init = [[y_pred_index]]
                word.append(y_pred_index)
            return word






