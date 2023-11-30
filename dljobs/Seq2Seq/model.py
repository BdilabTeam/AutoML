import tensorflow as tf
from tensorflow.keras.layers import Embedding


# 编码器
class Encoder(tf.keras.layers.Layer):
    def __init__(self, enc_units):
        super(Encoder, self).__init__()
        self.enc_units = enc_units

        # The embedding layer converts tokens to vectors
        # The GRU RNN layer processes those vectors sequentially.
        self.gru = tf.keras.layers.GRU(self.enc_units, return_sequences=True, return_state=True,
                                       recurrent_initializer='glorot_uniform')

    def call(self, tokens, state=None):
        output, state = self.gru(tokens, initial_state=state)

        # 返回处理后的序列（传给注意力头），内部状态（用于初始化解码器）
        return output, state


# 注意力
class BahdanauAttention(tf.keras.layers.Layer):
    def __init__(self, units):
        super().__init__()
        # the  Bahdanau attention
        self.W1 = tf.keras.layers.Dense(units, use_bias=False)
        self.W2 = tf.keras.layers.Dense(units, use_bias=False)

        self.attention = tf.keras.layers.AdditiveAttention()

    def call(self, query, value, mask):
        # From Eqn. (4), `W1@ht`.
        w1_query = self.W1(query)

        # From Eqn. (4), `W2@hs`.
        w2_key = self.W2(value)
        query_mask = tf.ones(tf.shape(query)[:-1], dtype=bool)
        value_mask = mask

        context_vector, attention_weights = self.attention(inputs=[w1_query, value, w2_key], mask=[query_mask, value_mask],
                                                           return_attention_scores=True)

        return context_vector, attention_weights


# 解码器
class Decoder(tf.keras.layers.Layer):
    def __init__(self, dec_units, output_vocab_size):
        super(Decoder, self).__init__()
        self.dec_units = dec_units
        self.output_vocab_size = output_vocab_size

        # The RNN keeps track of what's been generated so far.
        self.gru = tf.keras.layers.GRU(self.dec_units, return_sequences=True, return_state=True, recurrent_initializer='glorot_uniform')

        # The RNN output will be the query for the attention layer.
        self.attention = BahdanauAttention(self.dec_units)

        # converting `ct` to `at`
        self.Wc = tf.keras.layers.Dense(dec_units, activation=tf.math.tanh, use_bias=False)

        # This fully connected layer produces the logits for each
        # output token.
        self.fc = tf.keras.layers.Dense(self.output_vocab_size)

    def call(self, new_tokens, enc_output, mask, state=None):
        # Step 1. Lookup the embeddings
        # Step 2. Process one step with the RNN
        rnn_output, state = self.gru(new_tokens, initial_state=state)

        context_vector, attention_weights = self.attention(query=rnn_output, value=enc_output, mask=mask)
        context_and_rnn_output = tf.concat([context_vector, rnn_output], axis=-1)

        # Step 4. Eqn. (3): `at = tanh(Wc@[ct; ht])`
        attention_vector = self.Wc(context_and_rnn_output)
        # shape_checker(attention_vector, ('batch', 't', 'dec_units'))

        # Step 5. Generate logit predictions:
        logits = self.fc(attention_vector)
        # shape_checker(logits, ('batch', 't', 'output_vocab_size'))

        return logits, attention_weights, state


# 损失函数
class MaskedLoss(tf.keras.losses.Loss):
    def __init__(self):
        self.name = 'masked_loss'
        self.loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True, reduction='none')

    def __call__(self, y_true, y_pred):

        # Calculate the loss for each item in the batch.
        loss = self.loss(y_true, y_pred)

        # Mask off the losses on padding.
        mask = tf.cast(y_true != 0, tf.float32)
        loss *= mask

        # Return the total.
        return tf.reduce_sum(loss)


# 训练步骤
class TrainTranslator(tf.keras.Model):
    def __init__(self, embedding_dim, units, input_text_processor, output_text_processor):
        super().__init__()
        # Build the encoder and decoder
        encoder = Encoder(units)
        decoder = Decoder(units, output_text_processor.vocabulary_size())

        self.encoder = encoder
        self.decoder = decoder
        self.input_text_processor = input_text_processor
        self.output_text_processor = output_text_processor
        self.embedding_dim = embedding_dim
        self.max_vocab_size = input_text_processor.vocabulary_size()

    def train_step(self, inputs):
        return self._train_step(inputs)

    def _preprocess(self, input_text, target_text):
        # Convert the text to token IDs
        input_tokens = self.input_text_processor(input_text)
        target_tokens = self.output_text_processor(target_text)
        embedding = Embedding(self.max_vocab_size, self.embedding_dim)

        # Convert IDs to masks.
        input_mask = input_tokens != 0

        target_mask = target_tokens != 0

        input_tokens = embedding(input_tokens)
        target_embedding = embedding(target_tokens)
        # print(input_tokens.shape)
        # print(target_tokens.shape)

        return input_tokens, input_mask, target_tokens, target_mask, target_embedding

    def _train_step(self, inputs):
        input_text, target_text = inputs

        (input_tokens, input_mask, target_tokens, target_mask, target_embedding) = self._preprocess(input_text, target_text)

        max_target_length = tf.shape(target_tokens)[1].numpy()



        with tf.GradientTape() as tape:
            # Encode the input
            enc_output, enc_state = self.encoder(input_tokens)

            # Initialize the decoder's state to the encoder's final state.
            # This only works if the encoder and decoder have the same number of
            # units.
            dec_state = enc_state
            loss = tf.constant(0.0)

            for t in range(max_target_length - 1):
                # Pass in two tokens from the target sequence:
                # 1. The current input to the decoder.
                # 2. The target the target for the decoder's next prediction.
                new_tokens = target_tokens[:, t:t + 2]
                new_embedding = target_embedding[:, t:t + 2]
                step_loss, dec_state = self._loop_step(new_tokens, input_mask,
                                                       enc_output, dec_state, new_embedding)
                loss = loss + step_loss

            # Average the loss over all non padding tokens.
            average_loss = loss / tf.reduce_sum(tf.cast(target_mask, tf.float32))

        # Apply an optimization step
        variables = self.trainable_variables
        gradients = tape.gradient(average_loss, variables)
        self.optimizer.apply_gradients(zip(gradients, variables))

        # Return a dict mapping metric names to current value
        return {'batch_loss': average_loss}

    def _loop_step(self, new_tokens, input_mask, enc_output, dec_state, new_embedding):
        target_token = new_tokens[:, 1:2]
        input_embedding = new_embedding[:, 0:1]
        logits, attention_weights, dec_state = self.decoder(input_embedding, enc_output, input_mask, state=dec_state)


        # print(type(target_token), target_token, target_token.shape)
        # print(type(logits), logits, logits.shape)
        # `self.loss` returns the total for non-padded tokens
        y = target_token
        y_pred = logits
        step_loss = self.loss(y, y_pred)

        return step_loss, dec_state

