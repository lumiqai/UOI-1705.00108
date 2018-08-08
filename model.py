import keras
from keras_contrib.layers import CRF
from keras_wc_embd import get_embedding_layer


def build_model(word_dict_len,
                char_dict_len,
                max_word_len,
                output_dim,
                bi_lm_model,
                rnn_1_dim=150,
                rnn_2_dim=150,
                rnn_type='gru',
                word_dim=300,
                char_dim=80,
                char_embd_dim=25,
                word_embd_weights=None):
    """Build model for NER.

    :param word_dict_len: The number of words in the dictionary.
    :param char_dict_len: The numbers of characters in the dictionary.
    :param max_word_len: The maximum length of a word in the dictionary.
    :param output_dim: The output dimension / number of NER types.
    :param bi_lm_model: The trained BiLM model.
    :param word_dim: The dimension of word embedding.
    :param rnn_1_dim: The dimension of RNN after word/char embedding.
    :param rnn_2_dim: The dimension of RNN after embedding and bidirectional language model.
    :param rnn_type: The type of the two RNN layers.
    :param char_dim: The final dimension of character embedding.
    :param char_embd_dim: The embedding dimension of characters before bidirectional RNN.
    :param word_embd_weights: Pre-trained embeddings for words.

    :return model: The built model.
    """
    inputs, embd_layer = get_embedding_layer(
        word_dict_len=word_dict_len,
        char_dict_len=char_dict_len,
        max_word_len=max_word_len,
        word_embd_dim=word_dim,
        char_hidden_dim=char_dim // 2,
        char_embd_dim=char_embd_dim,
        word_embd_weights=word_embd_weights,
    )
    if rnn_type == 'gru':
        rnn = keras.layers.GRU
    else:
        rnn = keras.layers.LSTM
    dropout_layer_1 = keras.layers.Dropout(rate=0.25, name='Dropout-1')(embd_layer)
    bi_rnn_layer_1 = keras.layers.Bidirectional(
        layer=rnn(
            units=rnn_1_dim,
            dropout=0.0,
            recurrent_dropout=0.0,
            return_sequences=True,
        ),
        name='Bi-RNN-1',
    )(dropout_layer_1)
    lm_layer = bi_lm_model.get_feature_layers(input_layer=inputs[0])
    embd_lm_layer = keras.layers.Concatenate(name='Embd-Bi-LM-Feature')([bi_rnn_layer_1, lm_layer])
    dropout_layer_2 = keras.layers.Dropout(rate=0.25, name='Dropout-2')(embd_lm_layer)
    bi_rnn_layer_2 = keras.layers.Bidirectional(
        layer=rnn(
            units=rnn_2_dim,
            dropout=0.0,
            recurrent_dropout=0.0,
            return_sequences=True,
        ),
        name='Bi-RNN-2',
    )(dropout_layer_2)
    dense_layer = keras.layers.Dense(units=output_dim, name='Dense')(bi_rnn_layer_2)

    crf_layer = CRF(
        units=output_dim,
        sparse_target=True,
        name='CRF',
    )
    model = keras.models.Model(inputs=inputs, outputs=crf_layer(dense_layer))
    model.compile(
        optimizer=keras.optimizers.Adam(),
        loss=crf_layer.loss_function,
        metrics=[crf_layer.accuracy],
    )
    return model
