""" The RNN model to transform text into music using Machine Translation """
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.python.keras.models import load_model
from tensorflow.python.keras.preprocessing.sequence import pad_sequences
from keras.layers import Input, LSTM, Embedding, Dense, Bidirectional, Concatenate
from keras.callbacks import ModelCheckpoint, EarlyStopping
from tensorflow.keras.utils import plot_model
from keras.models import Model
from matplotlib import pyplot as plt
import numpy as np
from string import digits
import contractions
import pickle
import glob
import re
import os


def load_datastructures(lyrics_path, notes_path):
    """ Loads the lyrics and notes data structures from the .pkl file at the given paths.
    :param lyrics_path: path to the .pkl file containing the lyrics data structure.
    :param notes_path: path to the .pkl file containing the notes data structure.
    :return: lyrics and notes data structures. """

    with open(lyrics_path, 'rb') as l_path:
        lyrics = pickle.load(l_path)
    l_path.close()

    with open(notes_path, 'rb') as n_path:
        notes = pickle.load(n_path)
    n_path.close()

    return lyrics, notes


def plot_training(history, training_data, validation_data, title, x_label, y_label):
    """ Plots and compare some training and validation data of the model.
    :param history: training history of the model.
    :param training_data: training data to plot.
    :param validation_data: validation data to plot.
    :param title: title of the plot.
    :param x_label: x-axis label of the plot.
    :param y_label: y-axis label of the plot. """

    # Get the training and validation loss values.
    plt.plot(history.history[training_data])
    plt.plot(history.history[validation_data])

    # Set title and labels.
    plt.title(title)
    plt.ylabel(y_label)
    plt.xlabel(x_label)
    plt.legend(['train', 'test'], loc='upper left')

    # Plot.
    plt.show()


def save_datastructures(path, dst_to_be_saved):
    """ Saves the given data structure to the .pkl file at the given path. """

    dst_to_be_saved.to_pickle(path)


class RNNModel:
    """ A class to load the Recurrent Neural Network model. """

    def __init__(self):
        """ Initialize the model by:
        1. Loading the necessary data structures.
        2. Cleaning the data structures.
        3. Building the tokenizers.
        4. Building the respective vocabularies.
        5. Padding up to a given length.
        6. Building the model. """

        # Load and process the data structures in memory.
        lyrics_path = 'flask_source/static/assets/pickle-dsts/songs_lyrics_split'
        notes_path = 'flask_source/static/assets/pickle-dsts/songs_notes_split'
        self.lyrics_dst, self.notes_dst = load_datastructures(lyrics_path, notes_path)
        self.lyrics = [' '.join(song) for song in self.lyrics_dst]  # make sentences out of lyrics.
        self.notes = [' '.join(song_notes) for song_notes in self.notes_dst]  # make sentences out of notes.
        self.clean_data_and_process()  # prepare the data structures for the learning.

        # Construct vocabularies for training and prediction.
        self.LYRICS_VOCAB_SIZE = 0
        self.NOTES_VOCAB_SIZE = 0
        self.max_song_length = 0
        self.max_notes_length = 0
        self.songs_words_index = None
        self.songs_index_words = None
        self.notes_words_index = None
        self.notes_index_words = None

        # Given the processed data, build the respective tokenizer(s).
        songs_tokenizer, notes_tokenizer, songs_encoded, notes_encoded = self.build_tokenizers()
        self.build_vocabularies(songs_tokenizer, notes_tokenizer)

        # The source and target are padded to the same length to allow for training.
        self.songs_padded, self.notes_padded = self.pad_given_length(songs_encoded, notes_encoded)

        # Build the encoder-decoder model structure
        self.model = None
        self.encoder_model = None
        self.decoder_model = None
        self.build_models()

    def clean_data_and_process(self):
        """ Clean the data structures and process them for the learning. """

        # Remove digits from lyrics
        remove_digits = str.maketrans('', '', digits)
        self.lyrics = [song.translate(remove_digits) for song in self.lyrics]

        # Add a 'start of song' and 'end of song' delimiter to every set of notes.
        self.notes = ['sos ' + x + ' eos' for x in self.notes]

    @staticmethod
    def tokenize(corpus, is_notes):
        """ Upon input of some text, return its tokenizer
        along with its encoded version. When tokenizing notes, text
        is not converted to lowercase and it is not filtered for
        punctuation (to avoid deleting chords).
        :param corpus: the text to tokenize.
        :param is_notes: whether the text is notes or lyrics.
        :return: the tokenizer and the encoded version of the text. """

        tokenizer = Tokenizer(lower=False, filters='') if is_notes else Tokenizer()
        tokenizer.fit_on_texts(corpus)

        return tokenizer, tokenizer.texts_to_sequences(corpus)

    def build_tokenizers(self):
        """ Build the tokenizers for the songs and notes.
        Also updates the vocabularies size and, the max song length
        and max notes length.
        :return: the tokenizers for the songs and notes, and their encoded version. """

        songs_tokenizer, songs_encoded = self.tokenize(self.lyrics, is_notes=False)
        notes_tokenizer, notes_encoded = self.tokenize(self.notes, is_notes=True)

        # Update variables with acquired information.
        # Adding +1 for NN layers compatibility.
        self.LYRICS_VOCAB_SIZE = len(songs_tokenizer.word_counts) + 1
        self.NOTES_VOCAB_SIZE = len(notes_tokenizer.word_counts) + 1
        self.max_song_length = self.find_max_length(songs_encoded)
        self.max_notes_length = self.find_max_length(notes_encoded)

        return songs_tokenizer, notes_tokenizer, songs_encoded, notes_encoded

    @staticmethod
    def find_max_length(encoded):
        """ Find the maximum length of a list of encoded sequences.
        :param encoded: the list of encoded sequences.
        :return: the maximum length of the encoded sequences. """

        max_len = 0
        for sequence in encoded:
            if len(sequence) > max_len:
                max_len = len(sequence)
        return max_len

    def pad_given_length(self, songs_encoded, notes_encoded):
        """ Pad the encoded sequences to the same length.
        :param songs_encoded: the encoded songs.
        :param notes_encoded: the encoded notes.
        :return: the padded encoded songs and notes. """

        songs_padded = pad_sequences(songs_encoded, maxlen=self.max_song_length, padding='post')
        notes_padded = pad_sequences(notes_encoded, maxlen=self.max_notes_length, padding='post')

        # Return the padded encoded sequences as numpy arrays to allow for training.
        songs_padded = np.array(songs_padded)
        notes_padded = np.array(notes_padded)

        return songs_padded, notes_padded

    def build_vocabularies(self, songs_tokenizer, notes_tokenizer):
        """ Build the vocabularies for the songs and notes.
        :param songs_tokenizer: the tokenizer for the songs.
        :param notes_tokenizer: the tokenizer for the notes. """

        self.songs_words_index = songs_tokenizer.word_index
        self.songs_index_words = songs_tokenizer.index_word
        self.notes_words_index = notes_tokenizer.word_index
        self.notes_index_words = notes_tokenizer.index_word

    def split_train_test_data(self):
        """ Split the data into train and test sets.
        :return: the train and test sets. """

        # 10% of the data for testing.
        X_train, X_test, y_train, y_test = train_test_split(self.songs_padded, self.notes_padded, test_size=0.1)

        # Uncomment to store the train and test sets.
        # self.save_datastructures('flask_source/static/assets/pickles/X_train', X_train)
        # self.save_datastructures('flask_source/static/assets/pickles/y_train', y_train)
        # self.save_datastructures('flask_source/static/assets/pickles/X_test', X_test)
        # self.save_datastructures('flask_source/static/assets/pickles/y_test', y_test)

        return X_train, X_test, y_train, y_test

    def load_model(self, path, X, y):
        """ Load the model from a saved file.
        :param path: the path to the saved model.
        :param X: the train set input.
        :param y: the train set output. """

        # If the model is present, load the latest (most updated training), else train a new one.
        if os.listdir(path):
            checkpoint_files = glob.iglob(path + '/*.h5')
            latest_checkpoint = max(checkpoint_files, key=os.path.getctime)
            self.model = load_model(latest_checkpoint)
        else:
            print("A new model has been loaded. This means no model was in memory. If this message appears after" +
                  " training, something must have gone wrong.")
            self.model = Model(X, y)

            # As this is a multi-class classification problem, the loss function utilised
            # make the training possible without the need of converting the variables
            # into one hot encodings. Accuracy is chosen as a metric to evaluate the model.
            self.model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

    def build_models(self):
        """ Build the machine translation models. The encoder and decoder models are built one at a time, and at the
        end are finally combined together in the inference model. This function follow closely the Keras documentation
        at: https://blog.keras.io/a-ten-minute-introduction-to-sequence-to-sequence-learning-in-keras.html
        :return: the model. """

        # Set up the encoder model:
        # 1. A mask is used for the embedding layer as the input was previously padded (zeroes are so ignored).
        # 2. A bidirectional layer is used to allow the model to learn from the input to the output and vice versa.
        # 3. Concatenate the composing internal cells and hidden states.
        encoder_inputs = Input(shape=(None,))

        # Embedding layer.
        encoder_embedding = Embedding(self.LYRICS_VOCAB_SIZE, 512, mask_zero=True)(encoder_inputs)

        # Bidirectional layer.
        encoder_bidirectional_lstm = Bidirectional(LSTM(256, return_state=True))
        encoder_output, forward_h, forward_c, backward_h, backward_c = encoder_bidirectional_lstm(encoder_embedding)

        # Concatenate the composing internal cells and hidden states.
        state_h_concatenated = Concatenate()([forward_h, backward_h])
        state_c_concatenated = Concatenate()([forward_c, backward_c])
        encoder_states = [state_h_concatenated, state_c_concatenated]

        # Set up the decoder model by:
        # 1. A mask is used for the embedding layer as the input was previously padded (zeroes are so ignored).
        # 2. An LSTM layer is now used instead; the states need to be doubled up as a consequence of this change.
        # 3. The softmax activation function is used to convert the predictions into probabilities.
        decoder_inputs = Input(shape=(None,))

        # Embedding layer.
        decoder_embedding = Embedding(self.NOTES_VOCAB_SIZE, 512, mask_zero=True)
        decoder_embedding_output = decoder_embedding(decoder_inputs)

        # LSTM layer.
        decoder_lstm = LSTM(512, return_state=True, return_sequences=True)
        decoder_outputs, h, c = decoder_lstm(decoder_embedding_output, initial_state=encoder_states)

        # Dense layer.
        decoder_dense = Dense(self.NOTES_VOCAB_SIZE, activation='softmax')
        decoder_outputs = decoder_dense(decoder_outputs)

        # Load the trained model if present, else train a new one.
        self.load_model('flask_source/static/assets/model-checkpoints', [encoder_inputs, decoder_inputs],
                        decoder_outputs)

        # Set up the inference model by:
        # 1. Creating a composite encoder model with the previous encoder inputs and states.
        # 2. Building the decoder input.
        # 3. Building the decoder output combining the output of the decoder lstm.
        self.encoder_model = Model(encoder_inputs, encoder_states)
        decoder_input_h = Input(shape=(512,))
        decoder_input_c = Input(shape=(512,))
        decoder_input = [decoder_input_h, decoder_input_c]

        # Decoder output.
        decoder_inference_output, inference_h, inference_c = decoder_lstm(decoder_embedding_output,
                                                                          initial_state=decoder_input)
        decoder_inference_states = [inference_h, inference_c]
        decoder_inference_output = decoder_dense(decoder_inference_output)

        self.decoder_model = Model([decoder_inputs] + decoder_input,
                                   [decoder_inference_output] + decoder_inference_states)

    @staticmethod
    def plot_model_structure(model):
        """ Plot the model structure.
        :param model: the model to plot. """

        plot_model(model, to_file='model_structure.png', show_shapes=True)

    def train(self, X_train, y_train, validation_data):
        """ Train the neural network and save the trained weights in a checkpoint
        folder to continue or load the training from the saved state.
        :param X_train: the train set input.
        :param y_train: the train set output.
        :param validation_data: the validation set.
        :return the history of the training. """

        checkpoint_filepath = "flask_source/static/assets/model-checkpoints/model-improvement-{epoch:02d}-{loss:.2f}.h5"

        # Only the best weights that maximise the
        # accuracy of the model are saved.
        model_checkpoint_callback = ModelCheckpoint(
            filepath=checkpoint_filepath,
            save_best_only=True,
            save_weights_only=False,
            verbose=0,
            monitor='val_accuracy',  # monitor validation accuracy (from the test set).
            mode='auto'
        )

        # Stop training if the validation accuracy does not improve over the last 3 epochs.
        early_stopping = EarlyStopping(monitor='val_accuracy', patience=3, verbose=0, mode='auto')
        callbacks_list = [model_checkpoint_callback, early_stopping]

        batch_size = 64
        _EPOCHS = 50

        return self.model.fit(X_train, y_train, epochs=_EPOCHS, batch_size=batch_size, validation_data=validation_data,
                              callbacks=callbacks_list)

    def fit(self):
        """ Split the data and fit the model. The splitting of the data is such because of the
        teacher forcing technique. Finally the model is plotted (after training). """

        X_train, X_test, y_train, y_test = self.split_train_test_data()

        # The teacher forcing technique is used to predict the next word in the sequence.
        # Both the train and test set are split in input and output for the respective decoder and
        # encoder models.
        encoder_input_train = X_train
        decoder_input_train = y_train[:, :-1]
        decoder_output_train = y_train[:, 1:]

        encoder_input_validation = X_test
        decoder_input_validation = y_test[:, :-1]
        decoder_output_validation = y_test[:, 1:]

        history = self.train([encoder_input_train, decoder_input_train], decoder_output_train,
                             ([encoder_input_validation, decoder_input_validation], decoder_output_validation))

        # Plot the model structure
        plot_training(history, 'accuracy', 'val_accuracy', 'Model Accuracy', 'epoch', 'accuracy')
        plot_training(history, 'loss', 'val_loss', 'Model Loss', 'epoch', 'loss')

    @staticmethod
    def clean_text(text):
        """ Clean an inputted text in the following way:
        1. Expand all contractions.
        2. Make the text lowercase.
        3. Remove all the punctuation
        4. Remove all the numbers
        :param text: the text to clean.
        :return the cleaned text. """

        # Expand contractions
        expanded_words = []
        for word in text.split():
            expanded_words.append(contractions.fix(word))
        expanded_text = ' '.join(expanded_words)

        # Rest of operations
        expanded_text = expanded_text.lower()
        expanded_text = re.sub(r'[^\w\s]', '', expanded_text)
        expanded_text = re.sub(r'\d+', '', expanded_text)

        return expanded_text

    def predict(self, inputted_text):
        """ Generate a prediction based on an inputted text.
        :param inputted_text: the text to use for prediction.
        :return the prediction. """

        words = self.clean_text(inputted_text).split()

        # Transform the input text in a sequence of integers based on the vocabulary.
        encoded_text = []
        for word in words:
            if word in self.songs_words_index:
                encoded_text.append(self.songs_words_index[word])

        # Use a numpy array to feed the model.
        encoded_text_arr = np.array([encoded_text])
        return self.predict_from_vector(encoded_text_arr)

    def predict_from_vector(self, input_vector):
        """ Generate a prediction based on an inputted vector. This function follow closely the Keras documentation at:
        https://blog.keras.io/a-ten-minute-introduction-to-sequence-to-sequence-learning-in-keras.html
        :param input_vector: the vector to use for prediction.
        :return the generate prediction. """

        # The input is encoded as a state vector.
        states_vector = self.encoder_model.predict(input_vector)

        # A target sequence of length 1 is generated.
        # This will serve as the "next input" for the decoder.
        # The first element of this sequence is the start token.
        target_sequence = np.zeros((1, 1))
        target_sequence[0, 0] = self.notes_words_index['sos']

        # Keep iterating until stop condition is reached. This builds up the prediction.
        stop_condition = False
        decoded_output = ''
        while not stop_condition:
            output_tokens, h, c = self.decoder_model.predict([target_sequence] + states_vector)

            # A new token is sampled.
            sampled_token_index = np.argmax(output_tokens[0, -1, :])

            if sampled_token_index == 0:
                break
            else:
                # Convert index number to a note
                predicted_notes = self.notes_index_words[sampled_token_index]

                # If the start of sequence was picked, pick again.
                while predicted_notes == 'sos':
                    predicted_notes = self.notes_index_words[sampled_token_index]
                sampled_char = predicted_notes

            # Expand prediction.
            decoded_output += ' ' + sampled_char

            # When the end token is predicted, or the length of the prediction
            # is greater than the length of the inputted vector, stop the loop.
            if sampled_char == 'eos' or len(decoded_output.split()) >= np.size(input_vector, 1):
                stop_condition = True

            # Update the target sequence.
            target_sequence = np.zeros((1, 1))
            target_sequence[0, 0] = sampled_token_index

            # Update states.
            states_vector = [h, c]

        return self.polish_output(decoded_output)

    @staticmethod
    def polish_output(output):
        """ Remove the end of string token if present.
        param output: the output from the predict function.
        :return: the polished output. """

        if ' eos' in output:
            return output[:-4]
        return output

    def retrieve_notes(self, input_vector):
        """ Retrieve the original notes from the inputted vector.
        :param input_vector: the vector to retrieve the notes from.
        :return: the notes. """

        notes = ''
        for encoding in input_vector:
            if encoding != 0:
                if encoding != self.notes_words_index['sos'] and encoding != self.notes_words_index['eos']:
                    notes += self.notes_index_words[encoding] + ' '
        return notes

    def retrieve_lyrics(self, input_vector):
        """ Retrieve the original lyrics from the inputted vector.
        :param input_vector: the vector to retrieve the lyrics from.
        :return: the lyrics. """

        lyrics = ''
        for encoding in input_vector:
            if encoding != 0:
                lyrics += self.songs_index_words[encoding] + ' '
        return lyrics


if __name__ == '__main__':
    # Load the model.
    model = RNNModel()
    model.fit()
