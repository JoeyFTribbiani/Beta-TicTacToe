from keras.models import Sequential, load_model
from keras.layers import Dense, Dropout
from keras.callbacks import ModelCheckpoint

class ValueNetwork(object):
    def __init__(self, n_input, hidden_dims ,n_output):
        self.model = Sequential()
        self.model.add(Dense(hidden_dims[0], input_shape=(n_input,), activation='relu'))
        self.model.add(Dropout(.4))
        for layer in hidden_dims[1:]:
            self.model.add(Dense(layer, activation='relu'))
            self.model.add(Dropout(.3))
        self.model.add(Dense(n_output, activation='relu'))
        
    def train(self, x, y, x_test, y_test, batch_size=1, epochs=100):
        self.model.compile(loss='mean_squared_error', optimizer='adam')
        model_save_callback = ModelCheckpoint('value_network.m', monitor='val_loss', verbose = 0, save_best_only=True)
        self.model.fit(x, y, batch_size=batch_size,shuffle=True, epochs=epochs, validation_data=(x_test, y_test), callbacks=[model_save_callback])

    def predict(self, x):
        return self.model.predict(x)    

    def load_model(self, model_path):
        self.model = load_model(model_path)
