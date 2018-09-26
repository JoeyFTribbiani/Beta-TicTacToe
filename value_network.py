from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.callbacks import ModelCheckpoint, EarlyStopping
from keras.optimizers import Adam

class ValueNetwork(object):
    def __init__(self, n_input, hidden_dims ,n_output):
        self.model = Sequential()
        self.model.add(Dense(hidden_dims[0], input_shape=(n_input,), activation='relu'))
        self.model.add(Dropout(.2))
        for i in range(1,len(hidden_dims)):
            self.model.add(Dense(hidden_dims[i], activation='relu'))
            if i != len(hidden_dims) - 1:
                self.model.add(Dropout(.3))
        self.model.add(Dense(n_output, activation='relu'))

    def coeff_determination(self, y_true, y_pred):
        from keras import backend as K
        SS_res =  K.sum(K.square( y_true-y_pred ))
        SS_tot = K.sum(K.square( y_true - K.mean(y_true) ) )
        return ( 1 - SS_res/(SS_tot + K.epsilon()) )
    
    def train(self, x, y, x_test, y_test, batch_size=1, epochs=100):
        self.model.compile(loss='mean_squared_error', optimizer="adam")
        model_save_callback = ModelCheckpoint('value_network.m', verbose = 0, save_best_only=True, save_weights_only=True)
        early_stopping = EarlyStopping(monitor='val_loss', min_delta = 0.001, patience = 50, verbose = 3)
        self.model.fit(x, y, batch_size=batch_size,shuffle=True, epochs=epochs, validation_data=(x_test, y_test), callbacks=[model_save_callback, early_stopping])

    def predict(self, x):
        return self.model.predict(x)    

    def load_model(self, model_path):
        self.model.load_weights(model_path)
