"""Model base class."""


class Model(object):
    def train(self, x_train, y_train):
        raise NotImplementedError("Please Implement this method")

    def predict(self, x):
        raise NotImplementedError("Please Implement this method")


class KerasModel(Model):
    def __init__(self):
        self.model = None
        self.batch_size = 32

    def _create_model(self, input_shape, num_categories):
        """To be implemented by subclass."""
        raise NotImplementedError("Please Implement this method")

    def train(self, x, y):
        """Train."""
        self.model = self._create_model(x[0].shape, y.shape[-1])
        self.model.compile(
            loss='categorical_crossentropy',
            optimizer='adam',
            metrics=['accuracy']
        )

        from keras.callbacks import EarlyStopping

        early_stopping = EarlyStopping(monitor='val_loss', patience=10)
        self.model.fit(
            x, y,
            callbacks=[early_stopping],
            verbose=1,
            validation_split=0.1,  # last 10% of data
            shuffle=True,
            nb_epoch=100,
            batch_size=self.batch_size
        )

    def predict(self, x):
        """Predict."""
        return self.model.predict_classes(x, batch_size=self.batch_size)
