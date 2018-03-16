class FaceModel:
    def __init__(self, callback):
        self.model = callback()

    def train_model(self, train_gen, val_gen, epochs):
        self.model = self.model.fit_generator(train_gen, epochs=epochs, validation_data=val_gen)
