class FaceModel:
    def __init__(self, callback):
        self.model = callback()

    def validation_train(self, train_data, valid_data, generators, epochs, batch_size):
        self.model = self.model.fit_generator(generators[0].flow(train_data[0], train_data[1], batch_size), epochs=epochs, validation_data=generators[1].flow(valid_data[0], valid_data[1]))
