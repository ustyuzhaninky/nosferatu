import os
import tempfile
import tensorflow as tf
from unittest import TestCase
import numpy as np
from tensorflow import keras
from nosferatu.neocortex.OSAR import TransformerShift

print("TensorFlow version: {}".format(tf.__version__))
print("Eager execution: {}".format(tf.executing_eagerly()))

class TestTransformerShift(TestCase):

    def test_build(self):
        
        tf.config.experimental_run_functions_eagerly(True)
        inputs = keras.layers.Input(shape=(10, 1,))
        memory = keras.layers.Input(shape=(10, 1,), batch_size=200)
        x = TransformerShift(
            100, 100, filters=2, name='TransformerShift')([inputs, memory])
        x = keras.layers.Dense(200, activation="relu", name='dense')(x)
        x = keras.layers.Dropout(0.1)(x)
        outputs = keras.layers.Dense(1, activation="softmax", name='Output')(x)

        model = keras.Model(inputs=[inputs, memory], outputs=outputs)
        model.compile('adam', 'mse')
        print(tempfile.gettempdir())
        model_path = os.path.join(tempfile.gettempdir(
        ), 'test_transformer_shift_%f.h5' % np.random.random())
        model.save_weights(model_path)
        model.load_weights(model_path)
        model.save(model_path)
        # model = keras.models.load_model(model_path, custom_objects={
        #                                 'TransformerShift': TransformerShift})
        model.summary()
        try:
            current_path = os.path.dirname(os.path.abspath(__file__))
            visual_path = os.path.join(current_path, 'test_build.jpg')
            keras.utils.plot_model(model, visual_path)
        except Exception as e:
            print(e)

    # def test_baseline_shifting(self):
    #     tf.config.experimental_run_functions_eagerly(True)

    #     inputs = keras.layers.Input(shape=(10, 1,))
    #     memory = keras.layers.Input(shape=(1, 1,), batch_size=20)
    #     outputs = TransformerShift(
    #         5, 5, filters=2, name='TransformerShift')([inputs, memory])
    #     model = keras.Model(inputs=[inputs, memory], outputs=outputs)
    #     model.compile('adam', 'mse')
    #     model.summary()

    #     nm = np.zeros((10, 1, 1), dtype=float)
    #     x_data = np.array([[[0.8]]])
        
    #     prediction = model.predict([x_data, nm])
        
    #     assert prediction[-1, ...] != np.array([[0.0]]), "Doing nothing"
    #     assert prediction[-1, ...] == x_data[0], "Not shifting"

    def test_fit_batch_changes(self):
        
        tf.config.experimental_run_functions_eagerly(True)
        inputs = keras.layers.Input(shape=(10, 1,))
        memory =  keras.layers.Input(shape=(10, 1,), batch_size=200)
        tshift = TransformerShift(
            100, 100, filters=2, name='TransformerShift')
        x = tshift([inputs, memory])
        x = keras.layers.Dense(200, activation="relu", name='dense')(x)
        x = keras.layers.Dropout(0.1)(x)
        outputs = keras.layers.Dense(1, activation="softmax", name='Output')(x)

        model = keras.Model(inputs=[inputs, memory], outputs=outputs)
        model.compile('adam', 'mse')
        model.summary()
        nm = np.zeros((200, 10, 1), dtype=float)
        x_data = np.sin(np.linspace(0, 100, 2000).reshape((200, 10, 1)))
        y_data = 1 - 2 * np.exp(-x_data)
        
        print(tshift.variables)
        print(len(tshift.variables))

        model.train_on_batch(
            [x_data[0].reshape((1, 10, 1)), nm], y_data[0].reshape((1, 10, 1)))
        # model.train_on_batch(
        #     [x_data[:3], nm], y_data[:3])
        # model.train_on_batch(
        #     [x_data[:10], nm], y_data[:10])
        # model.train_on_batch(
        #     [x_data, nm], y_data)
