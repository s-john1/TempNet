import tkinter
import customtkinter

import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

from threading import Thread, Event
from queue import Queue

import json
import sys
import os
import time
from PIL import Image


class TempNet:
    def __init__(self, accuracy_queue, epochs_queue, predict_queue, train_event, predict_event):
        # Accuracy multiplier
        self.accuracy = 1.0
        self.accuracy_queue = accuracy_queue

        # Number of epochs to train for
        self.epochs = 1
        self.epochs_queue = epochs_queue

        # Define the training model
        self.model = None

        self.predict_queue = predict_queue

        # Open temperature_readings.json
        with open('temperature_readings.json') as json_file:
            self.data = json.load(json_file)

        # Print out the number of temperature readings in the training data
        print("Number of temperature readings in the training data: " + str(len(self.data)))

        # Print out min and max values from the raw data
        print("Smallest temperature reading found in the training data: " + str(min(self.data)))
        print("Largest temperature reading found in the training data: " + str(max(self.data)))

        # Print out the mean and standard deviation of the raw data
        print("Mean of the training data: " + str(self.get_mean()))
        print("Standard deviation of training data: " + str(self.get_std()))

        # Visualize the raw training data using matplotlib
        self.visualise_data()

        # Train the neural network
        # self.train()

        self.run()

    def run(self):
        while True:
            if temp_net_ui.train_event.is_set():
                self.train()
                temp_net_ui.train_event.clear()

            if temp_net_ui.predict_event.is_set():
                self.predict()
                temp_net_ui.predict_event.clear()

            time.sleep(0.1)

    def visualise_data(self):
        # Visualize the raw training data using matplotlib
        plt.plot(self.data)
        plt.ylabel('Temperature')
        plt.xlabel('Time')
        plt.savefig('temperature_readings.png')

        # Get the mean of the training data

    def get_mean(self):
        return np.mean(self.data)

    # Get the standard deviation of the training data
    def get_std(self):
        return np.std(self.data)

    def train(self):
        if not self.epochs_queue.empty():
            new_value = self.epochs_queue.get()
            self.epochs = new_value
            # print("[TempNet] Epochs set to " + str(new_value))

        # Define the training data
        train_data = np.array(self.data, dtype=float)

        # Define the neural network
        self.model = tf.keras.Sequential([
            tf.keras.layers.Dense(16, input_shape=[1], activation='relu'),
            tf.keras.layers.Dense(1, activation='linear')
        ])
        self.model.compile(optimizer='adam', loss='mean_squared_error')

        # Train the neural network
        self.model.fit(train_data, train_data, epochs=self.epochs, verbose=1)

        # Test the neural network
        self.test()
        self.test_3d()

    def test(self):
        # Define the test data
        test_data = np.array([1, 2, 3, 5, 10, 15, 20, 25, 30, 35, 40, 50, 100], dtype=float)

        # Define the test results dictionary
        test_results = {}

        # Prepare lists to hold the coordinates and colors of the dots
        good_prediction_x = []
        good_prediction_y = []
        bad_prediction_x = []
        bad_prediction_y = []

        for temperature in test_data:
            prediction = self.model.predict([temperature])[0][0]
            is_bad = abs(prediction - temperature) > self.accuracy * self.get_std()
            test_results[temperature] = is_bad

            if is_bad:
                bad_prediction_x.append(temperature)
                bad_prediction_y.append(prediction)
            else:
                good_prediction_x.append(temperature)
                good_prediction_y.append(prediction)

        # Visualize the test results using matplotlib
        plt.figure()
        plt.scatter(good_prediction_x, good_prediction_y, c='green', label='Good Prediction')
        plt.scatter(bad_prediction_x, bad_prediction_y, c='red', label='Bad Prediction')
        plt.plot(test_data, test_data, 'r', label='Actual')
        plt.xlabel('Temperature')
        plt.ylabel('Prediction')
        plt.title('Temperature Prediction')
        plt.legend()
        plt.savefig('temperature_test.png')

    def test_3d(self):
        # Define the test data
        test_data = np.array([1, 2, 3, 5, 10, 15, 20, 25, 30, 35, 40, 50, 100], dtype=float)

        # Define the test results dictionary
        test_results = {}

        # Define the test results lists
        good_prediction_x = []
        good_prediction_y = []
        good_prediction_z = []

        bad_prediction_x = []
        bad_prediction_y = []
        bad_prediction_z = []

        for temperature in test_data:
            prediction = self.model.predict([temperature])[0][0]
            deviation = abs(prediction - temperature)
            std_dev = self.get_std()
            accuracy = self.accuracy
            is_bad = deviation > accuracy * std_dev
            test_results[temperature] = {
                'prediction': prediction,
                'deviation': deviation,
            }

            if is_bad:
                bad_prediction_x.append(temperature)
                bad_prediction_y.append(prediction)
                bad_prediction_z.append(deviation)
            else:
                good_prediction_x.append(temperature)
                good_prediction_y.append(prediction)
                good_prediction_z.append(deviation)

        # Visualize the test results using matplotlib 3D scatter plot
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')

        ax.scatter(good_prediction_x, good_prediction_y, good_prediction_z, c='green', marker='o',
                   label='Good Prediction')
        ax.scatter(bad_prediction_x, bad_prediction_y, bad_prediction_z, c='red', marker='o', label='Bad Prediction')
        ax.plot(test_data, test_data, np.zeros_like(test_data), 'r', label='Actual')

        ax.set_xlabel('Temperature')
        ax.set_ylabel('Prediction')
        ax.set_zlabel('Deviation')
        ax.set_title('Temperature Prediction with Deviation')
        ax.legend()
        plt.savefig('temperature_test_3d.png')

    def predict(self):
        if not self.accuracy_queue.empty():
            new_value = self.accuracy_queue.get()
            self.accuracy = new_value
            # print("[TempNet] Accuracy set to " + str(new_value))

        # Predict the temperature
        temperatures = self.predict_queue.get()
        temperatures_inputs = []
        for temperature in temperatures:
            temperatures_inputs.append([temperature])

        temperatures_list = []
        for temperature in temperatures:
            is_bad = self.is_bad_temperature(temperature, temperatures_inputs)
            temperatures_list.append({
                'temperature': temperature,
                'prediction': self.model.predict([temperature])[0][0],
                'is_bad': is_bad,
            })
            if is_bad:
                print(f"{temperature} is a bad temperature reading")
            else:
                print(f"{temperature} is a good temperature reading")

        self.plot_temperatures_predictions(temperatures_list)

    def is_bad_temperature(self, temperature, temperatures_inputs):
        # Sort the temperatures inputs in descending order
        temperatures_inputs.sort(reverse=True)

        prediction = self.model.predict([temperature])[0][0]
        is_bad = abs(prediction - temperature) > self.accuracy * self.get_std()
        if is_bad:
            return True

        deviations = []
        for temperature_input in temperatures_inputs:
            prediction = self.model.predict([temperature_input])[0][0]
            deviation = abs(temperature_input[0] - prediction)
            if deviation < self.accuracy * self.get_std():
                deviations.append(deviation)
        average_deviation = sum(deviations) / len(deviations)
        # print("Average deviation: " + str(average_deviation))

        deviation = abs(prediction - temperature)
        # print("Deviation: " + str(deviation) + " of " + str(temperature))

        if deviation > average_deviation + 5:
            return True

        return is_bad

    def plot_temperatures_predictions(self, temperatures_list):
        plt.style.use('dark_background')
        plt.figure()
        legend_added = False
        for temperature_data in temperatures_list:
            temperature = temperature_data['temperature']
            prediction = temperature_data['prediction']
            is_bad = temperature_data['is_bad']
            acceptable_range = (
            temperature - self.accuracy * self.get_std(), temperature + self.accuracy * self.get_std())

            plt.axhspan(acceptable_range[0], acceptable_range[1], color='green', alpha=0.2, label='Acceptable Range')
            plt.plot(temperature, prediction, 'ro', label='Prediction')
            plt.xlabel('Temperature')
            plt.ylabel('Prediction')
            plt.title('Temperature Prediction')
            if is_bad:
                plt.text(temperature, prediction, ' Bad', color='red')
            else:
                plt.text(temperature, prediction, ' Good', color='green')

            if not legend_added:
                plt.legend()
                legend_added = True

        plt.savefig('temperature_prediction.png', transparent=True)

        temp_net_ui.temperature_prediction_image = customtkinter.CTkImage(
            Image.open(os.path.join("temperature_prediction.png")), size=(325, 243.75))
        temp_net_ui.image_event.set()


class TempNetUI:
    def __init__(self):
        self.accuracy_queue = Queue()
        self.epochs_queue = Queue()
        self.train_event = Event()

        self.predict_queue = Queue()
        self.predict_event = Event()
        self.image_event = Event()

        self.temperature_prediction_image = None

        self.custom_stdout = False

    def run(self):
        customtkinter.set_appearance_mode("System")  # Modes: system (default), light, dark
        customtkinter.set_default_color_theme("blue")  # Themes: blue (default), dark-blue, green

        app = customtkinter.CTk()  # create CTk window like you do with the Tk window
        app.geometry("725x600")
        app.title("TempNet")
        app.resizable(False, False)

        self.draw_training_frame(app)
        self.draw_prediction_frame(app)

        app.mainloop()

    def draw_training_frame(self, app):
        # Create a frame on the left part of the window to hold the training parameters
        training_frame = customtkinter.CTkFrame(master=app, width=350, height=590)
        training_frame.place(relx=0.01, rely=0.01, anchor=tkinter.NW)

        training_frame_label = customtkinter.CTkLabel(master=training_frame, text="Training", font=("Helvetica", 24))
        training_frame_label.place(relx=0.5, rely=0.05, anchor=tkinter.CENTER)

        # Create an epoch slider label
        epoch_slider_label = customtkinter.CTkLabel(master=training_frame, text="Epochs (1)", anchor=tkinter.W,
                                                    justify=tkinter.LEFT)
        epoch_slider_label.place(relx=0.05, rely=0.1, anchor=tkinter.W)

        def epoch_slider_callback(value):
            value = int(value)
            epoch_slider_label.configure(text="Epochs (" + str(value) + ")")
            self.epochs_queue.queue.clear()
            self.epochs_queue.put(value)

        # Create an epoch slider
        epoch_slider = customtkinter.CTkSlider(master=training_frame, from_=1, to=5, number_of_steps=5,
                                               command=epoch_slider_callback, width=325,
                                               variable=tkinter.IntVar(value=1))
        epoch_slider.place(relx=0.5, rely=0.13, anchor=tkinter.CENTER)

        # Create a textbox to display the training progress
        training_progress_textbox = customtkinter.CTkTextbox(master=training_frame, width=325, height=440,
                                                             activate_scrollbars=False)
        training_progress_textbox.place(relx=0.5, rely=0.16, anchor=tkinter.N)

        def train_button_callback():
            class CustomStdout:
                def __init__(self, textbox, original_stdout):
                    self.textbox = textbox
                    self.original_stdout = original_stdout

                def write(self, message):
                    custom_message = message
                    if "ETA" in message or "[=" in message:
                        custom_message = message.split(']')[1].strip() + "\n"
                    self.textbox.insert(customtkinter.END, custom_message)

                    self.original_stdout.write(message)

                    training_progress_textbox.see(customtkinter.END)

                def flush(self):
                    self.original_stdout.flush()

            training_progress_textbox.delete(1.0, customtkinter.END)
            self.train_event.set()
            if not isinstance(sys.stdout, CustomStdout) and not self.custom_stdout:
                custom_stdout = CustomStdout(training_progress_textbox, sys.stdout)
                sys.stdout = custom_stdout
                self.custom_stdout = True

        # Create a train button
        train_button = customtkinter.CTkButton(master=training_frame, text="Train", command=train_button_callback,
                                               width=325)
        train_button.place(relx=0.5, rely=0.95, anchor=tkinter.CENTER)

    def draw_prediction_frame(self, app):
        # Create a frame on the right part of the window to hold the prediction parameters
        prediction_frame = customtkinter.CTkFrame(master=app, width=350, height=590)
        prediction_frame.place(relx=0.99, rely=0.01, anchor=tkinter.NE)

        prediction_frame_label = customtkinter.CTkLabel(master=prediction_frame, text="Prediction",
                                                        font=("Helvetica", 24))
        prediction_frame_label.place(relx=0.5, rely=0.05, anchor=tkinter.CENTER)

        # Create an accuracy slider label
        accuracy_slider_label = customtkinter.CTkLabel(master=prediction_frame, text="Accuracy (1)", anchor=tkinter.W,
                                                       justify=tkinter.LEFT)
        accuracy_slider_label.place(relx=0.05, rely=0.1, anchor=tkinter.W)

        def accuracy_slider_callback(value):
            accuracy_slider_label.configure(text="Accuracy (" + str(value) + ")")
            self.accuracy_queue.queue.clear()
            self.accuracy_queue.put(float(value))

        # Create an accuracy slider
        accuracy_slider = customtkinter.CTkSlider(master=prediction_frame, from_=0.001, to=1.0,
                                                  number_of_steps=10000000, command=accuracy_slider_callback, width=325,
                                                  variable=tkinter.DoubleVar(value=1.0))
        accuracy_slider.place(relx=0.5, rely=0.13, anchor=tkinter.CENTER)

        # Create an temperature input field label
        temperature_input_label = customtkinter.CTkLabel(master=prediction_frame, text="Temperature", anchor=tkinter.W,
                                                         justify=tkinter.LEFT)
        temperature_input_label.place(relx=0.05, rely=0.175, anchor=tkinter.W)

        # Create an temperature input field
        temperature_input = customtkinter.CTkEntry(master=prediction_frame, width=325,
                                                   textvariable=tkinter.StringVar(value="10, 10, 10, 10, 100"))
        temperature_input.place(relx=0.5, rely=0.225, anchor=tkinter.CENTER)

        def wait_for_image_event():
            while not self.image_event.is_set():
                time.sleep(0.5)

            if os.path.isfile("temperature_prediction.png"):
                temperature_prediction_button = customtkinter.CTkButton(
                    master=prediction_frame,
                    image=self.temperature_prediction_image,
                    width=325,
                    height=243.75,
                    text="",
                    border_spacing=0,
                    border_width=0,
                    hover=False,
                    fg_color="transparent"
                )
                temperature_prediction_button.place(relx=0.5, rely=0.7, anchor=tkinter.CENTER)
            self.image_event.clear()

        def predict_button_callback():
            # Separate the input into a list of temperatures
            temperatures = temperature_input.get().split(', ')
            temperatures = [float(temperature) for temperature in temperatures]
            self.predict_queue.queue.clear()
            self.predict_queue.put(temperatures)
            self.predict_event.set()

            # Create a thread to wait for the image event
            wait_for_image_thread = Thread(target=wait_for_image_event)
            wait_for_image_thread.start()

        # Create a predict button
        predict_button = customtkinter.CTkButton(master=prediction_frame, text="Predict",
                                                 command=predict_button_callback, width=325)
        predict_button.place(relx=0.5, rely=0.95, anchor=tkinter.CENTER)


temp_net_ui = TempNetUI()

# Create an instance of the TempNet in a new thread
temp_net_thread = Thread(target=TempNet, args=(temp_net_ui.accuracy_queue, temp_net_ui.epochs_queue,
                                               temp_net_ui.predict_queue, temp_net_ui.train_event,
                                               temp_net_ui.predict_event))
temp_net_thread.start()

temp_net_ui.run()