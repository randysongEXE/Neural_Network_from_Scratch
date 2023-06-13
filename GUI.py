import tkinter as tk
from tkinter import filedialog
from PIL import Image, ImageTk
from NeuralNetwork import NeuralNetwork
import numpy as np
import os


class GUI:

    def __init__(self, neural_network):
        self.nn = neural_network
        self.window = tk.Tk()
        self.window.title("Neural Network GUI")

        self.canvas = tk.Canvas(self.window, width=300, height=300)
        self.canvas.pack()

        self.button = tk.Button(self.window, text="Select an image", command=self.load_image)
        self.button.pack()

        self.label = tk.Label(self.window, text="Prediction: ")
        self.label.pack()

        self.window.mainloop()

    def load_image(self):
        file_path = filedialog.askopenfilename()

        # Check if file path is valid
        if not os.path.exists(file_path):
            self.label["text"] = "File not found. Please select a valid image file."
            return

        # Try to open and process the image
        try:
            img = Image.open(file_path).convert("L")
            img = img.resize((28, 28))  # resize to the input size of your neural network
            img_arr = np.array(img).reshape(-1) / 255.0
        except Exception as e:
            self.label["text"] = f"Error processing image: {str(e)}"
            return

        # Try to classify the image
        try:
            prediction = self.nn.single_pass(img_arr)
            predicted_class = np.argmax(prediction)
        except Exception as e:
            self.label["text"] = f"Error classifying image: {str(e)}"
            return

        # Display the selected image
        try:
            img = img.resize((300, 300))  # resize for display
            img_tk = ImageTk.PhotoImage(img)
            self.canvas.create_image(20, 20, anchor='nw', image=img_tk)
            self.canvas.image = img_tk
        except Exception as e:
            self.label["text"] = f"Error displaying image: {str(e)}"
            return

        self.label["text"] = f"Prediction: {predicted_class}"


if __name__ == '__main__':
    # Replace "dataset" with the path to your dataset if it's not in a directory named "dataset"
    train_images, test_images, train_labels_one_hot, train_labels, test_labels_one_hot, test_labels, number_of_total_pixels = image_preprocessor(
        28, 10, "dataset")

    # Creating and training a neural network
    nn = NeuralNetwork([784, 128, 64, 10], 0.001)
    nn.train(train_images, train_labels_one_hot, 10)

    # Initializing the GUI
    gui = NeuralNetworkGUI(nn)



