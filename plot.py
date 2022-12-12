from NeuralNetwork import NeuralNetwork
import matplotlib.pyplot as plt
from configs import *


def plot_iter_image(num):
    iter = []
    image = []
    for i in range(num):
        n = NeuralNetwork()
        n.learn(pngs_path, 8, 0.25, 0.0001, 0.01, num, True)
        iter.append(len(n.errors)*1024)
        image.append("image " + str(i))
    plt.bar(image, iter)
    plt.show()


def plot_iter_compress_rate(num):
    iter = []
    compress_rate = []
    for i in range(2, num):
        n = NeuralNetwork()
        n.learn(pngs_path, 8, i / 10, 0.0001, 0.01, 0, True)
        iter.append(len(n.errors) * 1024)
        compress_rate.append(str(i/10))
    plt.plot(compress_rate, iter)
    plt.show()


def plot_iter_learning_rate(num):
    iter = []
    learning_rate = []
    for i in range(1, num):
        n = NeuralNetwork()
        rate = i * 0.000005
        n.learn(pngs_path, 8, 0.25, rate, 0.01, 0, True)
        iter.append(len(n.errors) * 1024)
        learning_rate.append(rate)
    plt.plot(learning_rate, iter)
    plt.show()


def plot_iter_max_error(num):
    iter = []
    max_error = []
    for i in range(1, num):
        n = NeuralNetwork()
        error = i * 0.002
        n.learn(pngs_path, 8, 0.25, 0.0001, error, 0, True)
        iter.append(len(n.errors) * 1024)
        max_error.append(error)
    plt.plot(max_error, iter)
    plt.show()
