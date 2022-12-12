import numpy
from NeuralNetwork import NeuralNetwork
from configs import *
from plot import *

if __name__ == '__main__':
    # обучение нейронной сети
    # n = NeuralNetwork()
    # n.learn(pngs_path, 8, 0.25, 0.0001, 0.001, 169, True)

    # демонстрация сжатия и восстановления изображения
    n = NeuralNetwork(weights841_path, weights842_path)
    matrix = n.compress('pngs/1083.png', False)
    numpy.save(matrix_path, matrix)
    img = n.uncompress(matrix, False)
    plt.imsave(test_img_path, img)

    # таблицы
    #plot_iter_image(10)
    #plot_iter_compress_rate(10)
    #plot_iter_learning_rate(10)
    #plot_iter_max_error(10)
