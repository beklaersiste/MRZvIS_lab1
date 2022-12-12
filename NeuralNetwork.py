import numpy
from configs import *
import glob
from PIL import Image
import matplotlib.pyplot as plt


class NeuralNetwork:
    def __init__(self, weights1_path=None, weights2_path=None):
        if weights1_path is not None and weights2_path is not None:
            self.weights1 = numpy.load(weights1_path)
            self.weights2 = numpy.load(weights2_path)
        self.errors = []

    def learn(self, image_path, size, compress_rate, learn_rate, max_error=0, used_image_num=-1, use_numpy=False):
        self.weights1 = numpy.random.rand(size**2*3, int(size**2*3*compress_rate)) * 2 - 1
        self.weights2 = self._transpose(self.weights1) if not use_numpy else self.weights1.T
        files = glob.glob(image_path + '*')
        for i in range(len(files)):
            if used_image_num > -1:
                i = used_image_num
            learn_set = self._fragment(size, numpy.asarray(Image.open(files[i]).convert('RGB')))
            result_set = []
            learn_mse = []
            for input in learn_set:
                x = numpy.array([input])                                                                                # Xi                нейроны первого слоя
                y = self._mult_matrix(x, self.weights1) if not use_numpy else x @ self.weights1                         # Yi = Xi * W       вычисление нейронов второго слоя
                output = self._mult_matrix(y, self.weights2) if not use_numpy else y @ self.weights2                    # Xi' = Yi * W'     вычисление нейронов последнего слоя
                delta_x = output - x                                                                                    # deltaX = Xi' - Xi
                learning_rate1 = learn_rate #1 / (1 + self._mult_matrix(x, self._transpose(x))[0][0])                   # a1 = 1 / (1 + Xi * (Xi)T)  вычисление адаптивного шага
                learning_rate2 = 1 / (1 + self._mult_matrix(y, self._transpose(y))[0][0]) if not use_numpy else \
                    1 / (1 + y @ y.T)[0][0]                                                                             # a2 = 1 / (1 + Yi * (Yi)T)  вычисление адаптивного шага
                e = self._mse(delta_x)                                                                                  #  среднеквадратич ошибка
                self.weights1 = self.weights1 - learning_rate1 * self._mult_matrix( \
                    self._mult_matrix(self._transpose(x), delta_x), self._transpose(self.weights2)) if not use_numpy else \
                    self.weights1 - learning_rate1 * x.T @ delta_x @ self.weights2.T                                    # W(t+1) = W(t) - a1 * (Xi)T * deltaX * (W'(t))T    коректировка весов
                self.weights2 = self.weights2 - learning_rate2 * self._mult_matrix(self._transpose(y), delta_x) \
                    if not use_numpy else self.weights2 - learning_rate2 * y.T @ delta_x                                # W'(t+1) = W'(t) - a2 * (Yi)T * deltaX      коректировка весов
                learn_mse.append(e)
                #print(e, '\t', len(learn_mse))
                result_set.append(output[0])
            plt.imsave(learn_img_path, self._reestablish(result_set))
            print(sum(learn_mse) / len(learn_mse))
            self.errors.append(sum(learn_mse) / len(learn_mse))
            if sum(learn_mse) / len(learn_mse) < max_error:
                break
        numpy.save(weights1_path, self.weights1)
        numpy.save(weights2_path, self.weights2)

    def compress(self, image_path, use_numpy=False):
        size = int((len(self.weights1)//3)**0.5)
        input_set = self._fragment(size, numpy.asarray(Image.open(image_path).convert('RGB')))
        result_set = []
        for input in input_set:
            x = numpy.array([input])                                                                    # Xi                нейроны первого слоя
            y = self._mult_matrix(x, self.weights1) if not use_numpy else x @ self.weights1             # Yi = Xi * W       вычисление нейронов второго слоя
            result_set.append(y)
        return result_set   # возвращает numpy матрицу

    def uncompress(self, matrix, use_numpy=False):
        result_set = []
        for input in matrix:                                                                                # Yi                нейроны второго слоя
            output = self._mult_matrix(input, self.weights2) if not use_numpy else input @ self.weights2    # Xi' = Yi * W'     вычисление нейронов последнего слоя
            result_set.append(output[0])
        return self._reestablish(result_set)    # возвращает матрицу значений 0-1

    def _transpose(self, matrix):
        result = numpy.zeros((len(matrix[0]), len(matrix)))
        for i in range(len(matrix)):
            for j in range(len(matrix[0])):
                result[j][i] = matrix[i][j]
        return result

    def _mult_matrix(self, matrix1, matrix2):
        if len(matrix1[0]) == len(matrix2):
            result = numpy.zeros((len(matrix1), len(matrix2[0])))
            for i in range(len(matrix1)):
                for j in range(len(matrix2[0])):
                    for k in range(len(matrix1[0])):
                        result[i][j] += matrix1[i][k]*matrix2[k][j]
            return result

    def _mse(self, delta_x):
        result = 0
        for i in delta_x[0]:
            result += i**2
        return result/len(delta_x[0])

    def _fragment(self, size, image):
        result = []
        count = int (len(image) // size)
        for i in range(count):
            for j in range(count):
                x = i*size
                y = j*size
                result.append((image[x:x+size, y:y+size]).reshape(3*size**2))
        return numpy.array(result) / 255 * 2 - 1

    def _reestablish(self, array):
        size = int((len(array[0])//3)**0.5)
        length = int(len(array)**0.5)
        result = numpy.zeros((size * length, size * length, 3))
        for i in range(len(array)):
            pix = 0
            while pix < len(array[i]):
                x = i % length * size + (pix // 3) % size
                y = i // length * size + (pix // 3) // size
                pixel = numpy.zeros(3)
                for color in range(3):
                    pixel[color] = array[i][pix + color]
                    pixel[color] = pixel[color] if pixel[color] >= -1 else -1
                    pixel[color] = pixel[color] if pixel[color] <= 1 else 1
                result[y, x] = pixel
                pix += 3
        return (result + 1) / 2

