import numpy as np
import librosa
import glob
import os


def normalize(matrix):
    return matrix / matrix.max(axis=0)


class Distance(object):
    def __init__(self, a: np.ndarray, b: np.ndarray):
        self._a = a
        self._b = b
        # for euclidean_distance we must ensure a and b have the same shape
        assert(self._a.shape == self._b.shape)

        self._distance_matrix_shape = (a.shape[1], b.shape[1])
        self._euclidean_distance_matrix = None
        self._dtw_distance_matrix = None
        self._construct_euclidean_distance_matrix()
        self._construct_dtw_distance_matrix()

    def get_dtw_distance(self):
        return self._dtw_distance_matrix[-1, -1]

    def get_euclidean_distance(self):
        return np.trace(self._euclidean_distance_matrix) / self._distance_matrix_shape[0]

    def _construct_dtw_distance_matrix(self):
        self._dtw_distance_matrix = np.zeros(self._distance_matrix_shape)

        # first row and column
        self._dtw_distance_matrix[0, 0] = self._euclidean_distance_matrix[0, 0]

        for i in range(1, self._distance_matrix_shape[0]):
            self._dtw_distance_matrix[i, 0] = self._dtw_distance_matrix[i - 1, 0] + \
                                              self._euclidean_distance_matrix[i, 0]

        for j in range(1, self._distance_matrix_shape[1]):
            self._dtw_distance_matrix[0, j] = self._dtw_distance_matrix[0, j - 1] + \
                                              self._euclidean_distance_matrix[0, j]

        # rest of the matrix
        for i in range(1, self._distance_matrix_shape[0]):
            for j in range(1, self._distance_matrix_shape[1]):
                choices = self._dtw_distance_matrix[i - 1, j - 1], \
                          self._dtw_distance_matrix[i, j - 1], \
                          self._dtw_distance_matrix[i - 1, j]
                self._dtw_distance_matrix[i, j] = min(choices) + self._euclidean_distance_matrix[i, j]

    def _construct_euclidean_distance_matrix(self):
        self._euclidean_distance_matrix = np.zeros(self._distance_matrix_shape)
        for i in range(self._distance_matrix_shape[0]):
            for j in range(self._distance_matrix_shape[1]):
                self._euclidean_distance_matrix[i, j] = np.sqrt(np.sum((self._a[:, i] - self._b[:, j]) ** 2))


def load_mfcc(file_path):
    y, sr = librosa.load(file_path, sr=None)
    return normalize(librosa.feature.mfcc(y=y, sr=sr))


def load_training_set(training_set_path):
    training_set = []
    for train_file_path in glob.glob("%s/*/*.wav" % training_set_path):
        training_set.append((load_mfcc(train_file_path), os.path.basename(os.path.dirname(train_file_path))))
    return training_set


def load_test_set(test_set_path):
    test_set = []
    for test_file_path in glob.glob("%s/*.wav" % test_set_path):
        test_set.append(load_mfcc(test_file_path))
    return test_set


def main():
    first_test = load_test_set("test_set")[0]
    for train_example, train_classification in load_training_set("training_set"):
        distance = Distance(first_test, train_example)
        print(distance.get_dtw_distance(), distance.get_euclidean_distance())


if __name__ == "__main__":
    main()
