import numpy as np
import librosa
import glob
import os


class Distance(object):
    def __init__(self, a: np.ndarray, b: np.ndarray):
        self._a = a
        self._b = b

        # for euclidean_distance we must ensure a and b have the same shape
        assert (self._a.shape == self._b.shape)

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
    return librosa.feature.mfcc(y=y, sr=sr)


def dirname_to_number(dirname: str):
    if dirname == "one":
        return "1"
    if dirname == "two":
        return "2"
    if dirname == "three":
        return "3"
    if dirname == "four":
        return "4"
    if dirname == "five":
        return "5"


def training_set(training_set_path):
    for train_file_path in glob.glob("%s/*/*.wav" % training_set_path):
        yield load_mfcc(train_file_path), dirname_to_number(os.path.basename(os.path.dirname(train_file_path)))


def test_set(test_set_path):
    for test_file_path in glob.glob("%s/*.wav" % test_set_path):
        yield load_mfcc(test_file_path), os.path.basename(test_file_path)


def create_output_file(results):
    with open("output.txt", "w") as f:
        for result in results:
            f.write(" - ".join(result) + "\n")


def main():
    results = []
    for test_example, test_file_path in test_set("test_set"):
        minimal_euclidean_distance = (float("inf"), None)
        minimal_dtw_distance = (float("inf"), None)

        for train_example, train_classification in training_set("training_set"):
            distance = Distance(test_example, train_example)
            euclidean_distance = distance.get_euclidean_distance()
            dtw_distance = distance.get_dtw_distance()
            if euclidean_distance < minimal_euclidean_distance[0]:
                minimal_euclidean_distance = (euclidean_distance, train_classification)
            if dtw_distance < minimal_dtw_distance[0]:
                minimal_dtw_distance = (dtw_distance, train_classification)

        results.append((test_file_path, minimal_euclidean_distance[1], minimal_dtw_distance[1]))
    print(results)
    create_output_file(results)


if __name__ == "__main__":
    main()
