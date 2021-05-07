import numpy as np


class ClassificationResult:
    LABELS = {
        'name': 'Name',
        'duration': 'Duration [s]',
        'tp': 'True positive',
        'fp': 'False positive',
        'fn': 'False negative',
        'tn': 'True negative',
        'accuracy': 'Accuracy',
        'sensitivity': 'Sensitivity',
        'specificity': 'Specificity'
    }

    def __init__(self, name, input_image, expected_result, actual_result, duration):
        self.__input_image = input_image
        self.__expected_result = expected_result
        self.__actual_result = actual_result

        self.__data = dict()
        self.__add_data('name', name)
        self.__add_data('duration', duration)

        self.__calculate_statistics()

    @property
    def input_image(self):
        return self.__input_image

    @property
    def expected_result(self):
        return self.__expected_result

    @property
    def actual_result(self):
        return self.__actual_result

    @property
    def error_matrix(self):
        return self.__error_matrix

    @property
    def data(self):
        for k, v in self.__data.items():
            yield ClassificationResult.LABELS[k], v

    def __add_data(self, key, value):
        self.__data[key] = value

    def __calculate_statistics(self):
        equal = self.__expected_result == self.__actual_result
        notequal = self.__expected_result != self.__actual_result
        positive = self.__expected_result == 1
        negative = self.__expected_result == 0

        self.__generate_error_matrix(equal, notequal, positive, negative)
        self.__calculate_confusion_matrix(equal, notequal, positive, negative)
        self.__calculate_accuracy()
        self.__calculate_sensitivity()
        self.__calculate_specificity()

    def __generate_error_matrix(self, equal, notequal, positive, negative):
        matrix = np.zeros((*self.__actual_result.shape, 3))
        matrix[equal & positive, :] = 1
        matrix[notequal & positive, 2] = 1
        matrix[notequal & negative, 0] = 1
        self.__error_matrix = matrix

    def __calculate_confusion_matrix(self, equal, notequal, positive, negative):
        self.__add_data('tp', (equal & positive).sum())
        self.__add_data('fp', (notequal & positive).sum())
        self.__add_data('fn', (notequal & negative).sum())
        self.__add_data('tn', (equal & negative).sum())

    def __calculate_accuracy(self):
        tp = self.__data['tp']
        fp = self.__data['fp']
        fn = self.__data['fn']
        tn = self.__data['tn']
        self.__add_data('accuracy', (tp + tn) / (tp + tn + fp + fn))

    def __calculate_sensitivity(self):
        tp = self.__data['tp']
        fn = self.__data['fn']
        self.__add_data('sensitivity', tp / (tp + fn))

    def __calculate_specificity(self):
        fp = self.__data['fp']
        tn = self.__data['tn']
        self.__add_data('specificity', tn / (fp + tn))
