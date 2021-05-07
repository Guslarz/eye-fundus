import time
import os
import joblib
from sys import stdout
from sklearn.neighbors import KNeighborsClassifier

from image_processing import *
from features import FeatureExtractorAll, FeatureExtractorRandom, PixelExtractorRandom


class ClassificationResult:
    LABELS = {
        'name': 'Name',
        'duration': 'Duration',
        'tp': 'True positive',
        'fp': 'False positive',
        'fn': 'False negative',
        'tn': 'True negative',
        'accuracy': 'Accuracy',
        'sensitivity': 'Sensitivity',
        'specificity': 'Specificity'
    }
    DEFAULT_FORMAT = "%s"
    PERCENT_FORMAT = "%.1f%%"
    FORMATS = {
        'duration': "%.1f s",
        'accuracy': PERCENT_FORMAT,
        'sensitivity': PERCENT_FORMAT,
        'specificity': PERCENT_FORMAT
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
            yield ClassificationResult.LABELS[k], \
                  ClassificationResult.FORMATS.get(k, ClassificationResult.DEFAULT_FORMAT) % v

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
        matrix[notequal & positive, 0] = 1
        matrix[notequal & negative, 2] = 1
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
        self.__add_data('accuracy', 100.0 * (tp + tn) / (tp + tn + fp + fn))

    def __calculate_sensitivity(self):
        tp = self.__data['tp']
        fn = self.__data['fn']
        self.__add_data('sensitivity', 100.0 * tp / (tp + fn))

    def __calculate_specificity(self):
        fp = self.__data['fp']
        tn = self.__data['tn']
        self.__add_data('specificity', 100.0 * tn / (fp + tn))


class Classifier:
    MODEL_DIR = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'model')

    def train(self, train_generator, validation_generator):
        raise NotImplementedError()

    def classify(self, name, raw_image, input_image, expected_result):
        stdout.write("Classify...")
        start_time = time.time()
        actual_result = self._inner_classify(input_image)
        duration = time.time() - start_time
        stdout.write("\rClassified \n")
        return ClassificationResult(name, raw_image, expected_result, actual_result, duration)

    def _inner_classify(self, input_image):
        raise NotImplementedError()

    def save(self):
        raise NotImplementedError()

    def load(self):
        raise NotImplementedError()

    @staticmethod
    def create_model_directory():
        if not os.path.exists(Classifier.MODEL_DIR):
            os.mkdir(Classifier.MODEL_DIR)


class ImageProcessingClassifier(Classifier):
    def train(self, train_generator, validation_generator):
        pass

    def _inner_classify(self, input_image):
        image = input_image.copy()
        channels = rgb_split(image)

        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        image = cv2.cvtColor(channels[1], cv2.COLOR_BGR2GRAY)
        image = clahe.apply(image)
        image = normalize(image, 0.28, 98)

        image = clear_data(image)
        image = correcting(image)

        image = put_mask(image, create_mask(channels[0]))
        return image

    def save(self):
        pass

    def load(self):
        pass


class ImageProcessingClassifierTurboExtra(ImageProcessingClassifier):
    def _inner_classify(self, input_image):
        image = input_image.copy()
        channels = rgb_split(image)

        mask = create_mask(channels[0])

        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        img_green = cv2.cvtColor(put_mask(channels[1], mask), cv2.COLOR_BGR2GRAY)
        img_clahe = clahe.apply(img_green)

        veils = find_veils(img_clahe)
        veils = put_mask(veils, mask)

        veils_fan = normalize(veils.copy(), 0.28, 98)
        veils_fan = contrast(veils_fan, 0.25)
        return veils_fan


class KnnClassifier(Classifier):
    MODEL_FILE = os.path.join(Classifier.MODEL_DIR, 'knn')
    PATCH_SIZE = 5
    PATCH_COUNT = 10000

    def __init__(self, patch_size=PATCH_SIZE, patch_count=PATCH_COUNT):
        self.__model = KNeighborsClassifier()
        self.__patch_size = patch_size
        self.__patch_count = patch_count

    def train(self, train_generator, validation_generator):
        x, y = [], []
        seed = int(time.time())
        feature_extractor = FeatureExtractorRandom(
            patch_size=self.__patch_size, patch_count=self.__patch_count, seed=seed)
        pixel_extractor = PixelExtractorRandom(count=self.__patch_count, seed=seed)
        for i, (image, mask) in enumerate(train_generator):
            stdout.write("\rTraining features: %.1f%%" % (100.0 * (i + 1) / train_generator.size))
            stdout.flush()
            x.extend(feature_extractor.extract(image))
            y.extend(pixel_extractor.extract(mask))
        stdout.write("\nTrain...")
        stdout.flush()
        self.__model.fit(x, y)
        stdout.write("\rTrained\n")

    def _inner_classify(self, input_image):
        feature_extractor = FeatureExtractorAll(patch_size=self.__patch_size)
        return self.__model.predict([*feature_extractor.extract(input_image)])\
            .reshape(input_image.shape)

    def save(self):
        Classifier.create_model_directory()
        joblib.dump(self.__model, KnnClassifier.MODEL_FILE)

    def load(self):
        self.__model = joblib.load(KnnClassifier.MODEL_FILE)


class ClassifierFactory:
    @staticmethod
    def create_classifier(cls, load=False, train_generator=None, validation_generator=None, **kwargs):
        classifier = cls(**kwargs)
        if load:
            classifier.load()
        else:
            classifier.train(train_generator, validation_generator)
            classifier.save()
        return classifier
