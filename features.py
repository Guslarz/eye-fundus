import numpy as np
from scipy.stats import moment


class PointGenerator:
    def __init__(self, count, seed=None):
        self.__count = count
        self.__random_state = np.random.RandomState(seed=seed)

    def points(self, shape):
        return zip(
            self.__random_state.randint(shape[0], size=self.__count),
            self.__random_state.randint(shape[1], size=self.__count),
        )


class Extractor:
    def extract(self, img):
        raise NotImplementedError()


class FeatureExtractor(Extractor):
    def __init__(self, patch_size):
        super().__init__()
        self._patch_size = patch_size
        self._center = (patch_size // 2, patch_size // 2)

    def extract(self, img):
        padding = self._patch_size // 2
        height, width = img.shape
        padded_img = np.zeros((height + self._patch_size - 1, width + self._patch_size - 1))
        padded_img[padding:-padding, padding:-padding] = img

        for patch in self._extract_patches(padded_img, img.shape):
            yield self.__patch_to_features(patch)

    def _extract_patches(self, img, shape):
        raise NotImplementedError()

    def __patch_to_features(self, patch):
        return [
            patch[self._center],
            np.mean(patch),
            np.std(patch),
            moment(patch, moment=3, axis=None)
        ]

    def _patch_at(self, img, y, x):
        return img[y:(y + self._patch_size), x:(x + self._patch_size)]


class FeatureExtractorAll(FeatureExtractor):
    def _extract_patches(self, img, shape):
        for y in range(shape[0]):
            for x in range(shape[1]):
                yield self._patch_at(img, y, x)


class FeatureExtractorRandom(FeatureExtractor):
    def __init__(self, patch_size, patch_count, seed):
        super().__init__(patch_size)
        self.__point_generator = PointGenerator(patch_count, seed)

    def _extract_patches(self, img, shape):
        for y, x in self.__point_generator.points(shape):
            yield self._patch_at(img, y, x)


class PixelExtractor(Extractor):
    def extract(self, img):
        raise NotImplementedError()


class PixelExtractorAll(PixelExtractor):
    def extract(self, img):
        for value in img[:, :]:
            yield value


class PixelExtractorRandom(PixelExtractor):
    def __init__(self, count, seed):
        super().__init__()
        self.__point_generator = PointGenerator(count, seed)

    def extract(self, img):
        for y, x in self.__point_generator.points(img.shape):
            yield img[y, x]
