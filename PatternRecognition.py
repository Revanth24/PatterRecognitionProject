# -*- coding: utf-8 -*-

from keras.datasets import mnist
from tensorflow.keras.utils import to_categorical
from sklearn import svm, metrics
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from abc import ABC, abstractmethod
import matplotlib.pyplot as plt
import numpy as np


class MNISTClassificationBaseModel(ABC):

  class_count = 10
  original_dataset_shape = (28, 28)
  classes = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]

  @abstractmethod
  def __init__(self, balanced=False, asym=False, noise_ratio=0):
    (self._x_train, self._y_train), (self._x_test, self._y_test) = mnist.load_data()
    self._original_dataset = (np.copy(self._x_train), np.copy(self._y_train),
                              np.copy(self._x_test), np.copy(self._y_test))

    if asym:
      self.induce_asym_noise(noise_ratio)

  def get_original_dataset(self):
    return self._original_dataset;

  def get_training_image(self):
    return self._x_train

  def get_training_label(self):
    return self._y_train

  def get_test_image(self):
    return self._x_test

  def get_test_label(self):
    return self._y_test

  def scale_pixels(self):
    # Convert from integers to float
    self._x_train = self._x_train.astype('float32')
    self._x_test = self._x_test.astype('float32')

    # Normalize pixels from range [0, 255] to [0, 1]
    self._x_train /= 255.0
    self._x_test /= 255.0

  def show_class_distribution(self, original_dataset=False):
    if original_dataset:
      unique, counts = np.unique(self.get_original_dataset()[1], return_counts=True)
      plt.bar(unique, counts)
      # unique, counts = np.unique(self.get_original_dataset()[3], return_counts=True)
      # plt.bar(unique, counts)
    else:
      unique, counts = np.unique(self.get_training_label(), return_counts=True)
      plt.bar(unique, counts)
      # unique, counts = np.unique(self.get_test_label(), return_counts=True)
      # plt.bar(unique, counts)

    plt.title('Class Distribution')
    plt.xlabel('Class')
    plt.ylabel('Frequency')

    plt.show()

  def print_accuracy(self, predicted):
    print('Accuracy for the predicted labels : ' , accuracy_score(s.get_test_label(), predicted) * 100)

  def display_confusion_matrix(self, predicted):
    cm = metrics.confusion_matrix(s.get_test_label(), predicted,
                                  labels=MNISTClassificationBaseModel.classes)
    disp = metrics.ConfusionMatrixDisplay(confusion_matrix=cm,
                                          display_labels=MNISTClassificationBaseModel.classes)
    disp.plot()
    plt.show()

  def display_samples(self, images, actual_labels, predicted_labels, is_label_pre_processed):

    images = images.reshape(-1, self.get_model_shape())
    predicted_labels = predicted_labels.reshape(len(images))
    actual_labels = actual_labels.reshape(len(images))

    l = len(images)

    while True:
      if l <= 0:
        break

      cols = min(5, l)

      subset_images, images = np.asarray(images[:cols]), np.asarray(images[cols:])
      subset_predicted_labels, predicted_labels = np.asarray(predicted_labels[:cols]), np.asarray(predicted_labels[cols:])
      subset_actual_labels, actual_labels = np.asarray(actual_labels[:cols]), np.asarray(actual_labels[cols:])

      _, axes = plt.subplots(nrows=1, ncols=cols, figsize=(10, 3),
                          constrained_layout=True)

      for ax, image, actual_label, predicted_label in zip(axes, subset_images, subset_actual_labels, subset_predicted_labels):
        ax.set_axis_off()
        ax.imshow(image.reshape(MNISTClassificationBaseModel.original_dataset_shape), cmap='gray')
        if is_label_pre_processed:
          ax.set_title('Actual: %i, Predicted: %i' % (np.argmax(actual_label), np.argmax(predicted_label)))
        else:
          ax.set_title('Actual: %i, Predicted: %i' % (actual_label, predicted_label))

      plt.show()

      l -=cols

  def _get_misclassified_indices(self, predicted):
    return np.nonzero(predicted != self.get_test_label())

  def get_misclassified(self, predicted):
    misclassified_indices = np.asarray(self._get_misclassified_indices(predicted))

    misclassified_test_image = s.get_test_image()[misclassified_indices]
    misclassified_predicted_label = predicted[misclassified_indices]
    misclassified_actual_label = s.get_test_label()[misclassified_indices]

    return misclassified_test_image, misclassified_actual_label, misclassified_predicted_label

  def induce_asym_noise(self, noise_ratio=0):
    source_class = [7, 2, 3, 5, 6]
    target_class = [1, 7, 8, 6, 5]
    y_train_clean = np.copy(self.get_training_label())

    for s, t in zip(source_class, target_class):
      cls_idx = np.where(y_train_clean == s)[0]
      # print('cls_idx',cls_idx)
      n_noisy = int(noise_ratio * cls_idx.shape[0] / 100)
      # print('n_noisy',n_noisy)
      noisy_sample_index = np.random.choice(cls_idx, n_noisy, replace=False)
      # print(noisy_sample_index)
      self._y_train[noisy_sample_index] = t

    print("Print noisy label generation statistics:")
    for i in range(MNISTClassificationBaseModel.class_count):
      n_noisy = np.sum(self.get_training_label() == i)
      print("Noisy class %s, has %s samples." % (i, n_noisy))

  @abstractmethod
  def shape_data(self):
    pass

  @abstractmethod
  def get_model_shape(self):
    pass

  @abstractmethod
  def pre_process_label(self):
    pass

  @abstractmethod
  def _define_model(self):
    pass

class LRModel(MNISTClassificationBaseModel):

  shape = 784

  def __init__(self, balanced=False, asym=False, noise_ratio=0):
    # Call the parent constructor to load the mninst dataset
    super().__init__(balanced, asym, noise_ratio)

  def get_model_shape(self):
    return LRModel.shape

  def shape_data(self):
    self._x_train = self._x_train.reshape(-1, LRModel.shape)
    self._x_test = self._x_test.reshape(-1, LRModel.shape)

  def pre_process_label(self):
    print('No label pre processing is required')

  def _define_model(self):
    return LogisticRegression(solver='lbfgs', max_iter=1000)

  def train_model(self):
    clf = self._define_model()
    # Learn the digits on the train subset
    clf.fit(self.get_training_image(), self.get_training_label())

    # Predict the value of the digit on the test subset
    predicted = clf.predict(self.get_test_image())

    return clf, predicted

class SVMModel(MNISTClassificationBaseModel):

  shape = 784

  def __init__(self, balanced=False, asym=False, noise_ratio=0):
    # Call the parent constructor to load the mninst dataset
    super().__init__(balanced, asym, noise_ratio)

  def get_model_shape(self):
    return SVMModel.shape

  def shape_data(self):
    self._x_train = self._x_train.reshape(-1, SVMModel.shape)
    self._x_test = self._x_test.reshape(-1, SVMModel.shape)

  def pre_process_label(self):
    print('No label pre processing is required')

  def _define_model(self):
    return svm.SVC()
    # return svm.SVC(kernel='rbf', gamma=1, C=10)

  def train_model(self):
    clf = self._define_model()
    # Learn the digits on the train subset
    clf.fit(self.get_training_image(), self.get_training_label())

    # Predict the value of the digit on the test subset
    predicted = clf.predict(self.get_test_image())

    return clf, predicted

s = SVMModel(balanced=False, asym=False, noise_ratio=40)
s.show_class_distribution(True)
s.scale_pixels()
s.shape_data()
clf, predicted = s.train_model()
s.display_confusion_matrix(predicted)
s.print_accuracy(predicted)
print(f"Classification report for classifier {clf}:\n"
    f"{metrics.classification_report(s.get_test_label(), predicted)}\n")

test_image, actual_label, predicted_label = s.get_misclassified(predicted)
s.display_samples(test_image, actual_label, predicted_label, False)
