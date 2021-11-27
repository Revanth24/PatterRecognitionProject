# -*- coding: utf-8 -*-

from keras.datasets import mnist
from imblearn.under_sampling import RandomUnderSampler
from tensorflow.keras.utils import to_categorical
from sklearn import metrics
from sklearn.metrics import accuracy_score
from abc import ABC, abstractmethod
import matplotlib.pyplot as plt
import numpy as np

np.random.seed(123)

class MNISTClassificationBaseModel(ABC):

  class_count = 10
  original_dataset_shape = (28, 28)
  classes = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]

  @abstractmethod
  def __init__(self, balanced=False, noise_type='No Noise', noise_ratio=0):
    (self._x_train, self._y_train), (self._x_test, self._y_test) = mnist.load_data()
    self._original_dataset = (np.copy(self._x_train), np.copy(self._y_train),
                              np.copy(self._x_test), np.copy(self._y_test))

    self.shape_data()

    if balanced is True:
      self.balance_dataset()

    if noise_type == 'asym':
      self.induce_asym_noise(noise_ratio)
      clean_selected = np.argwhere(self.get_training_label() == self.get_original_dataset()[1]).reshape((-1,))
      noisy_selected = np.argwhere(self.get_training_label() != self.get_original_dataset()[1]).reshape((-1,))
      print("#correct labels: %s, #incorrect labels: %s" % (len(clean_selected), len(noisy_selected)))
    elif noise_type == 'sym':
      self.induce_sym_noise(noise_ratio)
      clean_selected = np.argwhere(self.get_training_label() == self.get_original_dataset()[1]).reshape((-1,))
      noisy_selected = np.argwhere(self.get_training_label() != self.get_original_dataset()[1]).reshape((-1,))
      print("#correct labels: %s, #incorrect labels: %s" % (len(clean_selected), len(noisy_selected)))
    elif noise_type == 'No Noise':
      pass
    else:
      raise ValueError('%s noise type is not recognized. Value can be asym, sym or No Noise' % noise_type)

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
    print('Accuracy for the predicted labels : ' , accuracy_score(self.get_test_label(), predicted) * 100)

  def display_confusion_matrix(self, predicted):
    cm = metrics.confusion_matrix(self.get_test_label(), predicted,
                                  labels=MNISTClassificationBaseModel.classes)
    disp = metrics.ConfusionMatrixDisplay(confusion_matrix=cm,
                                          display_labels=MNISTClassificationBaseModel.classes)
    disp.plot()
    plt.show()

  def display_samples(self, images, actual_labels, predicted_labels, is_label_pre_processed):

    images = images.reshape(*self.get_model_shape())
    predicted_labels = predicted_labels.squeeze()
    actual_labels = actual_labels.squeeze()

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
    misclassified_test_image = self.get_test_image()[misclassified_indices]
    misclassified_predicted_label = predicted[misclassified_indices]
    misclassified_actual_label = self.get_test_label()[misclassified_indices]
    return misclassified_test_image, misclassified_actual_label, misclassified_predicted_label

  def balance_dataset(self):
    rus = RandomUnderSampler(random_state=123)
    self._x_train, self._y_train = rus.fit_resample(self.get_training_image(), self.get_training_label())

  def induce_asym_noise(self, noise_ratio=40):
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

  def other_class(self, n_classes, current_class):
    if current_class < 0 or current_class >= n_classes:
        error_str = "class_ind must be within the range (0, nb_classes - 1)"
        raise ValueError(error_str)

    other_class_list = list(range(n_classes))
    other_class_list.remove(current_class)
    other_class = np.random.choice(other_class_list)
    return other_class

  def induce_sym_noise(self, noise_ratio=40):
    y_train_clean = np.copy(self.get_training_label())
    n_samples = self.get_training_label().shape[0]
    n_noisy = int(noise_ratio * n_samples / 100)
    # print(n_samples, n_noisy)
    class_index = [np.where(y_train_clean == i)[0] for i in range(MNISTClassificationBaseModel.class_count)]
    class_noisy = int(n_noisy / 10)

    noisy_idx = []
    for d in range(MNISTClassificationBaseModel.class_count):
      noisy_class_index = np.random.choice(class_index[d], class_noisy, replace=False)
      noisy_idx.extend(noisy_class_index)

    for i in noisy_idx:
      self._y_train[i] = self.other_class(n_classes=MNISTClassificationBaseModel.class_count,
                                     current_class=self.get_training_label()[i])

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
