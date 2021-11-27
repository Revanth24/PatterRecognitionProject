# -*- coding: utf-8 -*-
from PatternRecognition import *
from sklearn.linear_model import LogisticRegression

class LRModel(MNISTClassificationBaseModel):

  shape = (-1, 784)

  def __init__(self, balanced=False, noise_type='No Noise', noise_ratio=0):
    # Call the parent constructor to load the mninst dataset
    super().__init__(balanced, noise_type, noise_ratio)

  def get_model_shape(self):
    return LRModel.shape

  def shape_data(self):
    self._x_train = self._x_train.reshape(*LRModel.shape)
    self._x_test = self._x_test.reshape(*LRModel.shape)

  def pre_process_label(self):
    print('No label pre processing is required')

  def _define_model(self):
    return LogisticRegression(solver='lbfgs', max_iter=1000)

  def train_and_predict(self):
    clf = self._define_model()
    # Learn the digits on the train subset
    clf.fit(self.get_training_image(), self.get_training_label())

    # Predict the value of the digit on the test subset
    predicted = clf.predict(self.get_test_image())

    return clf, predicted

if __name__ == "__main__":
  print('Original Imbalanced with No noise')
  lr_imbalanced = LRModel(balanced=False, noise_type='No Noise', noise_ratio=0)
  lr_imbalanced.scale_pixels()
  lr_imbalanced.show_class_distribution(False)
  clf, predicted = lr_imbalanced.train_and_predict()
  lr_imbalanced.display_confusion_matrix(predicted)
  lr_imbalanced.print_accuracy(predicted)
  print(f"Classification report for classifier {clf}:\n"
      f"{metrics.classification_report(lr_imbalanced.get_test_label(), predicted)}\n")
  # test_image, actual_label, predicted_label = lr_imbalanced.get_misclassified(predicted)
  # lr_imbalanced.display_samples(test_image, actual_label, predicted_label, False)

  print('Imbalanced with Symmetrical Noise')
  sym_non_balanced = LRModel(balanced=False, noise_type='sym', noise_ratio=40)
  sym_non_balanced.scale_pixels()
  sym_non_balanced.show_class_distribution(False)
  clf2, predicted2 = sym_non_balanced.train_and_predict()
  sym_non_balanced.display_confusion_matrix(predicted2)
  sym_non_balanced.print_accuracy(predicted2)
  print(f"Classification report for classifier {clf2}:\n"
    f"{metrics.classification_report(sym_non_balanced.get_test_label(), predicted2)}\n")

  print('Imbalanced with Asymmetrical Noise')
  asym_non_balanced = LRModel(balanced=False, noise_type='asym', noise_ratio=40)
  asym_non_balanced.scale_pixels()
  asym_non_balanced.show_class_distribution(False)
  clf3, predicted3 = asym_non_balanced.train_and_predict()
  asym_non_balanced.display_confusion_matrix(predicted3)
  asym_non_balanced.print_accuracy(predicted3)
  print(f"Classification report for classifier {clf3}:\n"
    f"{metrics.classification_report(asym_non_balanced.get_test_label(), predicted3)}\n")

  print('Balanced with No Noise')
  lr_balanced = LRModel(balanced=True, noise_type='No Noise', noise_ratio=0)
  lr_balanced.scale_pixels()
  lr_balanced.show_class_distribution(False)
  clf6, predicted6 = lr_balanced.train_and_predict()
  lr_balanced.display_confusion_matrix(predicted6)
  lr_balanced.print_accuracy(predicted6)
  print(f"Classification report for classifier {clf6}:\n"
    f"{metrics.classification_report(lr_balanced.get_test_label(), predicted6)}\n")

  print('Balanced with Symmetrical Noise')
  sym_balanced = LRModel(balanced=True, noise_type='sym', noise_ratio=40)
  sym_balanced.scale_pixels()
  sym_balanced.show_class_distribution(False)
  clf4, predicted4 = sym_balanced.train_and_predict()
  sym_balanced.display_confusion_matrix(predicted4)
  sym_balanced.print_accuracy(predicted4)
  print(f"Classification report for classifier {clf4}:\n"
    f"{metrics.classification_report(sym_balanced.get_test_label(), predicted4)}\n")

  print('Balanced with Asymmetrical Noise')
  asym_balanced = LRModel(balanced=True, noise_type='asym', noise_ratio=40)
  asym_balanced.scale_pixels()
  asym_balanced.show_class_distribution(False)
  clf5, predicted5 = asym_balanced.train_and_predict()
  asym_balanced.display_confusion_matrix(predicted5)
  asym_balanced.print_accuracy(predicted5)
  print(f"Classification report for classifier {clf5}:\n"
    f"{metrics.classification_report(asym_balanced.get_test_label(), predicted5)}\n")
