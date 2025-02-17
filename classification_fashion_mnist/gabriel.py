import numpy as np
import matplotlib.pyplot as plt
from numpy.ma import mean
import pandas as pd
from skimage import measure
from sklearn.model_selection import train_test_split, GroupKFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.metrics import roc_auc_score
from sklearn.decomposition import PCA
from sklearn.tree import DecisionTreeClassifier

def load_data(path):
    """
    Loads the data from the given path and returns the original images, reshaped images,
    padded images, padded images left, padded images right, padded images top, padded images bottom,
    binary images, binary images left, binary images right, binary images top, binary images bottom,
    and labels.
    """
    original_images = np.load(path)
    original_image = (original_images[:, :-1] - np.mean(original_images[:, :-1], axis=1)[:, np.newaxis]) / np.std(original_images[:, :-1], axis=1)[:, np.newaxis]
    reshaped_images = original_images[:, :28 * 28].reshape(-1, 28, 28)
    padded_images = np.pad(reshaped_images, pad_width=((0, 0), (1, 1), (1, 1)), mode='constant', constant_values=0)
    padded_images_left = padded_images[:, :, :15]
    padded_images_right = padded_images[:, :, 15:]
    padded_images_top = padded_images[:, :15, :]
    padded_images_bottom = padded_images[:, 15:, :]
    # threshold_values = np.array([filters.threshold_mean(image)/8 for image in padded_images])
    threshold_value = 10
    binary_images = padded_images > threshold_value
    binary_images_top = padded_images_top > threshold_value
    binary_images_bottom = padded_images_bottom > threshold_value
    binary_images_left = padded_images_left > threshold_value
    binary_images_right = padded_images_right > threshold_value
    labels = original_images[:, -1]

    return original_images, reshaped_images, padded_images, padded_images_left, padded_images_right, padded_images_top, padded_images_bottom, binary_images, binary_images_left, binary_images_right, binary_images_top, binary_images_bottom, labels


def calculate_circumferences(binary_images, binary_images_left, binary_images_right, binary_images_top, binary_images_bottom):
    """
    Calculates all circumferences of each binary image and its left, right, top, and bottom halves.
    """
    horizontal_transitions = np.sum(binary_images[:, 1:, :] != binary_images[:, :-1, :], axis=(1, 2))
    vertical_transitions = np.sum(binary_images[:, :, 1:] != binary_images[:, :, :-1], axis=(1, 2))
    total_transitions = horizontal_transitions + vertical_transitions

    horizontal_transitions_left = np.sum(binary_images_left[:, 1:, :] != binary_images_left[:, :-1, :], axis=(1, 2))
    vertical_transitions_left = np.sum(binary_images_left[:, :, 1:] != binary_images_left[:, :, :-1], axis=(1, 2))
    total_transitions_left = horizontal_transitions_left + vertical_transitions_left

    horizontal_transitions_right = np.sum(binary_images_right[:, 1:, :] != binary_images_right[:, :-1, :], axis=(1, 2))
    vertical_transitions_right = np.sum(binary_images_right[:, :, 1:] != binary_images_right[:, :, :-1], axis=(1, 2))
    total_transitions_right = horizontal_transitions_right + vertical_transitions_right

    horizontal_transitions_top = np.sum(binary_images_top[:, 1:, :] != binary_images_top[:, :-1, :], axis=(1, 2))
    vertical_transitions_top = np.sum(binary_images_top[:, :, 1:] != binary_images_top[:, :, :-1], axis=(1, 2))
    total_transitions_top = horizontal_transitions_top + vertical_transitions_top

    horizontal_transitions_bottom = np.sum(binary_images_bottom[:, 1:, :] != binary_images_bottom[:, :-1, :], axis=(1, 2))
    vertical_transitions_bottom = np.sum(binary_images_bottom[:, :, 1:] != binary_images_bottom[:, :, :-1], axis=(1, 2))
    total_transitions_bottom = horizontal_transitions_bottom + vertical_transitions_bottom

    return total_transitions, horizontal_transitions, vertical_transitions, total_transitions_left, horizontal_transitions_left, vertical_transitions_left, total_transitions_right, horizontal_transitions_right, vertical_transitions_right, total_transitions_top, horizontal_transitions_top, vertical_transitions_top, total_transitions_bottom, horizontal_transitions_bottom, vertical_transitions_bottom


def calculate_widths(binary_images, binary_images_left, binary_images_right, binary_images_top, binary_images_bottom):
    """
    Calculates all widths of each binary image and its left, right, top, and bottom halves.
    Widths includes max width, mean width, and variance of width.
    """
    leftmost = np.argmax(binary_images[:, 1:-1, :], axis=2)
    rightmost = np.argmax(np.flip(binary_images[:, 1:-1, :], axis=2), axis=2)
    widths = 29 - (leftmost + rightmost)
    max_widths = np.max(widths, axis=1)
    mean_widths = np.mean(widths, axis=1)
    variance_widths = np.var(widths, axis=1)

    leftmost_left = np.argmax(binary_images_left[:, 1:-1, :], axis=2)
    rightmost_left = np.argmax(np.flip(binary_images_left[:, 1:-1, :], axis=2), axis=2)
    widths_left = 15 - (leftmost_left + rightmost_left)
    max_widths_left = np.max(widths_left, axis=1)
    mean_widths_left = np.mean(widths_left, axis=1)
    variance_widths_left = np.var(widths_left, axis=1)

    leftmost_right = np.argmax(binary_images_right[:, 1:-1, :], axis=2)
    rightmost_right = np.argmax(np.flip(binary_images_right[:, 1:-1, :], axis=2), axis=2)
    widths_right = 15 - (leftmost_right + rightmost_right)
    max_widths_right = np.max(widths_right, axis=1)
    mean_widths_right = np.mean(widths_right, axis=1)
    variance_widths_right = np.var(widths_right, axis=1)

    leftmost_top = np.argmax(binary_images_top[:, 1:, :], axis=2)
    rightmost_top = np.argmax(np.flip(binary_images_top[:, 1:, :], axis=2), axis=2)
    widths_top = 29 - (leftmost_top + rightmost_top)
    max_widths_top = np.max(widths_top, axis=1)
    mean_widths_top = np.mean(widths_top, axis=1)
    variance_widths_top = np.var(widths_top, axis=1)

    leftmost_bottom = np.argmax(binary_images_bottom[:, :-1, :], axis=2)
    rightmost_bottom = np.argmax(np.flip(binary_images_bottom[:, :-1, :], axis=2), axis=2)
    widths_bottom = 29 - (leftmost_bottom + rightmost_bottom)
    max_widths_bottom = np.max(widths_bottom, axis=1)
    mean_widths_bottom = np.mean(widths_bottom, axis=1)
    variance_widths_bottom = np.var(widths_bottom, axis=1)

    return max_widths, mean_widths, variance_widths, max_widths_left, mean_widths_left, variance_widths_left, max_widths_right, mean_widths_right, variance_widths_right, max_widths_top, mean_widths_top, variance_widths_top, max_widths_bottom, mean_widths_bottom, variance_widths_bottom


def calculate_heights(binary_images, binary_images_left, binary_images_right, binary_images_top, binary_images_bottom):
    """
    Calculates all heights of each binary image and its left, right, top, and bottom halves.
    Widths includes max height, mean height, and variance of height.
    """
    topmost = np.argmax(binary_images[:, :, 1:-1], axis=1)
    bottommost = np.argmax(np.flip(binary_images[:, :, 1:-1], axis=1), axis=1)
    heights = 29 - (topmost + bottommost)
    max_heights = np.max(heights, axis=1)
    mean_heights = np.mean(heights, axis=1)
    variance_heights = np.var(heights, axis=1)

    topmost_left = np.argmax(binary_images_left[:, :, 1:-1], axis=1)
    bottommost_left = np.argmax(np.flip(binary_images_left[:, :, 1:-1], axis=1), axis=1)
    heights_left = 29 - (topmost_left + bottommost_left)
    max_heights_left = np.max(heights_left, axis=1)
    mean_heights_left = np.mean(heights_left, axis=1)
    variance_heights_left = np.var(heights_left, axis=1)

    topmost_right = np.argmax(binary_images_right[:, :, 1:-1], axis=1)
    bottommost_right = np.argmax(np.flip(binary_images_right[:, :, 1:-1], axis=1), axis=1)
    heights_right = 29 - (topmost_right + bottommost_right)
    max_heights_right = np.max(heights_right, axis=1)
    mean_heights_right = np.mean(heights_right, axis=1)
    variance_heights_right = np.var(heights_right, axis=1)

    topmost_top = np.argmax(binary_images_top[:, :, 1:], axis=1)
    bottommost_top = np.argmax(np.flip(binary_images_top[:, :, 1:], axis=1), axis=1)
    heights_top = 29 - (topmost_top + bottommost_top)
    max_heights_top = np.max(heights_top, axis=1)
    mean_heights_top = np.mean(heights_top, axis=1)
    variance_heights_top = np.var(heights_top, axis=1)

    topmost_bottom = np.argmax(binary_images_bottom[:, :, :-1], axis=1)
    bottommost_bottom = np.argmax(np.flip(binary_images_bottom[:, :, :-1], axis=1), axis=1)
    heights_bottom = 29 - (topmost_bottom + bottommost_bottom)
    max_heights_bottom = np.max(heights_bottom, axis=1)
    mean_heights_bottom = np.mean(heights_bottom, axis=1)
    variance_heights_bottom = np.var(heights_bottom, axis=1)

    return max_heights, mean_heights, variance_heights, max_heights_left, mean_heights_left, variance_heights_left, max_heights_right, mean_heights_right, variance_heights_right, max_heights_top, mean_heights_top, variance_heights_top, max_heights_bottom, mean_heights_bottom, variance_heights_bottom


def pca_on_images(original_images, labels, n_components=71):
    """
    Perform PCA on the original images
    Using built-in PCA function, with a default of 71 components
    Components need to be adjusted based on our discussion from previous meeting
    """
    pca = PCA(n_components)
    original_images_pca = pca.fit_transform(original_images)
    pca_data_with_label = np.column_stack([labels, original_images_pca])
    return pca_data_with_label


def decision_tree_classifier(data_path):
    """
    Decision Tree Classifier for classification of images
    Standard stuff, nothing special, also prints some metrics
    """
    # need k fold cross validation
    df = pd.read_csv(data_path)
    X, y = df.iloc[:, 1:], df.iloc[:, 0]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.7, random_state=42)
    dt = DecisionTreeClassifier(max_depth=50)
    dt.fit(X_train, y_train)
    y_pred = dt.predict(X_test)
    y_pred_proba = dt.predict_proba(X_test)
    print('\nDecision Tree Classifier:')
    print('Confusion Matrix:\n', confusion_matrix(y_test, y_pred))
    print('Classification Report:\n',classification_report(y_test, y_pred))
    print('AUC Score:', round(roc_auc_score(y_test, y_pred_proba, multi_class='ovr'), 2))


def random_forest_classifier(data_path):
    """
    Random Forest Classifier for classification of images
    Standard stuff, nothing special, also prints some metrics
    """
    # need k fold cross validation
    number_of_folds = 5
    group_k_fold = GroupKFold(n_splits=number_of_folds)
    df = pd.read_csv(data_path)
    X, y = df.iloc[:, 1:], df.iloc[:, 0]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    rf = RandomForestClassifier(n_estimators=100, random_state=42)
    rf.fit(X_train, y_train)
    y_pred = rf.predict(X_test)
    y_pred_proba = rf.predict_proba(X_test)
    print('\nRandom Forest Classifier:')
    print('Confusion Matrix:\n', confusion_matrix(y_test, y_pred))
    print('Classification Report:\n',classification_report(y_test, y_pred))
    print('AUC Score:', round(roc_auc_score(y_test, y_pred_proba, multi_class='ovr'), 4))



def random_forest_classifier_kfold(data_path):
    # To be implemented
    df = pd.read_csv(data_path)
    rf = RandomForestClassifier(n_estimators=100, random_state=42)
    group_k_fold = GroupKFold(n_splits=5)
    X, y = df.iloc[:, 1:], df.iloc[:, 0]


def evaluate_classifiers(data_path):
    """
    Evaluate the performance of the classifiers
    Current classifiers: Random Forest and Decision Tree
    """
    # Need to do k fold and cross validation
    df = pd.read_csv(data_path)
    features, labels = df.iloc[:, 1:], df.iloc[:, 0]
    classifiers = [RandomForestClassifier(), DecisionTreeClassifier()]
    runs = 5
    number_of_folds = 5
    number_of_classifiers = len(classifiers)


def visualize_data(padded_images, binary_images):
    """
    Visualizes the padded images and their binary versions
    Needs to be implemented: Display the data next to the images
    """
    contours = [measure.find_contours(image) for image in binary_images]
    fig, axes = plt.subplots(5, 6, figsize=(10, 8))
    for i in range(5):
        axes[i, 0].imshow(padded_images[i], cmap='gray')
        axes[i, 1].imshow(binary_images[i], cmap='gray')
        axes[i, 2].imshow(padded_images[i], cmap='gray')
        axes[i, 3].imshow(padded_images[i+5], cmap='gray')
        axes[i, 4].imshow(binary_images[i+5], cmap='gray')
        axes[i, 5].imshow(padded_images[i+5], cmap='gray')
        for contour in contours[i]:
            axes[i, 2].plot(contour[:, 1], contour[:, 0], linewidth=2)
        for contour in contours[i+5]:
            axes[i, 5].plot(contour[:, 1], contour[:, 0], linewidth=2)
    for ax in axes.ravel():
        ax.axis('off')
    plt.tight_layout()
    plt.show()


def data_manipulator():
    original_images, reshaped_images, padded_images, padded_images_left, padded_images_right, padded_images_top, \
            padded_images_bottom, binary_images, binary_images_left, binary_images_right, binary_images_top, \
            binary_images_bottom, labels = load_data('fashion_train.npy')

    total_transitions, horizontal_transitions, vertical_transitions, total_transitions_left, \
            horizontal_transitions_left, vertical_transitions_left, total_transitions_right, \
            horizontal_transitions_right, vertical_transitions_right, total_transitions_top, \
            horizontal_transitions_top, vertical_transitions_top, total_transitions_bottom, horizontal_transitions_bottom, \
            vertical_transitions_bottom = calculate_circumferences(binary_images, binary_images_left,
                                                                   binary_images_right, binary_images_top,
                                                                   binary_images_bottom)

    max_widths, mean_widths, variance_widths, max_widths_left, mean_widths_left, variance_widths_left, \
            max_widths_right, mean_widths_right, variance_widths_right, max_widths_top, mean_widths_top, \
            variance_widths_top, max_widths_bottom, mean_widths_bottom, \
            variance_widths_bottom = calculate_widths(binary_images, binary_images_left, binary_images_right,
                                                      binary_images_top, binary_images_bottom)

    max_heights, mean_heights, variance_heights, max_heights_left, mean_heights_left, variance_heights_left, \
            max_heights_right, mean_heights_right, variance_heights_right, max_heights_top, mean_heights_top, \
            variance_heights_top, max_heights_bottom, mean_heights_bottom, \
            variance_heights_bottom = calculate_heights(binary_images, binary_images_left, binary_images_right,
                                                        binary_images_top, binary_images_bottom)

    circumference_data = np.column_stack([labels, total_transitions, horizontal_transitions, vertical_transitions,
                            total_transitions_left, horizontal_transitions_left, vertical_transitions_left,
                            total_transitions_right, horizontal_transitions_right, vertical_transitions_right,
                            total_transitions_top, horizontal_transitions_top, vertical_transitions_top,
                            total_transitions_bottom, horizontal_transitions_bottom, vertical_transitions_bottom])

    width_data = np.column_stack([labels, max_widths, mean_widths, variance_widths,
                            max_widths_left, mean_widths_left, variance_widths_left,
                            max_widths_right, mean_widths_right, variance_widths_right,
                            max_widths_top, mean_widths_top, variance_widths_top,
                            max_widths_bottom, mean_widths_bottom, variance_widths_bottom])

    height_data = np.column_stack([labels, max_heights, mean_heights, variance_heights,
                            max_heights_left, mean_heights_left, variance_heights_left,
                            max_heights_right, mean_heights_right, variance_heights_right,
                            max_heights_top, mean_heights_top, variance_heights_top,
                            max_heights_bottom, mean_heights_bottom, variance_heights_bottom])

    circumference_column_names = ['labels', 'total_transitions', 'horizontal_transitions', 'vertical_transitions',
                    'total_transitions_left', 'horizontal_transitions_left', 'vertical_transitions_left',
                    'total_transitions_right', 'horizontal_transitions_right', 'vertical_transitions_right',
                    'total_transitions_top', 'horizontal_transitions_top', 'vertical_transitions_top',
                    'total_transitions_bottom', 'horizontal_transitions_bottom', 'vertical_transitions_bottom']

    width_column_names = ['labels', 'max_widths', 'mean_widths', 'variance_widths',
                    'max_widths_left', 'mean_widths_left', 'variance_widths_left',
                    'max_widths_right', 'mean_widths_right', 'variance_widths_right',
                    'max_widths_top', 'mean_widths_top', 'variance_widths_top',
                    'max_widths_bottom', 'mean_widths_bottom', 'variance_widths_bottom']

    height_column_names = ['labels', 'max_heights', 'mean_heights', 'variance_heights',
                    'max_heights_left', 'mean_heights_left', 'variance_heights_left',
                    'max_heights_right', 'mean_heights_right', 'variance_heights_right',
                    'max_heights_top', 'mean_heights_top', 'variance_heights_top',
                    'max_heights_bottom', 'mean_heights_bottom', 'variance_heights_bottom']

    pca_data = pca_on_images(original_images, labels, 150)
    df_circumference = pd.DataFrame(circumference_data, columns=circumference_column_names)
    df_circumference.to_csv('circumference_features.csv', index=False)
    df_width = pd.DataFrame(width_data, columns=width_column_names)
    df_width.to_csv('width_features.csv', index=False)
    df_height = pd.DataFrame(height_data, columns=height_column_names)
    df_height.to_csv('height_features.csv', index=False)
    df_pca = pd.DataFrame(pca_data)
    df_pca.to_csv('pca_features.csv', index=False)
    df_template = pd.read_csv('template_matching.csv')
    df_template.iloc[:, 0] = df_circumference.iloc[:, 0]
    df_pca.to_csv('template_features.csv', index=False)

    df_combined = pd.concat([df_circumference, df_width.iloc[:, 1:], df_height.iloc[:, 1:], df_pca.iloc[:, 1:], df_template.iloc[:, 1:]], axis=1)
    df_combined.to_csv('combined_features.csv', index=False)

    # visualize_data(padded_images, binary_images)


class DecisionTree:
    def __init__(self, max_depth=1000, min_samples_split=7):
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.tree = None

    def fit(self, X, y):
        self.tree = self._build_tree(X, y)

    def predict(self, X):
        return np.array([self._predict_single_input(x, self.tree) for x in X])

    def _build_tree(self, X, y, depth=0):
        num_samples, num_features = X.shape
        num_classes = len(set(y))
        if (depth >= self.max_depth or num_samples < self.min_samples_split or num_classes == 1):
            return self._create_leaf(y)
        best_gini, best_idx, best_thresh = np.inf, None, None
        for idx in range(num_features):
            thresholds = np.unique(X[:, idx])
            for thresh in thresholds:
                gini = self._gini_index(X[:, idx], y, thresh)
                if gini < best_gini:
                    best_gini, best_idx, best_thresh = gini, idx, thresh
        if best_idx is not None:
            left_indices = X[:, best_idx] < best_thresh
            right_indices = ~left_indices
            left_child = self._build_tree(X[left_indices], y[left_indices], depth + 1)
            right_child = self._build_tree(X[right_indices], y[right_indices], depth + 1)
            return {"feature_idx": best_idx, "threshold": best_thresh,
                    "left": left_child, "right": right_child}
        return self._create_leaf(y)

    def _gini_index(self, feature_column, y, threshold):
        left_indices = feature_column < threshold
        right_indices = ~left_indices
        left_gini = self._gini(y[left_indices])
        right_gini = self._gini(y[right_indices])
        left_ratio = np.sum(left_indices) / len(y)
        right_ratio = 1 - left_ratio
        return left_ratio * left_gini + right_ratio * right_gini

    def _gini(self, y):
        classes = np.unique(y)
        gini = 1.0
        for cls in classes:
            p = np.sum(y == cls) / len(y)
            gini -= p ** 2
        return gini

    def _create_leaf(self, y):
        return np.bincount(y).argmax()

    def _predict_single_input(self, x, tree):
        if isinstance(tree, dict):
            feature_idx = tree["feature_idx"]
            threshold = tree["threshold"]
            if x[feature_idx] < threshold:
                return self._predict_single_input(x, tree["left"])
            else:
                return self._predict_single_input(x, tree["right"])
        else:
            return tree


def dt_from_scratch(data_path):
    df = pd.read_csv(data_path)
    X, y = df.iloc[:, 1:], df.iloc[:, 0]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.7, random_state=42)
    tree = DecisionTree(max_depth=50)
    tree.fit(np.array(X_train), np.array(y_train))
    y_pred = tree.predict(np.array(X_test))
    print("\nDecision Tree Classifier from 'scratch':")
    print('Confusion Matrix:\n', confusion_matrix(y_test, y_pred))
    print('Classification Report:\n',classification_report(y_test, y_pred))
    # print('AUC Score:', round(roc_auc_score(y_test, y_pred_proba, multi_class='ovr'), 2))


def main():
    """
    Question: Should we implement Convolutional Neural Network for template matching?
    Labels: t-shirt=0, pants=1, long-sleeve=2, dress=3, other-shirt=4
    """
    # data_manipulator()
    # decision_tree_classifier('combined_features.csv')
    random_forest_classifier('combined_features.csv')
    # random_forest_classifier_kfold('circumference_features.csv')
    # evaluate_classifiers('combined_features.csv')
    # dt_from_scratch('combined_features.csv')

    # Use tensorflow for ffnn instead of pytorch as torch sucks (according to keli)


if __name__ == '__main__':
    main()
