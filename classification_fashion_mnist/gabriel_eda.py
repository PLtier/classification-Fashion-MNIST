import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from skimage import measure
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
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
    df = pd.read_csv(data_path)
    X, y = df.iloc[:, 1:], df.iloc[:, 0]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    dt = DecisionTreeClassifier()
    dt.fit(X_train, y_train)
    y_pred = dt.predict(X_test)
    print('Confusion Matrix:\n', confusion_matrix(y_test, y_pred))
    print('Classification Report:\n',classification_report(y_test, y_pred))


def random_forest_classifier(data_path):
    """
    Random Forest Classifier for classification of images
    Standard stuff, nothing special, also prints some metrics
    """
    df = pd.read_csv(data_path)
    X, y = df.iloc[:, 1:], df.iloc[:, 0]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    rf = RandomForestClassifier(n_estimators=100, random_state=42)
    rf.fit(X_train, y_train)
    y_pred = rf.predict(X_test)
    print('Confusion Matrix:\n', confusion_matrix(y_test, y_pred))
    print('Classification Report:\n',classification_report(y_test, y_pred))


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


def main():
    """
    Question: Should we implement Convolutional Neural Network for template matching?
    Labels: t-shirt=0, pants=1, long-sleeve=2, dress=3, other-shirt=4
    """

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

    pca_data = pca_on_images(original_images, labels, 71)

    df_circumference = pd.DataFrame(circumference_data, columns=circumference_column_names)
    df_circumference.to_csv('../data/processed/circumference_features.csv', index=False)

    df_width = pd.DataFrame(width_data, columns=width_column_names)
    df_width.to_csv('../data/processed/width_features.csv', index=False)

    df_height = pd.DataFrame(height_data, columns=height_column_names)
    df_height.to_csv('../data/processed/height_features.csv', index=False)

    df_pca = pd.DataFrame(pca_data)
    df_pca.to_csv('../data/processed/pca_features.csv', index=False)

    df_combined = pd.concat([df_circumference, df_width.iloc[:, 1:], df_height.iloc[:, 1:], df_pca.iloc[:, 1:]], axis=1)
    df_combined.to_csv('../data/processed/combined_features.csv', index=False)

    decision_tree_classifier('../data/processed/combined_features.csv')
    random_forest_classifier('../data/processed/combined_features.csv')
    # visualize_data(padded_images, binary_images)


if __name__ == '__main__':
    main()
