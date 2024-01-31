import os
import numpy as np
from keras.applications import MobileNet
from keras.applications.mobilenet import preprocess_input
from keras.utils import load_img, img_to_array
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.svm import SVC
import pickle
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from PIL import Image

# model_MobileNet = VGG16(weights='imagenet', include_top=False) #97
model_MobileNet = MobileNet(weights='imagenet', include_top=False)  # 100


def extract_features(images):
    features = []
    for img_path in images:
        img = load_img(img_path, target_size=(224, 224))
        x = img_to_array(img)
        x = np.expand_dims(x, axis=0)
        x = preprocess_input(x)
        features.append(model_MobileNet.predict(x).flatten())
    return np.array(features)


def train_validation_data():
    num_classes = 20
    train_images = []
    train_labels = []
    validation_images = []
    validation_labels = []

    for i in range(1, num_classes + 1):
        train_dir = "Data/Product Classification_83/" + str(i) + "/Train"
        validation_dir = "Data/Product Classification_83/" + str(i) + "/Validation"
        class_name = str(i)

        for filename in os.listdir(train_dir):
            if not filename.endswith(('.bmp', '.jpeg', '.jpg', '.png', '.tif', '.tiff')):
                continue
            img_path1 = os.path.join(train_dir, filename)
            train_images.append(img_path1)
            train_labels.append(class_name)

        for filename in os.listdir(validation_dir):
            if not filename.endswith(('.bmp', '.jpeg', '.jpg', '.png', '.tif', '.tiff')):
                continue
            img_path2 = os.path.join(validation_dir, filename)
            validation_images.append(img_path2)
            validation_labels.append(class_name)

    train_features = extract_features(train_images)
    pickle.dump(train_features, open('train_features.pkl', 'wb'))

    validation_features = extract_features(validation_images)

    lr = LogisticRegression(max_iter=2000)
    lr.fit(train_features, train_labels)
    train_predictions = lr.predict(train_features)
    train_accuracy = accuracy_score(train_labels, train_predictions)
    validation_predictions = lr.predict(validation_features)
    accuracy = accuracy_score(validation_labels, validation_predictions)
    print(f"\n\033[92mAccuracy Logistic Regression on train data: {train_accuracy} \033[0m")
    print(f"\033[92mAccuracy Logistic Regression on validation data: {accuracy} \033[0m \n")

    svm = SVC(kernel='linear', C=0.1, probability=True)
    svm.fit(train_features, train_labels)
    train_predictions2 = svm.predict(train_features)
    train_accuracy2 = accuracy_score(train_labels, train_predictions2)
    svm_predictions = svm.predict(validation_features)
    svm_accuracy = accuracy_score(validation_labels, svm_predictions)
    print(f"\033[94mAccuracy SVM on train data: {train_accuracy2} \033[0m")
    print(f"\033[94mAccuracy SVM on validation data: {svm_accuracy} \033[0m\n")

    pickle.dump(lr, open('logistic_regression_model.pkl', 'wb'))
    pickle.dump(svm, open('svm_model.pkl', 'wb'))


# train_validation_data()

logistic_regression = pickle.load(open('logistic_regression_model.pkl', 'rb'))
svm_model = pickle.load(open('svm_model.pkl', 'rb'))

test_images = []
test_labels = []


def visualization(images, true_labels, predictions, model_name):
    num_images = len(images)
    plt.figure(figsize=(20, 4))

    for i in range(num_images):
        plt.subplot(6, 7, i + 1)
        img = mpimg.imread(images[i])
        img_pil = Image.fromarray((img * 255).astype('uint8'))
        img = img_pil.resize((150, 130))
        plt.imshow(img)
        plt.axis('off')
        plt.text(0.5, -0.2, f"T: {true_labels[i]}, P: {predictions[i]}",
                 ha='center', fontsize=8, color='black', transform=plt.gca().transAxes)
    plt.suptitle(f"'{model_name} Predictions'\n(T->True Label, P->predictions)\n", fontsize=14)
    plt.show()


def test_data():
    test_dir = "Data/Test Samples Classification"
    for class_name in os.listdir(test_dir):
        class_path = os.path.join(test_dir, class_name)
        for filename in os.listdir(class_path):
            if not filename.endswith(('.bmp', '.jpeg', '.jpg', '.png', '.tif', '.tiff')):
                continue
            img_path = os.path.join(class_path, filename)
            test_images.append(img_path)
            test_labels.append(class_name)

    test_features = extract_features(test_images)
    logistic_regression_predictions = logistic_regression.predict(test_features)
    accuracy1 = accuracy_score(test_labels, logistic_regression_predictions)
    svm_predictions_test = svm_model.predict(test_features)
    accuracy2 = accuracy_score(test_labels, svm_predictions_test)
    visualization(test_images, test_labels, logistic_regression_predictions, "Logistic Regression")
    visualization(test_images, test_labels, svm_predictions_test, "SVM")
    print(f"\033[96mAccuracy Logistic Regression on test data: {accuracy1} \033[0m")
    print(f"\033[96mAccuracy SVM on test data: {accuracy2} \033[0m")


test_data()
