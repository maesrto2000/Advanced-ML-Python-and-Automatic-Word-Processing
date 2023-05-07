import cv2
from imutils import paths
from sklearn.model_selection import train_test_split
from sklearn.svm import LinearSVC
import numpy as np

# Определение функции для извлечения гистограммы
def extract_histogram(image, bins=(8, 8, 8)):
    hist = cv2.calcHist([image], [0, 1, 2], None, bins, [0, 256, 0, 256, 0, 256])
    cv2.normalize(hist, hist)
    return hist.flatten()

# Создание массивов гистограмм и меток
data = []
labels = []
for imagePath in sorted(list(paths.list_images("/content/drive/MyDrive/train"))):
    image = cv2.imread(imagePath)
    hist = extract_histogram(image)
    label = 0 if "cat" in imagePath else 1
    data.append(hist)
    labels.append(label)

# Разделение выборки на тренировочную и тестовую части
trainData, testData, trainLabels, testLabels = train_test_split(
    np.array(data), np.array(labels), test_size=0.25, random_state=3)

# Обучение классификатора LinearSVC с C = 1.05
model = LinearSVC(C=1.05, random_state=3)
model.fit(trainData, trainLabels)

# Оценка точности классификатора на тестовой выборке
accuracy = model.score(testData, testLabels)
print("Accuracy: {:.2f}%".format(accuracy * 100))

# Получение коэффициентов гиперплоскости
coef = model.coef_
intercept = model.intercept_
#print("Coefficient of hyperplane:", coef)
print("Theta 131:", coef[0, 130])
print("Theta 131:", coef[0, 291])
print("Theta 131:", coef[0, 37])
