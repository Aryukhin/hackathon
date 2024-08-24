import warnings
import pandas as pd
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from imblearn.over_sampling import SMOTE
import matplotlib.pyplot as plt

# Отключаем предупреждения
warnings.filterwarnings("ignore")

# Read
df = pd.read_csv('creditcard.csv')

amount_val = df['Amount'].values
time_val = df['Time'].values

# Инициализируем скейлеры
standard_scaler = StandardScaler()
robust_scaler = RobustScaler()

# Применяем RobustScaler для масштабирования столбцов Amount и Time
df['scaled_amount'] = robust_scaler.fit_transform(df[['Amount']])
df['scaled_time'] = robust_scaler.fit_transform(df[['Time']])

# Удаляем исходные столбцы Amount и Time
df.drop(columns=['Time', 'Amount'], inplace=True)

scaled_amount = df['scaled_amount']
scaled_time = df['scaled_time']

df.drop(['scaled_amount', 'scaled_time'], axis=1, inplace=True)
df.insert(0, 'scaled_amount', scaled_amount)
df.insert(1, 'scaled_time', scaled_time)


print(df.head())

print('Легитимные транзакции', round(df['Class'].value_counts()[0]/len(df) * 100,2), '% от общего количества транзакций')
print('Мошеннические транзакции', round(df['Class'].value_counts()[1]/len(df) * 100,2), '% от общего количества транзакицй')

X = df.drop('Class', axis=1)
y = df['Class']


# Разделение данных
sss = StratifiedKFold(n_splits=5, random_state=None, shuffle=False)

for train_index, test_index in sss.split(X, y):
    print("Train:", train_index, "Test:", test_index)
    original_Xtrain, original_Xtest = X.iloc[train_index], X.iloc[test_index]
    original_ytrain, original_ytest = y.iloc[train_index], y.iloc[test_index]



original_Xtrain = original_Xtrain.values
original_Xtest = original_Xtest.values
original_ytrain = original_ytrain.values
original_ytest = original_ytest.values


train_unique_label, train_counts_label = np.unique(original_ytrain, return_counts=True)
test_unique_label, test_counts_label = np.unique(original_ytest, return_counts=True)
print('-' * 100)

print('Label Distributions: \n')
print(train_counts_label/ len(original_ytrain))
print(test_counts_label/ len(original_ytest))


df = df.sample(frac=1)

fraud_df = df.loc[df['Class'] == 1]
non_fraud_df = df.loc[df['Class'] == 0][:492]

normal_distributed_df = pd.concat([fraud_df, non_fraud_df])

new_df = normal_distributed_df.sample(frac=1, random_state=42)

new_df.head()



# # -----> V14 Removing Outliers (Highest Negative Correlated with Labels)
v14_fraud = new_df['V14'].loc[new_df['Class'] == 1].values
q25, q75 = np.percentile(v14_fraud, 25), np.percentile(v14_fraud, 75)
print('Quartile 25: {} | Quartile 75: {}'.format(q25, q75))
v14_iqr = q75 - q25
print('iqr: {}'.format(v14_iqr))

v14_cut_off = v14_iqr * 1.5
v14_lower, v14_upper = q25 - v14_cut_off, q75 + v14_cut_off
print('Cut Off: {}'.format(v14_cut_off))
print('V14 Lower: {}'.format(v14_lower))
print('V14 Upper: {}'.format(v14_upper))

outliers = [x for x in v14_fraud if x < v14_lower or x > v14_upper]
print('Feature V14 Outliers for Fraud Cases: {}'.format(len(outliers)))
print('V10 outliers:{}'.format(outliers))

new_df = new_df.drop(new_df[(new_df['V14'] > v14_upper) | (new_df['V14'] < v14_lower)].index)
print('----' * 44)

# -----> V12 removing outliers from fraud transactions
v12_fraud = new_df['V12'].loc[new_df['Class'] == 1].values
q25, q75 = np.percentile(v12_fraud, 25), np.percentile(v12_fraud, 75)
v12_iqr = q75 - q25

v12_cut_off = v12_iqr * 1.5
v12_lower, v12_upper = q25 - v12_cut_off, q75 + v12_cut_off
print('V12 Lower: {}'.format(v12_lower))
print('V12 Upper: {}'.format(v12_upper))
outliers = [x for x in v12_fraud if x < v12_lower or x > v12_upper]
print('V12 outliers: {}'.format(outliers))
print('Feature V12 Outliers for Fraud Cases: {}'.format(len(outliers)))
new_df = new_df.drop(new_df[(new_df['V12'] > v12_upper) | (new_df['V12'] < v12_lower)].index)
print('Number of Instances after outliers removal: {}'.format(len(new_df)))
print('----' * 44)


# Removing outliers V10 Feature
v10_fraud = new_df['V10'].loc[new_df['Class'] == 1].values
q25, q75 = np.percentile(v10_fraud, 25), np.percentile(v10_fraud, 75)
v10_iqr = q75 - q25

v10_cut_off = v10_iqr * 1.5
v10_lower, v10_upper = q25 - v10_cut_off, q75 + v10_cut_off
print('V10 Lower: {}'.format(v10_lower))
print('V10 Upper: {}'.format(v10_upper))
outliers = [x for x in v10_fraud if x < v10_lower or x > v10_upper]
print('V10 outliers: {}'.format(outliers))
print('Feature V10 Outliers for Fraud Cases: {}'.format(len(outliers)))
new_df = new_df.drop(new_df[(new_df['V10'] > v10_upper) | (new_df['V10'] < v10_lower)].index)
print('Number of Instances after outliers removal: {}'.format(len(new_df)))


X = new_df.drop('Class', axis=1)
y = new_df['Class']


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


X_train = X_train.values
X_test = X_test.values
y_train = y_train.values
y_test = y_test.values


# Создание объекта SMOTE
sm = SMOTE(sampling_strategy='minority', random_state=42)

# Применение SMOTE для создания сбалансированного набора данных
Xsm_train, ysm_train = sm.fit_resample(original_Xtrain, original_ytrain)


# Определяем количество входных признаков
n_inputs = Xsm_train.shape[1]

# Определяем модель в PyTorch
class OversampleModel(nn.Module):
    def __init__(self, n_inputs):
        super(OversampleModel, self).__init__()
        self.fc1 = nn.Linear(n_inputs, n_inputs)  # Первый полносвязный слой
        self.relu1 = nn.ReLU()  # Активация ReLU
        self.fc2 = nn.Linear(n_inputs, 32)  # Второй полносвязный слой
        self.relu2 = nn.ReLU()  # Активация ReLU
        self.fc3 = nn.Linear(32, 2)  # Третий полносвязный слой (выходной)

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu1(x)
        x = self.fc2(x)
        x = self.relu2(x)
        x = self.fc3(x)  # Не применяем Softmax здесь
        return x

# Создаем экземпляр модели
oversampling_model = OversampleModel(n_inputs)

# Преобразуем данные в тензоры
Xsm_train_tensor = torch.tensor(Xsm_train, dtype=torch.float32)
ysm_train_tensor = torch.tensor(ysm_train, dtype=torch.long)

# Создаем DataLoader для обучения
train_data = TensorDataset(Xsm_train_tensor, ysm_train_tensor)
train_loader = DataLoader(train_data, batch_size=300, shuffle=True)

# Создаем модель
n_inputs = Xsm_train.shape[1]  # Количество входных признаков

# Определяем функцию потерь и оптимизатор
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(oversampling_model.parameters(), lr=0.001)

# Обучение модели
epochs = 20
for epoch in range(epochs):
    oversampling_model.train()  # Устанавливаем модель в режим обучения
    running_loss = 0.0
    correct = 0
    total = 0

    for inputs, labels in train_loader:
        optimizer.zero_grad()  # Обнуляем градиенты

        # Прямой проход
        outputs = oversampling_model(inputs)
        loss = criterion(outputs, labels)

        # Обратный проход и оптимизация
        loss.backward()
        optimizer.step()

        # Статистика
        running_loss += loss.item() * inputs.size(0)  # Накопление потерь
        _, predicted = torch.max(outputs, 1)  # Получаем предсказанные метки
        total += labels.size(0)  # Общее количество меток
        correct += (predicted == labels).sum().item()  # Количество правильных предсказаний

    # Среднее значение потерь и точности за эпоху
    epoch_loss = running_loss / len(train_loader.dataset)
    epoch_accuracy = correct / total
    print(f'Epoch [{epoch+1}/{epochs}], Loss: {epoch_loss:.4f}, Accuracy: {epoch_accuracy:.4f}')


# Переводим модель в режим оценки
oversampling_model.eval()

# Преобразуем тестовые данные в тензор
original_Xtest_tensor = torch.tensor(original_Xtest, dtype=torch.float32)

# Прогнозирование
with torch.no_grad():
    oversample_predictions = oversampling_model(original_Xtest_tensor)
    oversample_fraud_predictions = torch.argmax(oversample_predictions, dim=1).numpy()


# Построение матриц ошибок
oversample_smote = confusion_matrix(original_ytest, oversample_fraud_predictions)
actual_cm = confusion_matrix(original_ytest, original_ytest)
labels = ['No Fraud', 'Fraud']

fig, axs = plt.subplots(2, 2, figsize=(16, 8))

# Построение матрицы ошибок после SMOTE
ConfusionMatrixDisplay(confusion_matrix=oversample_smote, display_labels=labels).plot(ax=axs[0, 0], cmap=plt.cm.Oranges)
axs[0, 0].set_title("OverSample (SMOTE) \n Confusion Matrix")

# Построение матрицы ошибок с 100% точностью
ConfusionMatrixDisplay(confusion_matrix=actual_cm, display_labels=labels).plot(ax=axs[0, 1], cmap=plt.cm.Greens)
axs[0, 1].set_title("Confusion Matrix \n (with 100% accuracy)")

torch.save(oversampling_model, 'full_oversampling_model.pth')
torch.save(oversampling_model.state_dict(), 'oversampling_model_weights.pth')

plt.show()

