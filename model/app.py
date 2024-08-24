from flask import Flask, request, render_template, jsonify
import torch
import pandas as pd
from sklearn.preprocessing import RobustScaler
import numpy as np
import torch.nn as nn

app = Flask(__name__)

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

# Загрузка модели
model = OversampleModel(n_inputs=30)  # Укажите правильное количество входных признаков

# Загрузите только веса модели
model.load_state_dict(torch.load('oversampling_model_weights.pth', map_location=torch.device('cpu')))

# Установите модель в режим оценки
model.eval()


# Инициализация скейлера
robust_scaler = RobustScaler()  # Убедитесь, что скейлер был обучен и сохранен


def preprocess_data(data):
    # Преобразуем данные в числовые значения
    data = data.apply(pd.to_numeric, errors='coerce')
    data.fillna(0, inplace=True)

    # Применяем RobustScaler для столбцов Amount и Time
    if 'Amount' in data.columns and 'Time' in data.columns:
        data['scaled_amount'] = robust_scaler.fit_transform(data[['Amount']])
        data['scaled_time'] = robust_scaler.fit_transform(data[['Time']])

        # Удаляем исходные столбцы
        data.drop(columns=['Amount', 'Time'], inplace=True)

        # Переставляем столбцы в нужном порядке
        scaled_amount = data['scaled_amount']
        scaled_time = data['scaled_time']
        data.drop(['scaled_amount', 'scaled_time'], axis=1, inplace=True)
        data.insert(0, 'scaled_amount', scaled_amount)
        data.insert(1, 'scaled_time', scaled_time)

    return data

@app.route('/', methods=['GET', 'POST'])
def index():
    prediction = None
    if request.method == 'POST':
        if 'file' in request.files:
            file = request.files['file']
            if file.filename.endswith('.csv'):
                try:
                    data = pd.read_csv(file)
                    data = preprocess_data(data)

                    # Преобразуем данные в тензоры
                    inputs = torch.tensor(data.values, dtype=torch.float32)

                    # Прогнозирование
                    with torch.no_grad():
                        outputs = model(inputs)
                        predictions = torch.argmax(outputs, dim=1).numpy()

                    prediction = predictions.tolist()
                except Exception as e:
                    return str(e), 400
            else:
                return 'Файл должен быть в формате CSV', 400
        else:
            try:
                # Получите данные от пользователя
                data = {
                    'Time': float(request.form.get('time')),
                    'V1': float(request.form.get('v1')),
                    'V2': float(request.form.get('v2')),
                    'V3': float(request.form.get('v3')),
                    'V4': float(request.form.get('v4')),
                    'V5': float(request.form.get('v5')),
                    'V6': float(request.form.get('v6')),
                    'V7': float(request.form.get('v7')),
                    'V8': float(request.form.get('v8')),
                    'V9': float(request.form.get('v9')),
                    'V10': float(request.form.get('v10')),
                    'V11': float(request.form.get('v11')),
                    'V12': float(request.form.get('v12')),
                    'V13': float(request.form.get('v13')),
                    'V14': float(request.form.get('v14')),
                    'V15': float(request.form.get('v15')),
                    'V16': float(request.form.get('v16')),
                    'V17': float(request.form.get('v17')),
                    'V18': float(request.form.get('v18')),
                    'V19': float(request.form.get('v19')),
                    'V20': float(request.form.get('v20')),
                    'V21': float(request.form.get('v21')),
                    'V22': float(request.form.get('v22')),
                    'V23': float(request.form.get('v23')),
                    'V24': float(request.form.get('v24')),
                    'V25': float(request.form.get('v25')),
                    'V26': float(request.form.get('v26')),
                    'V27': float(request.form.get('v27')),
                    'V28': float(request.form.get('v28')),
                    'Amount': float(request.form.get('amount'))
                }

                # Создайте DataFrame из введенных данных
                df = pd.DataFrame([data])

                # Предобработка данных
                df = preprocess_data(df)

                # Преобразуем данные в тензоры
                inputs = torch.tensor(df.values, dtype=torch.float32)

                # Прогнозирование
                with torch.no_grad():
                    outputs = model(inputs)
                    prediction = torch.argmax(outputs, dim=1).numpy()[0]

            except Exception as e:
                return str(e), 400

    return render_template('index.html', prediction=prediction)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)

