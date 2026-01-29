# hh_pipeline

## Домашнее задание 1: Пайплайн (CSV → NPY)
 
## Цепочка ответственности
load_csv → parse_salary → clean_outliers → extract_features → save_npy

## Установка
pip install -r requirements.txt

## Использование
python app.py path/to/hh.csv

## Результат
x_data.npy (признаки: sex, age, education)
y_data.npy (зарплата рублях)

## Тест
python app.py hh.csv
ls *.npy

## Домашнее задание 2: Регрессионная модель

### Установка
pip install -r requirements.txt

### Обучение модели (сохранение весов в resources/)
python train_model.py

### Предсказание зарплат (CLI)
python app path/to/x_data.npy

### Артефакты
- resources/salary_model.joblib — сохранённая обученная модель (веса)

### Пример:
python app x_data.npy

