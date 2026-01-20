# hh_pipeline

## Цепочка ответственности
load_csv → parse_salary → clean_outliers → extract_features → save_npy

## Установка
pip install pandas numpy scikit-learn

## Использование
python app.py path/to/hh.csv

## Результат
x_data.npy (признаки: sex, age, education)
y_data.npy (зарплата рублях)

## Тест
python app.py hh.csv
ls *.npy
