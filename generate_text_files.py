import pandas as pd

train_parquet = pd.read_parquet('Datasets/opus-100_en-pt/train-00000-of-00001.parquet', engine='pyarrow')
val_parquet = pd.read_parquet('Datasets/opus-100_en-pt/validation-00000-of-00001.parquet', engine='pyarrow')
test_parquet = pd.read_parquet('Datasets/opus-100_en-pt/test-00000-of-00001.parquet', engine='pyarrow')

train_en = []
train_pt = []
val_en = []
val_pt = []
test_en = []
test_pt = []

for idx in range(len(train_parquet)):
    train_en.append(train_parquet['translation'][idx]['en'])
    train_pt.append(train_parquet['translation'][idx]['pt'])

for idx in range(len(val_parquet)):
    val_en.append(val_parquet['translation'][idx]['en'])
    val_pt.append(val_parquet['translation'][idx]['pt'])

for idx in range(len(test_parquet)):
    test_en.append(test_parquet['translation'][idx]['en'])
    test_pt.append(test_parquet['translation'][idx]['pt'])

with open('Datasets/opus-100_en-pt/text_files/train.en', 'w', encoding='utf-8') as f:
    f.writelines([f"{x}\n" for x in train_en])

with open('Datasets/opus-100_en-pt/text_files/train.pt', 'w', encoding='utf-8') as f:
    f.writelines([f"{x}\n" for x in train_pt])

with open('Datasets/opus-100_en-pt/text_files/val.en', 'w', encoding='utf-8') as f:
    f.writelines([f"{x}\n" for x in val_en])

with open('Datasets/opus-100_en-pt/text_files/val.pt', 'w', encoding='utf-8') as f:
    f.writelines([f"{x}\n" for x in val_pt])

with open('Datasets/opus-100_en-pt/text_files/test.en', 'w', encoding='utf-8') as f:
    f.writelines([f"{x}\n" for x in test_en])

with open('Datasets/opus-100_en-pt/text_files/test.pt', 'w', encoding='utf-8') as f:
    f.writelines([f"{x}\n" for x in test_pt])