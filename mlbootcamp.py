# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import GradientBoostingRegressor

import random
random.seed(666)


# рассчитаем ошибку
def mean_absolute_percentage_error(y_true, y_pred):
    ind = y_true > 1
    return np.mean(np.abs((y_true[ind] - y_pred[ind]) / y_true[ind]))


def loss_func(y_true, y_pred):
    return mean_absolute_percentage_error(y_true, y_pred)


print('Load data ...')
all_train = pd.read_csv('x_train.csv')
all_res = pd.read_csv('y_train.csv')
all_test = pd.read_csv('x_test.csv')

all_train.ix[:, 'TARGET'] = all_res['time']

cols = [
    'n',
    'Sequential_read_128B_by128',
    'k',
    'Random_write_3MB_by128',
    'cpuCount',
    'Sequential_write_32kB_by128',
    'Random_read_9MB_by128',
    'm',
    'SeqRead_20kB_by256',
    'cpuCores',
    'Sequential_read_48MB_by128',
    'Random_read_4MB_by32',
    'Random_write_32MB_by128',
    'Random_read_2MB_by32',
    'SeqCopy10MB_by128',
    'BMI',
    'm_div_n',
    'magic',
    'cpuExtra1',
    'cpuExtra2',
    'Random_write_bypassing_cache_6kB_by128',
    'Sequential_read_192kB_by32',
]

print('Preprocess ...')
# обработаем категории
lc_os = LabelEncoder()
lc_cpu = LabelEncoder()

lc_os.fit(all_train['os'].values)
lc_cpu.fit(all_train['cpuFull'].values)

all_train.ix[:, 'c1'] = all_train['TARGET'] / (all_train['m'] * all_train['n'] * all_train['k'])
all_test.ix[:, 'c1'] = 0
# посчитаем медианы
all_train_median = all_train[['c1', 'os', 'cpuFull']].groupby(['os', 'cpuFull'], as_index=False).median()


def preprocess_data(df):
    df.ix[:, 'cpuExtra1'] = 0
    df.ix[df['cpuFull'] == 'Intel(R) Core(TM) i3-2310M CPU @ 2.10GHz', 'cpuExtra1'] = 1

    df.ix[:, 'cpuExtra2'] = 0
    df.ix[df['cpuFull'] == 'Intel(R) Atom(TM) CPU N550   @ 1.50GHz', 'cpuExtra2'] = 1

    # это прогнозное время
    df = pd.merge(df, all_train_median, on=['os', 'cpuFull'], how='left', suffixes=('', '_med'))
    df.ix[:, 'test_mdeian'] = df['c1_med'] * df['m'] * df['n'] * df['k']

    df.ix[:, 'os'] = lc_os.transform(df['os'].values)
    df.ix[:, 'cpuFull'] = lc_cpu.transform(df['cpuFull'].values)

    # это новые удачные признаки
    df.ix[:, 'm_div_n'] = df['m'] / df['n']
    df.ix[:, 'magic'] = df['k'] * df['m'] * df['n'] / (df['cpuCount'] * df['cpuCount'])

    df['mnk'] = df['m'] * df['n'] * df['k']
    return df


all_train = preprocess_data(all_train)
all_test = preprocess_data(all_test)

X_train = all_train[cols].values
y_train = all_train['TARGET'].values
y_train_c = all_train['n'].values * all_train['m'].values * all_train['k'].values
y_train_c1 = all_train['test_mdeian'].values
train = all_train


# это три основные модели
params_est = {'n_estimators': 370,
              'subsample': 0.961,
              'learning_rate': 0.076,
              'min_samples_split': 18.0,
              'max_depth': 6,
              'min_samples_leaf': 8.0,
              'random_state': 1,
              'loss': 'lad', }

bst1 = GradientBoostingRegressor(**params_est)
bst1.fit(X_train, y_train / y_train_c1)

params_est = {'n_estimators': 680,
              'subsample': 0.902,
              'learning_rate': 0.076,
              'min_samples_split': 14.0,
              'alpha': 0.29,
              'max_depth': 9,
              'min_samples_leaf': 5.0,
              'loss': 'quantile',
              'random_state': 1}

bst2 = GradientBoostingRegressor(**params_est)
bst2.fit(X_train, y_train / y_train_c1)

params_est = {'n_estimators': 430,
              'subsample': 0.978,
              'learning_rate': 0.086,
              'min_samples_split': 19.0,
              'max_depth': 6,
              'min_samples_leaf': 10.0,
              'loss': 'lad',
              'random_state': 1}

bst3 = GradientBoostingRegressor(**params_est)
bst3.fit(X_train, y_train / y_train_c1)

# это веса для отдельной модели
all_train.ix[:, 'w'] = 1
all_train.ix[all_train['os'] == 15, 'w'] = 4

# это отдельная модель для сложной os = 15
params_est = {'n_estimators': 480,
              'subsample': 0.881,
              'learning_rate': 0.197,
              'min_samples_split': 3.0,
              'max_depth': 7,
              'min_samples_leaf': 2.0,
              'loss': 'lad',
              'random_state': 1}

bst4 = GradientBoostingRegressor(**params_est)
bst4.fit(X_train, np.log(y_train / y_train_c), sample_weight=train['w'])

X_test = all_test[cols].values
y_pred0 = bst1.predict(X_test) * all_test['test_mdeian']
y_pred1 = bst2.predict(X_test) * all_test['test_mdeian']
y_pred2 = bst3.predict(X_test) * all_test['test_mdeian']
y_pred3 = np.exp(bst4.predict(X_test)) * all_test['k'].values * all_test['m'].values * all_test['n'].values

y_pred0 = np.asarray([max(i, 1.000083) for i in y_pred0])
y_pred1 = np.asarray([max(i, 1.000083) for i in y_pred1])
y_pred2 = np.asarray([max(i, 1.000083) for i in y_pred2])
y_pred3 = np.asarray([max(i, 1.000083) for i in y_pred3])

# совмещаем результаты
all_test.ix[:, 'os15'] = y_pred3
all_test.ix[:, 'TARGET'] = (y_pred1 + y_pred2 + y_pred0) / 3.0
all_test.ix[all_test['os'] == 15, 'TARGET'] = all_test['os15'][all_test['os'] == 15]

print('Write ...')
sub_df = pd.DataFrame(data=all_test[['TARGET']], columns=['TARGET'])
sub_df.to_csv('submit.csv', index=False, header=False)
exit()


