import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
from itertools import product
from tqdm import tqdm
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import VotingClassifier
from sklearn.externals import joblib
pd.options.mode.chained_assignment = None


categories = pd.read_csv("item_categories.csv")
items = pd.read_csv("items.csv")
shops = pd.read_csv("shops.csv")
train = pd.read_csv("sales_train_v2.csv")
sample_submission = pd.read_csv("sample_submission.csv")
test = pd.read_csv("test.csv")
testID = test['ID']


#информация о датасетах
# print("______Train______")
# print(train.info())
# print("___Categories____")
# print(categories.info())
# print("______Items______")
# print(items.info())
# print("______Shops______")
# print(shops.info())

#размер тестовой и тренеровочной выборки
# print(train.shape)
# print(test.shape)

#содержание столбцов таблиц
# print(train.head())
# print(test.head())

#проверка NaN и Null значений в выборках
# print(train.isnull().sum())
# print(test.isnull().sum())

#удаление дубликатов строк
print("Train size before dropping duplicates:",  train.shape)
train.drop_duplicates(subset=["date", "date_block_num", "shop_id", "item_id", "item_cnt_day"],
                      keep="first",
                      inplace=True)
print("Train size after dropping duplicates:", train.shape)

#проверка остальных csv файлов на NaN и null на значения
# print(shops.isnull().sum())
# print(items.isnull().sum())
# print(categories.isnull().sum())

#Searcing unreasonable values
# print("____price_____")
# print("price_min: ", train.item_price.min())
# print("price_max: ", train.item_price.max())
# print("price_mean: ", train.item_price.mean())
# print("price_median: ", train.item_price.median())
# print("____cnt_____")
# print("cnt_min: ",train.item_cnt_day.min())
# print("cnt_min: ",train.item_cnt_day.max())
# print("cnt_min: ",train.item_cnt_day.mean())
# print("cnt_min: ",train.item_cnt_day.median())

# dif1 = train.item_price.value_counts().sort_index(ascending=False)
# print(dif1)
# dif2 = train.item_cnt_day.value_counts().sort_index(ascending=False)
# print(dif2)
# ds = dif1.to_frame().reset_index()
# ds.columns.values[1] = "number"
# ds.columns.values[0] = "item_price"
# sns.set(style="ticks")
# sns.lmplot("item_price", "number", ds)
#
# plt.show()

print("Train size before deleting unreasonable values:", train.shape)
train = train[(train.item_price > 0) & (train.item_price < 300000) & (train.item_cnt_day > 0)]
print("Train size after deleting unreasonable values:", train.shape)

# формат даты
print('Data format changing start')
train['date'] = pd.to_datetime(train['date'], format='%d.%m.%Y')
print('Data format changing ended')


print("Agregating data")
# Создаем наборы уникальных занчений товар-магазин для каждого месяца
month_table = []
for block_num in train["date_block_num"].unique():
    month_shops = train[train['date_block_num']==block_num]['shop_id'].unique()
    month_items = train[train['date_block_num']==block_num]['item_id'].unique()
    month_table.append(np.array(list(product(*[month_shops, month_items, [block_num]])),dtype='int32'))



# Создаем из месячных таблиц товар-магазин Pandas dataframe
indexes = ['shop_id', 'item_id', 'date_block_num']
month_table = pd.DataFrame(np.vstack(month_table), columns = indexes, dtype=np.int32)


train['item_cnt_day'] = train['item_cnt_day'].clip(0, 20)
# Расчет объема продажи товаров за месяц
month_cnt = train.groupby(indexes)['item_cnt_day'].agg(['sum']).reset_index().rename(columns = {'sum': 'item_cnt_month'})
month_cnt['item_cnt_month'] = month_cnt['item_cnt_month'].astype(int)

# добавление данных в месячную таблицу
new_train = pd.merge(month_table,month_cnt,how='left',on=indexes).fillna(0)
new_train['item_cnt_month'] = new_train['item_cnt_month'].astype(int)

# сортировка данных
new_train.sort_values(['date_block_num', 'shop_id', 'item_id'], inplace=True)


# # проверка правильности
# print(train['item_cnt_day'].sum())
# print(new_train['item_cnt_month'].sum())
# print(month_cnt['item_cnt_month'].sum())

print("Aggreating ended")


# добавление item_category_id
new_train = new_train.merge(items[['item_id', 'item_category_id']], on=['item_id'], how = 'left')
test = test.merge(items[['item_id', 'item_category_id']], on=['item_id'], how = 'left')

new_train = new_train.drop_duplicates(subset=["shop_id", "item_id", "date_block_num"] )

# print('_____Train______')
# print(new_train)
#
#
# print("_____Test_______")
# print(test)


# print(new_train.reset_index().groupby(['item_id', 'date_block_num', 'shop_id']).mean())




def TrainTestUnion(new_train, test):
    # объединение тестовой и тренировочной выборок
    test["date_block_num"] = 34
    train_test = pd.concat([new_train, test], axis=0)
    train_test = train_test.drop(columns=['ID'])
    # print(train_test)
    train_test = train_test.fillna(0)
    return train_test

index = ['shop_id', 'item_id', 'item_category_id', 'date_block_num']


# Добавление даты
def DataFeatures(train_test):
    dates_train = train[['date', 'date_block_num']].drop_duplicates()
    dates_test = dates_train[dates_train['date_block_num'] == 34-12]
    dates_test['date_block_num'] = 34
    dates_test['date'] = dates_test['date'] + pd.DateOffset(years=1)
    dates_all = pd.concat([dates_train, dates_test])

    dates_all['dow'] = dates_all['date'].dt.dayofweek
    dates_all['year'] = dates_all['date'].dt.year
    dates_all['month'] = dates_all['date'].dt.month
    dates_all = pd.get_dummies(dates_all, columns=['dow'])
    dow_col = ['dow_' + str(x) for x in range(7)]
    date_features = dates_all.groupby(['year', 'month', 'date_block_num'])[dow_col].agg('sum').reset_index()
    date_features['days_of_month'] = date_features[dow_col].sum(axis=1)
    date_features['year'] = date_features['year'] - 2013

    date_features = date_features[['month', 'year', 'days_of_month', 'date_block_num']]

    train_test = train_test.merge(date_features, on='date_block_num', how='left')
    # print(train_test)
    date_columns = date_features.columns.difference(set(index))
    # print(date_columns)
    return train_test, date_columns

# Scale features
def ScaleFeatures(train_test, date_columns, index):
    train = train_test[train_test['date_block_num']!=train_test['date_block_num'].max()]
    test = train_test[train_test['date_block_num']==train_test['date_block_num'].max()]
    scalar = StandardScaler()
    col = ['date_block_num']
    feature_columns = list(set(index + list(date_columns)).difference(col))
    train[feature_columns] = scalar.fit_transform(train[feature_columns])
    test[feature_columns] = scalar.transform(test[feature_columns])
    train_test = pd.concat([train, test], axis=0)
    # print(train_test)
    return train_test


def ModelBuilder(train_test):
    kfold = StratifiedKFold(n_splits=5, shuffle=True)
    classifiers = []
    random_state = 0
    classifiers.append(SVC(random_state=random_state))
    classifiers.append(DecisionTreeClassifier(random_state=random_state))
    classifiers.append(AdaBoostClassifier(DecisionTreeClassifier(random_state=random_state), random_state=random_state,
                                           learning_rate=0.1))
    classifiers.append(RandomForestClassifier(random_state=random_state))
    classifiers.append(ExtraTreesClassifier(random_state=random_state))
    classifiers.append(GradientBoostingClassifier(random_state=random_state))
    classifiers.append(MLPClassifier(random_state=random_state))
    classifiers.append(KNeighborsClassifier())
    classifiers.append(LogisticRegression(random_state=random_state))
    classifiers.append(LinearDiscriminantAnalysis())
    cv_results = []
    train_test = train_test[train_test['date_block_num']!=train_test['date_block_num'].max()]
    X_train = train_test[['item_category_id', 'item_id', 'shop_id', 'month', 'year', 'days_of_month']][:100000]
    # print('__________X_train________')
    # print(X_train)
    Y_train = (train_test['item_cnt_month'][:100000])
    # print('__________Y_train________')
    # print(Y_train)
    for classifier in tqdm(classifiers):
        cv_results.append(cross_val_score(classifier, X_train, y=Y_train, scoring="accuracy", cv=kfold, n_jobs=1))
    cv_means = []
    cv_std = []
    for cv_result in cv_results:
        cv_means.append(cv_result.mean())
        cv_std.append(cv_result.std())

    cv_res = pd.DataFrame(
        {"CrossValMeans": cv_means, "CrossValerrors": cv_std, "Algorithm": ["SVC", "DecisionTree", "AdaBoost",
                                                                            "RandomForest", "ExtraTrees",
                                                                            "GradientBoosting",
                                                                            "MultipleLayerPerceptron", "KNeighboors",
                                                                            "LogisticRegression",
                                                                            "LinearDiscriminantAnalysis"]})

    print(cv_res)
    g = sns.barplot("CrossValMeans", "Algorithm", data=cv_res, palette="Set3", orient="h", **{'xerr': cv_std})
    g.set_xlabel("Mean Accuracy")
    g = g.set_title("Cross validation scores")
    plt.show()

train_test = TrainTestUnion(new_train, test)
train_test, date_columns = DataFeatures(train_test)
train_test = ScaleFeatures(train_test, date_columns, index)
# print(train_test)
# DelBadNumbers(train_test)


# print(train_test.keys())
# ModelBuilder(train_test)
print("Разбиение данных")
train = train_test[(train_test['date_block_num']!=train_test['date_block_num'].max())
                    & ((((((((((train_test["date_block_num"] == 6) | (train_test["date_block_num"] == 7)) |
                    (train_test["date_block_num"] == 8)) | (train_test["date_block_num"] == 9)) |
                    (train_test["date_block_num"] == 10)) | (train_test["date_block_num"] == 18)) |
                    (train_test["date_block_num"] == 19)) | (train_test["date_block_num"] == 20)) |
                    (train_test["date_block_num"] == 21)) | (train_test["date_block_num"] == 22))]
X_train = train[['item_category_id', 'item_id', 'shop_id', 'month', 'year', 'days_of_month']]
Y_train = train['item_cnt_month']
test = train_test[train_test['date_block_num']==train_test['date_block_num'].max()]
test = test[['item_category_id', 'item_id', 'shop_id', 'month', 'year', 'days_of_month']]
print("Разбиение окончено")

random_state = 0
classifiers = [('svc', SVC(random_state=random_state)), ('gbc', GradientBoostingClassifier(random_state=random_state)),
               ('mlp', MLPClassifier(random_state=random_state)),
               ('LogisticRegression', LogisticRegression(random_state=random_state))]
VotClassifier = VotingClassifier(classifiers)
mlp = MLPClassifier(random_state=random_state)
print("Начало обучения")
mlp = mlp.fit(X_train, Y_train)
print("Обучение окончено")

print("Начало предсказания")
test_cnt = pd.Series(mlp.predict(test), name="count")
print("Предсказание окончено")
results = pd.concat([testID, test_cnt],axis=1)

results.to_csv("Kaggle.csv",index=False)