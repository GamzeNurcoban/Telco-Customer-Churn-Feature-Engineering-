
##############################
# Telco Customer Churn Feature Engineering
##############################

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from catboost import CatBoostClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score,roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV, cross_validate
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import MinMaxScaler, LabelEncoder, StandardScaler, RobustScaler
from sklearn.neighbors import LocalOutlierFactor
from sklearn.ensemble import RandomForestClassifier
import warnings
warnings.simplefilter(action="ignore")

pd.set_option('display.max_columns', None)
pd.set_option('display.width', 170)
pd.set_option('display.max_rows', None)
pd.set_option('display.float_format', lambda x: '%.3f' % x)

df = pd.read_csv("DSMLBC8/Haftalar/Week 6/Projects/Telco/Telco-Customer-Churn.csv")
df.head()
df.shape


##################################
# GÖREV 1: KEŞİFCİ VERİ ANALİZİ
##################################

##################################
# ADIM 1: GENEL RESİM
##################################

def check_df(dataframe, head=5):
    print("##################### Shape #####################")
    print(dataframe.shape)
    print("##################### Types #####################")
    print(dataframe.dtypes)
    print("##################### Head #####################")
    print(dataframe.head(head))
    print("##################### Tail #####################")
    print(dataframe.tail(head))
    print("##################### NA #####################")
    print(dataframe.isnull().sum())
    print("##################### Quantiles #####################")
    print(dataframe.quantile([0, 0.05, 0.50, 0.95, 0.99, 1]).T)

check_df(df)


# TotalCharges sayısal bir değişken olmalı
df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors='coerce')

# Bağımlı değişkenimizi binary değişkene çevirelim. (Encode da edilebilir.)
df["Churn"] = df["Churn"].apply(lambda x : 1 if x == "Yes" else 0)
# ya da
# df.loc[df["Churn"]=="Yes","Churn"] = 1
# df.loc[df["Churn"]=="No","Churn"] = 0


##################################
# ADIM 2: NUMERİK VE KATEGORİK DEĞİŞKENLERİN YAKALANMASI
##################################

def grab_col_names(dataframe, cat_th=10, car_th=20):
    """

    Veri setindeki kategorik, numerik ve kategorik fakat kardinal değişkenlerin isimlerini verir.
    Not: Kategorik değişkenlerin içerisine numerik görünümlü kategorik değişkenler de dahildir.

    Parameters
    ------
        dataframe: dataframe
                Değişken isimleri alınmak istenilen dataframe
        cat_th: int, optional
                numerik fakat kategorik olan değişkenler için sınıf eşik değeri
        car_th: int, optional
                kategorik fakat kardinal değişkenler için sınıf eşik değeri

    Returns
    ------
        cat_cols: list
                Kategorik değişken listesi
        num_cols: list
                Numerik değişken listesi
        cat_but_car: list
                Kategorik görünümlü kardinal değişken listesi

    Examples
    ------
        import seaborn as sns
        df = sns.load_dataset("iris")
        print(grab_col_names(df))


    Notes
    ------
        cat_cols + num_cols + cat_but_car = toplam değişken sayısı
        num_but_cat cat_cols'un içerisinde.

    """
    # cat_cols, cat_but_car
    cat_cols = [col for col in dataframe.columns if dataframe[col].dtypes == "O"]
    num_but_cat = [col for col in dataframe.columns if dataframe[col].nunique() < cat_th and dataframe[col].dtypes != "O"]
    cat_but_car = [col for col in dataframe.columns if dataframe[col].nunique() > car_th and dataframe[col].dtypes == "O"]
    cat_cols = cat_cols + num_but_cat
    cat_cols = [col for col in cat_cols if col not in cat_but_car]

    # num_cols
    num_cols = [col for col in dataframe.columns if dataframe[col].dtypes != "O"]
    num_cols = [col for col in num_cols if col not in num_but_cat]

    print(f"Observations: {dataframe.shape[0]}")
    print(f"Variables: {dataframe.shape[1]}")
    print(f'cat_cols: {len(cat_cols)}')
    print(f'num_cols: {len(num_cols)}')
    print(f'cat_but_car: {len(cat_but_car)}')
    print(f'num_but_cat: {len(num_but_cat)}')

    return cat_cols, num_cols, cat_but_car


cat_cols, num_cols, cat_but_car = grab_col_names(df)

cat_cols
df[cat_cols].head()

num_cols
df[num_cols].head()

cat_but_car
df[cat_but_car].head()



##################################
# ADIM 3: KATEGORİK DEĞİŞKENLERİN ANALİZİ
##################################

def cat_summary(dataframe, col_name, plot=False):
    print(pd.DataFrame({col_name: dataframe[col_name].value_counts(),
                        "Ratio": 100 * dataframe[col_name].value_counts() / len(dataframe)}))
    print("##########################################")
    if plot:
        sns.countplot(x=dataframe[col_name], data=dataframe)
        plt.show()

for col in cat_cols:
    cat_summary(df, col,plot=True)



##################################
# ADIM 3: NUMERİK DEĞİŞKENLERİN ANALİZİ
##################################

def num_summary(dataframe, numerical_col, plot=False):
    quantiles = [0.05, 0.10, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90, 0.95, 0.99]
    print(dataframe[numerical_col].describe(quantiles).T)

    if plot:
        dataframe[numerical_col].hist(bins=20)
        plt.xlabel(numerical_col)
        plt.title(numerical_col)
        plt.show()

for col in num_cols:
    num_summary(df, col, plot=True)




##################################
# ADIM 4: NUMERİK DEĞİŞKENLERİN TARGET GÖRE ANALİZİ
##################################

def target_summary_with_num(dataframe, target, numerical_col):
    print(dataframe.groupby(target).agg({numerical_col: "mean"}), end="\n\n\n")

for col in num_cols:
    target_summary_with_num(df, "Churn", col)


##################################
# ADIM 4: KATEGORİK DEĞİŞKENLERİN TARGET GÖRE ANALİZİ
##################################


def target_summary_with_cat(dataframe, target, categorical_col):
    print(categorical_col)
    print(pd.DataFrame({"TARGET_MEAN": dataframe.groupby(categorical_col)[target].mean(),
                        "Count": dataframe[categorical_col].value_counts(),
                        "Ratio": 100 * dataframe[categorical_col].value_counts() / len(dataframe)}), end="\n\n\n")

for col in cat_cols:
    target_summary_with_cat(df, "Churn", col)

# Kategorik değişkenleri "Churn" özelinde görselleştirip inceleyelim
for col in cat_cols:
    graph=pd.crosstab(index=df['Churn'],columns=df[col]).plot.bar(figsize=(7,4), rot=0)
    plt.show()



##################################
# ADIM 5: AYKIRI GÖZLEM ANALİZİ
##################################

def outlier_thresholds(dataframe, col_name, q1=0.10, q3=0.90):
    quartile1 = dataframe[col_name].quantile(q1)
    quartile3 = dataframe[col_name].quantile(q3)
    interquantile_range = quartile3 - quartile1
    up_limit = quartile3 + 1.5 * interquantile_range
    low_limit = quartile1 - 1.5 * interquantile_range
    return low_limit, up_limit


def check_outlier(dataframe, col_name):
    low_limit, up_limit = outlier_thresholds(dataframe, col_name)
    if dataframe[(dataframe[col_name] > up_limit) | (dataframe[col_name] < low_limit)].any(axis=None):
        return True
    else:
        return False

for col in num_cols:
    print(col,": ", check_outlier(df, col))

# Sayısal değişkenlerde aykırı değerin olmadığını gözlemliyoruz.



##################################
# ADIM 6: EKSİK GÖZLEM ANALİZİ
##################################

def missing_values_table(dataframe, na_name=False):
    na_columns = [col for col in dataframe.columns if dataframe[col].isnull().sum() > 0]

    n_miss = dataframe[na_columns].isnull().sum().sort_values(ascending=False)
    ratio = (dataframe[na_columns].isnull().sum() / dataframe.shape[0] * 100).sort_values(ascending=False)
    missing_df = pd.concat([n_miss, np.round(ratio, 2)], axis=1, keys=['n_miss', 'ratio'])
    print(missing_df, end="\n")

    if na_name:
        return na_columns


missing_values_table(df)


# Sadece "TotalCharges" değişkeninde 11 değerin eksik olduğunu gözlemliyoruz.

# TotalCharges değişkenini float'a çevirmeden önce NaN değer görünmüyordu. Bu durumu gözlemleyelim.

index_nan = df[df.isnull().any(axis=1)].index
df2 = pd.read_csv("DSMLBC8/Haftalar/Week 6/Projects/Telco/Telco-Customer-Churn.csv")
df2.iloc[index_nan]
# Bu değerler totalcharges'ta da boş görünüyor.


# TotalCharges değerleri NaN olan tüm müşterilerin tenure değerleri de 0. Ayrıca hiçbiri churn olmamış.
# Bu da bize bu müşterilerin firmanın yeni müşterisi oldukları bilgisini veriyor.
# Buna emin olmak için:
df[df["tenure"] == 0]
# Tenure değeri 0 olan tüm müşteriler aynı zamanda TotalCharges'ı NaN olan müşteriler.


##################################
# ADIM 7: KORELASYON
##################################

df[num_cols].corr()

# Korelasyon Matrisi
f, ax = plt.subplots(figsize=[18, 13])
sns.heatmap(df[num_cols].corr(), annot=True, fmt=".2f", ax=ax, cmap="magma")
ax.set_title("Correlation Matrix", fontsize=20)
plt.show()

# Tenure ve Monthly Charges arasındaki korelasyon düşük çıkarken Total Charges arasındaki ilişki çok yüksek çıktı.
# Ayrıca Monthly Charges ile Total Charges arasında beklenen bir korelasyon var.

df.corrwith(df["Churn"]).sort_values(ascending=False)




##################################
# GÖREV 2: FEATURE ENGINEERING
##################################

##################################
# ADIM 1: EKSİK DEĞERLER İÇİN İŞLEMLER
##################################

# TotalCharges değişkeninde 11 adet eksik gözlemimiz vardı.
# Toplam içinde çok az sayıda olduğu için silinebilir. 1 Aylık ödemeleri yazılabilir ya da hiç ödeme yapmadıkları için 0 yazılabilir. Sadece NaN olanların median'ı ile doldurulabilir.

# Biz 1 aylık ödemelerini yazalım.
df["TotalCharges"].fillna(df.iloc[index_nan]["MonthlyCharges"], inplace=True)

# Diğer alternatifler:


# df["TotalCharges"].dropna(inplace=True)

# Tüm NaN'lere 0 yazmak:
# df["TotalCharges"].fillna(0, inplace=True)


df.isnull().sum().any() #False

##################################
# ADIM 1: AYKIRI DEĞERLER İÇİN İŞLEMLER
##################################

for col in num_cols:
    print(col,": ", check_outlier(df, col))

# Aykırı değer olmadığını gözlemledik.


##################################
# BASE MODEL KURULUMU
##################################

dff = df.copy()
cat_cols = [col for col in cat_cols if col not in ["Churn"]]
cat_cols

def one_hot_encoder(dataframe, categorical_cols, drop_first=False):
    dataframe = pd.get_dummies(dataframe, columns=categorical_cols, drop_first=drop_first)
    return dataframe
dff = one_hot_encoder(dff, cat_cols, drop_first=True)


y = dff["Churn"]
X = dff.drop(["Churn","customerID"], axis=1)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=17)

catboost_model = CatBoostClassifier(verbose=False, random_state=12345).fit(X_train, y_train)
y_pred = catboost_model.predict(X_test)


print(f"Accuracy: {round(accuracy_score(y_pred, y_test), 4)}")
print(f"Recall: {round(recall_score(y_pred,y_test),4)}")
print(f"Precision: {round(precision_score(y_pred,y_test), 4)}")
print(f"F1: {round(f1_score(y_pred,y_test), 4)}")
print(f"Auc: {round(roc_auc_score(y_pred,y_test), 4)}")

# Accuracy: 0.7913
# Recall: 0.6468
# Precision: 0.5105
# F1: 0.5706
# Auc: 0.7388



##################################
# ADIM 2: YENİ DEĞİŞKENLER OLUŞTURMA
##################################

# Tenure değişkeninden yıllık kategorik değişken oluşturma
df.loc[(df["tenure"]>=0) & (df["tenure"]<=12),"NEW_TENURE_YEAR"] = "0-1 Year"
df.loc[(df["tenure"]>12) & (df["tenure"]<=24),"NEW_TENURE_YEAR"] = "1-2 Year"
df.loc[(df["tenure"]>24) & (df["tenure"]<=36),"NEW_TENURE_YEAR"] = "2-3 Year"
df.loc[(df["tenure"]>36) & (df["tenure"]<=48),"NEW_TENURE_YEAR"] = "3-4 Year"
df.loc[(df["tenure"]>48) & (df["tenure"]<=60),"NEW_TENURE_YEAR"] = "4-5 Year"
df.loc[(df["tenure"]>60) & (df["tenure"]<=72),"NEW_TENURE_YEAR"] = "5-6 Year"


# Kontratı 1 veya 2 yıllık müşterileri Engaged olarak belirtme
df["NEW_Engaged"] = df["Contract"].apply(lambda x: 1 if x in ["One year","Two year"] else 0)

# Herhangi bir destek, yedek veya koruma almayan kişiler
df["NEW_noProt"] = df.apply(lambda x: 1 if (x["OnlineBackup"] != "Yes") or (x["DeviceProtection"] != "Yes") or (x["TechSupport"] != "Yes") else 0, axis=1)

# Aylık sözleşmesi bulunan ve genç olan müşteriler
df["NEW_Young_Not_Engaged"] = df.apply(lambda x: 1 if (x["NEW_Engaged"] == 0) and (x["SeniorCitizen"] == 0) else 0, axis=1)


# Kişinin toplam aldığı servis sayısı
df['NEW_TotalServices'] = (df[['PhoneService', 'InternetService', 'OnlineSecurity',
                                       'OnlineBackup', 'DeviceProtection', 'TechSupport',
                                       'StreamingTV', 'StreamingMovies']]== 'Yes').sum(axis=1)


# Herhangi bir streaming hizmeti alan kişiler
df["NEW_FLAG_ANY_STREAMING"] = df.apply(lambda x: 1 if (x["StreamingTV"] == "Yes") or (x["StreamingMovies"] == "Yes") else 0, axis=1)

# Kişi otomatik ödeme yapıyor mu?
df["NEW_FLAG_AutoPayment"] = df["PaymentMethod"].apply(lambda x: 1 if x in ["Bank transfer (automatic)","Credit card (automatic)"] else 0)

# ortalama aylık ödeme
df["NEW_AVG_Charges"] = df["TotalCharges"] / (df["tenure"] + 1)

# Güncel Fiyatın ortalama fiyata göre artışı
df["NEW_Increase"] = df["NEW_AVG_Charges"] / df["MonthlyCharges"]

# Servis başına ücret
df["NEW_AVG_Service_Fee"] = df["MonthlyCharges"] / (df['NEW_TotalServices'] + 1)


# Şirket hizmet sektöründe yer aldığı için verdiği hizmetin kalitesinden memnuniyet durumu önemli.
# Memnuniyet durumunu tahmin edebilecek değişkenler oluşturalım.
# Öncelikle contract değişkenini rahat kullanabilmek adına sayısal değişkene çevirelim.

df.loc[(df['Contract'] == "Month-to-month" ), "NEW_CONTRACT"] = 1
df.loc[(df['Contract'] == "One year" ), "NEW_CONTRACT"] = 12
df.loc[(df['Contract'] == "Two year" ), "NEW_CONTRACT"] = 24


df.head()
df.shape


##################################
# ADIM 3: ENCODING İŞLEMLERİ
##################################

# Yeniden değişkenlerimizi türlerine göre ayıralım.

cat_cols, num_cols, cat_but_car = grab_col_names(df)

# NEW_TotalServices değişkeni cat_cols arasında yer almış fakat numeric bir değişken onun yerini değiştirelim.
cat_cols.remove("NEW_TotalServices")
num_cols.append("NEW_TotalServices")

# Churn bağımlı değişkenimiz olduğu için onu encode etmemize şu an için gerek yok.
cat_cols.remove("Churn")

def one_hot_encoder(dataframe, categorical_cols, drop_first=False):
    dataframe = pd.get_dummies(dataframe, columns=categorical_cols, drop_first=drop_first)
    return dataframe

df = one_hot_encoder(df, cat_cols, drop_first=True)

df.head()
df.shape


##################################
# ADIM 4: NUMERİK DEĞİŞKENLER İÇİN STANDARTLAŞTIRMA İŞLEMLERİ
##################################

scaler = RobustScaler() # Medyanı çıkar iqr'a böl.
df[num_cols] = scaler.fit_transform(df[num_cols])
df.head()




##################################
# MODELLEME-CatBoost
##################################

y = df["Churn"]
X = df.drop(["Churn","customerID"], axis=1)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=17)

catboost_model = CatBoostClassifier(verbose=False, random_state=12345).fit(X_train, y_train)
y_pred = catboost_model.predict(X_test)

print(f"Accuracy: {round(accuracy_score(y_pred, y_test), 2)}")
print(f"Recall: {round(recall_score(y_pred,y_test),2)}")
print(f"Precision: {round(precision_score(y_pred,y_test), 2)}")
print(f"F1: {round(f1_score(y_pred,y_test), 2)}")
print(f"Auc: {round(roc_auc_score(y_pred,y_test), 2)}")

# Accuracy: 0.83
# Recall: 0.75
# Precision: 0.56
# F1: 0.64
# Auc: 0.8

# Base Model
# Accuracy: 0.7913
# Recall: 0.6468
# Precision: 0.5105
# F1: 0.5706
# Auc: 0.7388

# Accuracy = Doğru sınıflandırma oranı
# Precision = Pozitif sınıf tahminlerinin başarı oranı
# Recall = Pozitif sınıfın doğru tahmin edilme oranı
# F1 Score = Precision ve recall'un harmonik ortalaması
# AUC Score = ROC eğrisinin altında kalan alanı ifade eder, Tüm sınıflandırma eşikleri için toplu bir performans ölçüsüdür. 1'e ne kadar yakınsa o kadar iyi diyebiliriz.


def plot_feature_importance(importance,names,model_type):
    # Create arrays from feature importance and feature names
    feature_importance = np.array(importance)
    feature_names = np.array(names)

    # Create a DataFrame using a Dictionary
    data = {'feature_names': feature_names, 'feature_importance': feature_importance}
    fi_df = pd.DataFrame(data)

    # Sort the DataFrame in order decreasing feature importance
    fi_df.sort_values(by=['feature_importance'], ascending=False, inplace=True)

    # Define size of bar plot
    plt.figure(figsize=(15, 10))
    # Plot Searborn bar chart
    sns.barplot(x=fi_df['feature_importance'], y=fi_df['feature_names'])
    # Add chart labels
    plt.title(model_type + ' FEATURE IMPORTANCE')
    plt.xlabel('FEATURE IMPORTANCE')
    plt.ylabel('FEATURE NAMES')
    plt.show()

plot_feature_importance(catboost_model.get_feature_importance(), X.columns, 'CATBOOST')


##################################
# MODELLEME-RandomForest
##################################

rf_model = RandomForestClassifier(random_state=46).fit(X_train, y_train)
y_pred = rf_model.predict(X_test)

print(f"Accuracy: {round(accuracy_score(y_pred, y_test), 2)}")
print(f"Recall: {round(recall_score(y_pred,y_test),2)}")
print(f"Precision: {round(precision_score(y_pred,y_test), 2)}")
print(f"F1: {round(f1_score(y_pred,y_test), 2)}")
print(f"Auc: {round(roc_auc_score(y_pred,y_test), 2)}")

# Accuracy: 0.82
# Recall: 0.73
# Precision: 0.54
# F1: 0.62
# Auc: 0.78
