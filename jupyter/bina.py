import numpy as np
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
from mlxtend.frequent_patterns import apriori, association_rules

data = pd.read_excel('orders.xlsx')
data.head()

data.columns
data['Pada Tanggal'] = pd.to_datetime(data['Pada Tanggal'])

#memcah kolom produk menajdi baris baru
data['Menu'] = data['Menu'].str.split(', ')
data = data.explode('Menu')
#cek duplikat
duplicate_mask = data.duplicated()
print(data[duplicate_mask])
#cek missing value
sel_kosong = data.isnull()
if sel_kosong.any().any():
    print("Terdapat setidaknya satu sel kosong dalam tabel.")
else:
    print("Tidak ada sel kosong dalam tabel.")
    #normalisasi
data['Menu'] = data['Menu'].str.lower()
data
#menampilkan seluruh data
basket=(data.groupby(['No','Menu'])['Quantity']
        .sum().unstack().reset_index().fillna(0) #menjumlahkan, membuat tabel, mereset index, menganti missing value dengan 0
        .set_index('No')) #menetapan index mnejadi no
basket
#one hot encoding
def hot_encode(x):
  if(x<=0):
    return 0
  if(x>=0):
    return 1
#menampilkan hasil one hot encoding
basket_encode = basket.applymap(hot_encode)
basket = basket_encode
basket
#model
frq_items = apriori(basket, min_support=0.01, use_colnames=True)
rules = association_rules(frq_items, metric="lift", min_threshold=1)
rules = rules.sort_values(['confidence','lift'],ascending=[False,False])
rules
plt.figure(figsize=(10, 6))
plt.bar(range(len(rules)), rules['support'], tick_label=rules['antecedents'].apply(lambda x: list(x)[0] if isinstance(x, frozenset) else x)
.astype(str) + ' -> ' + rules['consequents'].apply(lambda x: list(x)[0] if isinstance(x, frozenset) else x)
.astype(str))
plt.xlabel('Association Rules')
plt.ylabel('Support')
plt.title('Support for Association Rules')
plt.xticks(rotation=90)
plt.show()

plt.figure(figsize=(10, 6))
plt.bar(range(len(rules)), rules['confidence'].apply(lambda x: list(x)[0] if isinstance(x, frozenset) else x), tick_label=rules['antecedents'].apply(lambda x: list(x)[0] if isinstance(x, frozenset) else x)
.astype(str) + ' -> ' + rules['consequents'].apply(lambda x: list(x)[0] if isinstance(x, frozenset) else x)
.astype(str))
plt.xlabel('Association Rules')
plt.ylabel('Confidence')
plt.title('Confidence for Association Rules')
plt.xticks(rotation=90)
plt.show()

plt.figure(figsize=(10, 6))
plt.scatter(rules['support'], rules['confidence'], c=rules['lift'], cmap='viridis', s=np.multiply(rules['lift'], 1000))
plt.xlabel('Support')
plt.ylabel('Confidence')
plt.title('Support vs Confidence with Lift')
plt.colorbar(label='Lift')
plt.show()

heatmap_data = rules.pivot_table(index='antecedents', columns='consequents', values='lift')

# Membuat heatmap
plt.figure(figsize=(10, 6))
sns.heatmap(heatmap_data, annot=True, cmap='coolwarm', fmt=".3f", linewidths=.5)
plt.title('Heatmap of Lift Values for Association Rules')
plt.show()
