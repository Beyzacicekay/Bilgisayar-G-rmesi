import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# 1. Veri setini yükleme
df = pd.read_csv("AmesHousing.csv")  # AmesHousing.csv dosyasını aynı klasöre koyduğundan emin ol
print("Veri başarıyla yüklendi!\n")

# 2. Veri seti hakkında temel bilgiler
print("Veri setinin boyutları:", df.shape, "\n")
print("İlk 5 satır:")
print(df.head(), "\n")

print("Sütun isimleri ve türleri:")
print(df.dtypes, "\n")

# 3. Eksik veri analizi
missing_values = df.isnull().sum()
missing_percentages = (missing_values / len(df)) * 100
print("Eksik veri sayıları:")
print(missing_values[missing_values > 0], "\n")
print("Eksik veri oranları (%):")
print(missing_percentages[missing_percentages > 0], "\n")

# 4. Sütun isimlerini düzenleme (boşlukları kaldırma)
df.columns = df.columns.str.replace(' ', '')  # Sütun isimlerindeki boşlukları kaldırır
print("Güncellenmiş sütun isimleri:")
print(df.columns, "\n")

# 5. Eksik değerlerin doldurulması
# Sayısal veri için ortanca (median) kullanımı
if 'LotFrontage' in df.columns:  # Sütunun varlığını kontrol et
    df['LotFrontage'] = df['LotFrontage'].fillna(df['LotFrontage'].median())

# Kategorik veri için en sık geçen değer (mode) kullanımı
if 'Alley' in df.columns:
    df['Alley'] = df['Alley'].fillna(df['Alley'].mode()[0])

# 6. Eksik verilerin başarıyla doldurulduğunu kontrol etme
print("Eksik veri işlemi tamamlandı.")
missing_values_after = df.isnull().sum()
print("Eksik veri sayıları (işlem sonrası):")
print(missing_values_after[missing_values_after > 0], "\n")

# 7. Temel veri görselleştirme
plt.figure(figsize=(10, 6))
sns.heatmap(df.isnull(), cbar=False, cmap='viridis')
plt.title("Eksik Verilerin Görselleştirilmesi")
plt.show()

# 8. Veri setinde hedef değişkeni analiz etme (SalePrice)
plt.figure(figsize=(8, 5))
sns.histplot(df['SalePrice'], kde=True, color="blue")
plt.title("Ev Fiyatlarının Dağılımı (SalePrice)")
plt.xlabel("Ev Fiyatı")
plt.ylabel("Frekans")
plt.show()

# 9. Kategorik bir değişkenin analizi (örneğin MSZoning)
plt.figure(figsize=(8, 5))
sns.countplot(data=df, x='MSZoning', palette='pastel')
plt.title("Zonlama Türlerine Göre Dağılım")
plt.xlabel("Zonlama Türü")
plt.ylabel("Frekans")
plt.show()

# 10. Veri setini kaydetme (eksik değerler temizlendikten sonra)
df.to_csv("Cleaned_AmesHousing.csv", index=False)
print("Temizlenmiş veri seti 'Cleaned_AmesHousing.csv' olarak kaydedildi.")
