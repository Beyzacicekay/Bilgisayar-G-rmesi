import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import tkinter as tk
from tkinter import messagebox, ttk

# 1. Veri setini yükleme ve sütun isimlerini temizleme
try:
    df = pd.read_csv("AmesHousing.csv")
    print("Veri başarıyla yüklendi!")
except FileNotFoundError:
    print("Hata: AmesHousing.csv dosyası bulunamadı. Dosyanın doğru konumda olduğundan emin olun.")
    exit()

# Sütun isimlerini temizleyelim
df.columns = df.columns.str.strip().str.replace(' ', '').str.lower()

# 2. Kullanılacak özellikler ve hedef değişkeni belirleme
target = 'saleprice'  # Hedef değişken
categorical_feature = 'neighborhood'
features = ['overallqual', 'grlivarea', 'yearbuilt']  # Spesifik diğer özellikler

# 'Neighborhood' kontrolü
if categorical_feature not in df.columns or 'yearbuilt' not in df.columns:
    print("Hata: Gerekli sütunlar veri setinde bulunamadı.")
    exit()

# 3. Eksik değerleri doldurma
df.fillna(0, inplace=True)

# 4. Kategorik sütunu dönüştürme
df = pd.get_dummies(df, columns=[categorical_feature], drop_first=True)

# 5. Dummy değişkenleri ekleme
dummy_columns = [col for col in df.columns if categorical_feature in col]
features.extend(dummy_columns)

# 6. Veri setini bağımsız (X) ve bağımlı (y) değişkenler olarak ayırma
X = df[features]
y = df[target]

# 7. Eğitim ve test verisi olarak ayırma
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 8. Model oluşturma ve eğitme
model = LinearRegression()
model.fit(X_train, y_train)

# 9. Tahmin yapma
y_pred = model.predict(X_test)

# 10. Model performansını değerlendirme
mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print("\nModel Performansı:")
print(f"Mean Absolute Error (MAE): {mae:.2f}")
print(f"Mean Squared Error (MSE): {mse:.2f}")
print(f"R^2 Score: {r2:.2f}")

# 11. GUI ile kullanıcıdan giriş alarak tahmin yapma
def predict_price():
    try:
        neighborhood = neighborhood_combo.get()
        overallqual = int(combo_quality.get())
        grlivarea = float(entry_area.get())
        yearbuilt = int(combo_yearbuilt.get())

        # Dummy değişkenler için eksik sütunları tamamlayın
        neighborhood_columns = [col for col in X.columns if 'neighborhood_' in col]
        neighborhood_data = {col: 0 for col in neighborhood_columns}
        neighborhood_key = f'neighborhood_{neighborhood}'
        if neighborhood_key in neighborhood_data:
            neighborhood_data[neighborhood_key] = 1

        input_data = {
            'overallqual': overallqual,
            'grlivarea': grlivarea,
            'yearbuilt': yearbuilt,
            **neighborhood_data
        }

        input_df = pd.DataFrame([input_data])
        predicted_price = model.predict(input_df)
        result_label.config(text=f"Tahmini Fiyat: ${predicted_price[0]:,.2f}")
    except ValueError:
        messagebox.showerror("Hata", "Lütfen geçerli bir sayı girin.")
    except Exception as e:
        messagebox.showerror("Hata", f"Bir hata oluştu: {e}")

# 12. GUI oluşturma
root = tk.Tk()
root.title("Ev Fiyat Tahmin Aracı")
root.geometry("600x400")
root.configure(bg="#F7F7F7")

frame = ttk.Frame(root, padding=15, style="Card.TFrame")
frame.pack(fill="both", padx=20, pady=20)

# Stil tanımlaması
style = ttk.Style()
style.configure("Card.TFrame", background="white", relief="groove", borderwidth=2)
style.configure("TButton", font=("Arial", 10, "bold"), background="#4CAF50", foreground="white")
style.configure("TLabel", font=("Arial", 10), background="white")
style.map("TButton", foreground=[('active', 'black')])

# Mahalle Seçimi
ttk.Label(frame, text="Mahalle:").grid(row=0, column=0, sticky="w", pady=5, padx=5)
unique_neighborhoods = sorted([col.split('_')[1] for col in X.columns if 'neighborhood_' in col])
neighborhood_combo = ttk.Combobox(frame, values=unique_neighborhoods, state="readonly", width=25)
neighborhood_combo.grid(row=0, column=1, pady=5, padx=5)
neighborhood_combo.current(0)

# Genel Kalite
ttk.Label(frame, text="Genel Kalite (1-10):").grid(row=1, column=0, sticky="w", pady=5, padx=5)
combo_quality = ttk.Combobox(frame, values=list(range(1, 11)), state="readonly", width=25)
combo_quality.grid(row=1, column=1, pady=5, padx=5)
combo_quality.current(0)

# Yaşam Alanı
ttk.Label(frame, text="Yaşam Alanı (m²):").grid(row=2, column=0, sticky="w", pady=5, padx=5)
entry_area = ttk.Entry(frame, width=27)
entry_area.grid(row=2, column=1, pady=5, padx=5)

# İnşa Yılı
ttk.Label(frame, text="İnşa Yılı:").grid(row=3, column=0, sticky="w", pady=5, padx=5)
unique_years = sorted(df['yearbuilt'].unique())
combo_yearbuilt = ttk.Combobox(frame, values=unique_years, state="readonly", width=25)
combo_yearbuilt.grid(row=3, column=1, pady=5, padx=5)
combo_yearbuilt.current(0)

# Tahmin Yap Butonu
button_predict = ttk.Button(frame, text="Tahmin Yap", command=predict_price, style="TButton")
button_predict.grid(row=4, column=0, columnspan=2, pady=20)

# Sonuç Etiketi
result_label = tk.Label(frame, text="", font=("Arial", 12, "bold"), fg="green", bg="white")
result_label.grid(row=5, column=0, columnspan=2, pady=10)

root.mainloop()
