import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from autogluon.tabular import TabularPredictor
import joblib

# Wczytaj dane
df = pd.read_csv('./data/cleaned.csv')

# Wydziel kolumny numeryczne
num_cols = ["Year", "Engine_Size", "Mileage", "Doors", "Owner_Count"]

# Standaryzacja tylko kolumn numerycznych
scaler = StandardScaler()
df[num_cols] = scaler.fit_transform(df[num_cols])

train_df, test_df = train_test_split(df, test_size=0.2, random_state=42)


# Trenowanie mniejszych modeli z użyciem AutoGluon
predictor = TabularPredictor(label="Price", path="agModel/").fit(
    train_data=train_df,

    # Preset zbalansowany między jakością a szybkością uczenia
    presets="medium_quality_faster_train",

    # Wykluczenie cięższych lub niepotrzebnych typów modeli
    excluded_model_types=[
        "KNeighborsClassifier",         # KNN – mało efektywny przy dużych zbiorach
        "NeuralNetTorchClassifier",     # Sieci neuronowe – wolniejsze i cięższe
        "CatBoostClassifier",           # Podobny do LightGBM, ale cięższy
        "ExtraTreesClassifier",
        "FastText",                     # Modele tekstowe – zbędne w tym przypadku
        "ImagePredictor",               # Nie używamy obrazów
        "TabTransformerClassifier",     # Sieci transformerowe dla danych tabularycznych
        "TextPredictor",                # Predykcja tekstu – niepotrzebna
        "NeuralNetFastAI"               # Inna wersja sieci neuronowych
    ],

    # Użyte algorytmy – tylko lekkie modele boostujące + możliwość ensemblingu wagowego
    hyperparameters={
        "GBM": {},                      # LightGBM – lekki, wydajny model do danych tabularycznych
        "XGB": {},                      # XGBoost – szybki i skuteczny, ale mniejszy niż CatBoost
        # Umożliwia wybranie najlepszego modelu spośród tych bez ensemblingu
        "ENS_WEIGHTED": {}
    },

    # Wyłączenie baggingu – brak wielu foldów, zmniejszenie liczby plików
    num_bag_folds=0,

    # Wyłączenie stacking – brak wielu poziomów modeli ensemble
    num_stack_levels=0,
)


# Zapisz scaler
joblib.dump(scaler, "./agModel/scaler.pkl")
results = predictor.evaluate(test_df)

# Wyświetlenie wyników
print("Best model:", predictor.model_best)
for metric, value in results.items():
    print(f"{metric}: {value:.4f}")

# Ważność cech
importance_df = predictor.feature_importance(
    data=train_df, model=predictor.model_best)
print("\nFeature importance:")
print(importance_df[["importance"]])
