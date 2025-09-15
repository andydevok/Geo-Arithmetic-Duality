# ==============================================================================
# Celda Única
# GENERADOR DE LA MATRIZ DE CONFUSIÓN FINAL (17 MUESTRAS)
# GENERADONPORQUE NO LO ENCONTRABA, TIENE OTROS COLORES
# Misión:
#   Cargar los resultados del test de estrés y generar únicamente el gráfico
#   de la matriz de confusión definitiva para el paper.
# ==============================================================================
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from google.colab import drive
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.metrics import confusion_matrix, accuracy_score
from xgboost import XGBClassifier

print("--- Cargando entorno ---")
try:
    # --- Montar Drive y cargar el dataset del Stress Test ---
    drive.mount('/content/drive', force_remount=True)
    dataset_path = '/content/drive/MyDrive/PvsNP/hodge_stress_test_features_large_v2.3.csv'
    df_stress_test = pd.read_csv(dataset_path)
    print("✔ Dataset del test de estrés cargado.")

    # --- Recrear el set de datos de prueba exacto ---
    X = df_stress_test.drop('label', axis=1).values
    y = df_stress_test['label'].values
    # Usamos random_state=42 para garantizar que obtenemos las mismas 17 muestras
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42, stratify=y
    )
    print(f"✔ Set de prueba de {len(y_test)} muestras reconstruido.")

    # --- Re-entrenar el clasificador ---
    pipeline = make_pipeline(
        StandardScaler(),
        XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=42)
    )
    print("-> Entrenando clasificador...")
    pipeline.fit(X_train, y_train)
    y_pred = pipeline.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print("✔ Clasificador listo.")

    # --- Generar y guardar la figura final ---
    print("\n--- Generando Matriz de Confusión Definitiva ---")
    cm = confusion_matrix(y_test, y_pred)
    plt.style.use('dark_background')
    plt.figure(figsize=(9, 7))
    sns.heatmap(cm, annot=True, fmt='d', cmap='coolwarm',
                xticklabels=['Chaotic', 'Algebraic'], yticklabels=['Chaotic', 'Algebraic'],
                annot_kws={"size": 16}, linewidths=.5, linecolor='black')
    
    title_text = f'Figure 2: Classifier Performance on the Stress Test\nTest Accuracy: {accuracy:.2%}'
    plt.title(title_text, fontsize=18, pad=20)
    plt.xlabel('Predicted Label', fontsize=14)
    plt.ylabel('True Label', fontsize=14)
    
    output_filename = "final_confusion_matrix_17_samples.png"
    plt.savefig(output_filename, dpi=300, bbox_inches='tight')
    plt.show()
    
    print(f"\n✔ ¡Éxito! La matriz de confusión final ha sido guardada como '{output_filename}'")

except FileNotFoundError:
    print(f"\n❌ ERROR: No se encontró el archivo del dataset '{dataset_path}'.")
    print("Asegúrate de que el script v2.3 se haya ejecutado correctamente al menos una vez.")
except Exception as e:
    print(f"\n❌ Ocurrió un error: {e}")
