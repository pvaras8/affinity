from utils.data_preprocessing import prepare_data
from configs.config_rf import RF_PARAMS
from models.model_rf import RandomForestModel
from sklearn.metrics import mean_absolute_error, r2_score
import pickle


# Carga de datos
Xtr, Ytr, Xdev, Ydev, Xte, Yte = prepare_data('data/baseBuche.csv')

# Instanciar y entrenar el modelo Random Forest
rf = RandomForestModel(**RF_PARAMS)
rf.fit(Xtr, Ytr)

# Guardar el modelo entrenado usando pickle
modelo_filename = 'models/buche/rf_buche_model.pkl'
with open(modelo_filename, 'wb') as file:
    pickle.dump(rf, file)

print(f"Modelo guardado con éxito en {modelo_filename}")

# Evaluar el modelo en el conjunto de entrenamiento
Ytr_pred = rf.predict(Xtr)
mae_train = mean_absolute_error(Ytr.cpu().numpy(), Ytr_pred)
r2_train = r2_score(Ytr.cpu().numpy(), Ytr_pred)

print(f"Métricas en el conjunto de entrenamiento:")
print(f"MAE: {mae_train:.4f}")
print(f"R²: {r2_train:.4f}")

# Evaluar el modelo en el conjunto de validación
Ydev_pred = rf.predict(Xdev)
mae_dev = mean_absolute_error(Ydev.cpu().numpy(), Ydev_pred)
r2_dev = r2_score(Ydev.cpu().numpy(), Ydev_pred)

print(f"Métricas en el conjunto de validación:")
print(f"MAE: {mae_dev:.4f}")
print(f"R²: {r2_dev:.4f}")

# Evaluar el modelo en el conjunto de prueba
Yte_pred = rf.predict(Xte)
mae_test = mean_absolute_error(Yte.cpu().numpy(), Yte_pred)
r2_test = r2_score(Yte.cpu().numpy(), Yte_pred)

print(f"Métricas en el conjunto de prueba:")
print(f"MAE: {mae_test:.4f}")
print(f"R²: {r2_test:.4f}")
