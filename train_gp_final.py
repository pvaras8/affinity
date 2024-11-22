from models.model_gp import GaussianProcessModel
from configs.config_gp import GP_PARAMS
from utils.data_preprocessing import prepare_data
import pickle
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score


# Carga de datos
Xtr, Ytr, Xdev, Ydev, Xte, Yte = prepare_data('data/baseBuche.csv')

# Escalador para los datos objetivo
scaler_y = StandardScaler()
Ytr_scaled = scaler_y.fit_transform(Ytr.reshape(-1, 1))

# Entrenamiento del modelo con datos escalados
gp = GaussianProcessModel(**GP_PARAMS)
gp.fit(Xtr, Ytr_scaled.ravel())
# Realizar predicciones en el conjunto de entrenamiento y desescalar
Y_pred_train_scaled = gp.predict(Xtr)
Y_pred_train = scaler_y.inverse_transform(Y_pred_train_scaled.reshape(-1, 1))

# Calcular métricas para el conjunto de entrenamiento
train_mae = mean_absolute_error(Ytr, Y_pred_train)
train_r2 = r2_score(Ytr, Y_pred_train)

# Realizar predicciones en el conjunto de desarrollo y desescalar
Y_pred_dev_scaled = gp.predict(Xdev)
Y_pred_dev = scaler_y.inverse_transform(Y_pred_dev_scaled.reshape(-1, 1))

# Calcular métricas para el conjunto de desarrollo
dev_mae = mean_absolute_error(Ydev, Y_pred_dev)
dev_r2 = r2_score(Ydev, Y_pred_dev)
val_loss = mean_squared_error(Ydev, Y_pred_dev)

# Realizar predicciones en el conjunto de prueba y desescalar
Y_pred_test_scaled = gp.predict(Xte)
Y_pred_test = scaler_y.inverse_transform(Y_pred_test_scaled.reshape(-1, 1))

# Calcular métricas para el conjunto de prueba
test_mae = mean_absolute_error(Yte, Y_pred_test)
test_r2 = r2_score(Yte, Y_pred_test)

# Imprimir las métricas deseadas
print(f"Val_loss (MSE) en el conjunto de desarrollo: {val_loss}")
print(f"MAE en el conjunto de desarrollo: {dev_mae}")
print(f"R2 en el conjunto de desarrollo: {dev_r2}")

modelo_filename = 'models/buche/gaussian_buche_model.pkl'
scaler_filename = 'models/buche/gaussian_buche_scaler.pkl'

with open(modelo_filename, 'wb') as file:
    pickle.dump(gp, file)

with open(scaler_filename, 'wb') as file:
    pickle.dump(scaler_y, file)

print(f"Modelo guardado con éxito en {modelo_filename}")
print(f"Escalador guardado con éxito en {scaler_filename}")


