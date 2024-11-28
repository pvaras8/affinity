from models.model_gp import GaussianProcessModel
from configs.config_gp import GP_PARAMS
from utils.data_preprocessing import prepare_data
import pickle
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# Carga de datos (usar NumPy directamente)
Xtr, Ytr, Xdev, Ydev, Xte, Yte = prepare_data('data/baseDyrk1a.csv', use_torch=False)

# Escalador para los datos objetivo
scaler_y = StandardScaler()
Ytr_scaled = scaler_y.fit_transform(Ytr.reshape(-1, 1))  # Escalar solo los datos de entrenamiento

# Instanciar y entrenar el modelo con datos escalados
gp = GaussianProcessModel(**GP_PARAMS)
gp.fit(Xtr, Ytr_scaled.ravel())

# Función auxiliar para desescalar y calcular métricas
def evaluate_model(model, X, Y, scaler):
    """
    Realiza predicciones, desescala los valores y calcula las métricas de evaluación.

    Args:
        model: Modelo entrenado.
        X: Datos de entrada.
        Y: Datos reales.
        scaler: Escalador usado para los datos objetivo.

    Returns:
        dict: Métricas calculadas.
    """
    # Predicciones escaladas
    Y_pred_scaled = model.predict(X)
    # Desescalar las predicciones
    Y_pred = scaler.inverse_transform(Y_pred_scaled.reshape(-1, 1))
    # Calcular métricas
    metrics = {
        'MAE': mean_absolute_error(Y, Y_pred),
        'R2': r2_score(Y, Y_pred),
        'MSE': mean_squared_error(Y, Y_pred),
    }
    return metrics

# Evaluar en los conjuntos de entrenamiento, desarrollo y prueba
train_metrics = evaluate_model(gp, Xtr, Ytr, scaler_y)
dev_metrics = evaluate_model(gp, Xdev, Ydev, scaler_y)
test_metrics = evaluate_model(gp, Xte, Yte, scaler_y)

# Imprimir las métricas
print(f"Conjunto de entrenamiento - MAE: {train_metrics['MAE']:.4f}, R2: {train_metrics['R2']:.4f}")
print(f"Conjunto de desarrollo - MAE: {dev_metrics['MAE']:.4f}, R2: {dev_metrics['R2']:.4f}, MSE: {dev_metrics['MSE']:.4f}")
print(f"Conjunto de prueba - MAE: {test_metrics['MAE']:.4f}, R2: {test_metrics['R2']:.4f}")

# Guardar el modelo y el escalador
modelo_filename = 'models/dyrk1a/gaussian_dyrk1a_model.pkl'
scaler_filename = 'models/dyrk1a/gaussian_dyrk1a_scaler.pkl'

with open(modelo_filename, 'wb') as file:
    pickle.dump(gp, file)

with open(scaler_filename, 'wb') as file:
    pickle.dump(scaler_y, file)

print(f"Modelo guardado con éxito en {modelo_filename}")
print(f"Escalador guardado con éxito en {scaler_filename}")
