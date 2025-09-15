# analisis_regresion.py

import matplotlib.pyplot as plt

# Funciones 
def mean(values):
    return sum(values) / len(values)

def covariance(x, y, x_mean, y_mean):
    return sum((x[i] - x_mean) * (y[i] - y_mean) for i in range(len(x)))

def variance(values, mean_value):
    return sum((v - mean_value) ** 2 for v in values)

def mse(y_true, y_pred):
    return sum((yt - yp) ** 2 for yt, yp in zip(y_true, y_pred)) / len(y_true)


# Entrenamiento 
def linear_regression(x, y, lam=0.0):
    x_mean = mean(x)
    y_mean = mean(y)
    b1 = covariance(x, y, x_mean, y_mean) / (variance(x, x_mean) + lam)  # pendiente
    b0 = y_mean - b1 * x_mean
    return b0, b1

def predict(x, b0, b1):
    return [b0 + b1 * val for val in x]


# Programa
if __name__ == "__main__":
    # Datos originales
    x = [1, 2, 3, 4, 5]
    y = [2, 4, 5, 4, 5]

    x_train, y_train = [1, 2, 3], [2, 4, 5]
    x_test, y_test = [4], [4]
    x_val, y_val = [5], [5]

    # Entrenamos modelo 
    b0, b1 = linear_regression(x_train, y_train)
    print(f"Modelo sin regularización: y = {b0:.2f} + {b1:.2f}x")

    # Predicciones
    y_pred_train = predict(x_train, b0, b1)
    y_pred_test = predict(x_test, b0, b1)
    y_pred_val = predict(x_val, b0, b1)

    # Métricas
    print("\nMétricas sin regularización:")
    print(f"Train MSE: {mse(y_train, y_pred_train):.2f}")
    print(f"Test MSE: {mse(y_test, y_pred_test):.2f}")
    print(f"Validation MSE: {mse(y_val, y_pred_val):.2f}")

    # Modelo
    b0_r, b1_r = linear_regression(x_train, y_train, lam=0.1)
    print(f"\nModelo con regularización: y = {b0_r:.2f} + {b1_r:.2f}x")

    y_pred_val_r = predict(x_val, b0_r, b1_r)
    print(f"Validation MSE con regularización: {mse(y_val, y_pred_val_r):.2f}")

    
    # Gráficas
    plt.scatter(x_train, y_train, color="blue", label="Train")
    plt.scatter(x_test, y_test, color="green", label="Test")
    plt.scatter(x_val, y_val, color="red", label="Validation")

   
    x_line = list(range(1, 6))
    y_line = predict(x_line, b0, b1)
    plt.plot(x_line, y_line, label="Modelo", color="black")

  
    y_line_r = predict(x_line, b0_r, b1_r)
    plt.plot(x_line, y_line_r, label="Modelo Regularizado", color="orange", linestyle="--")

    plt.xlabel("x (horas de estudio)")
    plt.ylabel("y (calificación)")
    plt.title("Regresión Lineal - Desempeño del modelo")
    plt.legend()
    plt.show()
