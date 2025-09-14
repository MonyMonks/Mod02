
def mean(values):
    return sum(values) / len(values)

def covariance(x, y, x_mean, y_mean):
    result = 0
    for i in range(len(x)):
        result += (x[i] - x_mean) * (y[i] - y_mean)
    return result

def variance(values, mean_value):
    result = 0
    for i in range(len(values)):
        result += (values[i] - mean_value) ** 2
    return result

# Entrenamiento del modelo
def linear_regression(x, y):
    x_mean = mean(x)
    y_mean = mean(y)

    b1 = covariance(x, y, x_mean, y_mean) / variance(x, x_mean)  # pendiente β
    b0 = y_mean - b1 * x_mean                                   # intercepto α
    return b0, b1

# Predicción
def predict(x, b0, b1):
    return b0 + b1 * x

# Programa principal
if __name__ == "__main__":
    # Datos de ejemplo 
    x = [1, 2, 3, 4, 5]          # horas de estudio
    y = [2, 4, 5, 4, 5]          # calificación obtenida

    # Entrenamos el modelo
    b0, b1 = linear_regression(x, y)

    print("Modelo entrenado:")
    print(f"y = {b0:.2f} + {b1:.2f}x")

    # Predicciones de prueba
    test_x = [6, 7, 8]
    print("\nPredicciones:")
    for val in test_x:
        pred = predict(val, b0, b1)
        print(f"x={val} -> y={pred:.2f}")

    # Predicción manual con input del usuario
    while True:
        entrada = input("\nIngresa un valor de x (o escribe 'salir'): ")
        if entrada.lower() == "salir":
            break
        try:
            val = float(entrada)
            pred = predict(val, b0, b1)
            print(f"Predicción: y ≈ {pred:.2f}")
        except ValueError:
            print("Por favor ingresa un número válido.")
