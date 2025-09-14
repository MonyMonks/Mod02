# Implementación de regresión lineal usando scikit-learn

from sklearn.linear_model import LinearRegression


# Programa principal
if __name__ == "__main__":
    x = [[1], [2], [3], [4], [5]]  # sklearn requiere lista de listas (matriz)
    y = [2, 4, 5, 4, 5]

    # Modelo
    modelo = LinearRegression()
    modelo.fit(x, y)

    # Parámetros
    print("Modelo entrenado:")
    print(f"y = {modelo.intercept_:.2f} + {modelo.coef_[0]:.2f}x")

    # Predicciones
    test_x = [[6], [7], [8]]
    predicciones = modelo.predict(test_x)

    print("\nPredicciones:")
    for val, pred in zip(test_x, predicciones):
        print(f"x={val[0]} -> y={pred:.2f}")

    # Predicción manual con input del usuario
    while True:
        entrada = input("\nIngresa un valor de x (o escribe 'salir'): ")
        if entrada.lower() == "salir":
            break
        try:
            val = float(entrada)
            pred = modelo.predict([[val]])
            print(f"Predicción: y ≈ {pred[0]:.2f}")
        except ValueError:
            print("Por favor ingresa un número válido.")
