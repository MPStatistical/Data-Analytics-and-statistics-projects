import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

# Número de partículas y de pasos
N = 100
pasos = 1000

# Crear posiciones aleatorias para las partículas en 2D
particulas = np.random.rand(N, 2)

# Tamaño de los pasos de movimiento propuesto
tamaño_paso = 0.05

# Almacenar las posiciones de las partículas a lo largo del tiempo
historial_posiciones = np.zeros((pasos, N, 2))
historial_posiciones[0] = particulas

# Crear la figura y los ejes
fig, ax = plt.subplots(figsize=(8, 8))

# Configurar los límites y las etiquetas de los ejes
ax.set_xlim(0, 1)
ax.set_ylim(0, 1)
ax.set_xlabel('Posición X')
ax.set_ylabel('Posición Y')
ax.set_title('Simulación de Monte Carlo de un Gas Ideal')

# Crear el scatter plot para las partículas
scatter = ax.scatter(particulas[:, 0], particulas[:, 1], color='b', edgecolors='r')

# Función para actualizar las posiciones de las partículas y el gráfico
def actualizar(i):
    for j in range(N):
        # Proponer un nuevo movimiento para esta partícula
        movimiento_propuesto = particulas[j] + tamaño_paso * np.random.randn(2)

        # Aplicar condiciones de contorno periódicas
        movimiento_propuesto = movimiento_propuesto % 1

        # Aceptar el movimiento
        particulas[j] = movimiento_propuesto

    # Actualizar las posiciones en el gráfico
    scatter.set_offsets(particulas)

# Crear la animación
ani = animation.FuncAnimation(fig, actualizar, frames=pasos, interval=50)

# Mostrar la animación
plt.show()


