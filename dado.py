import matplotlib.pyplot as plt
import matplotlib.animation as animation
import numpy as np

# Configuración inicial del histograma
values = np.arange(1, 7)  # Valores posibles del dado (1 al 6)
frequencies = np.zeros(6)  # Frecuencias iniciales (todas en cero)

# Configuración de la animación
fig, ax = plt.subplots()
ax.set_title('Experimento de lanzar un dado')
ax.set_xlabel('Valor')
ax.set_ylabel('Frecuencia')
ax.set_xticks(values)
rects = ax.bar(values, frequencies, align='center')

experiment_count = 0  # Contador de experimentos realizados
animation_count = 0  # Contador de animaciones realizadas

def update_hist(frame):
    global experiment_count, animation_count
    
    # Generar un resultado aleatorio del dado
    result = np.random.randint(1, 7)
    
    # Actualizar las frecuencias
    frequencies[result - 1] += 1
    
    # Actualizar las alturas de las barras del histograma
    for rect, frequency in zip(rects, frequencies):
        rect.set_height(frequency)
    
    # Actualizar el título del gráfico con el número de experimentos realizados
    experiment_count += 1
    if experiment_count % 50 == 0:
        animation_count += 1
        ax.set_title(f'Experimento de lanzar un dado (Experimento {animation_count * 50})')
    
    # Ajustar el límite del eje y para mostrar correctamente todas las barras
    ax.set_ylim(0, np.max(frequencies) + 1)

# Configurar la animación
ani = animation.FuncAnimation(fig, update_hist, frames=2000, interval=0.001, repeat=False)

# Mejorar el diseño del histograma
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

# Mostrar la animación
plt.show()




