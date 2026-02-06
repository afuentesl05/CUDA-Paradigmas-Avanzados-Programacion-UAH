# Sesión 1 – Device Properties (Explicación teórica)

Para la realización de esta práctica se ha utilizado la CUDA Runtime API, concretamente la función `cudaGetDeviceProperties()`, que permite obtener las características hardware de la tarjeta gráfica compatible con CUDA instalada en el sistema. El dispositivo detectado corresponde a una NVIDIA GeForce RTX 3060.

## Nombre del dispositivo

El nombre del dispositivo identifica el modelo concreto de la GPU utilizada. En este caso, se trata de una NVIDIA GeForce RTX 3060, una tarjeta gráfica basada en la arquitectura Ampere, diseñada para cómputo paralelo y aceleración mediante CUDA.

## Número máximo de hilos por bloque

Este valor indica el número máximo de hilos que puede contener un bloque de ejecución en CUDA. Para esta GPU, el máximo es **1024 hilos por bloque**, lo que implica que cualquier kernel lanzado no puede superar este límite al definir la configuración de ejecución (`<<<grid, block>>>`). Este parámetro condiciona directamente el paralelismo a nivel de bloque.

## Dimensiones máximas de hilos por bloque (x, y, z)

Las dimensiones máximas de los hilos dentro de un bloque indican cómo pueden organizarse los hilos en hasta tres dimensiones:

- **Dimensión X:** 1024
- **Dimensión Y:** 1024
- **Dimensión Z:** 64

Aunque el producto total no puede superar los 1024 hilos por bloque, estas dimensiones permiten definir bloques 1D, 2D o 3D, lo que resulta especialmente útil para problemas matriciales o espaciales.

## Dimensiones máximas del grid (x, y, z)

El grid representa el conjunto total de bloques lanzados para un kernel. Las dimensiones máximas del grid determinan cuántos bloques pueden lanzarse en cada eje. Estos valores permiten escalar la ejecución a millones de bloques, garantizando que el modelo de programación CUDA pueda adaptarse a problemas de gran tamaño.

## Memoria global total disponible

La memoria global corresponde a la memoria principal de la GPU, accesible por todos los hilos y bloques. En este caso, la GPU dispone de **12 GB de memoria global**, utilizada para almacenar datos de entrada, salida y estructuras de gran tamaño. Aunque es de acceso relativamente lento en comparación con otros tipos de memoria, su capacidad es significativamente mayor.

## Memoria compartida (shared) por bloque

La memoria compartida es una memoria rápida y de baja latencia, compartida por todos los hilos dentro de un mismo bloque. Esta GPU dispone de **49.152 bytes de memoria compartida por bloque**, lo que permite optimizar el rendimiento de los kernels reduciendo accesos a memoria global mediante reutilización de datos.

## Número máximo de registros por bloque

Los registros son la memoria más rápida disponible en CUDA y se asignan por hilo. El valor mostrado (**65.536 registros por bloque**) indica el máximo número de registros que pueden utilizarse dentro de un bloque completo. Un uso excesivo de registros puede reducir la ocupación del multiprocesador, por lo que este parámetro es clave para el análisis del rendimiento.

## Número máximo de hilos por SM (Streaming Multiprocessor)

Este valor indica el máximo número de hilos que pueden residir simultáneamente en un SM. Para esta GPU, el límite es **1536 hilos por SM**. Este parámetro es fundamental para determinar la ocupación del SM y el grado de paralelismo real que puede alcanzarse durante la ejecución de un kernel.

## Conclusión

El análisis de las propiedades del dispositivo permite comprender las limitaciones y capacidades hardware de la GPU utilizada, lo cual resulta esencial para diseñar kernels CUDA eficientes. Conocer parámetros como el número máximo de hilos, la memoria compartida disponible y la capacidad de los multiprocesadores facilita la toma de decisiones a la hora de organizar la ejecución paralela y optimizar el rendimiento de las aplicaciones CUDA.
