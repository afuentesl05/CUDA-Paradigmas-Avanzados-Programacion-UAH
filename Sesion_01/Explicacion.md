# Sesión 1 – Device Properties (Explicación teórica)
## Tarea 1: Device Properties

### Descripción
Práctica utilizando la CUDA Runtime API, concretamente la función `cudaGetDeviceProperties()`, para obtener las características hardware de la GPU (NVIDIA GeForce RTX 3060).

### Propiedades del Dispositivo

| Propiedad | Valor | Descripción |
|-----------|-------|-------------|
| Nombre | NVIDIA GeForce RTX 3060 | Arquitectura Ampere |
| Máx. hilos/bloque | 1024 | Límite de threads por bloque |
| Dim. X máxima | 1024 | Dimensión X máxima de hilos |
| Dim. Y máxima | 1024 | Dimensión Y máxima de hilos |
| Dim. Z máxima | 64 | Dimensión Z máxima de hilos |
| Memoria global | 12 GB | Memoria principal de GPU |
| Shared memory/bloque | 49.152 bytes | Memoria rápida por bloque |
| Registros/bloque | 65.536 | Memoria más rápida disponible |
| Máx. hilos/SM | 1536 | Límite por Streaming Multiprocessor |

### Conceptos Clave
- **Memoria global**: Accesible por todos los hilos, de baja latencia relativa
- **Memoria compartida**: Rápida y de baja latencia, compartida dentro del bloque
- **Registros**: Memoria más rápida, asignada por hilo

---

## Tarea 2: Estructura Básica de un Kernel CUDA

### Objetivo
Comprender la estructura de un kernel CUDA, el modelo jerárquico (grid, bloques, hilos) y calcular identificadores globales.

### Configuración de Ejecución
```cuda
<<<3, 4>>>
```
- **3 bloques** en el grid
- **4 hilos por bloque**
- **12 hilos totales**

### Fórmula del Identificador Global
```cuda
int globalThreadId = blockIdx.x * blockDim.x + threadIdx.x;
```

### Notas Importantes
- Los identificadores (`blockIdx`, `blockDim`, `threadIdx`) solo existen en código device
- Usar `cudaDeviceSynchronize()` para sincronizar ejecución
- El orden de ejecución no está garantizado en paralelo

# Sesión 01 – Tarea 3: Sumar un valor X a un vector 1D

## Objetivo
El objetivo de esta tarea es implementar un kernel CUDA que, dado un **vector
unidimensional**, sume un valor constante **X** a todos sus elementos.  
La tarea se divide en dos partes para analizar cómo influye la configuración
de ejecución del kernel (número de bloques e hilos) en la organización del
paralelismo.

---

## Descripción del problema
Dado un vector 1D de tamaño `n`, se desea aplicar la operación:

```v[i] = v[i] + X
```

para cada uno de sus elementos, utilizando programación paralela mediante CUDA.

---

## Kernel CUDA

```cpp
__global__ void addX(int* d_v, int n, int X) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;  // índice global del hilo
    if (i < n) {
        d_v[i] += X;
    }
}
```

## Explicación

Cada hilo calcula su índice global a partir de su identificador de bloque
y su identificador local.

La condición if (i < n) garantiza que no se acceda fuera de los límites del
vector cuando el número total de hilos supera el tamaño del problema.

Cada hilo es responsable de modificar un único elemento del vector.