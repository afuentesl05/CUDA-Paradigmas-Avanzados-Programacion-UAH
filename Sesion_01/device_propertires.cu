#include <cuda_runtime.h>
#include <iostream>
#include <iomanip>

static void checkCuda(cudaError_t err, const char* msg) {
    if (err != cudaSuccess) {
        std::cerr << "[CUDA ERROR] " << msg << " -> "
            << cudaGetErrorString(err) << " (" << (int)err << ")\n";
        std::exit(EXIT_FAILURE);
    }
}

int main() {
    int deviceCount = 0;
    checkCuda(cudaGetDeviceCount(&deviceCount), "cudaGetDeviceCount");

    if (deviceCount == 0) {
        std::cout << "No se detectan dispositivos CUDA.\n";
        return 0;
    }

    std::cout << "Dispositivos CUDA detectados: " << deviceCount << "\n\n";

    for (int dev = 0; dev < deviceCount; ++dev) {
        cudaDeviceProp p{};
        checkCuda(cudaGetDeviceProperties(&p, dev), "cudaGetDeviceProperties");

        // Nombre del device
        std::cout << "=== GPU #" << dev << " ===\n";
        std::cout << "Nombre del Device: " << p.name << "\n";

        // Número máximo de hilos por bloque
        std::cout << "Max hilos por bloque: " << p.maxThreadsPerBlock << "\n";

        // Dimensiones máximas de hilos por bloque (x,y,z)
        std::cout << "Max dimensiones de hilos por bloque (x,y,z): "
            << p.maxThreadsDim[0] << ", "
            << p.maxThreadsDim[1] << ", "
            << p.maxThreadsDim[2] << "\n";

        // Dimensiones máximas del grid (x,y,z)
        std::cout << "Max dimensiones del grid (x,y,z): "
            << p.maxGridSize[0] << ", "
            << p.maxGridSize[1] << ", "
            << p.maxGridSize[2] << "\n";

        // Memoria global máxima disponible
        double globalGB = static_cast<double>(p.totalGlobalMem) / (1024.0 * 1024.0 * 1024.0);
        std::cout << "Memoria global total (GB): " << std::fixed << std::setprecision(2)
            << globalGB << "\n";

        // Memoria compartida por bloque
        std::cout << "Shared memory por bloque (bytes): " << p.sharedMemPerBlock << "\n";

        // Registros por bloque (ojo: es el máximo de registros asignables por bloque)
        std::cout << "Registros max por bloque: " << p.regsPerBlock << "\n";

        // Número máximo de hilos en un SM:
        // CUDA Runtime no da “maxThreadsPerSM” directo como campo único universal en todas las docs antiguas,
        // pero sí da warpSize y límites de ocupación (p.maxThreadsPerMultiProcessor existe en toolkits modernos).
        std::cout << "Max hilos por SM (multi-processor): " << p.maxThreadsPerMultiProcessor << "\n";

        std::cout << "\n";
    }

    return 0;
}
