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

        // N�mero m�ximo de hilos por bloque
        std::cout << "Max hilos por bloque: " << p.maxThreadsPerBlock << "\n";

        // Dimensiones m�ximas de hilos por bloque (x,y,z)
        std::cout << "Max dimensiones de hilos por bloque (x,y,z): "
            << p.maxThreadsDim[0] << ", "
            << p.maxThreadsDim[1] << ", "
            << p.maxThreadsDim[2] << "\n";

        // Dimensiones m�ximas del grid (x,y,z)
        std::cout << "Max dimensiones del grid (x,y,z): "
            << p.maxGridSize[0] << ", "
            << p.maxGridSize[1] << ", "
            << p.maxGridSize[2] << "\n";

        // Memoria global m�xima disponible
        double globalGB = static_cast<double>(p.totalGlobalMem) / (1024.0 * 1024.0 * 1024.0);
        std::cout << "Memoria global total (GB): " << std::fixed << std::setprecision(2)
            << globalGB << "\n";

        // Memoria compartida por bloque
        std::cout << "Shared memory por bloque (bytes): " << p.sharedMemPerBlock << "\n";

        // Registros por bloque 
        std::cout << "Registros max por bloque: " << p.regsPerBlock << "\n";

        // N�mero m�ximo de hilos en un SM:
        // CUDA Runtime no da �maxThreadsPerSM� directo como campo �nico universal en todas las docs antiguas,
        // pero s� da warpSize y l�mites de ocupaci�n (p.maxThreadsPerMultiProcessor existe en toolkits modernos).
        std::cout << "Max hilos por SM (multi-processor): " << p.maxThreadsPerMultiProcessor << "\n";

        std::cout << "\n";
    }

    return 0;
}
