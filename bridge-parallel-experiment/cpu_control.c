#include <stdint.h>
#include <omp.h>
uint64_t cpu_fpcr(void) { uint64_t value; __asm__ volatile("mrs %0, fpcr" : "=r"(value)); return value; }
uint64_t cpu_fpsr(void) { uint64_t value; __asm__ volatile("mrs %0, fpsr" : "=r"(value)); return value; }
int cpu_omp_max_threads(void) { return omp_get_max_threads(); }
int cpu_omp_thread_limit(void) { return omp_get_thread_limit(); }
