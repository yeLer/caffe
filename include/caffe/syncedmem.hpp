#ifndef CAFFE_SYNCEDMEM_HPP_
#define CAFFE_SYNCEDMEM_HPP_

#include <cstdlib>

// 如果包含intel开发的MKL线性代数库，将其包含在内
#ifdef USE_MKL
  #include "mkl.h"
#endif

#include "caffe/common.hpp"

namespace caffe {

// If CUDA is available and in GPU mode, host memory will be allocated pinned,
// using cudaMallocHost. It avoids dynamic pinning for transfers (DMA).
// The improvement in performance seems negligible in the single GPU case,
// but might be more significant for parallel training. Most importantly,
// it improved stability for large models on many GPUs.
inline void CaffeMallocHost(void** ptr, size_t size, bool* use_cuda) {
#ifndef CPU_ONLY
  if (Caffe::mode() == Caffe::GPU) {
    CUDA_CHECK(cudaMallocHost(ptr, size));
    *use_cuda = true;
    return;
  }
#endif
#ifdef USE_MKL
  *ptr = mkl_malloc(size ? size:1, 64);
#else
  *ptr = malloc(size);
#endif
  *use_cuda = false;
  CHECK(*ptr) << "host allocation of size " << size << " failed";
}

inline void CaffeFreeHost(void* ptr, bool use_cuda) {
#ifndef CPU_ONLY
  if (use_cuda) {
    CUDA_CHECK(cudaFreeHost(ptr));
    return;
  }
#endif
#ifdef USE_MKL
  mkl_free(ptr);
#else
  free(ptr);
#endif
}


/**
 * @brief Manages memory allocation and synchronization between the host (CPU)
 *        and device (GPU).
 * SyncedMemory类用于管理管理在CPU及GPU上的内存分配与同步。
 * TODO(dox): more thorough description.
 */
class SyncedMemory {
 public:
  SyncedMemory(); // 无参的构造函数
  explicit SyncedMemory(size_t size); // 带参的构造函数
  ~SyncedMemory(); // 析构函数，如果有cpu数据，则释放对应的内存空间；如果有gpu数据，则释放对应的内存空间；
  const void* cpu_data(); //以不可变方式获取cpu数据的指针
  void set_cpu_data(void* data); //将cpu数据的指针指向data,并将数据的状态更改为HEAD_AT_CPU
  const void* gpu_data();//以不可变方式获取gpu数据的指针
  void set_gpu_data(void* data);//将gpu数据的指针指向data,并将数据的状态更改为HEAD_AT_GPU
  void* mutable_cpu_data(); //以可变方式获取cpu数据的指针,并将数据的状态更改为HEAD_AT_CPU
  void* mutable_gpu_data(); //以可变方式获取gpu数据的指针,并将数据的状态更改为HEAD_AT_GPU
  /*
  状态同步的枚举
  UNINITIALIZED：数据未初始化
  HEAD_AT_CPU：数据在cpu
  HEAD_AT_GPU:数据在gpu
  SYNCED:数据在cpu及gpu
  */
  enum SyncedHead { UNINITIALIZED, HEAD_AT_CPU, HEAD_AT_GPU, SYNCED };
  SyncedHead head() const { return head_; }
  size_t size() const { return size_; }

#ifndef CPU_ONLY
  void async_gpu_push(const cudaStream_t& stream);
#endif

 private:
  void check_device();

  void to_cpu();  // 将数据从显存移动到内存
  void to_gpu();  //将数据从内存移动到显存
  void* cpu_ptr_;
  void* gpu_ptr_;
  size_t size_;
  SyncedHead head_;
  bool own_cpu_data_;
  bool cpu_malloc_use_cuda_;
  bool own_gpu_data_;
  int device_;

  DISABLE_COPY_AND_ASSIGN(SyncedMemory);
};  // class SyncedMemory

}  // namespace caffe

#endif  // CAFFE_SYNCEDMEM_HPP_
