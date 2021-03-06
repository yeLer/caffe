#ifndef CAFFE_UTIL_MATH_FUNCTIONS_H_
#define CAFFE_UTIL_MATH_FUNCTIONS_H_

#include <stdint.h>
#include <cmath>  // for std::fabs and std::signbit

// GLog库，它是google的一个开源的日志库
#include "glog/logging.h"

#include "caffe/common.hpp"
#include "caffe/util/device_alternate.hpp"

// 文件里包含两种库，一个是Intel MKL，一个是OpenBLAS，这里用的是OpenBLAS
#include "caffe/util/mkl_alternate.hpp"

namespace caffe {

// Caffe gemm provides a simpler interface to the gemm functions, with the
// limitation that the data has to be contiguous in memory.
// C=alpha*A*B+beta*C
// A,B,C 是输入矩阵（一维数组格式）
// CblasRowMajor :数据是行主序的（二维数据也是用一维数组储存的）
// TransA, TransB：是否要对A和B做转置操作（CblasTrans CblasNoTrans）
// M： A、C 的行数
// N： B、C 的列数
// K： A 的列数， B 的行数
// lda ： A的列数（不做转置）行数（做转置）
// ldb： B的列数（不做转置）行数（做转置）
template <typename Dtype>
void caffe_cpu_gemm(const CBLAS_TRANSPOSE TransA,
    const CBLAS_TRANSPOSE TransB, const int M, const int N, const int K,
    const Dtype alpha, const Dtype* A, const Dtype* B, const Dtype beta,
    Dtype* C);

// y=alpha*A*x+beta*y
// 其中X和Y是向量，A 是矩阵
// M：A 的行数
// N：A 的列数
template <typename Dtype>
void caffe_cpu_gemv(const CBLAS_TRANSPOSE TransA, const int M, const int N,
    const Dtype alpha, const Dtype* A, const Dtype* x, const Dtype beta,
    Dtype* y);

// Y=alpha*X+Y
// N：为X和Y中element的个数
template <typename Dtype>
void caffe_axpy(const int N, const Dtype alpha, const Dtype* X,
    Dtype* Y);
    
// Y=alpha*X+beta*Y
template <typename Dtype>
void caffe_cpu_axpby(const int N, const Dtype alpha, const Dtype* X,
    const Dtype beta, Dtype* Y);
    
// 从X中拷贝前N个元素到Y中
template <typename Dtype>
void caffe_copy(const int N, const Dtype *X, Dtype *Y);

// 将X中的前N个元素置为alpha    
template <typename Dtype>
void caffe_set(const int N, const Dtype alpha, Dtype *X);

//  一般为新申请的内存做初始化，功能是将buffer所指向内存中的每个字节的内容全部设置为c指定的ASCII值, count为块的大小    
inline void caffe_memset(const size_t N, const int alpha, void* X) {
  memset(X, alpha, N);  // NOLINT(caffe/alt_fn)
}
// 给X中的前N个元素分别加上常数alpha
template <typename Dtype>
void caffe_add_scalar(const int N, const Dtype alpha, Dtype *X);
    
// X = alpha*X
// N： X中element的个数
template <typename Dtype>
void caffe_scal(const int N, const Dtype alpha, Dtype *X);

template <typename Dtype>
void caffe_sqr(const int N, const Dtype* a, Dtype* y);

template <typename Dtype>
void caffe_sqrt(const int N, const Dtype* a, Dtype* y);
    
// 会调用mkl_alternate.hpp中的vsAdd、vsSub、vsMul、vsDiv、vdAdd、vdSub、vdMul、vdDiv函数
// caffe_add、 caffe_sub、 caffe_mul、 caffe_div 函数
// 这四个函数分别实现element-wise的加减乘除（y[i] = a[i] + - * \ b[i]）
template <typename Dtype>
void caffe_add(const int N, const Dtype* a, const Dtype* b, Dtype* y);

template <typename Dtype>
void caffe_sub(const int N, const Dtype* a, const Dtype* b, Dtype* y);

template <typename Dtype>
void caffe_mul(const int N, const Dtype* a, const Dtype* b, Dtype* y);

template <typename Dtype>
void caffe_div(const int N, const Dtype* a, const Dtype* b, Dtype* y);
    
// 会调用mkl_alternate.hpp中的vsPowx和vdPowx函数
// caffe_powx、 caffe_sqr、 caffe_exp、 caffe_abs 函数
// 同样是element-wise操作，分别是y[i] = a[i] ^ b， y[i] = a[i]^2，y[i] = exp(a[i] )，y[i] = |a[i]|
template <typename Dtype>
void caffe_powx(const int n, const Dtype* a, const Dtype b, Dtype* y);
    
// 返回一个unsignedint类型的随机数
unsigned int caffe_rng_rand();
    
// 在最大方向上，返回b可以表示的最接近的数值
template <typename Dtype>
Dtype caffe_nextafter(const Dtype b);
    
// 产生指定范围内的均匀分布随机数
template <typename Dtype>
void caffe_rng_uniform(const int n, const Dtype a, const Dtype b, Dtype* r);
    
// 产生高斯分布随机数
template <typename Dtype>
void caffe_rng_gaussian(const int n, const Dtype mu, const Dtype sigma,
                        Dtype* r);
    
// 产生伯努利分布随机数
template <typename Dtype>
void caffe_rng_bernoulli(const int n, const Dtype p, int* r);

template <typename Dtype>
void caffe_rng_bernoulli(const int n, const Dtype p, unsigned int* r);
    
// 会调用mkl_alternate.hpp中的vsSqr、vsExp、vsLn、vsAbs、vdSqr、vdExp、vdLn、vdAbs函数
template <typename Dtype>
void caffe_exp(const int n, const Dtype* a, Dtype* y);

template <typename Dtype>
void caffe_log(const int n, const Dtype* a, Dtype* y);

template <typename Dtype>
void caffe_abs(const int n, const Dtype* a, Dtype* y);
    
// 计算步长为1的内积
template <typename Dtype>
Dtype caffe_cpu_dot(const int n, const Dtype* x, const Dtype* y);
    
// 计算指定步长的内积
// 功能： 返回 vector X 和 vector Y 的内积。
// incx， incy ： 步长，即每隔incx 或 incy 个element 进行操作。
template <typename Dtype>
Dtype caffe_cpu_strided_dot(const int n, const Dtype* x, const int incx,
    const Dtype* y, const int incy);

// Returns the sum of the absolute values of the elements of vector x
// 计算向量x中前n个元素的绝对值之和    
template <typename Dtype>
Dtype caffe_cpu_asum(const int n, const Dtype* x);

// the branchless, type-safe version from
// http://stackoverflow.com/questions/1903954/is-there-a-standard-sign-function-signum-sgn-in-c-c
// 类似于正负号函数，仅返回-1或1
template<typename Dtype>
inline int8_t caffe_sign(Dtype val) {
  return (Dtype(0) < val) - (val < Dtype(0));
}

// The following two macros are modifications of DEFINE_VSL_UNARY_FUNC
//   in include/caffe/util/mkl_alternate.hpp authored by @Rowland Depp.
// Please refer to commit 7e8ef25c7 of the boost-eigen branch.
// Git cherry picking that commit caused a conflict hard to resolve and
//   copying that file in convenient for code reviewing.
// So they have to be pasted here temporarily.
// 一元函数，类似于mkl_alternate.hpp中的宏DEFINE_VSL_UNARY_FUNC，包括：
//  (1)、caffe_cpu_sign：正负号函数，输出-1、0、1；
//  (2)、caffe_cpu_sgnbit：作用类似于std::signbit，static_cast<bool>((std::signbit)(x))；x为负数输出为1，其它输出为0；
//  (3)、caffe_cpu_fabs：取绝对值，作用类似于std::fabs。
#define DEFINE_CAFFE_CPU_UNARY_FUNC(name, operation) \
  template<typename Dtype> \
  void caffe_cpu_##name(const int n, const Dtype* x, Dtype* y) { \
    CHECK_GT(n, 0); CHECK(x); CHECK(y); \
    for (int i = 0; i < n; ++i) { \
      operation; \
    } \
  }

// output is 1 for the positives, 0 for zero, and -1 for the negatives
DEFINE_CAFFE_CPU_UNARY_FUNC(sign, y[i] = caffe_sign<Dtype>(x[i]))

// This returns a nonzero value if the input has its sign bit set.
// The name sngbit is meant to avoid conflicts with std::signbit in the macro.
// The extra parens are needed because CUDA < 6.5 defines signbit as a macro,
// and we don't want that to expand here when CUDA headers are also included.
DEFINE_CAFFE_CPU_UNARY_FUNC(sgnbit, \
    y[i] = static_cast<bool>((std::signbit)(x[i])))

DEFINE_CAFFE_CPU_UNARY_FUNC(fabs, y[i] = std::fabs(x[i]))

template <typename Dtype>
void caffe_cpu_scale(const int n, const Dtype alpha, const Dtype *x, Dtype* y);

#ifndef CPU_ONLY  // GPU

// Decaf gpu gemm provides an interface that is almost the same as the cpu
// gemm function - following the c convention and calling the fortran-order
// gpu code under the hood.
template <typename Dtype>
void caffe_gpu_gemm(const CBLAS_TRANSPOSE TransA,
    const CBLAS_TRANSPOSE TransB, const int M, const int N, const int K,
    const Dtype alpha, const Dtype* A, const Dtype* B, const Dtype beta,
    Dtype* C);

template <typename Dtype>
void caffe_gpu_gemv(const CBLAS_TRANSPOSE TransA, const int M, const int N,
    const Dtype alpha, const Dtype* A, const Dtype* x, const Dtype beta,
    Dtype* y);

template <typename Dtype>
void caffe_gpu_axpy(const int N, const Dtype alpha, const Dtype* X,
    Dtype* Y);

template <typename Dtype>
void caffe_gpu_axpby(const int N, const Dtype alpha, const Dtype* X,
    const Dtype beta, Dtype* Y);

void caffe_gpu_memcpy(const size_t N, const void *X, void *Y);

template <typename Dtype>
void caffe_gpu_set(const int N, const Dtype alpha, Dtype *X);

inline void caffe_gpu_memset(const size_t N, const int alpha, void* X) {
#ifndef CPU_ONLY
  CUDA_CHECK(cudaMemset(X, alpha, N));  // NOLINT(caffe/alt_fn)
#else
  NO_GPU;
#endif
}

template <typename Dtype>
void caffe_gpu_add_scalar(const int N, const Dtype alpha, Dtype *X);

template <typename Dtype>
void caffe_gpu_scal(const int N, const Dtype alpha, Dtype *X);

#ifndef CPU_ONLY
template <typename Dtype>
void caffe_gpu_scal(const int N, const Dtype alpha, Dtype* X, cudaStream_t str);
#endif

template <typename Dtype>
void caffe_gpu_add(const int N, const Dtype* a, const Dtype* b, Dtype* y);

template <typename Dtype>
void caffe_gpu_sub(const int N, const Dtype* a, const Dtype* b, Dtype* y);

template <typename Dtype>
void caffe_gpu_mul(const int N, const Dtype* a, const Dtype* b, Dtype* y);

template <typename Dtype>
void caffe_gpu_div(const int N, const Dtype* a, const Dtype* b, Dtype* y);

template <typename Dtype>
void caffe_gpu_abs(const int n, const Dtype* a, Dtype* y);

template <typename Dtype>
void caffe_gpu_exp(const int n, const Dtype* a, Dtype* y);

template <typename Dtype>
void caffe_gpu_log(const int n, const Dtype* a, Dtype* y);

template <typename Dtype>
void caffe_gpu_powx(const int n, const Dtype* a, const Dtype b, Dtype* y);

template <typename Dtype>
void caffe_gpu_sqrt(const int n, const Dtype* a, Dtype* y);

// caffe_gpu_rng_uniform with two arguments generates integers in the range
// [0, UINT_MAX].
void caffe_gpu_rng_uniform(const int n, unsigned int* r);

// caffe_gpu_rng_uniform with four arguments generates floats in the range
// (a, b] (strictly greater than a, less than or equal to b) due to the
// specification of curandGenerateUniform.  With a = 0, b = 1, just calls
// curandGenerateUniform; with other limits will shift and scale the outputs
// appropriately after calling curandGenerateUniform.
template <typename Dtype>
void caffe_gpu_rng_uniform(const int n, const Dtype a, const Dtype b, Dtype* r);

template <typename Dtype>
void caffe_gpu_rng_gaussian(const int n, const Dtype mu, const Dtype sigma,
                            Dtype* r);

template <typename Dtype>
void caffe_gpu_rng_bernoulli(const int n, const Dtype p, int* r);

template <typename Dtype>
void caffe_gpu_dot(const int n, const Dtype* x, const Dtype* y, Dtype* out);

template <typename Dtype>
void caffe_gpu_asum(const int n, const Dtype* x, Dtype* y);

template<typename Dtype>
void caffe_gpu_sign(const int n, const Dtype* x, Dtype* y);

template<typename Dtype>
void caffe_gpu_sgnbit(const int n, const Dtype* x, Dtype* y);

template <typename Dtype>
void caffe_gpu_fabs(const int n, const Dtype* x, Dtype* y);

template <typename Dtype>
void caffe_gpu_scale(const int n, const Dtype alpha, const Dtype *x, Dtype* y);

#define DEFINE_AND_INSTANTIATE_GPU_UNARY_FUNC(name, operation) \
template<typename Dtype> \
__global__ void name##_kernel(const int n, const Dtype* x, Dtype* y) { \
  CUDA_KERNEL_LOOP(index, n) { \
    operation; \
  } \
} \
template <> \
void caffe_gpu_##name<float>(const int n, const float* x, float* y) { \
  /* NOLINT_NEXT_LINE(whitespace/operators) */ \
  name##_kernel<float><<<CAFFE_GET_BLOCKS(n), CAFFE_CUDA_NUM_THREADS>>>( \
      n, x, y); \
} \
template <> \
void caffe_gpu_##name<double>(const int n, const double* x, double* y) { \
  /* NOLINT_NEXT_LINE(whitespace/operators) */ \
  name##_kernel<double><<<CAFFE_GET_BLOCKS(n), CAFFE_CUDA_NUM_THREADS>>>( \
      n, x, y); \
}

#endif  // !CPU_ONLY

}  // namespace caffe

#endif  // CAFFE_UTIL_MATH_FUNCTIONS_H_
