// Copyright 2022 The Google Research Authors.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#ifndef SCANN_UTILS_PARALLEL_FOR_H_
#define SCANN_UTILS_PARALLEL_FOR_H_

#include <algorithm>
#include <atomic>
#include <cstdint>
#include <cstdlib>
#include <functional>
#include <limits>
#include <type_traits>
#include <utility>

#include "absl/base/thread_annotations.h"
#include "absl/functional/function_ref.h"
#include "absl/base/internal/spinlock.h"
#include "absl/synchronization/mutex.h"
#include "scann/oss_wrappers/scann_threadpool.h"
#include "scann/utils/types.h"
#include "tensorflow/core/lib/core/status.h"
#include "thread-pool/BS_thread_pool.hpp"
#include "SearchIndex/SearchIndexCommon.h"

namespace research_scann {

enum : size_t {
  kDynamicBatchSize = numeric_limits<size_t>::max(),
};

struct ParallelForOptions {
  size_t max_parallelism = numeric_limits<size_t>::max();
};

template <size_t kItersPerBatch = kDynamicBatchSize, typename SeqT,
          typename Function>
SCANN_INLINE void ParallelFor(SeqT seq, ThreadPool* pool, Function func,
                              ParallelForOptions opts = ParallelForOptions());

template <size_t kItersPerBatch = kDynamicBatchSize, typename SeqT,
          typename Function>
SCANN_INLINE Status
ParallelForWithStatus(SeqT seq, ThreadPool* pool, Function Func,
                      ParallelForOptions opts = ParallelForOptions()) {
  Status finite_check_status = OkStatus();

  std::atomic_bool is_ok_status{true};
  absl::Mutex mutex;
  ParallelFor(
      seq, pool,
      [&](size_t idx) {
        if (!is_ok_status.load(std::memory_order_relaxed)) {
          return;
        }
        Status status = Func(idx);
        if (!status.ok()) {
          absl::MutexLock lock(&mutex);
          finite_check_status = status;
          is_ok_status.store(false, std::memory_order_relaxed);
        }
      },
      opts);
  return finite_check_status;
}

struct ThreadPoolLevel {
  static int increment() { return ++level; }

  static int decrement() { return --level; }

  static int get() { return level; }

private:
  inline static thread_local int level = 0;
};

namespace parallel_for_internal {

template <size_t kItersPerBatch, typename SeqT, typename Function>
class TSLParallelForClosure : public std::function<void()> {
 public:
  static constexpr bool kIsDynamicBatch = (kItersPerBatch == kDynamicBatchSize);
  TSLParallelForClosure(SeqT seq, Function func)
      : func_(func),
        index_(*seq.begin()),
        range_end_(*seq.end()),
        reference_count_(1) {}

  SCANN_INLINE void RunParallel(ThreadPool* pool, size_t desired_threads) {
    DCHECK(pool);

    size_t n_threads =
        std::min<size_t>(desired_threads - 1, pool->get_thread_count());

    if (kIsDynamicBatch) {
      batch_size_ =
          SeqT::Stride() * std::max(1ul, desired_threads / 4 / n_threads);
    }

    reference_count_ += n_threads;
    while (n_threads--) {
      VLOG(1) << "ParallelFor schedule n_threads=" << n_threads;
      pool->submit([this]() { Run(); });
    }

    VLOG(1) << "Thread-" << std::this_thread::get_id() << " DoWork() start";
    DoWork();
    // main thread job done
    std::unique_lock<std::mutex> lock(mutex_);
    --reference_count_;
    // wait for all jobs to finish
    cv_.wait(lock, [this]() { return reference_count_ == 0; });
    VLOG(1) << "Thread-" << std::this_thread::get_id() << " DoWork() finish";
  }

  void Run() {
    VLOG(1) << "Thread-" << std::this_thread::get_id() << " Run() start";
    DoWork();
    // mark job as done
    // std::unique_lock<std::mutex> lock(mutex_);
    --reference_count_;
    cv_.notify_one();
    VLOG(1) << "Thread-" << std::this_thread::get_id() << " Run() finish";
  }

  SCANN_INLINE void DoWork() {
    const size_t range_end = range_end_;

    constexpr size_t kStaticBatchSize = SeqT::Stride() * kItersPerBatch;
    const size_t batch_size = kIsDynamicBatch ? batch_size_ : kStaticBatchSize;
    DCHECK_NE(batch_size, kDynamicBatchSize);
    DCHECK_EQ(batch_size % SeqT::Stride(), 0);

    for (;;) {
      const size_t batch_begin = index_.fetch_add(batch_size);
      VLOG(1) << "Thread-" << std::this_thread::get_id()
              << " starting batch at " << batch_begin << " with size "
              << batch_size << " and range end " << range_end;

      auto t0 = std::chrono::high_resolution_clock::now();
      const size_t batch_end = std::min(batch_begin + batch_size, range_end);
      if (ABSL_PREDICT_FALSE(batch_begin >= range_end)) {
        VLOG(1) << "Thread-" << std::this_thread::get_id() << " exit from DoWork()";
        break;
      }
      for (size_t idx : SeqWithStride<SeqT::Stride()>(batch_begin, batch_end)) {
        func_(idx);
      }
      auto t1 = std::chrono::high_resolution_clock::now();
      auto time_ms = std::chrono::duration_cast<std::chrono::milliseconds>(t1 - t0).count();

      VLOG(1) << "Thread-" << std::this_thread::get_id()
        << " finishing batch at " << batch_begin << " with size "
        << batch_size << " and range end " << range_end
        << " duration=" << time_ms << "ms";
    }
  }

 private:
  Function func_;

  std::atomic<size_t> index_;

  const size_t range_end_;

  std::mutex mutex_;

  std::condition_variable cv_;

  std::atomic<uint32_t> reference_count_;

  size_t batch_size_ = kItersPerBatch;
};

template <size_t kItersPerBatch, typename SeqT, typename Function>
class BSParallelForClosure : public std::function<void()> {
 public:
  static constexpr bool kIsDynamicBatch = (kItersPerBatch == kDynamicBatchSize);
  BSParallelForClosure(SeqT seq, Function func)
      : func_(func),
        index_(*seq.begin()),
        range_end_(*seq.end()) {}

  SCANN_INLINE void RunParallel(BS::thread_pool* pool, size_t desired_threads) {
    VLOG(1) << "Thread-" << std::this_thread::get_id() << " RunParallel() start";
    DCHECK(pool);

    size_t n_threads =
        std::min<size_t>(desired_threads - 1, pool->get_thread_count());

    if (kIsDynamicBatch) {
      batch_size_ =
          SeqT::Stride() * std::max(1ul, desired_threads / 4 / n_threads);
    }


    std::vector<std::future<void>> futures;
    while (n_threads--) {
      VLOG(1) << "ParallelFor schedule n_threads=" << n_threads;
      futures.push_back(pool->submit([this]() { Run(); }));
    }

    VLOG(1) << "Thread-" << std::this_thread::get_id() << " DoWork() start";
    DoWork();
    VLOG(1) << "Thread-" << std::this_thread::get_id() << " DoWork() finish";
    for (auto& f: futures) f.wait();
    VLOG(1) << "Thread-" << std::this_thread::get_id() << " RunParallel() finish";

    SIConfiguration::currentThreadCheckAndAbort();
  }

  void Run() {
    VLOG(1) << "Thread-" << std::this_thread::get_id() << " Run() start";
    DoWork();
    VLOG(1) << "Thread-" << std::this_thread::get_id() << " Run() finish";
  }

  SCANN_INLINE void DoWork() {
    CHECK_EQ(ThreadPoolLevel::increment(), 1) << "Nested BSParallelForClosure not supported";
    Search::OnExit on_exit([]() { ThreadPoolLevel::decrement(); });

    const size_t range_end = range_end_;

    constexpr size_t kStaticBatchSize = SeqT::Stride() * kItersPerBatch;
    const size_t batch_size = kIsDynamicBatch ? batch_size_ : kStaticBatchSize;
    DCHECK_NE(batch_size, kDynamicBatchSize);
    DCHECK_EQ(batch_size % SeqT::Stride(), 0);

    for (;;) {
      // handle index abort, only in the outmost level
      try {
        SIConfiguration::currentThreadCheckAndAbort();
      }
      catch(const std::exception& e) {
        // check index abort in main thread
        LOG(ERROR) << "Exception in thread " << std::this_thread::get_id()
                   << ": " << e.what() << "." << " Aborting now.";
        // set aborting_ flag for other threads
        aborting_ = true;
        break;
      }
      catch(...) {
        LOG(ERROR) << "Unknown exception in thread " << std::this_thread::get_id();
        break;
      }

      if (aborting_) {
        LOG(ERROR) << "Abort volunatarily in thread " << std::this_thread::get_id();
        break;
      }

      const size_t batch_begin = index_.fetch_add(batch_size);
      VLOG(1) << "Thread-" << std::this_thread::get_id()
              << " starting batch at " << batch_begin << " with size "
              << batch_size << " and range end " << range_end;

      auto t0 = std::chrono::high_resolution_clock::now();
      const size_t batch_end = std::min(batch_begin + batch_size, range_end);
      if (ABSL_PREDICT_FALSE(batch_begin >= range_end)) {
        VLOG(1) << "Thread-" << std::this_thread::get_id() << " exit from DoWork()";
        break;
      }
      for (size_t idx : SeqWithStride<SeqT::Stride()>(batch_begin, batch_end)) {
        func_(idx);
      }
      auto t1 = std::chrono::high_resolution_clock::now();
      auto time_ms = std::chrono::duration_cast<std::chrono::milliseconds>(t1 - t0).count();

      VLOG(1) << "Thread-" << std::this_thread::get_id()
        << " finishing batch at " << batch_begin << " with size "
        << batch_size << " and range end " << range_end
        << " duration=" << time_ms << "ms";
    }
  }

 private:
  Function func_;

  std::atomic<size_t> index_;

  const size_t range_end_;

  size_t batch_size_ = kItersPerBatch;

  std::atomic<bool> aborting_ = false;
};

template <size_t kItersPerBatch, typename SeqT, typename Function>
using ParallelForClosure = BSParallelForClosure<kItersPerBatch, SeqT, Function>;

}  // namespace parallel_for_internal

template <size_t kItersPerBatch, typename SeqT, typename Function>
SCANN_INLINE void ParallelFor(SeqT seq, ThreadPool* pool, Function func,
                              ParallelForOptions opts) {
  // only check & abort in the main thread and in the outermost call
  if (ThreadPoolLevel::get() == 0)
    SIConfiguration::currentThreadCheckAndAbort();

  constexpr size_t kMinItersPerBatch =
      kItersPerBatch == kDynamicBatchSize ? 1 : kItersPerBatch;
  const size_t desired_threads = std::min(
      opts.max_parallelism, DivRoundUp(*seq.end() - *seq.begin(),
                                       SeqT::Stride() * kMinItersPerBatch));
  // disable the thread pool if it's invoked recursively
  if (ThreadPoolLevel::get()) pool = nullptr;

  if (!pool || desired_threads <= 1) {
    for (size_t idx : seq) {
      func(idx);
    }
    return;
  }

  using parallel_for_internal::ParallelForClosure;
  auto closure =
      new ParallelForClosure<kItersPerBatch, SeqT, Function>(seq, func);
  closure->RunParallel(pool, desired_threads);
  delete closure;
}

struct QueryThreadPool {
  static void setTotalQueryThreads(int n) {
    std::lock_guard<std::mutex> lock(pool_mutex);
    if (pool_initialized) {
      LOG(WARNING) << "Thread pool has already been initialized, ignore setting threads";
      return;
    }
    absl::base_internal::SpinLockHolder guard(&thread_spinlock);
    pool_threads = n;
    available_threads = n;
  }

  static size_t getThreads(int desired) {
    absl::base_internal::SpinLockHolder guard(&thread_spinlock);
    // pool not initialized yet
    if (!pool_initialized) return 0;

    // no need for separate threads
    if (desired <= 1 || available_threads <= 0) return 0;
    // use at most half of the remaining threads
    int max_threads_to_use = (available_threads + 1) / 2;
    // calculate the number of threads to use (including current thread)
    int n = std::min(desired, max_threads_to_use + 1);
    // deduct the available threads
    available_threads -= n - 1;
    return n;
  }

  static void releaseThreads(size_t n) {
    absl::base_internal::SpinLockHolder guard(&thread_spinlock);
    if (!pool_initialized) LOG(FATAL) << "thread pool not initialized";

    // release the used threads
    if (n >= 1) available_threads += n - 1;
  }

  static BS::thread_pool* get() {
    if (pool_initialized.load()) {
      return pool.get();
    }
    // no available threads
    if (available_threads <= 0) return nullptr;
    // initialize the pool if it's not done yet
    std::lock_guard<std::mutex> lock(pool_mutex);
    if (!pool_initialized.load()) {
      LOG(INFO) << "Initializing query thread pool with " << pool_threads << " threads";
      if (pool_threads <= 0) LOG(FATAL) << "pool threads must be a positive number";
      pool = make_unique<BS::thread_pool>(pool_threads);
      pool_initialized = true;
    }
    return pool.get();
  }

private:
  inline static std::unique_ptr<BS::thread_pool> pool;
  inline static std::atomic<bool> pool_initialized{false};
  inline static std::mutex pool_mutex;
  inline static absl::base_internal::SpinLock thread_spinlock;

  inline static int pool_threads = -1;
  inline static std::atomic<int> available_threads{-1};
  inline static const std::string pool_name = "QueryThreadPool";
};

template <size_t kItersPerBatch, typename SeqT, typename Function>
SCANN_INLINE void QueryParallelFor(SeqT seq, Function func) {
  constexpr size_t kMinItersPerBatch =
      kItersPerBatch == kDynamicBatchSize ? 1 : kItersPerBatch;
  const size_t desired_threads = DivRoundUp(
    *seq.end() - *seq.begin(), SeqT::Stride() * kMinItersPerBatch);
  auto pool = QueryThreadPool::get();
  size_t n = QueryThreadPool::getThreads(desired_threads);

  if (!pool || n <= 1) {
    for (size_t idx : seq) {
      func(idx);
    }
    return;
  }

  using parallel_for_internal::BSParallelForClosure;
  auto closure =
      new BSParallelForClosure<kItersPerBatch, SeqT, Function>(seq, func);
  closure->RunParallel(pool, n);
  // TODO delete closure explicitly would cause double free
  delete closure;
  QueryThreadPool::releaseThreads(n);
}

}  // namespace research_scann

#endif
