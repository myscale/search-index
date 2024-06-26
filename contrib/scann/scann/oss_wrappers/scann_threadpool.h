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

#ifndef SCANN_OSS_WRAPPERS_SCANN_THREADPOOL_H_
#define SCANN_OSS_WRAPPERS_SCANN_THREADPOOL_H_

#define USE_TSL_THREADPOOL 0

#if USE_TSL_THREADPOOL

#include "tensorflow/core/lib/core/threadpool.h"

namespace research_scann {

using ::tensorflow::thread::ThreadPool;

}

#else // use BS thread pool

#include <thread-pool/BS_thread_pool.hpp>

namespace research_scann {

using ThreadPool = BS::thread_pool;

}

#endif

#endif
