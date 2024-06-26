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

#include "scann/utils/threads.h"

#include <algorithm>
#include <functional>
#include <string>
#include <utility>

#include "absl/time/time.h"

namespace research_scann {

unique_ptr<ThreadPool> StartThreadPool(const std::string& pool_name,
                                       ssize_t num_threads) {
  if (num_threads <= 0) {
    return nullptr;
  }

#if USE_TSL_THREADPOOL
  tensorflow::ThreadOptions options;
  options.stack_size = 1048576;
  auto pool = make_unique<ThreadPool>(tensorflow::Env::Default(), options,
                                      pool_name, num_threads);
#else // create a BS ThreadPool
  auto pool = make_unique<ThreadPool>(num_threads);
#endif

  return pool;
}

}  // namespace research_scann
