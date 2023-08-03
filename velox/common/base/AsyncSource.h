/*
 * Copyright (c) Facebook, Inc. and its affiliates.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#pragma once

#include <folly/Unit.h>
#include <folly/executors/QueuedImmediateExecutor.h>
#include <folly/futures/Future.h>
#include <functional>
#include <memory>

#include "velox/common/future/VeloxPromise.h"

namespace facebook::velox {

// A future-like object that prefabricates Items on an executor and
// allows consumer threads to pick items as they are ready. If the
// consumer needs the item before the executor started making it,
// the consumer will make it instead. If multiple consumers request
// the same item, exactly one gets it.
template <typename Item>
class AsyncSource {
 public:
  explicit AsyncSource(std::function<std::unique_ptr<Item>()> make)
      : make_(make) {}

  // Makes an item if it is not already made. To be called on a background
  // executor.
  void prepare() {
    std::function<std::unique_ptr<Item>()> make = nullptr;
    {
      // 加锁保护make_和making_
      std::lock_guard<std::mutex> l(mutex_);
      // 一个item只能由一个线程生成，如果已经确定一个线程生成item，将make_置成nullptr，防止其他线程用make重复生成item_
      if (!make_) {
        return;
      }
      // 指示已经在生成item
      making_ = true;
      // 将make_置成nullptr，防止其他线程用make重复生成item_
      std::swap(make, make_);
    }
    // 前面的逻辑已经确保了只有当前线程持有make，不会有竞争问题，不需要加锁
    // 其实上下文这么多代码都是为了将make逻辑从mutex_解放出来，make通常是一个很重的函数，如果在mutex_中执行，会有严重的竞争问题
    item_ = make();
    {
      std::lock_guard<std::mutex> l(mutex_);
      making_ = false;
      // promise_是生产者设置的，promise_存在，说明有人在等自己，setValue发出信号，通知消费者消费
      if (promise_) {
        promise_->setValue();
        promise_ = nullptr;
      }
    }
  }

  // Returns the item to the first caller and nullptr to subsequent callers. If
  // the item is preparing on the executor, waits for the item and otherwise
  // makes it on the caller thread.
  // 获取item
  std::unique_ptr<Item> move() {
    std::function<std::unique_ptr<Item>()> make = nullptr;
    ContinueFuture wait;
    {
      std::lock_guard<std::mutex> l(mutex_);
      // 如果已经生成了item，直接返回
      if (item_) {
        return std::move(item_);
      }
      // 已经有其他线程在等待item生成
      if (promise_) {
        // Somebody else is now waiting for the item to be made.
        return nullptr;
      }
      if (making_) {
        // 说明executor在正在通过prepare生成item，设置promise。executor生成完后，会使用promise通知自己
        promise_ = std::make_unique<ContinuePromise>();
        wait = promise_->getSemiFuture();
      } else {
        // 需要自己生成
        if (!make_) {
          return nullptr;
        }
        std::swap(make, make_);
      }
    }
    // Outside of mutex_.
    /// 这里不会有竞争问题，只会有一个线程拿到make，其他线程拿到的事swap后的nullptr
    /// 而拿到nullptr的线程直接return nullptr
    if (make) {
      // 这里是clangd识别不到，实际上可能走到这里
      // 当自己生生成item，执行make；如果是executor生成，make为空
      return make();
    }
    auto& exec = folly::QueuedImmediateExecutor::instance();
    // wait是一个semi，通过via指定executor后转变成Future
    // todo？为啥需要semi future而不直接使用future
    // 阻塞当前进程，直到wait完成。wait()和get()类似，只是不获取结果
    // todo？ 需要exec这个executor做什么？
    std::move(wait).via(&exec).wait();
    std::lock_guard<std::mutex> l(mutex_);
    return std::move(item_);
  }

  // If true, move() will not block. But there is no guarantee that somebody
  // else will not get the item first.
  bool hasValue() const {
    return item_ != nullptr;
  }

 private:
  std::mutex mutex_;
  // True if 'prepare() is making the item.
  bool making_{false};
  std::unique_ptr<ContinuePromise> promise_;
  std::unique_ptr<Item> item_;
  std::function<std::unique_ptr<Item>()> make_;
};
} // namespace facebook::velox
