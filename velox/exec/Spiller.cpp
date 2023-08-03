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

#include "velox/exec/Spiller.h"

#include "velox/common/base/AsyncSource.h"

#include <folly/ScopeGuard.h>
#include "velox/common/testutil/TestValue.h"

using facebook::velox::common::testutil::TestValue;

namespace facebook::velox::exec {

constexpr int kLogEveryN = 32;

Spiller::Spiller(
    Type type,
    RowContainer& container,
    RowContainer::Eraser eraser,
    RowTypePtr rowType,
    int32_t numSortingKeys,
    const std::vector<CompareFlags>& sortCompareFlags,
    const std::string& path,
    int64_t targetFileSize,
    memory::MemoryPool& pool,
    folly::Executor* executor)
    : Spiller(
          type,
          container,
          eraser,
          std::move(rowType),
          HashBitRange{},
          numSortingKeys,
          sortCompareFlags,
          path,
          targetFileSize,
          pool,
          executor) {}

Spiller::Spiller(
    Type type,
    RowContainer& container,
    RowContainer::Eraser eraser,
    RowTypePtr rowType,
    HashBitRange bits,
    int32_t numSortingKeys,
    const std::vector<CompareFlags>& sortCompareFlags,
    const std::string& path,
    int64_t targetFileSize,
    memory::MemoryPool& pool,
    folly::Executor* executor)
    : type_(type),
      container_(container),
      eraser_(eraser),
      bits_(bits),
      rowType_(std::move(rowType)),
      state_(
          path,
          bits.numPartitions(),
          numSortingKeys,
          sortCompareFlags,
          targetFileSize,
          pool,
          spillMappedMemory()),
      pool_(pool),
      executor_(executor) {
  // TODO: add to support kHashJoin type later.
  VELOX_CHECK_NE(
      type_,
      Type::kHashJoin,
      "Spiller type:{} is not supported yet",
      typeName(type_));
  /// kOrderBy spiller type must only have one partition.
  // todo: order by spill只有一个Partition
  VELOX_CHECK((type_ != Type::kOrderBy) || (state_.maxPartitions() == 1));
  // 为每一个Partition开辟一个SpillRun，Partition和spillRun一一对应
  spillRuns_.reserve(state_.maxPartitions());
  for (int i = 0; i < state_.maxPartitions(); ++i) {
    spillRuns_.emplace_back(spillMappedMemory());
  }
}

void Spiller::extractSpill(folly::Range<char**> rows, RowVectorPtr& resultPtr) {
  if (!resultPtr) {
    resultPtr =
        BaseVector::create<RowVector>(rowType_, rows.size(), &spillPool());
  } else {
    resultPtr->prepareForReuse();
    resultPtr->resize(rows.size());
  }
  auto result = resultPtr.get();
  auto& types = container_.columnTypes();
  for (auto i = 0; i < types.size(); ++i) {
    container_.extractColumn(rows.data(), rows.size(), i, result->childAt(i));
  }
  auto& aggregates = container_.aggregates();
  auto numKeys = types.size();
  for (auto i = 0; i < aggregates.size(); ++i) {
    aggregates[i]->finalize(rows.data(), rows.size());
    aggregates[i]->extractAccumulators(
        rows.data(), rows.size(), &result->childAt(i + numKeys));
  }
}

int64_t Spiller::extractSpillVector(
    SpillRows& rows,
    int32_t maxRows,
    int64_t maxBytes,
    RowVectorPtr& spillVector,
    size_t& nextBatchIndex) {
  auto limit = std::min<size_t>(rows.size() - nextBatchIndex, maxRows);
  assert(!rows.empty());
  int32_t numRows = 0;
  int64_t bytes = 0;
  // 统计需要Spill的行数
  for (; numRows < limit; ++numRows) {
    // 计算对应行的size，包括定长和变长
    bytes += container_.rowSize(rows[nextBatchIndex + numRows]);
    if (bytes > maxBytes) {
      // Increment because the row that went over the limit is part
      // of the result. We must spill at least one row.
      ++numRows;
      break;
    }
  }
  extractSpill(folly::Range(&rows[nextBatchIndex], numRows), spillVector);
  nextBatchIndex += numRows;
  return bytes;
}

namespace {
// A stream of ordered rows being read from the in memory
// container. This is the part of a spillable range that is not yet
// spilled when starting to produce output. This is only used for
// sorted spills since for hash join spilling we just use the data in
// the RowContainer as is.
class RowContainerSpillStream : public SpillStream {
 public:
  RowContainerSpillStream(
      RowTypePtr type,
      int32_t numSortingKeys,
      const std::vector<CompareFlags>& sortCompareFlags,
      memory::MemoryPool& pool,
      Spiller::SpillRows&& rows,
      Spiller& spiller)
      : SpillStream(std::move(type), numSortingKeys, sortCompareFlags, pool),
        rows_(std::move(rows)),
        spiller_(spiller) {
    if (!rows_.empty()) {
      nextBatch();
    }
  }

  uint64_t size() const override {
    // 0 means that 'this' does not own spilled data in files.
    return 0;
  }

 private:
  void nextBatch() override {
    // Extracts up to 64 rows at a time. Small batch size because may
    // have wide data and no advantage in large size for narrow data
    // since this is all processed row by row.
    static constexpr vector_size_t kMaxRows = 64;
    constexpr uint64_t kMaxBytes = 1 << 18;
    if (nextBatchIndex_ >= rows_.size()) {
      index_ = 0;
      size_ = 0;
      return;
    }
    spiller_.extractSpillVector(
        rows_, kMaxRows, kMaxBytes, rowVector_, nextBatchIndex_);
    size_ = rowVector_->size();
    index_ = 0;
  }

  Spiller::SpillRows rows_;
  Spiller& spiller_;
  size_t nextBatchIndex_ = 0;
};
} // namespace

std::unique_ptr<SpillStream> Spiller::spillStreamOverRows(int32_t partition) {
  VELOX_CHECK(spillFinalized_);
  VELOX_CHECK_LT(partition, state_.maxPartitions());

  if (!state_.isPartitionSpilled(partition)) {
    return nullptr;
  }
  ensureSorted(spillRuns_[partition]);
  return std::make_unique<RowContainerSpillStream>(
      rowType_,
      container_.keyTypes().size(),
      state_.sortCompareFlags(),
      pool_,
      std::move(spillRuns_[partition].rows),
      *this);
}

void Spiller::ensureSorted(SpillRun& run) {
  // The spill data of a hash join doesn't need to be sorted.
  // kAgg和kSort都需要排序
  if (!run.sorted && type_ != Type::kHashJoin) {
    std::sort(
        run.rows.begin(),
        run.rows.end(),
        [&](const char* left, const char* right) {
          return container_.compareRows(
                     left, right, state_.sortCompareFlags()) < 0;
        });
    run.sorted = true;
  }
}

std::unique_ptr<Spiller::SpillStatus> Spiller::writeSpill(
    int32_t partition,
    uint64_t maxBytes) {
  VELOX_CHECK_EQ(pendingSpillPartitions_.count(partition), 1);
  // Target size of a single vector of spilled content. One of
  // these will be materialized at a time for each stream of the
  // merge.
  constexpr int32_t kTargetBatchBytes = 1 << 18; // 256K

  RowVectorPtr spillVector;
  auto& run = spillRuns_[partition];
  try {
    // OrderBy和Agg Spill需要对Partition中的所有数据排序
    // TODO：？对于Sort而言，只需要SpillFile有序即可，而不是整个Partition有序
    ensureSorted(run);
    int64_t totalBytes = 0;
    size_t written = 0;
    while (written < run.rows.size()) {
      // 抽取spillRun中的数据到spillVector中
      /// 64行一批，每一批大小不超过kTargetBatchBytes
      totalBytes += extractSpillVector(
          run.rows, 64, kTargetBatchBytes, spillVector, written);
      // todo 写入文件，落盘的主要逻辑在SpillFileLists中
      state_.appendToPartition(partition, spillVector);
      if (totalBytes > maxBytes) {
        break;
      }
    }
    return std::make_unique<SpillStatus>(partition, written, nullptr);
  } catch (const std::exception& e) {
    // The exception is passed to the caller thread which checks this in
    // advanceSpill().
    return std::make_unique<SpillStatus>(
        partition, 0, std::current_exception());
  }
}

void Spiller::advanceSpill(uint64_t maxBytes) {
  // AsyncSource是velox封装的异步库
  std::vector<std::shared_ptr<AsyncSource<SpillStatus>>> writes;
  for (auto partition = 0; partition < spillRuns_.size(); ++partition) {
    if (pendingSpillPartitions_.count(partition) == 0) {
      continue;
    }
    writes.push_back(std::make_shared<AsyncSource<SpillStatus>>(
        [partition, this, maxBytes]() {
          // 真正写数据的逻辑
          return writeSpill(partition, maxBytes);
        }));
    // todo：如果executor_存在，就把writes.back()放到executor_中执行，但也可能不存在，此时的行为是什么？
    //  -> AsyncSource的功能之一：如果executor没有生成数据，在获取的时候自己生成，也就是说如果没有executor，在当前线程writeSpill
    // todo：这里只处理了writes.back()，前面的writes呢？ -> 此时在for循环内部，所以每一个Write都会被放到executor_中执行
    // 如果executor_（线程池）存在，每一个write都会被放到executor_中执行，由线程池执行writeSpill逻辑
    // 如果executor_不存在，AsyncSource会在当前线程获取结果时，由当前线程执行writeSpill逻辑
    if (executor_) {
      // 捕获writes.back()，并命名成新变量名source
      executor_->add([source = writes.back()]() { source->prepare(); });
    }
  }
  /// makeGuard是folly提供的RAII工具类，当离开作用域时，会自动调用makeGuard中的lambda表达式
  auto sync = folly::makeGuard([&]() {
    for (auto& write : writes) {
      // We consume the result for the pending writes. This is a
      // cleanup in the guard and must not throw. The first error is
      // already captured before this runs.
      try {
        // 只是清空write，并不获取数据，比如出现异常
        write->move();
      } catch (const std::exception& e) {
        // 如果清空过程中发生异常，直接忽略
      }
    }
  });

  /// 完成统计和清理工作
  for (auto& write : writes) {
    // todo: 这里可能阻塞，等待writes的落盘工作完成
    const auto result = write->move();

    if (result->error) {
      // 抛出异常，后面代码不再继续执行，但是makeGuard的代码是会继续执行的
      std::rethrow_exception(result->error);
    }
    auto numWritten = result->rowsWritten;
    spilledRows_ += numWritten;
    auto partition = result->partition;
    auto& run = spillRuns_[partition];
    auto spilled = folly::Range<char**>(run.rows.data(), numWritten);
    // 删除contain_中已经Spill的数据
    eraser_(spilled);
    if (!container_.numRows()) {
      // If the container became empty, free its memory.
      container_.clear();
    }
    // 删除SpillRun中的数据
    run.rows.erase(run.rows.begin(), run.rows.begin() + numWritten);
    if (run.rows.empty()) {
      // Run ends, start with a new file next time.
      run.clear();
      // flush磁盘
      state_.finishWrite(partition);
      pendingSpillPartitions_.erase(partition);
    }
  }
}

void Spiller::spill(uint64_t targetRows, uint64_t targetBytes) {
  // 算子调用noMoreInput后spillFinalized_会被设置成true
  // spill只会在addInput的时候触发，在noMoreInput之后一定不会spill
  VELOX_CHECK(!spillFinalized_);
  bool hasFilledRuns = false;
  for (;;) {
    // 获取内存中剩余的行数和Varchar内存
    auto rowsLeft = container_.numRows();
    auto spaceLeft = container_.stringAllocator().retainedSize() -
        container_.stringAllocator().freeSpace();
    // 每次spill后都会判断是否满足要求：1. container_数据Spill完 2. 内存达到目标，行数达到目标，Varchar内存达到目标
    if (rowsLeft == 0 || (rowsLeft <= targetRows && spaceLeft <= targetBytes)) {
      break;
    }
    // 首先，制定spill策略，需要spill哪些Partition，pendingSpillPartitions_维护要罗盘的Partition
    // 然后，调用advanceSpill进行落盘
    if (!pendingSpillPartitions_.empty()) {
      advanceSpill(state_.targetFileSize());
      // spill一个targetFileSize后，进入下一次循环，在下一个循环判断是否满足内存要求，满足内存要求就停止spill
      continue;
    }

    // 走到这里，说明还没有制定spill策略，或者已经按照之前的spill策略spill完了，需要制定spill策略

    // 将container_中的数据计算Hash分区，然后放到对应的SpillRun中
    // 每次spill()会填充一次
    // SpillRun是复用的，采用Lazy clear的方式，如果SpillRun中有上次的spill残留，清空
    if (!hasFilledRuns) {
      fillSpillRuns();
      hasFilledRuns = true;
    }

    // 制定spill策略
    while (rowsLeft > 0 && (rowsLeft > targetRows || spaceLeft > targetBytes)) {
      const int32_t partition = pickNextPartitionToSpill();
      if (partition == -1) {
        VELOX_FAIL(
            "No partition has no spillable data but still doesn't reach the spill target, target rows {}, target bytes {}, rows left {}, bytes left {}",
            targetRows,
            targetBytes,
            rowsLeft,
            spaceLeft);
        break;
      }
      if (!state_.isPartitionSpilled(partition)) {
        state_.setPartitionSpilled(partition);
      }
      VELOX_DCHECK_EQ(pendingSpillPartitions_.count(partition), 0);
      pendingSpillPartitions_.insert(partition);
      rowsLeft =
          std::max<int64_t>(0, rowsLeft - spillRuns_[partition].rows.size());
      spaceLeft =
          std::max<int64_t>(0, spaceLeft - spillRuns_[partition].numBytes);
    }
    // Quit this spill run if we have spilled all the partitions.
    if (pendingSpillPartitions_.empty()) {
      // 前面的while循环刚刚制定完Spill策略，但是并没有pendingSpillPartitions_，说明所有的Partition都Spill完了，退出for(;;)循环
      LOG_EVERY_N(WARNING, kLogEveryN)
          << spaceLeft << " bytes and " << rowsLeft
          << " rows left after spilled all partitions, spiller: " << toString();
      break;
    }
  }

  // Clear the non-spilling runs on exit.
  // 制定完spill策略后，并不是所有partition都需要spill，删除不需要的spill
  // todo: 看是否可以改进，在fillSpillRun的时候，只fill需要spill的Partition，而不是fill所有，此处再来删除
  clearNonSpillingRuns();
}

int32_t Spiller::pickNextPartitionToSpill() {
  VELOX_DCHECK_EQ(spillRuns_.size(), state_.maxPartitions());

  // Sort the partitions based on spiller type to pick.
  std::vector<int32_t> partitionIndices(spillRuns_.size());
  std::iota(partitionIndices.begin(), partitionIndices.end(), 0);
  // todo: 这里有重复排序的可能（虽然排序的开销相对较小），如果spill一个Partition后还不满足要求，则会再次调用pickNextPartitionToSpill，这里会再次排序
  std::sort(
      partitionIndices.begin(),
      partitionIndices.end(),
      [&](int32_t lhs, int32_t rhs) {
        // For non kHashJoin type, we always try to spill from the partition
        // with more spillable data first no matter it has spilled or not. For
        // kHashJoin, we will try to stick with the spilling partitions if they
        // have spillable data.
        //
        // NOTE: the picker loop below will skip the spilled partition which has
        // no spillable data.
        if (type_ == Type::kHashJoin &&
            state_.isPartitionSpilled(lhs) != state_.isPartitionSpilled(rhs)) {
          return state_.isPartitionSpilled(lhs);
        }
        return spillRuns_[lhs].numBytes > spillRuns_[rhs].numBytes;
      });
  for (auto partition : partitionIndices) {
    // todo: ? 只有pendingSpillPartitions_ empty才会走到这里
    if (pendingSpillPartitions_.count(partition) != 0) {
      continue;
    }
    // 为什么不是break？ 在kHashJoin的场景，已经spilled的优先级更高，但spilled的Partition的numBytes == 0，还需要spill后面的Partition
    if (spillRuns_[partition].numBytes == 0) {
      continue;
    }
    return partition;
  }
  return -1;
}

Spiller::SpillRows Spiller::finishSpill() {
  VELOX_CHECK(!spillFinalized_);
  spillFinalized_ = true;
  SpillRows rowsFromNonSpillingPartitions(
      0, memory::StlMappedMemoryAllocator<char*>(&spillMappedMemory()));
  fillSpillRuns(&rowsFromNonSpillingPartitions);
  return rowsFromNonSpillingPartitions;
}

void Spiller::clearSpillRuns() {
  for (auto& run : spillRuns_) {
    run.clear();
  }
}

void Spiller::fillSpillRuns(SpillRows* rowsFromNonSpillingPartitions) {
  // 每次spill()会填充一次，spillRuns_是复用的，所以需要清空上一次的残留
  // 每一次的spill都是对container_做全量的spillRun填充，这意味着这次残留在spillRuns_中的数据，下一次spill还会被再次填充到spillRuns_中,
  // 这样做的缺点是有冗余操作，但好处是每次清空状态，不用区分container_中哪些是上次残留的，哪些是新增的，简化代码
  clearSpillRuns();

  RowContainerIterator iterator;
  // Number of rows to hash and divide into spill partitions at a time.
  /// todo？ Sort并不需要计算Hash，因为就一个Partition
  // 批量计算Hash值
  constexpr int32_t kHashBatchSize = 4096;
  // 每一个元素都会被初始化成0
  std::vector<uint64_t> hashes(kHashBatchSize);
  std::vector<char*> rows(kHashBatchSize);
  for (;;) {
    // 抽数据到rows
    auto numRows = container_.listRows(
        &iterator, rows.size(), RowContainer::kUnlimited, rows.data());
    // Calculate hashes for this batch of spill candidates.
    auto rowSet = folly::Range<char**>(rows.data(), numRows);
    for (auto i = 0; i < container_.keyTypes().size(); ++i) {
      // 计算rowSet中对应行的hash值，存储到hashes.data()中，如果mix为true，会合并计算出
      // 的hash和result存储的数据，这里hashes.data()恒为0，所以没有意义
      container_.hash(i, rowSet, i > 0, hashes.data());
    }

    // Put each in its run.
    for (auto i = 0; i < numRows; ++i) {
      // TODO: consider to cache the hash bits in row container so we only need
      // to calculate them once.
      /// 对于OrderBy，并不不需要计算Hash，因为只有一个Partition
      const auto partition = (type_ == Type::kOrderBy)
          ? 0
          : bits_.partition(hashes[i], state_.maxPartitions());
      VELOX_DCHECK_GE(partition, 0);
      // If 'rowsFromNonSpillingPartitions' is not null, it is used to collect
      // the rows from non-spilling partitions when finishes spilling.
      // rowsFromNonSpillingPartitions只在finishSpill中传入，其他地方传入的都是nullptr
      // 在算子结束整个spill的时候，一次落盘都没有的Partition，不再落盘，维护在rowsFromNonSpillingPartitions（内存）中，
      // Partition在结束时没有落盘的原因在于占用的内存太小，没有必要罗盘成大量的小文件
      if (FOLLY_UNLIKELY(
              rowsFromNonSpillingPartitions != nullptr &&
              !state_.isPartitionSpilled(partition))) {
        rowsFromNonSpillingPartitions->push_back(rows[i]);
        continue;
      }
      spillRuns_[partition].rows.push_back(rows[i]);
      spillRuns_[partition].numBytes += container_.rowSize(rows[i]);
    }
    if (numRows == 0) {
      break;
    }
  }
}

void Spiller::clearNonSpillingRuns() {
  for (auto partition = 0; partition < spillRuns_.size(); ++partition) {
    // todo? 不需要这个判断？pendingSpillPartitions_.count(partition) == 0恒为true X
    // todo ?这里是一个改进点，在fillRun的时候只fill需要spill的Partition，不需要spill的Partition不需要fill
    if (pendingSpillPartitions_.count(partition) == 0) {
      spillRuns_[partition].clear();
    }
  }
}

std::string Spiller::toString() const {
  return fmt::format(
      "Type{}\tRowType:{}\tNum Partitions:{}\tFinalized:{}",
      typeName(type_),
      rowType_->toString(),
      state_.maxPartitions(),
      spillFinalized_);
}

// static
std::string Spiller::typeName(Type type) {
  switch (type) {
    case Type::kOrderBy:
      return "ORDER_BY";
    case Type::kHashJoin:
      return "HASH_JOIN";
    case Type::kAggregate:
      return "AGGREGATE";
    default:
      VELOX_UNREACHABLE("Unknown type: {}", static_cast<int>(type));
      return fmt::format("UNKNOWN TYPE: {}", static_cast<int>(type));
  }
}

// static
memory::MappedMemory& Spiller::spillMappedMemory() {
  // Return the top level instance. Since this too may be full,
  // another possibility is to return an emergency instance that
  // delegates to the process wide one and makes a file-backed mmap
  // if the allocation fails.
  return *memory::MappedMemory::getInstance();
}

// static
memory::MemoryPool& Spiller::spillPool() {
  static auto pool = memory::getDefaultScopedMemoryPool();
  return *pool;
}

} // namespace facebook::velox::exec
