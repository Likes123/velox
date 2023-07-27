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
#include <folly/executors/IOThreadPoolExecutor.h>
#include "velox/exec/tests/utils/RowContainerTestBase.h"

using namespace facebook::velox;
using namespace facebook::velox::exec;

class SpillerTest : public exec::test::RowContainerTestBase {
 protected:
  void testSpill(int32_t spillPct, bool makeError = false) {
    constexpr int32_t kNumRows = 100000;
    std::vector<char*> rows(kNumRows);
    RowVectorPtr batch;
    auto data = makeSpillData(kNumRows, rows, batch);
    // 每一行都有一个对应的hash值
    std::vector<uint64_t> hashes(kNumRows);
    auto keys = data->keyTypes();
    // Calculate a hash for every key in 'rows'.
    for (auto i = 0; i < keys.size(); ++i) {
      data->hash(
          i, folly::Range<char**>(rows.data(), kNumRows), i > 0, hashes.data());
    }

    // We divide the rows in 4 partitions according to 2 low bits of the hash.
    // 选用2的次幂，为的是可以直接&，计算hash值
    std::vector<std::vector<int32_t>> partitions(4);
    // 因为分成4个Partition，所以hash值的后两位只能是00,01,10,11，也就是&3
    for (auto i = 0; i < kNumRows; ++i) {
      // partition中存的是RowContainer中的行号
      partitions[hashes[i] & 3].push_back(i);
    }

    // 这里的Sort是为了后面验证Merge的正确与否，不是Spill的必要条件
    // We sort the rows in each partition in key order.
    for (auto& partition : partitions) {
      std::sort(
          partition.begin(),
          partition.end(),
          [&](int32_t leftIndex, int32_t rightIndex) {
            return data->compareRows(rows[leftIndex], rows[rightIndex]) < 0;
          });
    }

    auto tempDirectory = exec::test::TempDirectoryPath::create();
    // We spill 'data' in 4 sorted partitions.
    auto spiller = std::make_unique<Spiller>(
        *data,
        [&](folly::Range<char**> rows) { data->eraseRows(rows); },
        std::static_pointer_cast<const RowType>(batch->type()),
        HashBitRange(0, 2), // 暗示了分成4个Partition
        keys.size(),
        makeError ? "/bad/path" : tempDirectory->path,
        2000000,
        *pool_,
        executor());

    // We have a bit range of two bits , so up to 4 spilled partitions.
    EXPECT_EQ(4, spiller->state().maxPartitions());

    RowContainerIterator iter;

    // 一次spill 10%的数据，一共spill spillPct%的数据
    // We spill spillPct% of the data in 10% increments.
    auto initialBytes = data->allocatedBytes();
    auto initialRows = data->numRows();
    for (int32_t pct = 10; pct <= spillPct; pct += 10) {
      try {
        spiller->spill(
            initialRows - (initialRows * pct / 100),
            initialBytes - (initialBytes * pct / 100),
            iter);
        EXPECT_FALSE(makeError);
      } catch (const std::exception& e) {
        if (!makeError) {
          throw;
        }
        return;
      }
    }
    auto unspilledPartitionRows = spiller->finishSpill();
    if (spillPct == 100) {
      EXPECT_TRUE(unspilledPartitionRows.empty());
      EXPECT_EQ(0, data->numRows());
    }
    // We read back the spilled and not spilled data in each of the
    // partitions. We check that the data comes back in key order.
    for (auto partitionIndex = 0; partitionIndex < 4; ++partitionIndex) {
      if (!spiller->isSpilled(partitionIndex)) {
        continue;
      }

      // Spiller merge的作用，将一个Partition下的多个spill文件和RowContainer中的数据合并成一个有序的流
      // We make a merge reader that merges the spill files and the rows that
      // are still in the RowContainer.
      // merge最后一个Partition
      // merge是指merge一个Partition下的多个spillFile和RowContainer中的数据，不涉及Partition之间
      auto merge = spiller->startMerge(partitionIndex);

      // We read the spilled data back and check that it matches the sorted
      // order of the partition.
      // 取最后一个Partition
      auto& indices = partitions[partitionIndex];
      for (auto i = 0; i < indices.size(); ++i) {
        auto stream = merge->next();
        if (!stream) {
          FAIL() << "Stream ends after " << i << " entries";
          break;
        }
        EXPECT_TRUE(batch->equalValueAt(
            &stream->current(), indices[i], stream->currentIndex()));
        stream->pop();
      }
    }
  }

  // 返回一个拥有各种类型的数据的RowContainer，用于测试Spill
  std::unique_ptr<RowContainer> makeSpillData(
      int32_t numRows,
      std::vector<char*>& rows,
      RowVectorPtr& batch) {
    batch = makeDataset(
        ROW({
            {"bool_val", BOOLEAN()},
            {"tiny_val", TINYINT()},
            {"small_val", SMALLINT()},
            {"int_val", INTEGER()},
            {"long_val", BIGINT()},
            {"ordinal", BIGINT()},
            {"float_val", REAL()},
            {"double_val", DOUBLE()},
            {"string_val", VARCHAR()},
            {"array_val", ARRAY(VARCHAR())},
            {"struct_val",
             ROW({{"s_int", INTEGER()}, {"s_array", ARRAY(REAL())}})},
            {"map_val",
             MAP(VARCHAR(),
                 MAP(BIGINT(),
                     ROW({{"s2_int", INTEGER()}, {"s2_string", VARCHAR()}})))},
        }),
        numRows,
        [](RowVectorPtr /*rows&*/) {});
    const auto& types = batch->type()->as<TypeKind::ROW>().children();
    std::vector<TypePtr> keys;
    keys.insert(keys.begin(), types.begin(), types.begin() + 6);

    std::vector<TypePtr> dependents;
    dependents.insert(dependents.begin(), types.begin() + 6, types.end());
    // Set ordinal so that the sorted order is unambiguous

    auto ordinal = batch->childAt(5)->as<FlatVector<int64_t>>();
    for (auto i = 0; i < numRows; ++i) {
      ordinal->set(i, i);
    }
    // Make non-join build container so that spill runs are sorted. Note
    // that a distinct or group by hash table can have dependents if
    // some keys are known to be unique by themselves. Aggregation
    // spilling will be tested separately.
    auto data = makeRowContainer(keys, dependents, false);
    rows.resize(numRows);
    for (int i = 0; i < numRows; ++i) {
      rows[i] = data->newRow();
    }

    SelectivityVector allRows(numRows);
    for (auto column = 0; column < batch->childrenSize(); ++column) {
      DecodedVector decoded(*batch->childAt(column), allRows);
      for (auto index = 0; index < numRows; ++index) {
        data->store(decoded, index, rows[index], column);
      }
    }

    return data;
  }

  folly::IOThreadPoolExecutor* FOLLY_NONNULL executor() {
    static std::mutex mutex;
    std::lock_guard<std::mutex> l(mutex);
    if (!executor_) {
      executor_ = std::make_unique<folly::IOThreadPoolExecutor>(8);
    }
    return executor_.get();
  }

  std::unique_ptr<folly::IOThreadPoolExecutor> executor_;
};

TEST_F(SpillerTest, spilFew) {
  testSpill(10);
}

TEST_F(SpillerTest, spilMost) {
  testSpill(60);
}

TEST_F(SpillerTest, spillAll) {
  testSpill(100);
}

TEST_F(SpillerTest, error) {
  testSpill(100, true);
}
