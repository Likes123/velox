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

#include "velox/exec/Spill.h"
#include "velox/common/file/FileSystems.h"

namespace facebook::velox::exec {

std::atomic<int32_t> SpillStream::ordinalCounter_;

void SpillInput::next(bool /*throwIfPastEnd*/) {
  int32_t readBytes = std::min(input_->size() - offset_, buffer_->capacity());
  VELOX_CHECK_LT(0, readBytes, "Reading past end of spill file");
  setRange({buffer_->asMutable<uint8_t>(), readBytes, 0});
  input_->pread(offset_, readBytes, buffer_->asMutable<char>());
  offset_ += readBytes;
}

void SpillStream::pop() {
  if (++index_ >= size_) {
    setNextBatch();
  }
}

SpillFile::~SpillFile() {
  try {
    auto fs = filesystems::getFileSystem(path_, nullptr);
    fs->remove(path_);
  } catch (const std::exception& e) {
    LOG(ERROR) << "Error deleting spill file " << path_ << " : " << e.what();
  }
}

WriteFile& SpillFile::output() {
  if (!output_) {
    auto fs = filesystems::getFileSystem(path_, nullptr);
    output_ = fs->openFileForWrite(path_);
  }
  return *output_;
}

void SpillFile::startRead() {
  constexpr uint64_t kMaxReadBufferSize =
      (1 << 20) - AlignedBuffer::kPaddedSize; // 1MB - padding.
  VELOX_CHECK(!output_);
  VELOX_CHECK(!input_);
  auto fs = filesystems::getFileSystem(path_, nullptr);
  auto file = fs->openFileForRead(path_);
  auto buffer = AlignedBuffer::allocate<char>(
      std::min<uint64_t>(fileSize_, kMaxReadBufferSize), &pool_);
  input_ = std::make_unique<SpillInput>(std::move(file), std::move(buffer));
  nextBatch();
}

// 分批从SpillFile中读取数据
void SpillFile::nextBatch() {
  index_ = 0;
  if (input_->atEnd()) {
    size_ = 0;
    return;
  }
  VectorStreamGroup::read(input_.get(), &pool_, type_, &rowVector_);
  size_ = rowVector_->size();
}

WriteFile& SpillFileList::currentOutput() {
  if (files_.empty() || !files_.back()->isWritable() ||
      files_.back()->size() > targetFileSize_ * 1.5) {
    if (!files_.empty() && files_.back()->isWritable()) {
      // 最后一个文件可以写入，但是大小已经超过了targetFileSize_ * 1.5，先结束掉
      files_.back()->finishWrite();
    }
    // 创建新的SpillFile
    files_.push_back(std::make_unique<SpillFile>(
        type_,
        numSortingKeys_,
        sortCompareFlags_,
        fmt::format("{}-{}", path_, files_.size()),
        pool_));
  }
  // 可能是新创建的SpillFile，也可能是已经存在的SpillFile
  return files_.back()->output();
}

void SpillFileList::flush() {
  // batch_用于序列化
  if (batch_) {
    // 将序列化的数据写入到out
    IOBufOutputStream out(
        mappedMemory_, nullptr, std::max<int64_t>(64 * 1024, batch_->size()));
    batch_->flush(&out);
    batch_.reset();
    auto iobuf = out.getIOBuf();
    // 获取SpillFile，可能新建，也可能复用之前的，将out中的数据写入到SpillFile
    auto& file = currentOutput();
    for (auto& range : *iobuf) {
      file.append(std::string_view(
          reinterpret_cast<const char*>(range.data()), range.size()));
    }
  }
}

void SpillFileList::write(
    const RowVectorPtr& rows,
    const folly::Range<IndexRange*>& indices) {
  // batch_用于序列化，没有则创建
  if (!batch_) {
    batch_ = std::make_unique<VectorStreamGroup>(&mappedMemory_);
    batch_->createStreamTree(
        std::static_pointer_cast<const RowType>(rows->type()), 1000);
  }
  batch_->append(rows, indices);

  // 将batch_中的数据写入到SpillFile
  flush();
}

void SpillFileList::finishFile() {
  // 每一次write完就会flush，这里不需要?
  flush();
  if (files_.empty()) {
    return;
  }
  if (files_.back()->isWritable()) {
    files_.back()->finishWrite();
  }
}

uint64_t SpillFileList::spilledBytes() const {
  uint64_t bytes = 0;
  for (auto& file : files_) {
    bytes += file->size();
  }
  return bytes;
}

std::vector<std::string> SpillFileList::testingSpilledFilePaths() const {
  std::vector<std::string> spilledFiles;
  for (auto& file : files_) {
    spilledFiles.push_back(file->testingFilePath());
  }
  return spilledFiles;
}

void SpillState::setPartitionSpilled(int32_t partition) {
  VELOX_DCHECK_LT(partition, maxPartitions_);
  VELOX_DCHECK(!isPartitionSpilled_[partition]);
  isPartitionSpilled_[partition] = true;
}

void SpillState::appendToPartition(
    int32_t partition,
    const RowVectorPtr& rows) {
  VELOX_CHECK(isPartitionSpilled(partition));
  // Ensure that partition exist before writing.
  if (!files_.at(partition)) {
    files_[partition] = std::make_unique<SpillFileList>(
        std::static_pointer_cast<const RowType>(rows->type()),
        numSortingKeys_,
        sortCompareFlags_,
        fmt::format("{}-spill-{}", path_, partition),
        targetFileSize_,
        pool_,
        mappedMemory_);
  }

  IndexRange range{0, rows->size()};
  files_[partition]->write(rows, folly::Range<IndexRange*>(&range, 1));
}

std::unique_ptr<TreeOfLosers<SpillStream>> SpillState::startMerge(
    int32_t partition,
    std::unique_ptr<SpillStream>&& extra) {
  VELOX_CHECK_LT(partition, files_.size());
  std::vector<std::unique_ptr<SpillStream>> result;
  // 给定Partition下的每一个SpillFile作为多路归并的一路
  if (auto list = std::move(files_[partition]); list) {
    for (auto& file : list->files()) {
      file->startRead();
      result.push_back(std::move(file));
    }
  }
  VELOX_DCHECK_EQ(!result.empty(), isPartitionSpilled(partition));
  // extra是内存中没有落盘的数据
  if (extra != nullptr) {
    result.push_back(std::move(extra));
  }
  // Check if the partition is empty or not.
  if (FOLLY_UNLIKELY(result.empty())) {
    return nullptr;
  }
  // 使用败者树进行归并
  return std::make_unique<TreeOfLosers<SpillStream>>(std::move(result));
}

uint64_t SpillState::spilledBytes() const {
  uint64_t bytes = 0;
  for (auto& list : files_) {
    if (list) {
      bytes += list->spilledBytes();
    }
  }
  return bytes;
}

uint32_t SpillState::spilledPartitions() const {
  uint32_t numSpilledPartitions = 0;
  for (int i = 0; i < isPartitionSpilled_.size(); ++i) {
    if (isPartitionSpilled_[i]) {
      ++numSpilledPartitions;
    }
  }
  return numSpilledPartitions;
}

int64_t SpillState::spilledFiles() const {
  int64_t numFiles = 0;
  for (const auto& list : files_) {
    if (list != nullptr) {
      numFiles += list->spilledFiles();
    }
  }
  return numFiles;
}

std::vector<std::string> SpillState::testingSpilledFilePaths() const {
  std::vector<std::string> spilledFiles;
  for (const auto& list : files_) {
    if (list != nullptr) {
      const auto spilledFilesFromList = list->testingSpilledFilePaths();
      spilledFiles.insert(
          spilledFiles.end(),
          spilledFilesFromList.begin(),
          spilledFilesFromList.end());
    }
  }
  return spilledFiles;
}

} // namespace facebook::velox::exec
