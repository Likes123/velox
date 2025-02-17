# Copyright (c) Facebook, Inc. and its affiliates.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

add_library(
  velox_aggregates
  ApproxDistinctAggregate.cpp
  ApproxMostFrequentAggregate.cpp
  ApproxPercentileAggregate.cpp
  ArbitraryAggregate.cpp
  ArrayAggAggregate.cpp
  AverageAggregate.cpp
  BitwiseAggregates.cpp
  BoolAggregates.cpp
  CentralMomentsAggregates.cpp
  CountIfAggregate.cpp
  CovarianceAggregates.cpp
  ChecksumAggregate.cpp
  EntropyAggregates.cpp
  HistogramAggregate.cpp
  MapAggAggregate.cpp
  MapUnionAggregate.cpp
  MapUnionSumAggregate.cpp
  MinMaxAggregates.cpp
  MinMaxByAggregates.cpp
  MultiMapAggAggregate.cpp
  CountAggregate.cpp
  PrestoHasher.cpp
  SetAggregates.cpp
  SumAggregate.cpp
  ValueList.cpp
  VarianceAggregates.cpp
  MaxSizeForStatsAggregate.cpp
  SumDataSizeForStatsAggregate.cpp
  RegisterAggregateFunctions.cpp)

target_link_libraries(
  velox_aggregates
  velox_common_hyperloglog
  velox_exec
  velox_expression
  velox_presto_serializer
  velox_functions_aggregates
  velox_functions_lib
  velox_functions_util
  Folly::folly)

if(${VELOX_BUILD_TESTING})
  add_subdirectory(tests)
endif()

if(${VELOX_ENABLE_BENCHMARKS})
  add_subdirectory(benchmarks)
endif()
