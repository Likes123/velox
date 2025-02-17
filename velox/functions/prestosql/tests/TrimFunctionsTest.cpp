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
#include "velox/functions/prestosql/tests/utils/FunctionBaseTest.h"

using namespace std::string_literals;

namespace facebook::velox::functions {
namespace {

class TrimFunctionsTest : public test::FunctionBaseTest {
 protected:
  static std::string generateInvalidUtf8() {
    std::string str;
    str.resize(2);
    // Create corrupt data below.
    char16_t c = u'\u04FF';
    str[0] = (char)c;
    str[1] = (char)c;
    return str;
  }

  // Generate complex encoding with the format:
  // whitespace|unicode line separator|ascii|two bytes encoding|three bytes
  // encoding|four bytes encoding|whitespace|unicode line separator
  static std::string generateComplexUtf8(bool invalid = false) {
    std::string str;
    // White spaces
    str.append(" \u2028"s);
    if (invalid) {
      str.append(generateInvalidUtf8());
    }
    // Ascii
    str.append("hello");
    // two bytes
    str.append(" \u017F");
    // three bytes
    str.append(" \u4FE1");
    // four bytes
    std::string tmp;
    tmp.resize(4);
    tmp[0] = 0xF0;
    tmp[1] = 0xAF;
    tmp[2] = 0xA8;
    tmp[3] = 0x9F;
    str.append(" ").append(tmp);
    if (invalid) {
      str.append(generateInvalidUtf8());
    }
    // white spaces
    str.append("\u2028 ");
    return str;
  }
};

TEST_F(TrimFunctionsTest, trim) {
  // Making input vector
  std::string complexStr = generateComplexUtf8();
  std::string expectedComplexStr = complexStr.substr(4, complexStr.size() - 8);

  const auto trim = [&](std::optional<std::string> input) {
    return evaluateOnce<std::string>("trim(c0)", input);
  };

  EXPECT_EQ("facebook", trim("  facebook  "));
  EXPECT_EQ("facebook", trim("  facebook"));
  EXPECT_EQ("facebook", trim("facebook  "));
  EXPECT_EQ("facebook", trim("\n\nfacebook \n "));
  EXPECT_EQ("", trim(" \n"));
  EXPECT_EQ("", trim(""));
  EXPECT_EQ("", trim("    "));
  EXPECT_EQ("a", trim("  a  "));

  EXPECT_EQ(
      "\u4FE1\u5FF5 \u7231 \u5E0C\u671B",
      trim("\u4FE1\u5FF5 \u7231 \u5E0C\u671B \u2028 "));
  EXPECT_EQ(
      "\u4FE1\u5FF5 \u7231 \u5E0C\u671B",
      trim("\u4FE1\u5FF5 \u7231 \u5E0C\u671B  "));
  EXPECT_EQ(
      "\u4FE1\u5FF5 \u7231 \u5E0C\u671B",
      trim(" \u4FE1\u5FF5 \u7231 \u5E0C\u671B "));
  EXPECT_EQ(
      "\u4FE1\u5FF5 \u7231 \u5E0C\u671B",
      trim("  \u4FE1\u5FF5 \u7231 \u5E0C\u671B"));
  EXPECT_EQ(
      "\u4FE1\u5FF5 \u7231 \u5E0C\u671B",
      trim(" \u2028 \u4FE1\u5FF5 \u7231 \u5E0C\u671B"));

  EXPECT_EQ(expectedComplexStr, trim(complexStr));
  EXPECT_EQ(
      "Ψ\xFF\xFFΣΓΔA", trim("\u2028 \r \t \nΨ\xFF\xFFΣΓΔA \u2028 \r \t \n"));
}

TEST_F(TrimFunctionsTest, ltrim) {
  std::string complexStr = generateComplexUtf8();
  std::string expectedComplexStr = complexStr.substr(4, complexStr.size() - 4);

  const auto ltrim = [&](std::optional<std::string> input) {
    return evaluateOnce<std::string>("ltrim(c0)", input);
  };

  EXPECT_EQ("facebook", ltrim("facebook"));
  EXPECT_EQ("facebook ", ltrim("  facebook "));
  EXPECT_EQ("facebook \n", ltrim("\n\nfacebook \n"));
  EXPECT_EQ("", ltrim("\n"));
  EXPECT_EQ("", ltrim(" "));
  EXPECT_EQ("", ltrim("     "));
  EXPECT_EQ("a  ", ltrim("  a  "));
  EXPECT_EQ("facebo ok", ltrim(" facebo ok"));
  EXPECT_EQ("move fast", ltrim("\tmove fast"));
  EXPECT_EQ("move fast", ltrim("\r\t move fast"));
  EXPECT_EQ("hello", ltrim("\n\t\r hello"));

  EXPECT_EQ("\u4F60\u597D", ltrim(" \u4F60\u597D"));
  EXPECT_EQ("\u4F60\u597D ", ltrim(" \u4F60\u597D "));
  EXPECT_EQ(
      "\u4FE1\u5FF5 \u7231 \u5E0C\u671B  ",
      ltrim("\u4FE1\u5FF5 \u7231 \u5E0C\u671B  "));
  EXPECT_EQ(
      "\u4FE1\u5FF5 \u7231 \u5E0C\u671B ",
      ltrim(" \u4FE1\u5FF5 \u7231 \u5E0C\u671B "));
  EXPECT_EQ(
      "\u4FE1\u5FF5 \u7231 \u5E0C\u671B",
      ltrim("  \u4FE1\u5FF5 \u7231 \u5E0C\u671B"));
  EXPECT_EQ(
      "\u4FE1\u5FF5 \u7231 \u5E0C\u671B",
      ltrim(" \u2028 \u4FE1\u5FF5 \u7231 \u5E0C\u671B"));

  EXPECT_EQ(expectedComplexStr, ltrim(complexStr));
  EXPECT_EQ("Ψ\xFF\xFFΣΓΔA", ltrim("  \u2028 \r \t \n   Ψ\xFF\xFFΣΓΔA"));
}

TEST_F(TrimFunctionsTest, rtrim) {
  std::string complexStr = generateComplexUtf8();
  std::string expectedComplexStr = complexStr.substr(0, complexStr.size() - 4);

  const auto rtrim = [&](std::optional<std::string> input) {
    return evaluateOnce<std::string>("rtrim(c0)", input);
  };

  EXPECT_EQ("facebook", rtrim("facebook"));
  EXPECT_EQ(" facebook", rtrim(" facebook  "));
  EXPECT_EQ("\nfacebook", rtrim("\nfacebook \n\n"));
  EXPECT_EQ("", rtrim(" \n"));
  EXPECT_EQ("", rtrim(" "));
  EXPECT_EQ("", rtrim("     "));
  EXPECT_EQ("  a", rtrim("  a  "));
  EXPECT_EQ("facebo ok", rtrim("facebo ok "));
  EXPECT_EQ("move fast", rtrim("move fast\t"));
  EXPECT_EQ("move fast", rtrim("move fast\r\t "));
  EXPECT_EQ("hello", rtrim("hello\n\t\r "));

  EXPECT_EQ(" \u4F60\u597D", rtrim(" \u4F60\u597D"));
  EXPECT_EQ(" \u4F60\u597D", rtrim(" \u4F60\u597D "));
  EXPECT_EQ(
      "\u4FE1\u5FF5 \u7231 \u5E0C\u671B",
      rtrim("\u4FE1\u5FF5 \u7231 \u5E0C\u671B  "));
  EXPECT_EQ(
      " \u4FE1\u5FF5 \u7231 \u5E0C\u671B",
      rtrim(" \u4FE1\u5FF5 \u7231 \u5E0C\u671B "));
  EXPECT_EQ(
      "\u4FE1\u5FF5 \u7231 \u5E0C\u671B",
      rtrim("\u4FE1\u5FF5 \u7231 \u5E0C\u671B  "));
  EXPECT_EQ(
      "\u4FE1\u5FF5 \u7231 \u5E0C\u671B",
      rtrim("\u4FE1\u5FF5 \u7231 \u5E0C\u671B \u2028 "));

  EXPECT_EQ(expectedComplexStr, rtrim(complexStr));
  EXPECT_EQ("     Ψ\xFF\xFFΣΓΔA", rtrim("     Ψ\xFF\xFFΣΓΔA \u2028 \r \t \n"));
}

TEST_F(TrimFunctionsTest, trimCustomCharacters) {
  const auto trim = [&](const std::string& input, const std::string& chars) {
    return evaluateOnce<std::string>(
               "trim(c0, c1)",
               std::make_optional(input),
               std::make_optional(chars))
        .value();
  };

  const auto ltrim = [&](const std::string& input, const std::string& chars) {
    return evaluateOnce<std::string>(
               "ltrim(c0, c1)",
               std::make_optional(input),
               std::make_optional(chars))
        .value();
  };

  const auto rtrim = [&](const std::string& input, const std::string& chars) {
    return evaluateOnce<std::string>(
               "rtrim(c0, c1)",
               std::make_optional(input),
               std::make_optional(chars))
        .value();
  };

  // One custom trim character.
  EXPECT_EQ("es", trim("test", "t"));
  EXPECT_EQ("es", trim("tttesttt", "t"));
  EXPECT_EQ("est", ltrim("test", "t"));
  EXPECT_EQ("est", ltrim("tttest", "t"));
  EXPECT_EQ("tes", rtrim("test", "t"));
  EXPECT_EQ("tes", rtrim("testtt", "t"));
  EXPECT_EQ("", trim("tttttttt", "t"));

  // Empty list of custom trim characters.
  EXPECT_EQ("test", trim("test", ""));

  // Multiple custom trim characters.
  EXPECT_EQ("nan", trim("banana", "ab"));
  EXPECT_EQ("nan", trim("banana", "ba"));
  EXPECT_EQ("", trim("banana", "abn"));
  EXPECT_EQ("", trim("banana", "nba"));
  EXPECT_EQ("anana", trim("banana", "bn"));
  EXPECT_EQ("anana", trim("banana", "nb"));
}

} // namespace
} // namespace facebook::velox::functions