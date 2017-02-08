/**
 * Copyright (c) 2016-present, Facebook, Inc.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree. An additional grant
 * of patent rights can be found in the PATENTS file in the same directory.
 */

#ifndef FASTTEXT_DICTIONARY_H
#define FASTTEXT_DICTIONARY_H

#include <vector>
#include <string>
#include <istream>
#include <ostream>
#include <random>
#include <memory>
#include <stack>  

#include "args.h"
#include "real.h"

namespace fasttext {

typedef int32_t id_type;
enum class entry_type : int8_t {word=0, label=1};
enum class flag_time : int8_t {time=0, word=1};

struct entry {
  std::string word;
  int64_t count;
  entry_type type;
  std::vector<int32_t> subwords;
};

struct word_time{
    int64_t time;    // 计算所得第几个time_unit(e.g., 第2周， time = 2)
    std::vector<int32_t> wordsID; //
};

class Dictionary {
  private:
    static const int32_t MAX_VOCAB_SIZE = 30000000;
    static const int32_t MAX_LINE_SIZE = 1024;

    int32_t find(const std::string&) const;
    void initTableDiscard();
    void initNgrams();

    //std::shared_ptr<std::stack<char>> brackets_;//count how many square brackets
    mutable int brackets_;
    std::shared_ptr<Args> args_;
    std::vector<int32_t> word2int_;
    std::vector<entry> words_;
    std::vector<real> pdiscard_;
    int32_t size_;
    int32_t nwords_;
    int32_t nlabels_;
    int64_t ntokens_;

  public:
    static const std::string EOS;
    static const std::string BOW;
    static const std::string EOW;

    explicit Dictionary(std::shared_ptr<Args>);
    int32_t nwords() const;
    int32_t nlabels() const;
    int64_t ntokens() const;
    int32_t getId(const std::string&) const;
    entry_type getType(int32_t) const;
    bool discard(int32_t, real) const;
    std::string getWord(int32_t) const;
    const std::vector<int32_t>& getNgrams(int32_t) const;
    const std::vector<int32_t> getNgrams(const std::string&) const;
    void computeNgrams(const std::string&, std::vector<int32_t>&) const;
    uint32_t hash(const std::string& str) const;
    void add(const std::string&);
    bool readWord(std::istream&, std::string&) const;
    bool readWordTime(std::istream&, std::string&, flag_time&, int32_t&) const;
    void clearStack(std::shared_ptr<std::stack<char>>) const;
    void readFromFile(std::istream&);
    std::string getLabel(int32_t) const;
    void save(std::ostream&) const;
    void load(std::istream&);
    std::vector<int64_t> getCounts(entry_type) const;
    void addNgrams(std::vector<int32_t>&, int32_t) const;
    int32_t getLine(std::istream&, std::vector<int32_t>&,
                    std::vector<int32_t>&, std::minstd_rand&) const;
    int64_t timeConvert(std::string, std::string) const;
    int32_t getLineContext(std::istream&, std::vector<word_time>&,
                    std::vector<int32_t>&, std::minstd_rand&) const;
    void threshold(int64_t, int64_t);
};

}

#endif
