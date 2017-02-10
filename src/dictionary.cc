/**
 * Copyright (c) 2016-present, Facebook, Inc.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree. An additional grant
 * of patent rights can be found in the PATENTS file in the same directory.
 */

#include "dictionary.h"

#include <assert.h>

#include <algorithm>
#include <iostream>
#include <iterator>
#include <stack>
#include <unordered_map>

namespace fasttext {

const std::string Dictionary::EOS = "</s>";
const std::string Dictionary::BOW = "<";
const std::string Dictionary::EOW = ">";

Dictionary::Dictionary(std::shared_ptr<Args> args) {
  args_ = args;
  size_ = 0;
  nwords_ = 0;
  nlabels_ = 0;
  ntokens_ = 0;
  word2int_.resize(MAX_VOCAB_SIZE);
  //brackets_ = std::make_shared<std::stack<char>>();
  for (int32_t i = 0; i < MAX_VOCAB_SIZE; i++) {
    word2int_[i] = -1;
  }
}

int32_t Dictionary::find(const std::string& w) const {
  int32_t h = hash(w) % MAX_VOCAB_SIZE;
  while (word2int_[h] != -1 && words_[word2int_[h]].word != w) {
    h = (h + 1) % MAX_VOCAB_SIZE;
  }
  return h;
}

void Dictionary::add(const std::string& w) {
  int32_t h = find(w);
  ntokens_++;
  if (word2int_[h] == -1) {
    entry e;
    e.word = w;
    e.count = 1;
    e.type = (w.find(args_->label) == 0) ? entry_type::label : entry_type::word;
    words_.push_back(e);
    word2int_[h] = size_++;
  } else {
    words_[word2int_[h]].count++;
  }
}

int32_t Dictionary::nwords() const { return nwords_; }

int32_t Dictionary::nlabels() const { return nlabels_; }

int64_t Dictionary::ntokens() const { return ntokens_; }

const std::vector<int32_t>& Dictionary::getNgrams(int32_t i) const {
  assert(i >= 0);
  assert(i < nwords_);
  return words_[i].subwords;
}

const std::vector<int32_t> Dictionary::getNgrams(
    const std::string& word) const {
  int32_t i = getId(word);
  if (i >= 0) {
    return getNgrams(i);
  }
  std::vector<int32_t> ngrams;
  computeNgrams(BOW + word + EOW, ngrams);
  return ngrams;
}

int32_t Dictionary::getWordCount(int32_t i) { return words_[i].count; }

bool Dictionary::discard(int32_t id, real rand) const {
  assert(id >= 0);
  assert(id < nwords_);
  if (args_->model == model_name::sup) return false;
  return rand > pdiscard_[id];
}

int32_t Dictionary::getId(const std::string& w) const {
  int32_t h = find(w);
  return word2int_[h];
}

entry_type Dictionary::getType(int32_t id) const {
  assert(id >= 0);
  assert(id < size_);
  return words_[id].type;
}

std::string Dictionary::getWord(int32_t id) const {
  assert(id >= 0);
  assert(id < size_);
  return words_[id].word;
}

uint32_t Dictionary::hash(const std::string& str) const {
  uint32_t h = 2166136261;
  for (size_t i = 0; i < str.size(); i++) {
    h = h ^ uint32_t(str[i]);
    h = h * 16777619;
  }
  return h;
}

void Dictionary::computeNgrams(const std::string& word,
                               std::vector<int32_t>& ngrams) const {
  for (size_t i = 0; i < word.size(); i++) {
    std::string ngram;
    if ((word[i] & 0xC0) == 0x80) continue;
    for (size_t j = i, n = 1; j < word.size() && n <= args_->maxn; n++) {
      ngram.push_back(word[j++]);
      while (j < word.size() && (word[j] & 0xC0) == 0x80) {
        ngram.push_back(word[j++]);
      }
      if (n >= args_->minn && !(n == 1 && (i == 0 || j == word.size()))) {
        int32_t h = hash(ngram) % args_->bucket;
        ngrams.push_back(nwords_ + h);
      }
    }
  }
}

void Dictionary::initNgrams() {
  for (size_t i = 0; i < size_; i++) {
    std::string word = BOW + words_[i].word + EOW;
    words_[i].subwords.push_back(i);
    computeNgrams(word, words_[i].subwords);
  }
}

bool Dictionary::readWord(std::istream& in, std::string& word) const {
  char c;
  std::streambuf& sb = *in.rdbuf();
  word.clear();
  while ((c = sb.sbumpc()) != EOF) {
    // if (c == ' ' || c == '\n' || c == '\r' || c == '\t' || c == '\v' || c ==
    // '\f' || c == '\0')
    if (c == '\n' || c == ',') {
     if (word.empty()) {
        if (c == '\n') {
          word += EOS;
          return true;
        }
        continue;
      } else {
        if (c == '\n')
          sb.sungetc();
        return true;
      }
   }
    if (c != ' ' && c != '\r' && c != '\t' && c != '\v' && c != '\f' &&
        c != '\0') {
      if (c == '[') {
        brackets_++;
      } else if (c == ']') {
        if (brackets_ > 0) brackets_--;
      } else if (brackets_ == 3) {
        word.push_back(c);
      }
    }
  }
  // trigger eofbit
  in.get();
  return !word.empty();
}

bool Dictionary::readWordTime(std::istream& in, std::string& word, flag_time& flag, int32_t& nBrackets) const {
  char c;
  std::streambuf& sb = *in.rdbuf();
  word.clear();
  while ((c = sb.sbumpc()) != EOF) {
    if (c == '\n' || c == ',') {
      if (word.empty()) {
        if (c == '\n') {
          flag = flag_time::word;
          word += EOS;
          return true;
        }
        continue;
      } else {
        if (c == '\n')
          sb.sungetc();
        return true;
      }
    }
    if (c != ' ' && c != '\r' && c != '\t' && c != '\v' && c != '\f' && c != '\0') {
      //std::cout<<"c: " << c <<std::endl;
      //std::cout<<"brackets_->size():" << brackets_->size()<<std::endl;
      if (c == '[') {
        nBrackets++;
        //brackets_->push(c);
      } else if (c == ']') {
        //if (!brackets_->empty()) brackets_->pop();
        if (nBrackets > 0) nBrackets--;
      } else if (nBrackets == 2) {
        word.push_back(c);
        flag = flag_time::time;
      } else if (nBrackets == 3) {
        word.push_back(c);
        flag = flag_time::word;
      }
    }
  }
  // trigger eofbit
  in.get();
  return !word.empty();
}

void Dictionary::clearStack(std::shared_ptr<std::stack<char>> t) const {
  while (!t->empty()) {
    t->pop();
  }
}

void Dictionary::readFromFile(std::istream& in) {
  std::string word;
  int64_t minThreshold = 1;
  //clearStack(brackets_);
  while (readWord(in, word)) {
    add(word);
    if (ntokens_ % 1000000 == 0 && args_->verbose > 1) {
      std::cout << "\rRead " << ntokens_ / 1000000 << "M words" << std::flush;
    }
    if (size_ > 0.75 * MAX_VOCAB_SIZE) {
      minThreshold++;
      threshold(minThreshold, minThreshold);
    }
  }
  threshold(args_->minCount, args_->minCountLabel);
  initTableDiscard();
  initNgrams();
  if (args_->verbose > 0) {
    std::cout << "\rRead " << ntokens_ / 1000000 << "M words" << std::endl;
    std::cout << "Number of words:  " << nwords_ << std::endl;
    std::cout << "Number of labels: " << nlabels_ << std::endl;
  }
  if (size_ == 0) {
    std::cerr << "Empty vocabulary. Try a smaller -minCount value."
              << std::endl;
    exit(EXIT_FAILURE);
  }
}

void Dictionary::threshold(int64_t t, int64_t tl) {
  sort(words_.begin(), words_.end(), [](const entry& e1, const entry& e2) {
    if (e1.type != e2.type) return e1.type < e2.type;
    return e1.count > e2.count;
  });
  words_.erase(remove_if(words_.begin(), words_.end(),
                         [&](const entry& e) {
                           return (e.type == entry_type::word && e.count < t) ||
                                  (e.type == entry_type::label && e.count < tl);
                         }),
               words_.end());
  words_.shrink_to_fit();
  size_ = 0;
  nwords_ = 0;
  nlabels_ = 0;
  for (int32_t i = 0; i < MAX_VOCAB_SIZE; i++) {
    word2int_[i] = -1;
  }
  for (auto it = words_.begin(); it != words_.end(); ++it) {
    int32_t h = find(it->word);
    word2int_[h] = size_++;
    if (it->type == entry_type::word) nwords_++;
    if (it->type == entry_type::label) nlabels_++;
  }
}

void Dictionary::initTableDiscard() {
  pdiscard_.resize(size_);
  for (size_t i = 0; i < size_; i++) {
    real f = real(words_[i].count) / real(ntokens_);
    pdiscard_[i] = sqrt(args_->t / f) + args_->t / f;
  }
}

std::vector<int64_t> Dictionary::getCounts(entry_type type) const {
  std::vector<int64_t> counts;
  for (auto& w : words_) {
    if (w.type == type) counts.push_back(w.count);
  }
  return counts;
}

void Dictionary::addNgrams(std::vector<int32_t>& line, int32_t n) const {
  int32_t line_size = line.size();
  for (int32_t i = 0; i < line_size; i++) {
    uint64_t h = line[i];
    for (int32_t j = i + 1; j < line_size && j < i + n; j++) {
      h = h * 116049371 + line[j];
      line.push_back(nwords_ + (h % args_->bucket));
    }
  }
}

int32_t Dictionary::getLine(std::istream& in, std::vector<int32_t>& words,
                            std::vector<int32_t>& labels,
                            std::minstd_rand& rng) const {
  std::uniform_real_distribution<> uniform(0, 1);
  std::string token;
  int32_t ntokens = 0;
  words.clear();
  labels.clear();
  if (in.eof()) {
    in.clear();
    in.seekg(std::streampos(0));
  }
  while (readWord(in, token)) {
    int32_t wid = getId(token);
    if (wid < 0) continue;
    entry_type type = getType(wid);
    ntokens++;
    if (type == entry_type::word && !discard(wid, uniform(rng))) {
      words.push_back(wid);
    }
    if (type == entry_type::label) {
      labels.push_back(wid - nwords_);
    }
    if (words.size() > MAX_LINE_SIZE && args_->model != model_name::sup) break;
    if (token == EOS) break;
  }
  return ntokens;
}

int64_t Dictionary::timeConvert(std::string begin_time,
                                std::string current_time) const {
  int64_t time_u, result, b_time, c_time;
  if (args_->timeUnit == time_unit::day)
    time_u = 86400;
  else if (args_->timeUnit == time_unit::week)
    time_u = 7 * 24 * 3600;
  else if (args_->timeUnit == time_unit::month)
    time_u = 30 * 24 * 3600;
  else
    time_u = 365 * 24 * 3600;
  b_time = std::stof(begin_time);
  c_time = std::stof(current_time);
  result = int64_t((c_time - b_time) / float(time_u) + 0.5);
  return result;
}

int32_t Dictionary::getLineContext(std::istream& in,
                                   std::vector<word_time>& words_time,
                                   std::vector<int32_t>& labels,
                                   std::minstd_rand& rng) const {
  std::uniform_real_distribution<> uniform(0, 1);
  std::string token;
  flag_time flag;
  int32_t ntokens = 0;  // the number of visit, combing visits within a time
                        // unit
  words_time.clear();
  labels.clear();
  if (in.eof()) {
    in.clear();
    in.seekg(std::streampos(0));
  }
  //clearStack(brackets_);
  word_time wtime;
  wtime.time = -1;
  std::string begin_time;
  int64_t token_time;
  int32_t nBrackets = 0;
  while (readWordTime(in, token, flag, nBrackets)) {
    if (flag == flag_time::time) {
      //std::cout << "flag in getLineContext: " << int(flag) << std::endl;
      //std::cout << "wtime.time: " << wtime.time << std::endl;
      if (wtime.time == -1) {
        begin_time = token;
        wtime.time = 0;
        //std::cout << "wtime.time 1.5: " << wtime.time << std::endl;
      } 
      else {
        token_time = timeConvert(begin_time, token);
        if (wtime.time == token_time) {
          continue;
        }
        words_time.push_back(wtime);
        //ntokens += wtime.wordsID.size();
        //std::cout << "num of words in wordsID: " << wtime.wordsID.size() << std::endl;
        wtime.time = token_time;
        wtime.wordsID.clear();
      }
      //std::cout << "wtime.time 2: " << wtime.time << std::endl;
    }
    else {
      int32_t wid = getId(token);
      if (wid < 0) continue;
      entry_type type = getType(wid);
      //bool isDiscard = discard(wid, uniform(rng));
      //std::cout << "isDiscard: " << isDiscard << std::endl;
      ntokens++;
      if (type == entry_type::word && !discard(wid, uniform(rng))) {
        //std::cout << "wid: " << wid << std::endl;
        wtime.wordsID.push_back(wid);
      }
      if (type == entry_type::label) {
        labels.push_back(wid - nwords_);
      }
      if (words_time.size() > MAX_LINE_SIZE && args_->model != model_name::sup)
        break;
      if (token == EOS) {
        words_time.push_back(wtime);
        break;
      } 
    }
  }
  //std::cout << "size of words_time: " << words_time.size() << std::endl;
  //std::cout << "num of words: " << ntokens << std::endl;
  return ntokens;
}

std::string Dictionary::getLabel(int32_t lid) const {
  assert(lid >= 0);
  assert(lid < nlabels_);
  return words_[lid + nwords_].word;
}

void Dictionary::save(std::ostream& out) const {
  out.write((char*)&size_, sizeof(int32_t));
  out.write((char*)&nwords_, sizeof(int32_t));
  out.write((char*)&nlabels_, sizeof(int32_t));
  out.write((char*)&ntokens_, sizeof(int64_t));
  for (int32_t i = 0; i < size_; i++) {
    entry e = words_[i];
    out.write(e.word.data(), e.word.size() * sizeof(char));
    out.put(0);
    out.write((char*)&(e.count), sizeof(int64_t));
    out.write((char*)&(e.type), sizeof(entry_type));
  }
}

void Dictionary::load(std::istream& in) {
  words_.clear();
  for (int32_t i = 0; i < MAX_VOCAB_SIZE; i++) {
    word2int_[i] = -1;
  }
  in.read((char*)&size_, sizeof(int32_t));
  in.read((char*)&nwords_, sizeof(int32_t));
  in.read((char*)&nlabels_, sizeof(int32_t));
  in.read((char*)&ntokens_, sizeof(int64_t));
  for (int32_t i = 0; i < size_; i++) {
    char c;
    entry e;
    while ((c = in.get()) != 0) {
      e.word.push_back(c);
    }
    in.read((char*)&e.count, sizeof(int64_t));
    in.read((char*)&e.type, sizeof(entry_type));
    words_.push_back(e);
    word2int_[find(e.word)] = i;
  }
  initTableDiscard();
  initNgrams();
}
}
