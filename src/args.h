/**
 * Copyright (c) 2016-present, Facebook, Inc.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree. An additional grant
 * of patent rights can be found in the PATENTS file in the same directory.
 */

#ifndef FASTTEXT_ARGS_H
#define FASTTEXT_ARGS_H

#include <istream>
#include <ostream>
#include <string>
#include "real.h"

namespace fasttext {

enum class model_name : int {cbow=1, sg, sup};
enum class loss_name : int {hs=1, ns, softmax};
enum class time_unit : int {day=1, week, month, season, year};

class Args {
  public:
    Args();
    std::string input;
    std::string test;
    std::string output;
    double lr;
    int lrUpdateRate;
    int dim;
    int ws;
    int epoch;
    int minCount;
    int minCountLabel;
    int neg;
    int wordNgrams;
    loss_name loss;
    model_name model;
    time_unit timeUnit;
    int bucket;
    int minn;
    int maxn;
    int thread;
    double t;
    std::string label;
    int verbose;
    std::string pretrainedVectors;
    real beta_base;
    real delta;

    void parseArgs(int, char**);
    void printHelp();
    void save(std::ostream&);
    void load(std::istream&);
};

}

#endif
