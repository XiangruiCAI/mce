/**
 * Copyright (c) 2016-present, Facebook, Inc.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree. An additional grant
 * of patent rights can be found in the PATENTS file in the same directory.
 */

#include <iostream>

#include "args.h"
#include "med2vec.h"

using namespace fasttext;

void printUsage() {
  std::cout << "usage: fasttext <command> <args>\n\n"
            << "The commands supported by fasttext are:\n\n"
            << "  skipgram            train a skipgram model\n"
            << "  cbow                train a cbow model\n"
            << "  print-vectors       print vectors given a trained model\n"
            << std::endl;
}

void printPrintVectorsUsage() {
  std::cout << "usage: fasttext print-vectors <model>\n\n"
            << "  <model>      model filename\n"
            << std::endl;
}

void printVectors(int argc, char** argv) {
  if (argc != 3) {
    printPrintVectorsUsage();
    exit(EXIT_FAILURE);
  }
  FastText fasttext;
  fasttext.loadModel(std::string(argv[2]));
  fasttext.printVectors();
  exit(0);
}

void train(int argc, char** argv) {
  std::shared_ptr<Args> a = std::make_shared<Args>();
  a->parseArgs(argc, argv);
  FastText fasttext;
  fasttext.train(a);
}

int main(int argc, char** argv) {
  if (argc < 2) {
    printUsage();
    exit(EXIT_FAILURE);
  }
  std::string command(argv[1]);
  if (command == "skipgram" || command == "cbow") {
    train(argc, argv);
  } else if (command == "print-vectors") {
    printVectors(argc, argv);
  } else {
    printUsage();
    exit(EXIT_FAILURE);
  }
  return 0;
}
