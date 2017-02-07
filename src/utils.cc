/**
 * Copyright (c) 2016-present, Facebook, Inc.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree. An additional grant
 * of patent rights can be found in the PATENTS file in the same directory.
 */

#include "utils.h"

#include <ios>
#include <iostream>

namespace fasttext {

namespace utils {

  int64_t size(std::ifstream& ifs) {
    ifs.seekg(std::streamoff(0), std::ios::end);
    return ifs.tellg();
  }

  void seek(std::ifstream& ifs, int64_t pos) {
    ifs.clear();
    ifs.seekg(std::streampos(pos));
  }
  
  void seekToBOS(std::ifstream& ifs, int64_t pos) {
    ifs.clear();
    std::streambuf& sb = *ifs.rdbuf();
    int32_t off = 0;
    for (off = 0; ifs.seekg(std::streampos(pos - off)); off++) {
      char c = sb.sbumpc();
      if (c == '\n')
        break;
    }
    std::cout << "seek position: " << pos - off + 1 << std::endl;
    ifs.seekg(std::streampos(pos - off + 1));
  }
}

}
