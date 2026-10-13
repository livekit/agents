/**
 * Copyright (c) Microsoft Corporation. All rights reserved.
 * Licensed under the MIT License.
 */


#ifndef _FA_LIMITS_H_
#define _FA_LIMITS_H_

#include "FAConfig.h"
#include "FAFsmConst.h"

namespace BlingFire
{

///
/// This class defines global limits for input and output of all modules.
/// 

class FALimits {

public:
    enum {
        // general-purpose
        MaxArrSize = 1000000000,   // security improvement limit
        MaxChainSize = MaxArrSize, // security improvement limit
        MaxStateVal = MaxArrSize,  // security improvement limit
        MaxIwVal = MaxArrSize,     // security improvement limit
        MaxIw = MaxArrSize,        // security improvement limit

        // LDB
        MaxLdbDumpCount = 3 * FAFsmConst::FUNC_COUNT,

        // word related
        MaxWordLen = 300,          // word size in symbols
        MaxWordSize = MaxWordLen,  // word size in symbols

        // word-list related
        MaxWordCount = 1000,       // number words in (paradigm, sentence, etc.)
        MaxParadigmSize = MaxWordCount,

        // POS tags related
        MaxTagsPerWord = 100,      // number of POS tags per word
        MinTag = 1,                // the smallest tag value
        MaxTag = 65535,            // the biggest tag value
        MaxGram = 4,               // Maximum ngram order
    };
};

}

#endif
