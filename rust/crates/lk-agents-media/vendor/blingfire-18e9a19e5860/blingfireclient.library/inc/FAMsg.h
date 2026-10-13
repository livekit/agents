/**
 * Copyright (c) Microsoft Corporation. All rights reserved.
 * Licensed under the MIT License.
 */


#ifndef _FA_MSG_H_
#define _FA_MSG_H_

#include "FAConfig.h"

namespace BlingFire
{

class FAMsg {

public:

    static const char * const InternalError;
    static const char * const InitializationError;
    static const char * const InvalidParameters;
    static const char * const IndexIsOutOfRange;
    static const char * const LimitIsExceeded;
    static const char * const OutOfMemory;
    static const char * const IOError;
    static const char * const ReadError;
    static const char * const WriteError;
    static const char * const CorruptFile;
    static const char * const ResourceIsNotFound;
    static const char * const ObjectIsNotReady;
    static const char * const SyntaxError;

};

}

#endif
