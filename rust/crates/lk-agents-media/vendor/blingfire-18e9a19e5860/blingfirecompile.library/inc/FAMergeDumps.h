/**
 * Copyright (c) Microsoft Corporation. All rights reserved.
 * Licensed under the MIT License.
 */


#ifndef _FA_MERGEDUMPS_H_
#define _FA_MERGEDUMPS_H_

#include "FAConfig.h"
#include "FAArray_cont_t.h"

#include <iostream>

namespace BlingFire
{

class FAAllocatorA;

///
/// This class merges together memory dumps into a single file and adds
/// array of entry points.
///

class FAMergeDumps {

public:
    FAMergeDumps (FAAllocatorA * pAlloc);
    ~FAMergeDumps ();

public:
    // loads in one more dump file
    void AddDumpFile (const char * pFileName);
    // adds validation data and stores merged dump into an output stream
    void Save (std::ostream * pOs);
    // returns object into the initial state
    void Clear ();

private:
    // adds a dump with BIN file validation data
    void AddValidationDump ();

private:
    FAAllocatorA * m_pAlloc;
    FAArray_cont_t < unsigned char * > m_dumps;
    FAArray_cont_t < int > m_sizes;
};

}

#endif
