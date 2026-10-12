/**
 * Copyright (c) Microsoft Corporation. All rights reserved.
 * Licensed under the MIT License.
 */

#ifndef _FA_TESTMORPH_W2H_H_
#define _FA_TESTMORPH_W2H_H_

#include "FAConfig.h"
#include "FATestMorph.h"
#include "FAHyphInterpreter_t.h"
#include "FAChain2Num_hash.h"
#include "FAArray_cont_t.h"

namespace BlingFire
{

///
/// W2H quality test. Input format: see FATrWordIOTools_utf8.h for details.
///

class FATestMorph_w2h : public FATestMorph {

public:
    FATestMorph_w2h (FAAllocatorA * pAlloc);

public:
    void SetW2H (FAHyphInterpreter_t < int > * pHyph);
    // alternative rule-set should be used, false by default
    void SetUseAltW2H (const bool UseAltW2H);
    void Test (const char * pLineStr, const int LineLen);

private:
    inline const int Chain2Set (
            int * pChain, 
            const int Size
        );

private:
    FAHyphInterpreter_t < int > * m_pHyph;
    bool m_UseAltW2H;
};

}

#endif
