/**
 * Copyright (c) Microsoft Corporation. All rights reserved.
 * Licensed under the MIT License.
 */


#ifndef _FA_STATE2OWS_AMB_H_
#define _FA_STATE2OWS_AMB_H_

#include "FAConfig.h"
#include "FAState2Ows.h"
#include "FABitArray.h"
#include "FAArray_cont_t.h"

namespace BlingFire
{

class FAAllocatorA;
class FARSNfaA;

///
/// This class keeps State2Ows mapping only for ambiguous from
/// NFA point of view positions.
///

class FAState2Ows_amb : public FAState2Ows {

public:
    FAState2Ows_amb (FAAllocatorA * pAlloc);

public:
    // sets up reversed position NFA
    void SetRevPosNfa (const FARSNfaA * pRevPosNfa);
    // calculates ambiguous positions
    void Prepare ();
    // overrides SetOws, so only ambiguous positions are stored
    void SetOws (const int State, const int * pOws, const int Size);

private:

    const FARSNfaA * m_pRevPosNfa;
    FABitArray m_is_amb;
    FAArray_cont_t < int > m_ows;

};

}

#endif
