/**
 * Copyright (c) Microsoft Corporation. All rights reserved.
 * Licensed under the MIT License.
 */


#ifndef _FA_CMPTRBROWS_GREEDY_H_
#define _FA_CMPTRBROWS_GREEDY_H_

#include "FAConfig.h"
#include "FALessA.h"

namespace BlingFire
{

class FAMultiMapCA;

///
/// Compares output-weights associated with triangular brackets.
///
/// '<N' LESS -1 LESS 'N>'
/// '<N' LESS '<M' iff M < N
/// 'N>' LESS 'M>' iff N < M
///

class FACmpTrBrOws_greedy : public FALessA {

public:
    FACmpTrBrOws_greedy ();

public:
    void SetTrBrMap (const FAMultiMapCA * pOw2TrBrs);

public:
    const bool Less (const int Val1, const int Val2) const;

private:
    // <0 if TrBr1 < TrBr2
    // 0 if TrBr1 == TrBr2
    // >0 if TrBr1 > TrBr2
    static inline const int CmpTrBr (const int TrBr1, const int TrBr2);
    // -1 --> 0
    // <N --> -(N+1)
    // N> --> (N+1)
    static inline const int TrBr2Val (const int TrBr);

private:
    const FAMultiMapCA * m_pOw2TrBrs;
};

}

#endif
