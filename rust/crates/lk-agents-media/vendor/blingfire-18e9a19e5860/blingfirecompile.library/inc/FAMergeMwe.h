/**
 * Copyright (c) Microsoft Corporation. All rights reserved.
 * Licensed under the MIT License.
 */

#ifndef _FA_MERGEMWE_H_
#define _FA_MERGEMWE_H_

#include "FAConfig.h"

namespace BlingFire
{

class FARSDfaCA;
class FATaggedTextA;
class FATaggedTextCA;

///
/// This class merges MWE by the left-most-longest algorithm.
///

class FAMergeMwe {

public:
    FAMergeMwe ();

public:
    /// sets up simple word delimiter, ' ' is used by default
    void SetMweDelim (const int MweDelim);
    /// MWE DFA
    void SetRsDfa (const FARSDfaCA * pDfa);
    /// ignores case
    void SetIgnoreCase (const bool IgnoreCase);
    /// makes processing
    void Process (FATaggedTextA * pOut, const FATaggedTextCA * pIn) const;

private:
    /// returns how many tokens should be glued together starting from Pos
    inline const int GetTokenCount (
            const int Pos, 
            const FATaggedTextCA * pIn
        ) const;

private:
    const FARSDfaCA * m_pDfa;
    int m_MweDelim;
    bool m_IgnoreCase;
};

}

#endif
