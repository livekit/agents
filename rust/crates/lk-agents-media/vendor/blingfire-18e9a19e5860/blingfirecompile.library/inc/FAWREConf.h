/**
 * Copyright (c) Microsoft Corporation. All rights reserved.
 * Licensed under the MIT License.
 */

#ifndef _FA_WRECONF_H_
#define _FA_WRECONF_H_

#include "FAConfig.h"
#include "FAWREConfA.h"
#include "FAArray_cont_t.h"
#include "FAArray_p2ca.h"
#include "FAState2Ow.h"
#include "FAState2Ows.h"
#include "FAMealyDfa.h"
#include "FARSDfa_ro.h"
#include "FAMultiMap_ar_uniq.h"

namespace BlingFire
{

class FAAllocatorA;

///
/// Compiled WRE data container.
///

class FAWREConf : public FAWREConfA {

public:
    FAWREConf (FAAllocatorA * pAlloc);
    virtual ~FAWREConf ();

/// from FAWREConfCA
public:
    const int GetType () const;
    const int GetTokenType () const;
    const int GetTagOwBase () const;
    const FARSDfaCA * GetTxtDigDfa () const;
    const FAState2OwCA * GetTxtDigOws () const;
    const FAArrayCA * GetDictDig () const;
    const FARSDfaCA * GetDfa1 () const;
    const FARSDfaCA * GetDfa2 () const;
    const FAState2OwsCA * GetState2Ows () const;
    const FAMealyDfaCA * GetSigma1 () const;
    const FAMealyDfaCA * GetSigma2 () const;
    const FAMultiMapCA * GetTrBrMap () const;

/// from FAWREConfA (read)
public:
    void GetTxtDigititizer (
            const FARSDfaA ** ppDfa,
            const FAState2OwA ** ppState2Ow
        ) const;
    void GetDictDigitizer (
            const int ** ppId2Ow,
            int * pId2OwSize
        ) const;
    void GetDfa1 (const FARSDfaA ** ppDfa) const;
    void GetDfa2 (const FARSDfaA ** ppDfa) const;
    void GetState2Ows (const FAState2OwsA ** ppState2Ows) const;
    void GetSigma1 (const FAMealyDfaA ** ppSigma) const;
    void GetSigma2 (const FAMealyDfaA ** ppSigma) const;
    void GetTrBrMap (const FAMultiMapA ** ppTrBr) const;

/// from FAWREConfA (write)
public:
    void SetType (const int Type);
    void SetTokenType (const int TokenType);
    void SetTagOwBase (const int TagOwBase);
    void GetTxtDigititizer (FARSDfaA ** ppDfa, FAState2OwA ** ppState2Ow);
    void SetDictDigitizer (const int * pId2Ow, const int Size);
    void GetDfa1 (FARSDfaA ** ppDfa);
    void GetDfa2 (FARSDfaA ** ppDfa);
    void GetState2Ows (FAState2OwsA ** ppState2Ows);
    void GetSigma1 (FAMealyDfaA ** ppSigma);
    void GetSigma2 (FAMealyDfaA ** ppSigma);
    void GetTrBrMap (FAMultiMapA ** ppTrBr);
    void Clear ();

private:
    /// conf
    int m_Type;
    int m_TokenType;
    int m_TagOwBase;
    /// text digitizer
    FARSDfa_ro m_txt_dig_dfa;
    FAState2Ow m_txt_dig_ows;
    /// dict digitizer
    FAArray_cont_t < int > m_dct_dig_arr;
    FAArray_p2ca m_ca_arr;
    /// rules1
    FARSDfa_ro m_dfa1;
    FAState2Ows m_moore_ows1;
    FAMealyDfa m_sigma1;
    /// rules2
    FARSDfa_ro m_dfa2;
    FAMealyDfa m_sigma2;
    /// trbr map
    FAMultiMap_ar_uniq m_trbr;
};

}

#endif
