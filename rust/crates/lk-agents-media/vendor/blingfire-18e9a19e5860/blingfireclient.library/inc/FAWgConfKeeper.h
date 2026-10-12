/**
 * Copyright (c) Microsoft Corporation. All rights reserved.
 * Licensed under the MIT License.
 */


#ifndef _FA_WG_CONFKEEPER_H_
#define _FA_WG_CONFKEEPER_H_

#include "FAConfig.h"

namespace BlingFire
{

class FALDB;
class FARSDfa_pack_triv;
class FAState2Ows_pack_triv;
class FARSDfaCA;
class FAState2OwsCA;
class FAMultiMapCA;
class FAMultiMap_pack_fixed;

///
/// Keeps configuration for word-guesser module.
///
/// Note: Resource pointers may not be initialized and be NULL.
///

class FAWgConfKeeper {

public:
    FAWgConfKeeper ();
    ~FAWgConfKeeper ();

public:
    /// initialization
    void Initialize (const FALDB * pLDB, const int * pValues, const int Size);
    /// returns object into the initial state
    void Clear ();

public:
    const FARSDfaCA * GetRsDfa () const;
    const FAState2OwsCA * GetState2Ows () const;
    const int GetDirection () const;
    const int GetMaxLen () const;
    const bool GetNoTrUse () const;
    // returns tag value or -1 if it was not specified
    const int GetDefTag () const;
    const bool GetDictMode () const;
    const bool GetIgnoreCase () const;
    // the numrical value corresponding to 1 of the prob, prob guesser only
    const int GetMaxProb () const;
    const FAMultiMapCA * GetCharMap () const;
    const float GetMinProbVal () const;
    const float GetMaxProbVal () const;
    // returns true if logarithmic scale was used
    const bool GetIsLog () const;
    // returns the end-of-sequence (EOS) tag value
    const int GetEosTag () const;
    // returns the order parameter (e.g. n-gram order if ngrams are stored in the WG)
    // (returns -1 if was not specified)
    const int GetOrder () const;

private:
    FARSDfa_pack_triv * m_pDfa;
    FAState2Ows_pack_triv * m_pState2Ows;
    int m_Direction;
    int m_MaxLen;
    int m_DefTag;
    bool m_NoTrUse;
    bool m_DictMode;
    bool m_IgnoreCase;
    int m_MaxProb;
    FAMultiMap_pack_fixed * m_pCharMap;
    float m_MinProbVal;
    float m_MaxProbVal;
    bool m_fLogScale;
    int m_TagEos;
    int m_Order;
};

}

#endif
