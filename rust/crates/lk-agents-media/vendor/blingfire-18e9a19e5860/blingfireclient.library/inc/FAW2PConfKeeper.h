/**
 * Copyright (c) Microsoft Corporation. All rights reserved.
 * Licensed under the MIT License.
 */


#ifndef _FA_W2PCONFKEEPER_H_
#define _FA_W2PCONFKEEPER_H_

#include "FAConfig.h"

namespace BlingFire
{

class FALDB;
class FARSDfa_pack_triv;
class FAState2Ow_pack_triv;
class FAMultiMap_pack_fixed;
class FARSDfaCA;
class FAState2OwCA;
class FAMultiMapCA;

///
/// Keeps configuration for word-guesser module.
///
/// Note: Resource pointers may not be initialized and be NULL.
///

class FAW2PConfKeeper {

public:
    FAW2PConfKeeper ();
    ~FAW2PConfKeeper ();

public:
    /// LDB is used to get the data, initialization vector indicates which 
    /// W2P data to use
    void Initialize (const FALDB * pLDB, const int * pValues, const int Size);
    /// returns object into the initial state
    void Clear ();

public:
    const FARSDfaCA * GetRsDfa () const;
    const FAState2OwCA * GetState2Ow () const;
    const bool GetIgnoreCase () const;
    // the numrical value corresponding to 1 of the prob, prob guesser only
    const int GetMaxProb () const;
    const FAMultiMapCA * GetCharMap () const;
    const float GetMinProbVal () const;
    const float GetMaxProbVal () const;

private:
    const FALDB * m_pLDB;
    FARSDfa_pack_triv * m_pDfa;
    FAState2Ow_pack_triv * m_pState2Ow;
    bool m_IgnoreCase;
    int m_MaxProb;
    FAMultiMap_pack_fixed * m_pCharMap;
    float m_MinProbVal;
    float m_MaxProbVal;
};

}

#endif
