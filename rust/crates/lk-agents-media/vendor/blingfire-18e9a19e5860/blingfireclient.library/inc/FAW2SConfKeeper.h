/**
 * Copyright (c) Microsoft Corporation. All rights reserved.
 * Licensed under the MIT License.
 */


#ifndef _FA_W2SCONFKEEPER_H_
#define _FA_W2SCONFKEEPER_H_

#include "FAConfig.h"

namespace BlingFire
{

class FALDB;
class FARSDfa_pack_triv;
class FAState2Ow_pack_triv;
class FARSDfaCA;
class FAState2OwCA;

///
/// Keeps configuration for word-segmentaion module.
///
/// Note: Resource pointers may not be initialized and be NULL.
///

class FAW2SConfKeeper {

public:
    FAW2SConfKeeper ();
    ~FAW2SConfKeeper ();

public:
    /// this LDB will be used to get the data from
    void SetLDB (const FALDB * pLDB);
    /// initialization vector
    void Init (const int * pValues, const int Size);
    /// returns object into the initial state
    void Clear ();

public:
    const FARSDfaCA * GetRsDfa () const;
    const FAState2OwCA * GetState2Ow () const;
    const int GetMinInputLen () const;
    const int GetMinSegLen () const;
    const bool GetIgnoreCase () const;
    const int GetDirection () const;
    const int GetThreshold () const;

private:
    const FALDB * m_pLDB;
    FARSDfa_pack_triv * m_pDfa;
    FAState2Ow_pack_triv * m_pState2Ow;
    int m_MinInputLen;
    int m_MinSegLen;
    int m_Threshold;
    bool m_IgnoreCase;
    int m_Dir;
};

}

#endif
