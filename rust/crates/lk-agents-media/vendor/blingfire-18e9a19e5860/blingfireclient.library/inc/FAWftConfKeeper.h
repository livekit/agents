/**
 * Copyright (c) Microsoft Corporation. All rights reserved.
 * Licensed under the MIT License.
 */


#ifndef _FA_WFTCONFKEEPER_H_
#define _FA_WFTCONFKEEPER_H_

#include "FAConfig.h"

namespace BlingFire
{

class FALDB;
class FARSDfa_pack_triv;
class FAState2Ows_pack_triv;
class FAMultiMap_pack;
class FAMultiMap_pack_mph;
class FAGetIWs_pack_triv;
class FARSDfaCA;
class FAState2OwsCA;
class FAMultiMapCA;
class FAGetIWsCA;
class FAMultiMap_pack_fixed;

///
/// Keeps configuration for any of word-form-transformation modules.
///
/// Note: Resource pointers may not be initialized and be NULL.
///

class FAWftConfKeeper {

public:
    FAWftConfKeeper ();
    ~FAWftConfKeeper ();

public:
    /// LDB is used to get the data, initialization vector indicates which 
    /// WFT data to use
    void Initialize (const FALDB * pLDB, const int * pValues, const int Size);
    /// returns object into the initial state
    void Clear ();

public:
    const FARSDfaCA * GetRsDfa () const;
    const FAState2OwsCA * GetState2Ows () const;
    const FAMultiMapCA * GetActs () const;
    const FAGetIWsCA * GetIws () const;
    const bool GetNoTrUse () const;
    const bool GetDictMode () const;
    const bool GetIgnoreCase () const;
    const bool GetUseNfst () const;
    const FAMultiMapCA * GetCharMap () const;

private:
    const FALDB * m_pLDB;
    FARSDfa_pack_triv * m_pDfa;
    FAState2Ows_pack_triv * m_pState2Ows;
    const FAMultiMapCA * m_pActsA;
    FAMultiMap_pack * m_pActs_triv;
    FAMultiMap_pack_mph * m_pActs_mph;
    FAGetIWs_pack_triv * m_pIws;
    bool m_NoTrUse;
    bool m_DictMode;
    bool m_IgnoreCase;
    bool m_UseNfst;
    FAMultiMap_pack_fixed * m_pCharMap;
};

}

#endif
