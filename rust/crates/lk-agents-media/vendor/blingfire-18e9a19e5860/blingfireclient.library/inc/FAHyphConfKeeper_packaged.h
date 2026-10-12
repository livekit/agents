/**
 * Copyright (c) Microsoft Corporation. All rights reserved.
 * Licensed under the MIT License.
 */


#ifndef _FA_HYPHCONFKEEPER_H_
#define _FA_HYPHCONFKEEPER_H_

#include "FAConfig.h"

namespace BlingFire
{

class FALDB;
class FARSDfa_pack_triv;
class FAState2Ow_pack_triv;
class FAMultiMap_pack;
class FARSDfaCA;
class FAState2OwCA;
class FAMultiMapCA;
class FAMultiMap_pack_fixed;

///
/// Keeps dictionary object configuration and common containers.
///
/// Note: The pointers can be NULL, if yet not initialized
///

class FAHyphConfKeeper {

public:
    FAHyphConfKeeper ();
    ~FAHyphConfKeeper ();

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
    const FAMultiMapCA * GetI2Info () const;
    const bool GetIgnoreCase () const;
    const int GetLeftAnchor () const;
    const int GetRightAnchor () const;
    const int GetMinPatLen () const;
    const int GetHyphType () const;
    const bool GetNormSegs () const;
    const int GetNoHyphLen () const;
    const FAMultiMapCA * GetCharMap () const;

private:
    // input LDB
    const FALDB * m_pLDB;
    // data
    FARSDfa_pack_triv * m_pRsDfa;
    FAState2Ow_pack_triv * m_pState2Ow;
    FAMultiMap_pack * m_pI2Info;
    bool m_IgnoreCase;
    int m_MinPatLen;
    int m_LeftAnchor;
    int m_RightAnchor;
    int m_HyphType;
    bool m_NormSegs;
    int m_NoHyphLen;
    FAMultiMap_pack_fixed * m_pCharMap;

    enum {
        DefMinPatLen = 3,
        DefMaxNoHyphLen = 5,
    };
};

}

#endif
