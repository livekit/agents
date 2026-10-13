/**
 * Copyright (c) Microsoft Corporation. All rights reserved.
 * Licensed under the MIT License.
 */


#ifndef _FA_LADCONFKEEPER_H_
#define _FA_LADCONFKEEPER_H_

#include "FAConfig.h"

namespace BlingFire
{

class FALDB;
class FAMultiMapCA;
class FAMultiMap_pack_fixed;

///
/// Keeps LAD specific parameters
///

class FALadConfKeeper {

public:
    FALadConfKeeper ();
    ~FALadConfKeeper ();

public:
    /// LDB is used to get the data
    void Initialize (const FALDB * pLDB, const int * pValues, const int Size);
    /// returns object into the initial state
    void Clear ();

public:
    const int GetMaxTag () const;
    const int GetUnkTag () const;
    const int GetOrder () const;
    const int GetMinOrder () const;
    const int GetMaxCount () const;
    const int GetMinMatchRatio () const;
    const int GetMinWordMatchRatio () const;
    const FAMultiMapCA * GetCharMap () const;
    const FAMultiMapCA * GetC2SMap () const;
    const FAMultiMapCA * GetS2LMap () const;
    const int GetMinScriptTag () const;
    const int GetMaxScriptTag () const;

private:
    // n-gram order
    int m_Order;
    // n-gram min backoff order
    int m_MinOrder;
    // maximum amount of n-grams to use
    int m_MaxCount;
    /// maximum tag value
    int m_MaxTag;
    /// Unknown language tag value
    int m_UnkTag;
    /// percent of n-grams should match
    int m_MinMatchRatio;
    /// percent of words should match
    int m_MinWordMatchRatio;
    /// char maps
    FAMultiMap_pack_fixed * m_pCharMap;
    FAMultiMap_pack_fixed * m_pC2SMap;
    FAMultiMap_pack_fixed * m_pS2LMap;
    /// min/max script tag values
    int m_MinScriptTag;
    int m_MaxScriptTag;
};

}

#endif
