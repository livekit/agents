/**
 * Copyright (c) Microsoft Corporation. All rights reserved.
 * Licensed under the MIT License.
 */


#ifndef _FA_WBDCONFKEEPER_H_
#define _FA_WBDCONFKEEPER_H_

#include "FAConfig.h"

namespace BlingFire
{

class FALDB;
class FARSDfa_pack_triv;
class FAState2Ow_pack_triv;
class FAState2Ows_pack_triv;
class FAMultiMap_pack;
class FAMultiMap_pack_fixed;
class FARSDfaCA;
class FAMultiMapCA;
class FAState2OwCA;
class FAState2OwsCA;

///
/// Keeps dictionary object configuration and common containers.
///
/// Note: The pointers can be NULL, if yet not initialized
///

class FAWbdConfKeeper {

public:
    FAWbdConfKeeper ();
    ~FAWbdConfKeeper ();

public:
    /// initialization vector
    void Initialize (const FALDB * pLDB, const int * pValues, const int Size);
    /// returns object into the initial state
    void Clear ();

public:
    const FARSDfaCA * GetRsDfa () const;
    const FAState2OwCA * GetState2Ow () const;
    const FAState2OwsCA * GetState2Ows () const;
    const FAMultiMapCA * GetMMap () const;
    const bool GetIgnoreCase() const;
    const FAMultiMapCA * GetCharMap () const;
    /// return maximum call depth, if functions are used
    const int GetMaxDepth () const;
    const int GetWbdTagEos () const;
    const int GetWbdTagEop () const;
    const int GetWbdTagPunkt () const;
    const int GetWbdTagWord () const;
    const int GetWbdTagXWord () const;
    const int GetWbdTagSeg () const;
    const int GetWbdTagIgnore () const;
    const int GetMaxTag () const;
    /// returns an array of function initial states, the array contains -1
    /// for undefined function ids
    const unsigned int GetFnIniStates (const int ** ppFn2Ini) const;
    /// returns an optional map with action data, if this map does not exist returns NULL
    const FAMultiMapCA * GetActData () const;
    /// returns maximum allowed token length, FALimits::MaxWordLen is used by default
    const int GetMaxTokenLength () const;

public:
    // overrides RS Dfa from the LDB
    void SetRsDfa (const FARSDfaCA * pDfa);
    // overrides State -> Ow from the LDB
    void SetState2Ow (const FAState2OwCA * pState2Ow);
    // overrides Action map from the LDB
    void SetMMap (const FAMultiMapCA * pMMap);
    // overrides the ignore case from the LDB
    void SetIgnoreCase (const bool IgnoreCase);
    // overrides maximum allowed depth for function calls
    void SetMaxDepth (const int MaxRecDepth);
    // overrides action data map
    void SetActData (const FAMultiMapCA * pActData);

private:
    /// sets up rules' actions map
    inline void CalcFnIniStates ();

private:
    FARSDfa_pack_triv * m_pRsDfa;
    FAState2Ow_pack_triv * m_pState2Ow;
    FAState2Ows_pack_triv * m_pState2Ows;
    FAMultiMap_pack * m_pMMapTriv;
    FAMultiMap_pack_fixed * m_pCharMap;
    FAMultiMap_pack * m_pActData;
    bool m_IgnoreCase;
    int m_MaxDepth;
    int m_TagEos;
    int m_TagEop;
    int m_TagPunkt;
    int m_TagWord;
    int m_TagXWord;
    int m_TagSeg;
    int m_TagIgnore;
    int m_MaxTag;

    const FARSDfaCA * m_pRsDfaA;
    const FAState2OwCA * m_pState2OwA;
    const FAMultiMapCA * m_pMMapA;
    const FAMultiMapCA * m_pActDataCA;

    /// maps function id into an initial state, or to -1 if not valid
    int * m_pFn2Ini;
    unsigned int m_Fn2IniSize;

    /// maximum token length
    int m_MaxTokenLength;

    enum {
        DefMaxDepth = 5,
        MinActSize = 3,
        MaxFunctionId = 65536,
    };
};

}

#endif
