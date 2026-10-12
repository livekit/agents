/**
 * Copyright (c) Microsoft Corporation. All rights reserved.
 * Licensed under the MIT License.
 */


#ifndef _FA_W2V_INTERPRETER_T_H_
#define _FA_W2V_INTERPRETER_T_H_

#include "FAConfig.h"
#include "FAFsmConst.h"
#include "FALimits.h"
#include "FAMorphLDB_t_packaged.h"
#include "FANfstLookupTools_t.h"
#include "FASecurity.h"

namespace BlingFire
{

///
/// W2V (spelling variants) function run-time interpreter.
///

template < class Ty >
class FAW2VInterpreter_t {

public:
    /// creates uninitialized object
    FAW2VInterpreter_t (FAAllocatorA * pAlloc);
    ~FAW2VInterpreter_t ();

public:
    void SetLDB (const FAMorphLDB_t < Ty > * pLDB);

    /// word -> { variants }
    /// returns -1 in case of error
    const int ProcessW2V (
            const Ty * pIn,
            const int InSize,
            __out_ecount_opt(MaxOutSize) Ty * pOut,
            const int MaxOutSize
        );

private:
    // returns object into initial state
    inline void Clear ();
    // makes initialization and takes care of m_Ready_w2v flag
    inline void InitW2V ();

private:
    // keeps morphology resources
    const FAMorphLDB_t < Ty > * m_pLDB;
    FAAllocatorA * m_pAlloc;
    // true if object was initialized
    bool m_Ready_w2v;
    // word -> { variant } NFST
    FANfstLookupTools_t < Ty > * m_pNfstW2V;
};


template < class Ty >
FAW2VInterpreter_t< Ty >::FAW2VInterpreter_t (FAAllocatorA * pAlloc) :
    m_pLDB (NULL),
    m_pAlloc (pAlloc),
    m_Ready_w2v (false),
    m_pNfstW2V (NULL)
{}


template < class Ty >
FAW2VInterpreter_t< Ty >::~FAW2VInterpreter_t ()
{
    FAW2VInterpreter_t< Ty >::Clear ();
}


template < class Ty >
void FAW2VInterpreter_t< Ty >::SetLDB (const FAMorphLDB_t < Ty > * pLDB)
{
    m_pLDB = pLDB;

    m_Ready_w2v = false;
}


template < class Ty >
void FAW2VInterpreter_t< Ty >::Clear ()
{
    m_Ready_w2v = false;

    if (m_pNfstW2V) {
        delete m_pNfstW2V;
        m_pNfstW2V = NULL;
    }
}


template < class Ty >
void FAW2VInterpreter_t< Ty >::InitW2V ()
{
    DebugLogAssert (!m_Ready_w2v);

    if (!m_pLDB) {
        return;
    }

    const FAWftConfKeeper * pConf = m_pLDB->GetW2VConf ();
    if (!pConf) {
        return;
    }
    
    const bool UseNfst = pConf->GetUseNfst ();
    FAAssert (UseNfst, FAMsg::InitializationError);

    if (!m_pNfstW2V) {
        m_pNfstW2V = NEW FANfstLookupTools_t < Ty > (m_pAlloc);
    }

///    if (false == pConf->GetNoTrUse ()) {
///        m_pNfstW2V->SetInTr (m_pLDB->GetInTr ());
///        m_pNfstW2V->SetOutTr (m_pLDB->GetOutTr ());
///    }

    m_pNfstW2V->SetConf (pConf);

    m_Ready_w2v = pConf->GetRsDfa () && pConf->GetIws () && \
        pConf->GetActs ();
}


template < class Ty >
const int FAW2VInterpreter_t< Ty >::
    ProcessW2V (
            const Ty * pIn,
            const int InSize,
            __out_ecount_opt(MaxOutSize) Ty * pOut,
            const int MaxOutSize
        )
{
    if (!m_Ready_w2v) {
        InitW2V ();
        if (!m_Ready_w2v)
            return -1;
    } // of if (!m_Ready_w2v) ...

    // check for limits
    if (0 >= InSize || FALimits::MaxWordSize < InSize || !pIn || \
        (NULL == pOut && 0 != MaxOutSize))
        return -1;

    DebugLogAssert (m_pNfstW2V);
    const int OutSize = m_pNfstW2V->Process (pIn, InSize, pOut, MaxOutSize);

    return OutSize;
}

}

#endif
