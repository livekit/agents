/**
 * Copyright (c) Microsoft Corporation. All rights reserved.
 * Licensed under the MIT License.
 */


#include "blingfire-client_src_pch.h"
#include "FAConfig.h"
#include "FAParserConfKeeper.h"
#include "FAFsmConst.h"
#include "FALDB.h"
#include "FAWREConfCA.h"
#include "FAWREConf_pack.h"
#include "FAMultiMapCA.h"
#include "FAMultiMap_pack.h"
#include "FARSDfaCA.h"

namespace BlingFire
{

FAParserConfKeeper::FAParserConfKeeper () :
    m_pWre (NULL),
    m_pActs (NULL),
    m_pActData (NULL),
    m_IgnoreCase (false),
    m_MaxDepth (DefMaxDepth),
    m_MaxPassCount (DefMaxPassCount),
    m_pWreCA (NULL),
    m_pActsCA (NULL),
    m_pActDataCA (NULL),
    m_pFn2Ini (NULL),
    m_Fn2IniSize (0)
{}


FAParserConfKeeper::~FAParserConfKeeper ()
{
    FAParserConfKeeper::Clear ();
}


void FAParserConfKeeper::Initialize (const FALDB * pLDB, const int * pValues, const int Size)
{
    LogAssert (pLDB);
    LogAssert (pValues || 0 >= Size);

    FAParserConfKeeper::Clear ();

    for (int i = 0; i < Size; ++i) {

        const int Param = pValues [i];

        switch (Param) {

        case FAFsmConst::PARAM_DEPTH:
        {
            m_MaxDepth = pValues [++i];
            LogAssert (0 <= m_MaxDepth);
            break;
        }
        case FAFsmConst::PARAM_MAX_PASS_COUNT:
        {
            m_MaxPassCount = pValues [++i];
            LogAssert (0 < m_MaxPassCount);
            break;
        }
        case FAFsmConst::PARAM_IGNORE_CASE:
        {
            m_IgnoreCase = true;
            break;
        }
        case FAFsmConst::PARAM_WRE_CONF:
        {
            const int DumpNum = pValues [++i];
            const unsigned char * pDump = pLDB->GetDump (DumpNum);
            LogAssert (pDump);

            if (!m_pWre) {
                m_pWre = new FAWREConf_pack;
                LogAssert (m_pWre);
            }
            m_pWre->SetImage (pDump);
            break;
        }
        case FAFsmConst::PARAM_ACTS:
        {
            const int DumpNum = pValues [++i];
            const unsigned char * pDump = pLDB->GetDump (DumpNum);
            LogAssert (pDump);

            if (!m_pActs) {
                m_pActs = new FAMultiMap_pack;
                LogAssert (m_pActs);
            }
            m_pActs->SetImage (pDump);
            break;
        }
        case FAFsmConst::PARAM_ACT_DATA:
        {
            const int DumpNum = pValues [++i];
            const unsigned char * pDump = pLDB->GetDump (DumpNum);
            LogAssert (pDump);

            if (!m_pActData) {
                m_pActData = new FAMultiMap_pack;
                LogAssert (m_pActData);
            }
            m_pActData->SetImage (pDump);
            break;
        }

        default:
            LogAssert (false);
        }
    } // of for (int i = 0; ...

    LogAssert (m_pWre && m_pActs);

    m_pWreCA = m_pWre;
    m_pActsCA = m_pActs;
    m_pActDataCA = m_pActData;

    // calculates functions' initial states
    CalcFnInitialStates (m_pWre, m_pActsCA);
}


///
/// This method calculates initial states corresponding to each function
/// of the grammar. The functions are represented within the same automaton
/// except WREs corresponding to the function bodies are prefixed with 
/// "$ NNNN", where NNNN is the function name (a tag defined in the tagset
/// or a new one with the value FnId). In order to calulate initial states 
/// corresponding to each function this method matches $ NNNN with the WRE
///  and the resulting state is an initial state of the function NNNN.
////

inline const int FAParserConfKeeper::
    GetFnId2State (const int FnId, const FARSDfaCA * pDfa, const int TokenType, const int TagOwBase)
{
    DebugLogAssert (pDfa);
    DebugLogAssert (0 <= FnId);

    int State = pDfa->GetInitial ();

    /// Transition by $
    if (FAFsmConst::WRE_TT_TEXT & TokenType && -1 != State) {
      State = pDfa->GetDest (State, FAFsmConst::IW_R_ANCHOR);
    }
    if (FAFsmConst::WRE_TT_TAGS & TokenType && -1 != State) {
      State = pDfa->GetDest (State, FAFsmConst::IW_R_ANCHOR);
    }
    if (FAFsmConst::WRE_TT_DCTS & TokenType && -1 != State) {
      State = pDfa->GetDest (State, FAFsmConst::IW_R_ANCHOR);
    }
    LogAssert (-1 != State);

    /// Transition by FnId tag value
    if (FAFsmConst::WRE_TT_TEXT & TokenType && -1 != State) {
      State = pDfa->GetDest (State, FAFsmConst::IW_ANY);
    }
    if (FAFsmConst::WRE_TT_TAGS & TokenType && -1 != State) {
      State = pDfa->GetDest (State, FnId + TagOwBase);
    }
    if (FAFsmConst::WRE_TT_DCTS & TokenType && -1 != State) {
      State = pDfa->GetDest (State, FAFsmConst::IW_ANY);
    }
    LogAssert (-1 != State);

    return State;
}


inline void FAParserConfKeeper::
    CalcFnInitialStates (const FAWREConfCA * pWre, const FAMultiMapCA * pActs)
{
    LogAssert (pWre);

    const FARSDfaCA * pDfa = pWre->GetDfa1 ();

    if (!pActs || !pDfa) {
        // no error
        return;
    }

    // get token type mask
    const int TokenType = pWre->GetTokenType ();
    // get tag Ow base
    const int TagOwBase = pWre->GetTagOwBase ();

    const int * pAct = NULL;
    int ActSize = 0;
    int ActId = 0;
    int MaxFnId = -1;

    // iterate thru all the actions
    while (-1 != (ActSize = pActs->Get (ActId++, &pAct))) {

        // invalid action
        LogAssert (pAct && MinActSize <= ActSize);

        int i = ActSize;

        // just one tag
        if (MinActSize == ActSize && 0 != pAct [MinActSize - 1]) {
            continue;
        // delimiter and fn(s), but no tag
        } else if (MinActSize < ActSize && 0 == pAct [MinActSize - 1]) {
            i = MinActSize;
        // tag, delimiter and fn(s)
        } else if (MinActSize + 1 < ActSize && 0 == pAct [MinActSize]) {
            i = MinActSize + 1;
        } else {
            // invalid action
            LogAssert (false);
        }

        // get the maximum function id
        for (; i < ActSize; ++i) {

            const int FnId = pAct [i];
            // bad function id
            LogAssert (0 <= FnId);

            if (MaxFnId < FnId) {
                MaxFnId = FnId;
            }
        }
    } // of while (-1 != (ActSize = ...

    if (-1 == MaxFnId)
        return;

    // allocate array mapping FnId to its initial state
    m_Fn2IniSize = MaxFnId + 1;
    m_pFn2Ini = new int [m_Fn2IniSize];
    LogAssert (m_pFn2Ini);

    for (int j = 0; j <= MaxFnId; ++j) {
        m_pFn2Ini [j] = -1;
    }

    const int Initial = pDfa->GetInitial ();
    ActId = 0;

    // iterate thru all the actions again
    while (-1 != (ActSize = pActs->Get (ActId++, &pAct))) {

        int i = ActSize;

        // just one tag
        if (MinActSize == ActSize) {
            continue;
        // delimiter and fn(s), but no tag
        } else if (0 == pAct [MinActSize - 1]) {
            LogAssert (MinActSize < ActSize);
            i = MinActSize;
        // tag, delimiter and fn(s)
        } else {
            LogAssert (MinActSize + 1 < ActSize && 0 == pAct [MinActSize]);
            i = MinActSize + 1;
        }

        // get the maximum function id
        for (; i < ActSize; ++i) {

            const int FnId = pAct [i];
            if (-1 != m_pFn2Ini [FnId]) {
                continue;
            }

            // get the function's starting state, if not set yet
            if (0 != FnId) {
                const int FnInitial = GetFnId2State (FnId, pDfa, TokenType, TagOwBase);
                m_pFn2Ini [FnId] = FnInitial;
            } else {
                m_pFn2Ini [FnId] = Initial;
            }
        }
    } // of while (-1 != (ActSize = ...
}


void FAParserConfKeeper::Clear ()
{
    if (m_pWre) {
        delete m_pWre;
        m_pWre = NULL;
    }
    if (m_pActs) {
        delete m_pActs;
        m_pActs = NULL;
    }
    if (m_pActData) {
        delete m_pActData;
        m_pActData = NULL;
    }
    if (m_pFn2Ini) {
        delete [] m_pFn2Ini;
        m_pFn2Ini = NULL;
    }

    m_pWreCA = NULL;
    m_pActsCA = NULL;
    m_pActDataCA = NULL;
    m_IgnoreCase = false;
    m_MaxDepth = DefMaxDepth;
    m_MaxPassCount = DefMaxPassCount;
    m_Fn2IniSize = 0;
}


const FAWREConfCA * FAParserConfKeeper::GetWre () const
{
    return m_pWreCA;
}

const FAMultiMapCA * FAParserConfKeeper::GetActs () const
{
    return m_pActsCA;
}

const FAMultiMapCA * FAParserConfKeeper::GetActData () const
{
    return m_pActDataCA;
}

const bool FAParserConfKeeper::GetIgnoreCase () const
{
    return m_IgnoreCase;
}

const int FAParserConfKeeper::GetMaxDepth () const
{
    return m_MaxDepth;
}

const int FAParserConfKeeper::GetMaxPassCount () const
{
    return m_MaxPassCount;
}


void FAParserConfKeeper::SetWre (const FAWREConfCA * pWre)
{
    m_pWreCA = pWre;
}

void FAParserConfKeeper::SetActs (const FAMultiMapCA * pActs)
{
    m_pActsCA = pActs;
}

void FAParserConfKeeper::SetActData (const FAMultiMapCA * pActData)
{
    m_pActDataCA = pActData;
}

void FAParserConfKeeper::SetIgnoreCase (const bool IgnoreCase)
{
    m_IgnoreCase = IgnoreCase;
}

void FAParserConfKeeper::SetMaxDepth (const int MaxDepth)
{
    m_MaxDepth = MaxDepth;
}

const unsigned int FAParserConfKeeper::GetFnIniStates (const int ** ppFn2Ini) const
{
    LogAssert (ppFn2Ini);
    *ppFn2Ini = m_pFn2Ini;
    return m_Fn2IniSize;
}

}
