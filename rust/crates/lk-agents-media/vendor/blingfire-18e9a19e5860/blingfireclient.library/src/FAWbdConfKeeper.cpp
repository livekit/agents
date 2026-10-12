/**
 * Copyright (c) Microsoft Corporation. All rights reserved.
 * Licensed under the MIT License.
 */


#include "blingfire-client_src_pch.h"
#include "FAConfig.h"
#include "FAWbdConfKeeper.h"
#include "FAFsmConst.h"
#include "FALDB.h"
#include "FARSDfa_pack_triv.h"
#include "FAState2Ow_pack_triv.h"
#include "FAState2Ows_pack_triv.h"
#include "FAMultiMap_pack.h"
#include "FAMultiMap_pack_mph.h"
#include "FAMultiMap_pack_fixed.h"
#include "FALimits.h"

namespace BlingFire
{

FAWbdConfKeeper::FAWbdConfKeeper () :
    m_pRsDfa (NULL),
    m_pState2Ow (NULL),
    m_pState2Ows (NULL),
    m_pMMapTriv(NULL),
    m_pActData (NULL),
    m_IgnoreCase(false),
    m_pRsDfaA (NULL),
    m_pState2OwA (NULL),
    m_pMMapA (NULL),
    m_pCharMap (NULL),
    m_MaxDepth (DefMaxDepth),
    m_TagEos (0),
    m_TagEop (0),
    m_TagPunkt (0),
    m_TagWord (0),
    m_TagXWord (0),
    m_TagSeg (0),
    m_TagIgnore (0),
    m_MaxTag (0),
    m_pActDataCA (NULL),
    m_pFn2Ini (NULL),
    m_Fn2IniSize (0),
    m_MaxTokenLength (FALimits::MaxWordLen)
{}


FAWbdConfKeeper::~FAWbdConfKeeper ()
{
    FAWbdConfKeeper::Clear ();
}


void FAWbdConfKeeper::Initialize (const FALDB * pLDB, const int * pValues, const int Size)
{
    LogAssert (pLDB);
    LogAssert (pValues || 0 >= Size);

    int FsmType = FAFsmConst::TYPE_MOORE_DFA;
    const unsigned char * pFsmDump = NULL;

    FAWbdConfKeeper::Clear ();

    for (int i = 0; i < Size; ++i) {

        const int Param = pValues [i];

        switch (Param) {

        case FAFsmConst::PARAM_MAP_MODE:
        {
            const int MMType = pValues [++i];
            LogAssert (FAFsmConst::MODE_PACK_TRIV == MMType);
            break;
        }
        case FAFsmConst::PARAM_DEPTH:
        {
            m_MaxDepth = pValues [++i];
            LogAssert (0 <= m_MaxDepth);
            break;
        }
        case FAFsmConst::PARAM_MAX_LENGTH:
        {
            m_MaxTokenLength = pValues [++i];
            LogAssert (0 <= m_MaxTokenLength);
            break;
        }
        case FAFsmConst::PARAM_IGNORE_CASE:
        {
            m_IgnoreCase = true;
            break;
        }
        case FAFsmConst::PARAM_FSM_TYPE:
        {
            FsmType = pValues [++i];
            LogAssert (FAFsmConst::TYPE_MOORE_DFA == FsmType ||
                FAFsmConst::TYPE_MOORE_MULTI_DFA == FsmType);
            break;
        }
        case FAFsmConst::PARAM_FSM:
        {
            const int DumpNum = pValues [++i];
            pFsmDump = pLDB->GetDump (DumpNum);
            LogAssert (pFsmDump);

            if (!m_pRsDfa) {
                m_pRsDfa = new FARSDfa_pack_triv;
                LogAssert (m_pRsDfa);
            }
            m_pRsDfa->SetImage (pFsmDump);
            m_pRsDfaA = m_pRsDfa;

            break;
        }
        case FAFsmConst::PARAM_MULTI_MAP:
        {
            const int DumpNum = pValues [++i];
            const unsigned char * pDump = pLDB->GetDump (DumpNum);
            LogAssert (pDump);

            if (!m_pMMapTriv) {
                m_pMMapTriv = new FAMultiMap_pack;
                LogAssert (m_pMMapTriv);
            }
            m_pMMapTriv->SetImage (pDump);
            m_pMMapA = m_pMMapTriv;

            break;
        }
        case FAFsmConst::PARAM_CHARMAP:
        {
            const int DumpNum = pValues [++i];
            const unsigned char * pDump = pLDB->GetDump (DumpNum);
            LogAssert (pDump);

            if (!m_pCharMap) {
                m_pCharMap = new FAMultiMap_pack_fixed;
                LogAssert (m_pCharMap);
            }
            m_pCharMap->SetImage (pDump);

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
            m_pActDataCA = m_pActData;
            break;
        }
        case FAFsmConst::PARAM_PUNKT:
        {
            m_TagPunkt = pValues [++i];
            break;
        }
        case FAFsmConst::PARAM_EOS:
        {
            m_TagEos = pValues [++i];
            break;
        }
        case FAFsmConst::PARAM_EOP:
        {
            m_TagEop = pValues [++i];
            break;
        }
        case FAFsmConst::PARAM_WORD:
        {
            m_TagWord = pValues [++i];
            break;
        }
        case FAFsmConst::PARAM_XWORD:
        {
            m_TagXWord = pValues [++i];
            break;
        }
        case FAFsmConst::PARAM_SEG:
        {
            const int Tag = pValues [++i];
            m_TagSeg = Tag;
            break;
        }
        case FAFsmConst::PARAM_IGNORE:
        {
            m_TagIgnore = pValues [++i];
            break;
        }
        case FAFsmConst::PARAM_MAX_TAG:
        {
            m_MaxTag = pValues [++i];
            break;
        }

        default:
            LogAssert (false);
        }
    } // of for (int i = 0; ...

    /// create an appropriate reaction map, if needed
    if (NULL == pFsmDump) {
        return;
    }

    if (FAFsmConst::TYPE_MOORE_DFA == FsmType) {
        if (!m_pState2Ow) {
            m_pState2Ow = new FAState2Ow_pack_triv;
            LogAssert (m_pState2Ow);
        }
        m_pState2Ow->SetImage (pFsmDump);
        m_pState2OwA = m_pState2Ow;
    } else if (FAFsmConst::TYPE_MOORE_MULTI_DFA == FsmType) {
        if (!m_pState2Ows) {
            m_pState2Ows = new FAState2Ows_pack_triv;
            LogAssert (m_pState2Ows);
        }
        m_pState2Ows->SetImage (pFsmDump);
    } else {
        // wrong FsmType value
        LogAssert (false);
    }

    // calculates functions' initial states
    CalcFnIniStates ();
}


///
/// This method calculates initial states corresponding to each function
/// of the grammar. The functions are represented within the same automaton
/// except regular expressions corresponding to the function bodies are
/// prefixed with "$ \xNNNN", where \xNNNN is the function id. In order
/// to calulate initial states corresponding to each function this method
/// looks up the automaton by "$ FnId" for each FnId and stores 
/// corresponding states into the array m_pFn2Ini [FnId] = State. This is
/// done for optimization puposees to optimize a function call.
///

inline void FAWbdConfKeeper::CalcFnIniStates ()
{
    if (m_pMMapA && m_pRsDfaA) {

        // get the initial state
        const int Initial = m_pRsDfaA->GetInitial ();
        // get the right-anchor ($) state
        const int StateR = m_pRsDfaA->GetDest (Initial, FAFsmConst::IW_R_ANCHOR);

        // rules do not have any functions
        if (-1 == StateR)
            return;

        const int * pAct;
        int ActSize;
        int ActId = 0;
        int MaxFnId = -1;

        while (-1 != (ActSize = m_pMMapA->Get (ActId++, &pAct))) {

            // invalid action
            LogAssert (pAct && MinActSize <= ActSize);

            // skip left and right context
            int i = 2;
            // find the place where the function ids start
            for (;i < ActSize; ++i) {
                if (0 == pAct [i] && i + 1 < ActSize) {
                    // skip 0-delimiter
                    i++;
                    break;
                }
            }
            // update the maximum function id value
            for (; i < ActSize; ++i) {

                const int FnId = pAct [i];
                // bad function id
                LogAssert (0 <= FnId);

                if (MaxFnId < FnId) {
                    MaxFnId = FnId;
                }
            }
        } // of while (-1 != (ActSize = ...

        // rules do not have any functions
        if (-1 == MaxFnId)
            return;

        // too many functions, something is messed up
        LogAssert (MaxFnId <= MaxFunctionId);

        m_Fn2IniSize = MaxFnId + 1;
        m_pFn2Ini = new int [m_Fn2IniSize];
        LogAssert (m_pFn2Ini);

        // save the state for the "_main" function
        m_pFn2Ini [0] = Initial;

        // save the states for the rest of the functions
        for (int FnId = 1; FnId <= MaxFnId; ++FnId) {
            const int Dst = m_pRsDfaA->GetDest (StateR, FnId);
            LogAssert (0 <= Dst || -1 == Dst);
            m_pFn2Ini [FnId] = Dst;
        }

    } // of if (m_pMMapA && m_pRsDfaA) ...
}


void FAWbdConfKeeper::Clear ()
{
    if (m_pRsDfa) {
        delete m_pRsDfa;
        m_pRsDfa = NULL;
    }
    if (m_pState2Ow) {
        delete m_pState2Ow;
        m_pState2Ow = NULL;
    }
    if (m_pState2Ows) {
        delete m_pState2Ows;
        m_pState2Ows = NULL;
    }
    if (m_pMMapTriv) {
        delete m_pMMapTriv;
        m_pMMapTriv = NULL;
    }
    if (m_pCharMap) {
        delete m_pCharMap;
        m_pCharMap = NULL;
    }
    if (m_pActData) {
        delete m_pActData;
        m_pActData = NULL;
    }
    if (m_pFn2Ini) {
        delete [] m_pFn2Ini;
        m_pFn2Ini = NULL;
    }
    m_Fn2IniSize = 0;

    m_pMMapA = NULL;
    m_pState2OwA = NULL;
    m_pMMapA = NULL;
    m_pActDataCA = NULL;

    m_IgnoreCase = false;

    m_TagEos = 0;
    m_TagEop = 0;
    m_TagPunkt = 0;
    m_TagWord = 0;
    m_TagXWord = 0;
    m_TagSeg = 0;
    m_TagIgnore = 0;
    m_MaxTag = 0;

    m_MaxDepth = DefMaxDepth;

    m_MaxTokenLength = FALimits::MaxWordLen;
}


const FARSDfaCA * FAWbdConfKeeper::GetRsDfa () const
{
    return m_pRsDfaA;
}

const FAState2OwCA * FAWbdConfKeeper::GetState2Ow () const
{
    return m_pState2OwA;
}

const FAState2OwsCA * FAWbdConfKeeper::GetState2Ows () const
{
    return m_pState2Ows;
}

const FAMultiMapCA * FAWbdConfKeeper::GetMMap () const
{
    return m_pMMapA;
}

const FAMultiMapCA * FAWbdConfKeeper::GetActData () const
{
    return m_pActDataCA;
}

const bool FAWbdConfKeeper::GetIgnoreCase () const
{
    return m_IgnoreCase;
}

const int FAWbdConfKeeper::GetWbdTagEos () const
{
    return m_TagEos;
}

const int FAWbdConfKeeper::GetWbdTagEop () const
{
    return m_TagEop;
}

const int FAWbdConfKeeper::GetWbdTagWord () const
{
    return m_TagWord;
}

const int FAWbdConfKeeper::GetWbdTagPunkt () const
{
    return m_TagPunkt;
}

const int FAWbdConfKeeper::GetWbdTagXWord () const
{
    return m_TagXWord;
}

const int FAWbdConfKeeper::GetWbdTagSeg () const
{
    return m_TagSeg;
}

const int FAWbdConfKeeper::GetWbdTagIgnore () const
{
    return m_TagIgnore;
}

const int FAWbdConfKeeper::GetMaxDepth () const
{
    return m_MaxDepth;
}

const unsigned int FAWbdConfKeeper::GetFnIniStates (const int ** ppFn2Ini) const
{
    LogAssert (ppFn2Ini);
    *ppFn2Ini = m_pFn2Ini;
    return m_Fn2IniSize;
}

void FAWbdConfKeeper::SetRsDfa (const FARSDfaCA * pDfa)
{
    m_pRsDfaA = pDfa;
}

void FAWbdConfKeeper::SetState2Ow (const FAState2OwCA * pState2Ow)
{
    m_pState2OwA = pState2Ow;
}

void FAWbdConfKeeper::SetMMap (const FAMultiMapCA * pMMap)
{
    m_pMMapA = pMMap;
    // calculates functions' initial states
    CalcFnIniStates ();
}

void FAWbdConfKeeper::SetActData (const FAMultiMapCA * pActData)
{
    m_pActDataCA = pActData;
}

void FAWbdConfKeeper::SetIgnoreCase (const bool IgnoreCase)
{
    m_IgnoreCase = IgnoreCase;
}

void FAWbdConfKeeper::SetMaxDepth (const int MaxRecDepth)
{
    m_MaxDepth = MaxRecDepth;
    LogAssert (0 <= m_MaxDepth);
}

const FAMultiMapCA * FAWbdConfKeeper::GetCharMap () const
{
    return m_pCharMap;
}

const int FAWbdConfKeeper::GetMaxTag () const
{
    return m_MaxTag;
}

const int FAWbdConfKeeper::GetMaxTokenLength () const
{
    return m_MaxTokenLength;
}

}
