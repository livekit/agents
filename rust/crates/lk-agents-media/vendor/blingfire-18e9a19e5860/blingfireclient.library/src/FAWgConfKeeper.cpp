/**
 * Copyright (c) Microsoft Corporation. All rights reserved.
 * Licensed under the MIT License.
 */


#include "blingfire-client_src_pch.h"
#include "FAConfig.h"
#include "FAWgConfKeeper.h"
#include "FAFsmConst.h"
#include "FALimits.h"
#include "FALDB.h"
#include "FARSDfa_pack_triv.h"
#include "FAState2Ows_pack_triv.h"
#include "FAMultiMap_pack_fixed.h"
#include "FAUtils_cl.h"

namespace BlingFire
{

FAWgConfKeeper::FAWgConfKeeper () :
    m_pDfa (NULL),
    m_pState2Ows (NULL),
    m_Direction (FAFsmConst::DIR_L2R),
    m_MaxLen (-1),
    m_DefTag (-1),
    m_NoTrUse (false),
    m_DictMode (false),
    m_IgnoreCase (false),
    m_MaxProb (0),
    m_pCharMap (NULL),
    m_MinProbVal (0),
    m_MaxProbVal (0),
    m_fLogScale (false),
    m_TagEos (0),
    m_Order (-1)
{}


FAWgConfKeeper::~FAWgConfKeeper ()
{
    FAWgConfKeeper::Clear ();
}


void FAWgConfKeeper::
    Initialize (const FALDB * pLDB, const int * pValues, const int Size)
{
    LogAssert (pLDB);
    LogAssert (pValues || 0 >= Size);

    FAWgConfKeeper::Clear ();

    for (int i = 0; i < Size; ++i) {

        const int Param = pValues [i];

        switch (Param) {

        case FAFsmConst::PARAM_NO_TR:
        {
            m_NoTrUse = true;
            break;
        }
        case FAFsmConst::PARAM_EOS:
        {
            m_TagEos = pValues [++i];
            break;
        }
        case FAFsmConst::PARAM_ORDER:
        {
            m_Order = pValues [++i];
            LogAssert (0 < m_Order && FALimits::MaxWordLen >= m_Order);
            break;
        }
        case FAFsmConst::PARAM_LOG_SCALE:
        {
            m_fLogScale = true;
            break;
        }
        case FAFsmConst::PARAM_IGNORE_CASE:
        {
            m_IgnoreCase = true;
            break;
        }
        case FAFsmConst::PARAM_DICT_MODE:
        {
            m_DictMode = true;
            break;
        }
        case FAFsmConst::PARAM_MAX_PROB:
        {
            m_MaxProb = pValues [++i];
            LogAssert (0 < m_MaxProb);
            break;
        }
        case FAFsmConst::PARAM_TRIM:
        {
            m_MaxLen = pValues [++i];
            break;
        }
        case FAFsmConst::PARAM_DEFAULT_TAG:
        {
            m_DefTag = pValues [++i];
            LogAssert ((-1 == m_DefTag) || 
              (FALimits::MinTag <= m_DefTag && FALimits::MaxTag >= m_DefTag));
            break;
        }
        case FAFsmConst::PARAM_DIRECTION:
        {
            m_Direction = pValues [++i];
            break;
        }
        case FAFsmConst::PARAM_CHARMAP:
        {
            const int DumpNum = pValues [++i];
            const unsigned char * pDump = pLDB->GetDump (DumpNum);
            LogAssert (pDump);

            if (!m_pCharMap) {
                m_pCharMap = NEW FAMultiMap_pack_fixed;
                LogAssert (m_pCharMap);
            }
            m_pCharMap->SetImage (pDump);

            break;
        }
        case FAFsmConst::PARAM_FSM:
        {
            const int DumpNum = pValues [++i];
            const unsigned char * pDump = pLDB->GetDump (DumpNum);
            LogAssert (pDump);

            if (!m_pDfa) {
                m_pDfa = NEW FARSDfa_pack_triv;
                LogAssert (m_pDfa);
            }
            m_pDfa->SetImage (pDump);

            if (!m_pState2Ows) {
                m_pState2Ows = NEW FAState2Ows_pack_triv;
                LogAssert (m_pState2Ows);
            }
            m_pState2Ows->SetImage (pDump);

            break;
        }
        case FAFsmConst::PARAM_FLOAT_ARRAY:
        {
            const int DumpNum = pValues [++i];
            const unsigned char * pDump = pLDB->GetDump (DumpNum);
            LogAssert (pDump);

            const int FloatArrSize = *((const int * )pDump);
            LogAssert (2 == FloatArrSize);

            const float * pFloatArr = (const float *) (pDump + sizeof (int));

            m_MinProbVal = pFloatArr [0];
            m_MaxProbVal = pFloatArr [1];

            break;
        }
        default:
            LogAssert (false);
        }
    }
}


void FAWgConfKeeper::Clear ()
{
    if (m_pDfa) {
        delete m_pDfa;
        m_pDfa = NULL;
    }
    if (m_pState2Ows) {
        delete m_pState2Ows;
        m_pState2Ows = NULL;
    }
    if (m_pCharMap) {
        delete m_pCharMap;
        m_pCharMap = NULL;
    }

    m_Direction = FAFsmConst::DIR_L2R;
    m_MaxLen = -1;
    m_NoTrUse = false;
    m_IgnoreCase = false;
    m_MaxProb = 0;
    m_DictMode = false;
    m_MinProbVal = 0;
    m_MaxProbVal = 0;
    m_fLogScale = false;
    m_TagEos = 0;
    m_Order = -1;
}


const FARSDfaCA * FAWgConfKeeper::GetRsDfa () const
{
    return m_pDfa;
}

const FAState2OwsCA * FAWgConfKeeper::GetState2Ows () const
{
    return m_pState2Ows;
}

const int FAWgConfKeeper::GetDirection () const
{
    return m_Direction;
}

const int FAWgConfKeeper::GetMaxLen () const
{
    return m_MaxLen;
}

const int FAWgConfKeeper::GetDefTag () const
{
    return m_DefTag;
}

const bool FAWgConfKeeper::GetNoTrUse () const
{
    return m_NoTrUse;
}

const bool FAWgConfKeeper::GetIgnoreCase () const
{
    return m_IgnoreCase;
}

const int FAWgConfKeeper::GetMaxProb () const
{
    return m_MaxProb;
}

const FAMultiMapCA * FAWgConfKeeper::GetCharMap () const
{
    return m_pCharMap;
}

const bool FAWgConfKeeper::GetDictMode () const
{
    return m_DictMode;
}

const float FAWgConfKeeper::GetMinProbVal () const
{
    return m_MinProbVal;
}

const float FAWgConfKeeper::GetMaxProbVal () const
{
    return m_MaxProbVal;
}

const bool FAWgConfKeeper::GetIsLog () const
{
    return m_fLogScale;
}

const int FAWgConfKeeper::GetEosTag () const
{
    return m_TagEos;
}

const int FAWgConfKeeper::GetOrder () const
{
    return m_Order;
}

}
