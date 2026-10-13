/**
 * Copyright (c) Microsoft Corporation. All rights reserved.
 * Licensed under the MIT License.
 */


#include "blingfire-client_src_pch.h"
#include "FAConfig.h"
#include "FAW2PConfKeeper.h"
#include "FAFsmConst.h"
#include "FALDB.h"
#include "FARSDfa_pack_triv.h"
#include "FAState2Ow_pack_triv.h"
#include "FAMultiMap_pack_fixed.h"
#include "FAUtils_cl.h"

namespace BlingFire
{

FAW2PConfKeeper::FAW2PConfKeeper () :
    m_pLDB (NULL),
    m_pDfa (NULL),
    m_pState2Ow (NULL),
    m_IgnoreCase (false),
    m_MaxProb (0),
    m_pCharMap (NULL),
    m_MinProbVal (0),
    m_MaxProbVal (1)
{}


FAW2PConfKeeper::~FAW2PConfKeeper ()
{
    FAW2PConfKeeper::Clear ();
}


void FAW2PConfKeeper::Initialize (const FALDB * pLDB, const int * pValues, const int Size)
{
    LogAssert (pLDB);
    LogAssert (pValues || 0 >= Size);

    m_pLDB = pLDB;

    FAW2PConfKeeper::Clear ();

    for (int i = 0; i < Size; ++i) {

        const int Param = pValues [i];

        switch (Param) {

        case FAFsmConst::PARAM_IGNORE_CASE:
        {
            m_IgnoreCase = true;
            break;
        }
        case FAFsmConst::PARAM_MAX_PROB:
        {
            m_MaxProb = pValues [++i];
            LogAssert (0 < m_MaxProb);
            break;
        }
        case FAFsmConst::PARAM_CHARMAP:
        {
            const int DumpNum = pValues [++i];
            const unsigned char * pDump = m_pLDB->GetDump (DumpNum);
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
            const unsigned char * pDump = m_pLDB->GetDump (DumpNum);
            LogAssert (pDump);

            if (!m_pDfa) {
                m_pDfa = NEW FARSDfa_pack_triv;
                LogAssert (m_pDfa);
            }
            m_pDfa->SetImage (pDump);

            if (!m_pState2Ow) {
                m_pState2Ow = NEW FAState2Ow_pack_triv;
                LogAssert (m_pState2Ow);
            }
            m_pState2Ow->SetImage (pDump);

            break;
        }
        case FAFsmConst::PARAM_FLOAT_ARRAY:
        {
            const int DumpNum = pValues [++i];
            const unsigned char * pDump = m_pLDB->GetDump (DumpNum);
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


void FAW2PConfKeeper::Clear ()
{
    if (m_pDfa) {
        delete m_pDfa;
        m_pDfa = NULL;
    }
    if (m_pState2Ow) {
        delete m_pState2Ow;
        m_pState2Ow = NULL;
    }
    if (m_pCharMap) {
        delete m_pCharMap;
        m_pCharMap = NULL;
    }

    m_IgnoreCase = false;
    m_MaxProb = 0;
    m_MinProbVal = 0;
    m_MaxProbVal = 1;
}


const FARSDfaCA * FAW2PConfKeeper::GetRsDfa () const
{
    return m_pDfa;
}

const FAState2OwCA * FAW2PConfKeeper::GetState2Ow () const
{
    return m_pState2Ow;
}

const bool FAW2PConfKeeper::GetIgnoreCase () const
{
    return m_IgnoreCase;
}

const int FAW2PConfKeeper::GetMaxProb () const
{
    return m_MaxProb;
}

const FAMultiMapCA * FAW2PConfKeeper::GetCharMap () const
{
    return m_pCharMap;
}

const float FAW2PConfKeeper::GetMinProbVal () const
{
    return m_MinProbVal;
}

const float FAW2PConfKeeper::GetMaxProbVal () const
{
    return m_MaxProbVal;
}

}
