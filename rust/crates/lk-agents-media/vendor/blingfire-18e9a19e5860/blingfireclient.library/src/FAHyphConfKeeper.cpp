/**
 * Copyright (c) Microsoft Corporation. All rights reserved.
 * Licensed under the MIT License.
 */


#include "blingfire-client_src_pch.h"
#include "FAConfig.h"
#include "FAHyphConfKeeper_packaged.h"
#include "FAFsmConst.h"
#include "FALimits.h"
#include "FALDB.h"
#include "FARSDfa_pack_triv.h"
#include "FAState2Ow_pack_triv.h"
#include "FAMultiMap_pack.h"
#include "FAMultiMap_pack_fixed.h"

namespace BlingFire
{

FAHyphConfKeeper::FAHyphConfKeeper () :
    m_pLDB (NULL),
    m_pRsDfa (NULL),
    m_pState2Ow (NULL),
    m_pI2Info (NULL),
    m_IgnoreCase (false),
    m_MinPatLen (DefMinPatLen),
    m_LeftAnchor (FAFsmConst::IW_L_ANCHOR),
    m_RightAnchor (FAFsmConst::IW_R_ANCHOR),
    m_HyphType (FAFsmConst::HYPH_TYPE_DEFAULT),
    m_NormSegs (false),
    m_NoHyphLen (0),
    m_pCharMap (NULL)
{}


FAHyphConfKeeper::~FAHyphConfKeeper ()
{
    FAHyphConfKeeper::Clear ();
}


void FAHyphConfKeeper::SetLDB (const FALDB * pLDB)
{
    m_pLDB = pLDB;
}


void FAHyphConfKeeper::Init (const int * pValues, const int Size)
{
    LogAssert (m_pLDB && (pValues || 0 >= Size), "Internal error.");

    FAHyphConfKeeper::Clear ();

    for (int i = 0; i < Size; ++i) {

        const int Param = pValues [i];

        switch (Param) {

        case FAFsmConst::PARAM_IGNORE_CASE:
        {
            m_IgnoreCase = true;
            break;
        }
        case FAFsmConst::PARAM_NORMALIZE:
        {
            m_NormSegs = true;
            break;
        }
        case FAFsmConst::PARAM_MIN_LEN:
        {
            m_MinPatLen = pValues [++i];
            LogAssert (0 < m_MinPatLen && m_MinPatLen <= FALimits::MaxWordLen, \
                "Object cannot be initialized.");
            break;
        }
        case FAFsmConst::PARAM_MIN_LEN2:
        {
            m_NoHyphLen = pValues [++i];
            LogAssert (0 <= m_NoHyphLen && DefMaxNoHyphLen >= m_NoHyphLen, \
                "Object cannot be initialized.");
            break;
        }
        case FAFsmConst::PARAM_LEFT_ANCHOR:
        {
            m_LeftAnchor = pValues [++i];
            LogAssert (0 < m_LeftAnchor, "Object cannot be initialized.");
            break;
        }
        case FAFsmConst::PARAM_RIGHT_ANCHOR:
        {
            m_RightAnchor = pValues [++i];
            LogAssert (0 < m_RightAnchor, "Object cannot be initialized.");
            break;
        }
        case FAFsmConst::PARAM_HYPH_TYPE:
        {
            m_HyphType = pValues [++i];
            LogAssert (0 <= m_HyphType, "Object cannot be initialized.");
            LogAssert (m_HyphType < FAFsmConst::HYPH_TYPE_COUNT, \
                "Object cannot be initialized.");
            break;
        }
        case FAFsmConst::PARAM_FSM:
        {
            const int DumpNum = pValues [++i];
            const unsigned char * pDump = m_pLDB->GetDump (DumpNum);
            LogAssert (pDump, "Object cannot be initialized.");

            if (!m_pRsDfa)
                m_pRsDfa = NEW FARSDfa_pack_triv;
            m_pRsDfa->SetImage (pDump);

            if (!m_pState2Ow) {
                m_pState2Ow = NEW FAState2Ow_pack_triv;
            }
            m_pState2Ow->SetImage (pDump);

            break;
        }
        case FAFsmConst::PARAM_CHARMAP:
        {
            const int DumpNum = pValues [++i];
            const unsigned char * pDump = m_pLDB->GetDump (DumpNum);
            LogAssert (pDump, "Object cannot be initialized.");

            if (!m_pCharMap)
                m_pCharMap = NEW FAMultiMap_pack_fixed;
            m_pCharMap->SetImage (pDump);

            break;
        }
        case FAFsmConst::PARAM_MULTI_MAP:
        {
            const int DumpNum = pValues [++i];
            const unsigned char * pDump = m_pLDB->GetDump (DumpNum);
            LogAssert (pDump, "Object cannot be initialized.");

            if (!m_pI2Info)
                m_pI2Info = NEW FAMultiMap_pack;
            m_pI2Info->SetImage (pDump);

            break;
        }

        default:
            LogAssert (false, "Object cannot be initialized.");
        }
    } // of for (int i = 0; ...
}


void FAHyphConfKeeper::Clear ()
{
    if (m_pRsDfa) {
        delete m_pRsDfa;
        m_pRsDfa = NULL;
    }
    if (m_pState2Ow) {
        delete m_pState2Ow;
        m_pState2Ow = NULL;
    }
    if (m_pI2Info) {
        delete m_pI2Info;
        m_pI2Info = NULL;
    }
    if (m_pCharMap) {
        delete m_pCharMap;
        m_pCharMap = NULL;
    }

    m_IgnoreCase = false;
    m_MinPatLen = DefMinPatLen;
    m_LeftAnchor = FAFsmConst::IW_L_ANCHOR;
    m_RightAnchor = FAFsmConst::IW_R_ANCHOR;
    m_HyphType = FAFsmConst::HYPH_TYPE_DEFAULT;
    m_NormSegs = false;
    m_NoHyphLen = 0;
}


const FARSDfaCA * FAHyphConfKeeper::GetRsDfa () const
{
    return m_pRsDfa;
}


const FAState2OwCA * FAHyphConfKeeper::GetState2Ow () const
{
    return m_pState2Ow;
}


const FAMultiMapCA * FAHyphConfKeeper::GetI2Info () const
{
    return m_pI2Info;
}


const bool FAHyphConfKeeper::GetIgnoreCase () const
{
    return m_IgnoreCase;
}


const int FAHyphConfKeeper::GetMinPatLen () const
{
    return m_MinPatLen;
}


const int FAHyphConfKeeper::GetLeftAnchor () const
{
    return m_LeftAnchor;
}


const int FAHyphConfKeeper::GetRightAnchor () const
{
    return m_RightAnchor;
}


const int FAHyphConfKeeper::GetHyphType () const
{
    return m_HyphType;
}


const bool FAHyphConfKeeper::GetNormSegs () const
{
    return m_NormSegs;
}


const int FAHyphConfKeeper::GetNoHyphLen () const
{
    return m_NoHyphLen;
}

const FAMultiMapCA * FAHyphConfKeeper::GetCharMap () const
{
    return m_pCharMap;
}

}

