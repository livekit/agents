/**
 * Copyright (c) Microsoft Corporation. All rights reserved.
 * Licensed under the MIT License.
 */


#include "blingfire-client_src_pch.h"
#include "FAConfig.h"
#include "FAWftConfKeeper.h"
#include "FAFsmConst.h"
#include "FALDB.h"
#include "FARSDfa_pack_triv.h"
#include "FAState2Ows_pack_triv.h"
#include "FAMultiMap_pack.h"
#include "FAMultiMap_pack_mph.h"
#include "FAGetIWs_pack_triv.h"
#include "FAMultiMap_pack_fixed.h"
#include "FAUtils_cl.h"

namespace BlingFire
{

FAWftConfKeeper::FAWftConfKeeper () :
    m_pLDB (NULL),
    m_pDfa (NULL),
    m_pState2Ows (NULL),
    m_pActsA (NULL),
    m_pActs_triv (NULL),
    m_pActs_mph (NULL),
    m_pIws (NULL),
    m_NoTrUse (false),
    m_DictMode (false),
    m_IgnoreCase (false),
    m_UseNfst (false),
    m_pCharMap (NULL)
{}


FAWftConfKeeper::~FAWftConfKeeper ()
{
    FAWftConfKeeper::Clear ();
}


void FAWftConfKeeper::Initialize (const FALDB * pLDB, const int * pValues, const int Size)
{
    LogAssert (pLDB);
    LogAssert (pValues || 0 >= Size);

    m_pLDB = pLDB;

    FAWftConfKeeper::Clear ();

    int MapMode = FAFsmConst::MODE_PACK_TRIV;

    for (int i = 0; i < Size; ++i) {

        const int Param = pValues [i];

        switch (Param) {

        case FAFsmConst::PARAM_NO_TR:
        {
            m_NoTrUse = true;
            break;
        }
        case FAFsmConst::PARAM_IGNORE_CASE:
        {
            m_IgnoreCase = true;
            break;
        }
        case FAFsmConst::PARAM_USE_NFST:
        {
            m_UseNfst = true;
            break;
        }
        case FAFsmConst::PARAM_DICT_MODE:
        {
            m_DictMode = true;
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

            if (!m_UseNfst) {
                if (!m_pState2Ows) {
                    m_pState2Ows = NEW FAState2Ows_pack_triv;
                    LogAssert (m_pState2Ows);
                }
                m_pState2Ows->SetImage (pDump);
            } else {
                if (!m_pIws) {
                    m_pIws = NEW FAGetIWs_pack_triv;
                    LogAssert (m_pIws);
                }
                m_pIws->SetImage (pDump);
            }
            break;
        }
        case FAFsmConst::PARAM_MAP_MODE:
        {
            MapMode = pValues [++i];

            LogAssert (FAFsmConst::MODE_PACK_TRIV == MapMode || \
                FAFsmConst::MODE_PACK_MPH == MapMode);
            LogAssert (MapMode == FAFsmConst::MODE_PACK_TRIV || !m_UseNfst);

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
        case FAFsmConst::PARAM_ACTS:
        {
            const int DumpNum = pValues [++i];
            const unsigned char * pDump = m_pLDB->GetDump (DumpNum);
            LogAssert (pDump);

            if (FAFsmConst::MODE_PACK_TRIV == MapMode) {

                if (!m_pActs_triv) {
                    m_pActs_triv = NEW FAMultiMap_pack;
                    LogAssert (m_pActs_triv);
                }
                m_pActs_triv->SetImage (pDump);
                m_pActsA = m_pActs_triv;

            } else if (FAFsmConst::MODE_PACK_MPH == MapMode) {

                if (!m_pActs_mph) {
                    m_pActs_mph = NEW FAMultiMap_pack_mph;
                    LogAssert (m_pActs_mph);
                }
                m_pActs_mph->SetImage (pDump);
                m_pActsA = m_pActs_mph;
            }

            break;
        }
        default:
            LogAssert (false);

        } // of switch (Param) ...
    }
}


void FAWftConfKeeper::Clear ()
{
    if (m_pDfa) {
        delete m_pDfa;
        m_pDfa = NULL;
    }
    if (m_pState2Ows) {
        delete m_pState2Ows;
        m_pState2Ows = NULL;
    }
    if (m_pActs_triv) {
        delete m_pActs_triv;
        m_pActs_triv = NULL;
    }
    if (m_pActs_mph) {
        delete m_pActs_mph;
        m_pActs_mph = NULL;
    }
    if (m_pIws) {
        delete m_pIws;
        m_pIws = NULL;
    }
    if (m_pCharMap) {
        delete m_pCharMap;
        m_pCharMap = NULL;
    }
    m_pActsA = NULL;
    m_NoTrUse = false;
    m_IgnoreCase = false;
    m_DictMode = false;
    m_UseNfst = false;
}


const FARSDfaCA * FAWftConfKeeper::GetRsDfa () const
{
    return m_pDfa;
}

const FAState2OwsCA * FAWftConfKeeper::GetState2Ows () const
{
    return m_pState2Ows;
}

const FAMultiMapCA * FAWftConfKeeper::GetActs () const
{
    return m_pActsA;
}

const bool FAWftConfKeeper::GetNoTrUse () const
{
    return m_NoTrUse;
}

const bool FAWftConfKeeper::GetDictMode () const
{
    return m_DictMode;
}

const bool FAWftConfKeeper::GetIgnoreCase () const
{
    return m_IgnoreCase;
}

const FAGetIWsCA * FAWftConfKeeper::GetIws () const
{
    return m_pIws;
}

const bool FAWftConfKeeper::GetUseNfst () const
{
    return m_UseNfst;
}


const FAMultiMapCA * FAWftConfKeeper::GetCharMap () const
{
    return m_pCharMap;
}

}
