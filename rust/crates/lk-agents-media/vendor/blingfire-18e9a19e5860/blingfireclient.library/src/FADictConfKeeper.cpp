/**
 * Copyright (c) Microsoft Corporation. All rights reserved.
 * Licensed under the MIT License.
 */


#include "blingfire-client_src_pch.h"
#include "FAConfig.h"
#include "FADictConfKeeper.h"
#include "FAFsmConst.h"
#include "FALDB.h"
#include "FARSDfa_pack_triv.h"
#include "FAMealyDfa_pack_triv.h"
#include "FAState2Ow_pack_triv.h"
#include "FAArray_pack.h"
#include "FAMultiMap_pack.h"
#include "FAMultiMap_pack_mph.h"
#include "FAMultiMap_pack_fixed.h"

namespace BlingFire
{

FADictConfKeeper::FADictConfKeeper () :
    m_pLDB (NULL),
    m_FsmType (FAFsmConst::TYPE_MEALY_DFA),
    m_pRsDfa (NULL),
    m_pMealy (NULL),
    m_pState2Ow (NULL),
    m_pK2I (NULL),
    m_pI2Info_triv (NULL),
    m_pI2Info_mph (NULL),
    m_pI2Info_fixed (NULL),
    m_pI2Info (NULL),
    m_IgnoreCase (false),
    m_NoTrUse (true),
    m_Direction (FAFsmConst::DIR_L2R),
    m_pCharMap (NULL),
    m_TokAlgo (FAFsmConst::TOKENIZE_DEFAULT),
    m_IdOffset (0),
    m_UseRawBytes (false),
    m_fNoDummyPrefix (false)
{}


FADictConfKeeper::~FADictConfKeeper ()
{
    FADictConfKeeper::Clear ();
}


void FADictConfKeeper::SetLDB (const FALDB * pLDB)
{
    m_pLDB = pLDB;
}


void FADictConfKeeper::Init (const int * pValues, const int Size)
{
    LogAssert (m_pLDB);
    LogAssert (pValues || 0 >= Size);

    FADictConfKeeper::Clear ();

    int i2info_mode = FAFsmConst::MODE_PACK_TRIV;

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
        case FAFsmConst::PARAM_USE_BYTE_ENCODING:
        {
            m_UseRawBytes = true;
            break;
        }
        case FAFsmConst::PARAM_NO_DUMMY_PREFIX:
        {
            m_fNoDummyPrefix = true;
            break;
        }
        case FAFsmConst::PARAM_DIRECTION:
        {
            m_Direction = pValues [++i];

            LogAssert (FAFsmConst::DIR_L2R == m_Direction || \
                FAFsmConst::DIR_R2L == m_Direction);

            break;
        }
        case FAFsmConst::PARAM_TOKENIZATION_TYPE:
        {
            m_TokAlgo = pValues [++i];

            LogAssert (FAFsmConst::TOKENIZE_DEFAULT <= m_TokAlgo && \
                    FAFsmConst::TOKENIZE_COUNT > m_TokAlgo);

            break;
        }
        case FAFsmConst::PARAM_ID_OFFSET:
        {
            m_IdOffset = pValues [++i];
            break;
        }
        case FAFsmConst::PARAM_FSM_TYPE:
        {
            m_FsmType = pValues [++i];

            LogAssert (FAFsmConst::TYPE_MOORE_DFA == m_FsmType || \
                FAFsmConst::TYPE_MEALY_DFA == m_FsmType);

            break;
        }
        case FAFsmConst::PARAM_MAP_MODE:
        {
            i2info_mode = pValues [++i];

            LogAssert (FAFsmConst::MODE_PACK_TRIV == i2info_mode || \
                FAFsmConst::MODE_PACK_MPH == i2info_mode || \
                FAFsmConst::MODE_PACK_FIXED == i2info_mode);

            break;
        }
        case FAFsmConst::PARAM_FSM:
        {
            const int DumpNum = pValues [++i];
            const unsigned char * pDump = m_pLDB->GetDump (DumpNum);

            LogAssert (pDump);

            if (!m_pRsDfa) {
                m_pRsDfa = NEW FARSDfa_pack_triv;
            }
            m_pRsDfa->SetImage (pDump);

            if (FAFsmConst::TYPE_MEALY_DFA == m_FsmType) {

                if (!m_pMealy) {
                    m_pMealy = NEW FAMealyDfa_pack_triv;
                }
                m_pMealy->SetImage (pDump);

            } else {
                LogAssert (FAFsmConst::TYPE_MOORE_DFA == m_FsmType);

                if (!m_pState2Ow) {
                    m_pState2Ow = NEW FAState2Ow_pack_triv;
                }
                m_pState2Ow->SetImage (pDump);
            }
            break;
        }
        case FAFsmConst::PARAM_ARRAY:
        {
            const int DumpNum = pValues [++i];
            const unsigned char * pDump = m_pLDB->GetDump (DumpNum);

            LogAssert (pDump);

            if (!m_pK2I) {
                m_pK2I = NEW FAArray_pack;
            }
            m_pK2I->SetImage (pDump);

            break;
        }
        case FAFsmConst::PARAM_CHARMAP:
        {
            const int DumpNum = pValues [++i];
            const unsigned char * pDump = m_pLDB->GetDump (DumpNum);
            LogAssert (pDump);

            if (!m_pCharMap)
                m_pCharMap = NEW FAMultiMap_pack_fixed;
            m_pCharMap->SetImage (pDump);

            break;
        }
        case FAFsmConst::PARAM_MULTI_MAP:
        {
            const int DumpNum = pValues [++i];
            const unsigned char * pDump = m_pLDB->GetDump (DumpNum);

            LogAssert (pDump);

            if (FAFsmConst::MODE_PACK_TRIV == i2info_mode) {

                if (!m_pI2Info_triv) {
                    m_pI2Info_triv = NEW FAMultiMap_pack;
                }
                m_pI2Info_triv->SetImage (pDump);
                m_pI2Info = m_pI2Info_triv;

            } else if (FAFsmConst::MODE_PACK_FIXED == i2info_mode) {

                if (!m_pI2Info_fixed) {
                    m_pI2Info_fixed = NEW FAMultiMap_pack_fixed;
                }
                m_pI2Info_fixed->SetImage (pDump);
                m_pI2Info = m_pI2Info_fixed;

            } else {
                DebugLogAssert (FAFsmConst::MODE_PACK_MPH == i2info_mode);

                if (!m_pI2Info_mph) {
                    m_pI2Info_mph = NEW FAMultiMap_pack_mph;
                }
                m_pI2Info_mph->SetImage (pDump);
                m_pI2Info = m_pI2Info_mph;
            }
            break;
        }

        default:
            LogAssert (0);
        }
    } // of for (int i = 0; ...
}


void FADictConfKeeper::Clear ()
{
    if (m_pRsDfa) {
        delete m_pRsDfa;
        m_pRsDfa = NULL;
    }
    if (m_pMealy) {
        delete m_pMealy;
        m_pMealy = NULL;
    }
    if (m_pState2Ow) {
        delete m_pState2Ow;
        m_pState2Ow = NULL;
    }
    if (m_pK2I) {
        delete m_pK2I;
        m_pK2I = NULL;
    }
    if (m_pI2Info_triv) {
        delete m_pI2Info_triv;
        m_pI2Info_triv = NULL;
    }
    if (m_pI2Info_mph) {
        delete m_pI2Info_mph;
        m_pI2Info_mph = NULL;
    }
    if (m_pI2Info_fixed) {
        delete m_pI2Info_fixed;
        m_pI2Info_fixed = NULL;
    }
    if (m_pCharMap) {
        delete m_pCharMap;
        m_pCharMap = NULL;
    }

    m_IgnoreCase = false;
    m_NoTrUse = true;
    m_Direction = FAFsmConst::DIR_L2R;
    m_pI2Info = NULL;
    m_FsmType = FAFsmConst::TYPE_MEALY_DFA;
    m_TokAlgo = FAFsmConst::TOKENIZE_DEFAULT;
    m_IdOffset = 0;
    m_UseRawBytes = false;
    m_fNoDummyPrefix = false;
}


const int FADictConfKeeper::GetFsmType () const
{
    return m_FsmType;
}


const FARSDfaCA * FADictConfKeeper::GetRsDfa () const
{
    return m_pRsDfa;
}


const FAMealyDfaCA * FADictConfKeeper::GetMphMealy () const
{
    return m_pMealy;
}


const FAState2OwCA * FADictConfKeeper::GetState2Ow () const
{
    return m_pState2Ow;
}


const FAArrayCA * FADictConfKeeper::GetK2I () const
{
    return m_pK2I;
}


const FAMultiMapCA * FADictConfKeeper::GetI2Info () const
{
    return m_pI2Info;
}


const bool FADictConfKeeper::GetIgnoreCase () const
{
    return m_IgnoreCase;
}


const bool FADictConfKeeper::GetNoTrUse () const
{
    return m_NoTrUse;
}


const int FADictConfKeeper::GetDirection () const
{
    return m_Direction;
}


const FAMultiMapCA * FADictConfKeeper::GetCharMap () const
{
    return m_pCharMap;
}


const int FADictConfKeeper::GetTokAlgo () const
{
    return m_TokAlgo;
}


const int FADictConfKeeper::GetIdOffset () const
{
    return m_IdOffset;
}


const bool FADictConfKeeper::GetUseByteEncoding () const
{
    return m_UseRawBytes;
}


const bool FADictConfKeeper::GetNoDummyPrefix () const
{
    return m_fNoDummyPrefix;
}

void FADictConfKeeper::SetNoDummyPrefix(bool fNoDummyPrefix)
{
    m_fNoDummyPrefix = fNoDummyPrefix;
}

}
