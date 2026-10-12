/**
 * Copyright (c) Microsoft Corporation. All rights reserved.
 * Licensed under the MIT License.
 */


#include "blingfire-client_src_pch.h"
#include "FAConfig.h"
#include "FAW2SConfKeeper.h"
#include "FAFsmConst.h"
#include "FALDB.h"
#include "FARSDfa_pack_triv.h"
#include "FAState2Ow_pack_triv.h"
#include "FAUtils_cl.h"

namespace BlingFire
{

FAW2SConfKeeper::FAW2SConfKeeper () :
    m_pLDB (NULL),
    m_pDfa (NULL),
    m_pState2Ow (NULL),
    m_MinInputLen (-1),
    m_MinSegLen (-1),
    m_Threshold (-1),
    m_IgnoreCase (false),
    m_Dir (FAFsmConst::DIR_R2L)
{}


FAW2SConfKeeper::~FAW2SConfKeeper ()
{
    FAW2SConfKeeper::Clear ();
}


void FAW2SConfKeeper::SetLDB (const FALDB * pLDB)
{
    m_pLDB = pLDB;
}


void FAW2SConfKeeper::Init (const int * pValues, const int Size)
{
    LogAssert (m_pLDB && (pValues || 0 >= Size));

    FAW2SConfKeeper::Clear ();

    for (int i = 0; i < Size; ++i) {

        const int Param = pValues [i];

        switch (Param) {

        case FAFsmConst::PARAM_MIN_LEN:
        {
            m_MinInputLen = pValues [++i];
            break;
        }
        case FAFsmConst::PARAM_MIN_LEN2:
        {
            m_MinSegLen = pValues [++i];
            break;
        }
        case FAFsmConst::PARAM_THRESHOLD:
        {
            m_Threshold = pValues [++i];
            break;
        }
        case FAFsmConst::PARAM_IGNORE_CASE:
        {
            m_IgnoreCase = true;
            break;
        }
        case FAFsmConst::PARAM_DIRECTION:
        {
            m_Dir = pValues [++i];
            LogAssert (m_Dir == FAFsmConst::DIR_R2L 
                    || m_Dir == FAFsmConst::DIR_L2R);
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
        default:
            // initialization error
            LogAssert (false);
        }
    }
}


void FAW2SConfKeeper::Clear ()
{
    if (m_pDfa) {
        delete m_pDfa;
        m_pDfa = NULL;
    }
    if (m_pState2Ow) {
        delete m_pState2Ow;
        m_pState2Ow = NULL;
    }

    m_MinInputLen = -1;
    m_MinSegLen = -1;
    m_Threshold = -1;
    m_IgnoreCase = false;
    m_Dir = FAFsmConst::DIR_R2L;
}


const FARSDfaCA * FAW2SConfKeeper::GetRsDfa () const
{
    return m_pDfa;
}

const FAState2OwCA * FAW2SConfKeeper::GetState2Ow () const
{
    return m_pState2Ow;
}

const int FAW2SConfKeeper::GetMinInputLen () const
{
    return m_MinInputLen;
}

const int FAW2SConfKeeper::GetMinSegLen () const
{
    return m_MinSegLen;
}

const bool FAW2SConfKeeper::GetIgnoreCase () const
{
    return m_IgnoreCase;
}

const int FAW2SConfKeeper::GetDirection () const
{
    return m_Dir;
}

const int FAW2SConfKeeper::GetThreshold () const
{
    return m_Threshold;
}

}
