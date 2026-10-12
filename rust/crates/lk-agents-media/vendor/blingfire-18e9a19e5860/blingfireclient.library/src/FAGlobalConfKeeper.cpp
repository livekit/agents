/**
 * Copyright (c) Microsoft Corporation. All rights reserved.
 * Licensed under the MIT License.
 */


#include "blingfire-client_src_pch.h"
#include "FAConfig.h"
#include "FAGlobalConfKeeper_packaged.h"
#include "FALDB.h"
#include "FAFsmConst.h"
#include "FARSDfa_pack_triv.h"

namespace BlingFire
{

FAGlobalConfKeeper::FAGlobalConfKeeper () :
    m_pLDB (NULL),
    m_DoW2B (false),
    m_pDfa (NULL),
    m_pSuffDfa (NULL)
{}


FAGlobalConfKeeper::~FAGlobalConfKeeper ()
{
    FAGlobalConfKeeper::Clear ();
}


void FAGlobalConfKeeper::SetLDB (const FALDB * pLDB)
{
    m_pLDB = pLDB;
}


void FAGlobalConfKeeper::Init (const int * pValues, const int Size)
{
    LogAssert (m_pLDB, "Internal error.");
    LogAssert (pValues || 0 >= Size, "Internal error.");

    FAGlobalConfKeeper::Clear ();

    for (int i = 0; i < Size; ++i) {

        const int Param = pValues [i];

        switch (Param) {

        case FAFsmConst::PARAM_DO_W2B:
        {
            m_DoW2B = true;
            break;
        }
        case FAFsmConst::PARAM_PREF_FSM:
        {
            const int DumpNum = pValues [++i];
            const unsigned char * pDump = m_pLDB->GetDump (DumpNum);

            LogAssert (pDump, "Object cannot be initialized.");

            if (!m_pDfa)
                m_pDfa = NEW FARSDfa_pack_triv;
            m_pDfa->SetImage (pDump);

            break;
        }
        case FAFsmConst::PARAM_SUFFIX_FSM:
        {
            const int DumpNum = pValues [++i];
            const unsigned char * pDump = m_pLDB->GetDump (DumpNum);

            LogAssert (pDump, "Object cannot be initialized.");

            if (!m_pSuffDfa)
                m_pSuffDfa = NEW FARSDfa_pack_triv;
            m_pSuffDfa->SetImage (pDump);

            break;
        }

        /// no default: action, global configuration keeper
        /// as [global] can have arbitrary parameters this 
        /// class does not know about

        }
    } // of for (int i = 0; ...
}


void FAGlobalConfKeeper::Clear ()
{
    m_DoW2B = false;

    if (m_pDfa) {
        delete m_pDfa;
        m_pDfa = NULL;
    }
    if (m_pSuffDfa) {
        delete m_pSuffDfa;
        m_pSuffDfa = NULL;
    }
}


const bool FAGlobalConfKeeper::GetDoW2B () const
{
    return m_DoW2B;
}


const FARSDfaCA * FAGlobalConfKeeper::GetPrefixes () const
{
    return m_pDfa;
}


const FARSDfaCA * FAGlobalConfKeeper::GetSuffixes () const
{
    return m_pSuffDfa;
}

}
