/**
 * Copyright (c) Microsoft Corporation. All rights reserved.
 * Licensed under the MIT License.
 */


#include "blingfire-client_src_pch.h"
#include "FAConfig.h"
#include "FATsConfKeeper.h"
#include "FAFsmConst.h"
#include "FALDB.h"
#include "FAArray_pack.h"

namespace BlingFire
{

FATsConfKeeper::FATsConfKeeper () :
    m_pLDB (NULL),
    m_pArr (NULL),
    m_LogScale (false),
    m_MaxProb (0),
    m_MaxTag (0),
    m_FloatArr (NULL),
    m_FloatArrSize (0)
{}

FATsConfKeeper::~FATsConfKeeper ()
{
    FATsConfKeeper::Clear ();
}

void FATsConfKeeper::SetLDB (const FALDB * pLDB)
{
    m_pLDB = pLDB;
}

void FATsConfKeeper::Init (const int * pValues, const int Size)
{
    LogAssert (m_pLDB);
    LogAssert (pValues || 0 >= Size);

    FATsConfKeeper::Clear ();

    for (int i = 0; i < Size; ++i) {

        const int Param = pValues [i];

        switch (Param) {

        case FAFsmConst::PARAM_LOG_SCALE:
        {
            m_LogScale = true;
            break;
        }
        case FAFsmConst::PARAM_MAX_PROB:
        {
            m_MaxProb = pValues [++i];
            LogAssert (0 < m_MaxProb);
            break;
        }
        case FAFsmConst::PARAM_MAX_TAG:
        {
            m_MaxTag = pValues [++i];
            LogAssert ((-1 == m_MaxTag) || (FALimits::MinTag <= m_MaxTag &&
                FALimits::MaxTag >= m_MaxTag));
            break;
        }
        case FAFsmConst::PARAM_ARRAY:
        {
            const int DumpNum = pValues [++i];
            const unsigned char * pDump = m_pLDB->GetDump (DumpNum);
            LogAssert (pDump);

            if (!m_pArr) {
                m_pArr = new FAArray_pack;
                LogAssert (m_pArr);
            }
            m_pArr->SetImage (pDump);

            break;
        }
        case FAFsmConst::PARAM_FLOAT_ARRAY:
        {
            const int DumpNum = pValues [++i];
            const unsigned char * pDump = m_pLDB->GetDump (DumpNum);
            LogAssert (pDump);

            m_FloatArrSize = *((const int * )pDump);
            LogAssert (0 <= m_FloatArrSize);

            m_FloatArr = (const float *) (pDump + sizeof (int));

            break;
        }
        default:
            // unknown parameters in the configuration file
            LogAssert (false);
        }
    }
}

void FATsConfKeeper::Clear ()
{
    if (m_pArr) {
        delete m_pArr;
        m_pArr = NULL;
    }

    m_LogScale = false;
    m_MaxProb = 0;
    m_MaxTag = 0;
    m_FloatArr = NULL;
    m_FloatArrSize = 0;
}

const FAArrayCA * FATsConfKeeper::GetArr () const
{
    return m_pArr;
}

const int FATsConfKeeper::GetArr (const float ** ppArr) const
{
    DebugLogAssert (ppArr);
    *ppArr = m_FloatArr;
    return m_FloatArrSize;
}

const bool FATsConfKeeper::GetIsLog () const
{
    return m_LogScale;
}

const int FATsConfKeeper::GetMaxProb () const
{
    return m_MaxProb;
}

const int FATsConfKeeper::GetMaxTag () const
{
    return m_MaxTag;
}

}
