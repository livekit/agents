/**
 * Copyright (c) Microsoft Corporation. All rights reserved.
 * Licensed under the MIT License.
 */


#include "blingfire-client_src_pch.h"
#include "FAConfig.h"
#include "FAException.h"

namespace BlingFire
{

FAException::FAException (
        const char * pSourceName, 
        const int SourceLine) : 
    m_pSourceFile (pSourceName),
    m_SourceLine (SourceLine)
{
    CopyMsg (FAMsg::InternalError);
}

FAException::FAException (
        const char * pErrMsg, 
        const char * pSourceName, 
        const int SourceLine) : 
    m_pSourceFile (pSourceName),
    m_SourceLine (SourceLine)
{
    CopyMsg (pErrMsg);
}

inline void FAException::CopyMsg (const char * pErrMsg)
{
    int i = 0;
    for (; i < FALimits::MaxWordSize && pErrMsg; ++i) {
        if (0 == pErrMsg [i])
            break;
        m_ErrMsg [i] = pErrMsg [i];
    }
    m_ErrMsg [i] = 0;
}

const char * FAException::GetErrMsg () const
{
    return m_ErrMsg;
}

const char * FAException::GetSourceName () const
{
    return m_pSourceFile;
}

const int FAException::GetSourceLine () const
{
    return m_SourceLine;
}

}
