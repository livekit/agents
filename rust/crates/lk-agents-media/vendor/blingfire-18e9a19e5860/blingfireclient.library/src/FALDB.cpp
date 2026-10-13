/**
 * Copyright (c) Microsoft Corporation. All rights reserved.
 * Licensed under the MIT License.
 */


#include "blingfire-client_src_pch.h"
#include "FAConfig.h"
#include "FALDB.h"
#include "FAFsmConst.h"
#include "FAUtils_cl.h"

namespace BlingFire
{

FALDB::FALDB () :
    m_DumpCount (0)
{
}

FALDB::~FALDB ()
{}

void FALDB::SetImage (const unsigned char * pImgDump)
{
    m_DumpCount = 0;

    if (!pImgDump) {
        return;
    }

    const unsigned char * pArr = pImgDump;

    // get the number of dumps
    const int Count = *((const int *)pArr);
    DebugLogAssert (0 < Count);
    pArr += sizeof (int);

    // LDB configuration has more dumps than 
    // FALimits::MaxLdbDumpCount or the Count is negative
    LogAssert (0 <= Count && Count <= FALimits::MaxLdbDumpCount);

    // setup configuration image-dump, it is 0-th
    int Offset = *((const int *)pArr);
    DebugLogAssert (0 <= Offset);
    m_Conf.SetImage (pImgDump + Offset);

    // store the count
    m_DumpCount = Count;

    // setup dumps array
    for (int i = 0; i < Count; ++i) {

        Offset = *((const int *)pArr);
        DebugLogAssert (0 <= Offset);
        pArr += sizeof (int);

        m_Dumps [i] = (pImgDump + Offset);
        m_Offsets [i] = Offset;
    }

    const bool fIsValid = IsValidBinary ();
    LogAssert (fIsValid, "Invalid LDB binary file detected.");
}


const bool FALDB::IsValidBinary ()
{
    // see if the LDB bin file verification is required
    int fVerifyLdb = 0;
    GetValue (FAFsmConst::FUNC_GLOBAL, FAFsmConst::PARAM_VERIFY_LDB_BIN, &fVerifyLdb);

    if (fVerifyLdb)
    {
        // LDB should have at least two dumps if the verification is requested
        LogAssert (1 < m_DumpCount);
        // get the validation dump (the last one)
        const unsigned int * pVerify = (const unsigned int *) GetDump (m_DumpCount - 1);

        // get the expected values
        const unsigned int VaildationDataVersion = pVerify [FAFsmConst::VALIDATION_VERSION];

        if (0 == VaildationDataVersion)
        {
            const unsigned int ExpectedDataSize = pVerify [FAFsmConst::VALIDATION_SIZE];
            const unsigned int ExpectedDataHash = pVerify [FAFsmConst::VALIDATION_HASH];

            unsigned int DataSize = 0;
            unsigned int DataHash = 0;

            // iterate over all data dumps
            for (int i = 0; i < m_DumpCount - 1; ++i)
            {
                const int Size = m_Offsets [i + 1] - m_Offsets [i];

                if (0 > Size)
                {
                    return false;
                }

                DataSize += Size;
                DataHash = FAGetCrc32 (m_Dumps [i], (size_t) Size, DataHash);
            }

            // see if actual numbers match the expected onces
            if (DataSize != ExpectedDataSize || 
                DataHash != ExpectedDataHash)
            {
                return false;
            }
        }
    }

    // return true if we don't know how to use the validation data or if the validation is not requested
    return true;
}


const FAMultiMapCA * FALDB::GetHeader () const
{
    return & m_Conf;
}

const int FALDB::GetDumpCount () const
{
    return m_DumpCount;
}

const unsigned char * FALDB::GetDump (const int Num) const
{
    LogAssert (0 <= Num && Num < m_DumpCount);
    const unsigned char * pDump = m_Dumps [Num];
    return pDump;
}

inline const bool FALDB::IsBooleanParam (const int Parameter)
{
    return Parameter == FAFsmConst::PARAM_REVERSE ||
           Parameter == FAFsmConst::PARAM_NO_TR ||
           Parameter == FAFsmConst::PARAM_IGNORE_CASE ||
           Parameter == FAFsmConst::PARAM_DICT_MODE ||
           Parameter == FAFsmConst::PARAM_NORMALIZE ||
           Parameter == FAFsmConst::PARAM_LOG_SCALE ||
           Parameter == FAFsmConst::PARAM_USE_NFST ||
           Parameter == FAFsmConst::PARAM_DO_W2B ||
           Parameter == FAFsmConst::PARAM_VERIFY_LDB_BIN ;
}

const bool FALDB::
    GetValue (const int Section, const int Parameter, int * pValue) const
{
    LogAssert (pValue);

    *pValue = 0;

    const int * pValues = NULL;
    const int Size = m_Conf.Get (Section, &pValues);

    for (int i = 0; i < Size; ++i) {

        const int NextParam = pValues [i];

        // see if this is a boolean parameter
        bool fIsBoolean = IsBooleanParam (NextParam);

        // if not boolean skip to the value
        if (!fIsBoolean) {
            i++;
            LogAssert (i < Size);
        }

        // see if the parameter id matches the requested parameter id
        if (NextParam == Parameter) {
            if (fIsBoolean) {
                *pValue = 1;
            } else {
                *pValue = pValues [i];
            }
            return true;
        }

    } // of for (int i = 0; i < Size; ++i) ...

    // see if the requested parameter is boolean
    if (IsBooleanParam (Parameter)) {
        DebugLogAssert (0 == *pValue);
        return true;
    }

    return false;
}

}
