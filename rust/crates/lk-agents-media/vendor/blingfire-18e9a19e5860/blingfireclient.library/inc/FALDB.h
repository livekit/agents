/**
 * Copyright (c) Microsoft Corporation. All rights reserved.
 * Licensed under the MIT License.
 */


#ifndef _FA_LDB_H_
#define _FA_LDB_H_

#include "FAConfig.h"
#include "FASetImageA.h"
#include "FAMultiMap_pack.h"
#include "FALimits.h"

namespace BlingFire
{

///
/// Base class for keeping linguistic resources.
///

class FALDB : public FASetImageA {

public:
    FALDB ();
    virtual ~FALDB ();

public:
    void SetImage (const unsigned char * pImgDump);

public:
    // returns configuration multi map, it contain runtime initialization
    // vectors for all sections described in the ldb.conf
    const FAMultiMapCA * GetHeader () const;

    // returns the amount of image dumps stored in the LDB file
    const int GetDumpCount () const;

    // returns pointer to the image dump memory by its index
    const unsigned char * GetDump (const int Num) const;

    // returns true if the parameter found, false otherwise
    // *pValue will contain parameter's value from the given section
    // it will be 1 if the parameter is boolean
    const bool GetValue (const int Section, const int Parameter, int* pValue) const;

private:
    static inline const bool IsBooleanParam (const int Parameter);

    // validates LDB bin file if the validation information is available
    const bool IsValidBinary ();

protected:
    // keeps configuration
    FAMultiMap_pack m_Conf;

    // keeps the array of resource dumps
    const unsigned char * m_Dumps [FALimits::MaxLdbDumpCount];

    // and array of offsets where the dumps start
    unsigned int m_Offsets [FALimits::MaxLdbDumpCount];

    // number of dumps
    int m_DumpCount;

};

}

#endif
