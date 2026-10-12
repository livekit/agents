/**
 * Copyright (c) Microsoft Corporation. All rights reserved.
 * Licensed under the MIT License.
 */


#ifndef _FA_GCLDB_H_
#define _FA_GCLDB_H_

#include "FAConfig.h"
#include "FALDB.h"
#include "FAWREConf_pack.h"

namespace BlingFire
{

class FAAllocatorA;

///
/// Keeps all resources needed for GC rules execution.
///

class FAGcLDB : public FALDB {

public:
    FAGcLDB (FAAllocatorA * pAlloc);
    virtual ~FAGcLDB ();

public:
    void SetImage (const unsigned char * pImgDump);
    const FAWREConfCA * GetCommon () const;
    const FAWREConfCA * GetRule (const int i) const;

private:
    void Clear ();

private:
    /// common Moore WRE
    FAWREConf_pack m_common_wre;
    /// Mealy WREs each one per rule, NULL if does not exist
    FAWREConf_pack ** m_ppMealyWres;
    /// rule count
    int m_Count;
};

}

#endif
