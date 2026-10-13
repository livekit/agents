/**
 * Copyright (c) Microsoft Corporation. All rights reserved.
 * Licensed under the MIT License.
 */

#ifndef _FA_ARRAY_P2CA_H_
#define _FA_ARRAY_P2CA_H_

#include "FAConfig.h"
#include "FAArrayCA.h"

namespace BlingFire
{

///
/// int [] --> FAArrayCA
///

class FAArray_p2ca : public FAArrayCA {

public:
    FAArray_p2ca ();

public:
    const int GetAt (const int Idx) const;
    const int GetCount () const;
    void SetArray (const int * pA, const int Count);

private:
    const int * m_pA;
    int m_Count;
};

}

#endif
