/**
 * Copyright (c) Microsoft Corporation. All rights reserved.
 * Licensed under the MIT License.
 */


#ifndef _FA_ACTIONSARRAY_STATIC_H_
#define _FA_ACTIONSARRAY_STATIC_H_

#include "FAConfig.h"
#include "FAActionsArrayA.h"

namespace BlingFire
{

///
/// Implementation based on static array.
///

class FAActionsArray_static : public FAActionsArrayA {

public:
    FAActionsArray_static (const FAActionsA ** pActsArr, const int Count);
    virtual ~FAActionsArray_static ();

public:
    const int GetStageCount () const;
    const FAActionsA * GetStage (const int Num) const;

private:
    const FAActionsA ** m_pActsArr;
    const int m_Count;
};

}

#endif
