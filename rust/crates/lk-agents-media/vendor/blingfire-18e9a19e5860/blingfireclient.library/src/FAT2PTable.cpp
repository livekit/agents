/**
 * Copyright (c) Microsoft Corporation. All rights reserved.
 * Licensed under the MIT License.
 */


#include "blingfire-client_src_pch.h"
#include "FAConfig.h"
#include "FAT2PTable.h"
#include "FATsConfKeeper.h"

namespace BlingFire
{

FAT2PTable::FAT2PTable () :
    m_pArr (NULL),
    m_Size (0)
{}


FAT2PTable::~FAT2PTable ()
{}


void FAT2PTable::SetConf (const FATsConfKeeper * pConf)
{
    m_pArr  = NULL;
    m_Size = 0;

    if (!pConf) {
        return;
    }

    m_Size = pConf->GetArr (&m_pArr);

    LogAssert (m_pArr && 0 < m_Size);
}

}
