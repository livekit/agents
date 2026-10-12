/**
 * Copyright (c) Microsoft Corporation. All rights reserved.
 * Licensed under the MIT License.
 */


#ifndef _FA_STEMMERLDB_H_
#define _FA_STEMMERLDB_H_

#include "FAConfig.h"
#include "FALDB.h"
#include "FAWftConfKeeper.h"
#include "FAW2PConfKeeper.h"

namespace BlingFire
{

///
/// Keeps all the resources needed for the stemmer.
///

class FAStemmerLDB : public FALDB {

public:
    FAStemmerLDB ();
    ~FAStemmerLDB ();

public:
    void SetImage (const unsigned char * pImgDump);

public:
    const FAWftConfKeeper * GetW2BConf () const;
    const FAWftConfKeeper * GetB2WConf () const;
    const FAWftConfKeeper * GetWT2BConf () const;
    const FAWftConfKeeper * GetB2WTConf () const;
    const FAW2PConfKeeper * GetW2PConf () const;

private:
    /// returns object into the initial state, (automatically called from
    /// the SetImage)
    void Clear ();
    void Init ();

private:
    // WFT data
    FAWftConfKeeper m_w2b;
    FAWftConfKeeper m_b2w;
    FAWftConfKeeper m_wt2b;
    FAWftConfKeeper m_b2wt;
    // W2P data
    FAW2PConfKeeper m_w2p;

    // flags
    bool m_fW2B;
    bool m_fB2W;
    bool m_fWT2B;
    bool m_fB2WT;
    bool m_fW2P;
};

}

#endif
