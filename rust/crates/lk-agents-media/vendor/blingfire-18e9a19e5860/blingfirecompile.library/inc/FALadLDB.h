/**
 * Copyright (c) Microsoft Corporation. All rights reserved.
 * Licensed under the MIT License.
 */


#ifndef _FA_LADLDB_H_
#define _FA_LADLDB_H_

#include "FAConfig.h"
#include "FALDB.h"
#include "FAWgConfKeeper.h"
#include "FAWbdConfKeeper.h"
#include "FALadConfKeeper.h"

namespace BlingFire
{

///
/// Keeps all the resources needed for the stemmer.
///

class FALadLDB : public FALDB {

public:
    FALadLDB ();
    ~FALadLDB ();

public:
    void SetImage (const unsigned char * pImgDump);

public:
    const FAWgConfKeeper * GetW2TPConf () const;
    const FAWgConfKeeper * GetN2TPConf () const;
    const FAWbdConfKeeper * GetU2LConf () const;
    const FALadConfKeeper * GetLadConf () const;

private:
    /// returns object into the initial state, (automatically called from
    /// the SetImage)
    void Clear ();
    void Init ();

private:
    // N2TP, W2TP data
    FAWgConfKeeper m_w2tp;
    FAWgConfKeeper m_n2tp;
    // U2L data
    FAWbdConfKeeper m_u2l;
    // LAD parameters
    FALadConfKeeper m_lad;

    // flags
    bool m_fW2TP;
    bool m_fN2TP;
    bool m_fU2L;
    bool m_fLAD;
};

}

#endif
