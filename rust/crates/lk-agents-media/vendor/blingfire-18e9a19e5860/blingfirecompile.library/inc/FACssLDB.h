/**
 * Copyright (c) Microsoft Corporation. All rights reserved.
 * Licensed under the MIT License.
 */


#ifndef _FA_CSSLDB_H_
#define _FA_CSSLDB_H_

#include "FAConfig.h"
#include "FALDB.h"
#include "FAWgConfKeeper.h"
#include "FADictConfKeeper.h"
#include "FAWreRulesConfKeeper.h"
#include "FAGlobalConfKeeper_packaged.h"

namespace BlingFire
{

class FAAllocatorA;


///
/// Keeps morphology resources. (The object is safe to share among threads,
//  after initialization). Every GetXXXConf returns NULL if corresponding
/// data does not exist.
///

class FACssLDB : public FALDB {

public:
    FACssLDB (FAAllocatorA * pAlloc);
    ~FACssLDB ();

public:
    void SetImage (const unsigned char * pImgDump);
    /// returns object into the initial state, (automatically called from the
    /// SetImage)
    void Clear ();

public:
    /// returns W --> {T/P}, word tag guesser data, can be NULL
    const FAWgConfKeeper * GetW2TPConf () const;
    /// returns tag dictionary data, can be NULL
    const FADictConfKeeper * GetTagDictConf () const;
    /// returns OIC rules data, can be NULL
    const FAWreRulesConfKeeper * GetOicConf () const;
    /// returns CSS WRE rules data, can be NULL
    const FAWreRulesConfKeeper * GetCssRulesConf () const;
    /// returns global configuration
    const FAGlobalConfKeeper * GetGlobalConf () const;

private:
    void Init ();

private:
    FAAllocatorA * m_pAlloc;
    // tag dict
    FADictConfKeeper m_tag_dict;
    // W2TP* data
    FAWgConfKeeper m_w2tp;
    // OIC
    FAWreRulesConfKeeper m_oic;
    // CSS rules
    FAWreRulesConfKeeper m_css_rules;
    /// global data
    FAGlobalConfKeeper m_global;

    // flags
    bool m_fTagDict;
    bool m_fW2TP;
    bool m_fOic;
    bool m_fCssRules;
    bool m_fGlobal;
};

}

#endif
