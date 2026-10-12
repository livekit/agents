/**
 * Copyright (c) Microsoft Corporation. All rights reserved.
 * Licensed under the MIT License.
 */


#ifndef _FA_MORPHLDB_T_H_
#define _FA_MORPHLDB_T_H_

#include "FAConfig.h"
#include "FALDB.h"
#include "FAFsmConst.h"
#include "FAWftConfKeeper.h"
#include "FAWgConfKeeper.h"
#include "FAW2SConfKeeper.h"
#include "FADictConfKeeper.h"
#include "FAHyphConfKeeper_packaged.h"
#include "FATrsConfKeeper_t.h"
#include "FAWbdConfKeeper.h"
#include "FAGlobalConfKeeper_packaged.h"
#include "FATsConfKeeper.h"
#include "FASecurity.h"

namespace BlingFire
{

///
/// Keeps morphology resources. (The object is safe to share among threads,
//  after initialization). Every GetXXXConf returns NULL if corresponding
/// data does not exist.
///
/// Notes:
///  1. Abbreviations:
///       W - word form(s) including base form,
///       B - base form(s),
///       T - POS tag
///       S - segments of a word
///       H - word's hyphenation points
///       Tr - transformation
///       TagDict - dictionary keeps sets of different tags
///       PosDict - dictionary keeps POS tags and probabilities
///       TP - Tag/P(T|W)
///       V - spelling variants
///  2. Ty is a character type.
///  3. If the resource does not exist then the pointer will be NULL
///

template < class Ty >
class FAMorphLDB_t : public FALDB {

public:
    FAMorphLDB_t ();
    ~FAMorphLDB_t ();

public:
    void SetImage (const unsigned char * pImgDump);

public:
    const FATransformCA_t < Ty > * GetInTr () const;
    const FATransformCA_t < Ty > * GetOutTr () const;
    const FAWgConfKeeper * GetW2TConf () const;
    const FAWgConfKeeper * GetB2TConf () const;
    const FADictConfKeeper * GetTagDictConf () const;
    const FADictConfKeeper * GetPosDictConf () const;
    const FADictConfKeeper * GetNormDictConf () const;
    const FADictConfKeeper * GetConcatDictConf () const;
    const FADictConfKeeper * GetEmissionDictConf () const;
    const FAW2SConfKeeper * GetW2SConf () const;
    const FAWftConfKeeper * GetW2BConf () const;
    const FAWftConfKeeper * GetB2WConf () const;
    const FAWftConfKeeper * GetWT2BConf () const;
    const FAWftConfKeeper * GetB2WTConf () const;
    const FAHyphConfKeeper * GetW2HConf () const;
    const FAHyphConfKeeper * GetW2HAltConf () const;
    const FAWgConfKeeper * GetW2TPConf () const;
    const FAWgConfKeeper * GetW2TPLConf () const;
    const FAWgConfKeeper * GetW2TPRConf () const;
    const FAWbdConfKeeper * GetWbdConf () const;
    const FAGlobalConfKeeper * GetGlobalConf () const;
    const FATsConfKeeper * GetT2PConf () const;
    const FATsConfKeeper * GetTT2PConf () const;
    const FATsConfKeeper * GetTTT2PConf () const;
    const FAWftConfKeeper * GetW2VConf () const;

    /// returns object into the initial state, (automatically called from the
    /// SetImage)
    void Clear ();

private:
    void Init ();

private:
    // transformations
    FATrsConfKeeper_t < Ty > m_trs;
    // B2T, W2T data
    FAWgConfKeeper m_w2t;
    FAWgConfKeeper m_b2t;
    // W2S data
    FAW2SConfKeeper m_w2s;
    // WFT data
    FAWftConfKeeper m_w2b;
    FAWftConfKeeper m_b2w;
    FAWftConfKeeper m_wt2b;
    FAWftConfKeeper m_b2wt;
    // tag dict
    FADictConfKeeper m_tag_dict;
    // W2H data
    FAHyphConfKeeper m_w2h;
    FAHyphConfKeeper m_w2h_alt;
    // POS (Tag/Prob) dict data
    FADictConfKeeper m_pos_dict;
    // W2TP* data
    FAWgConfKeeper m_w2tp;
    FAWgConfKeeper m_w2tpl;
    FAWgConfKeeper m_w2tpr;
    // Word breaker data
    FAWbdConfKeeper m_wbd;
    // global client specific configuration
    FAGlobalConfKeeper m_global;
    // T[T[T]] --> P data
    FATsConfKeeper m_t2p;
    FATsConfKeeper m_tt2p;
    FATsConfKeeper m_ttt2p;
    // normalization dict
    FADictConfKeeper m_norm_dict;
    // concatenation dict
    FADictConfKeeper m_concat;
    // NE emission dict
    FADictConfKeeper m_emission;
    // W2V (spelling variants) data
    FAWftConfKeeper m_w2v;

    // flags
    bool m_fTrs;
    bool m_fW2T;
    bool m_fB2T;
    bool m_fW2S;
    bool m_fW2B;
    bool m_fB2W;
    bool m_fWT2B;
    bool m_fB2WT;
    bool m_fTagDict;
    bool m_fW2H;
    bool m_fW2HAlt;
    bool m_fPosDict;
    bool m_fW2TP;
    bool m_fW2TPL;
    bool m_fW2TPR;
    bool m_fWbd;
    bool m_fNormDict;
    bool m_fConcat;
    bool m_fGlobal;
    bool m_fEmission;
    bool m_fT2P;
    bool m_fTT2P;
    bool m_fTTT2P;
    bool m_fW2V;
};


template < class Ty >
FAMorphLDB_t< Ty >::FAMorphLDB_t () :
    FALDB (),
    m_fTrs (false),
    m_fW2T (false),
    m_fB2T (false),
    m_fW2S (false),
    m_fW2B (false),
    m_fB2W (false),
    m_fWT2B (false),
    m_fB2WT (false),
    m_fTagDict (false),
    m_fW2H (false),
    m_fW2HAlt (false),
    m_fPosDict (false),
    m_fW2TP (false),
    m_fW2TPL (false),
    m_fW2TPR (false),
    m_fWbd (false),
    m_fNormDict (false),
    m_fConcat (false),
    m_fGlobal (false),
    m_fEmission (false),
    m_fT2P (false),
    m_fTT2P (false),
    m_fTTT2P (false),
    m_fW2V (false)
{}


template < class Ty >
FAMorphLDB_t< Ty >::~FAMorphLDB_t ()
{
    FAMorphLDB_t< Ty >::Clear ();
}


template < class Ty >
void FAMorphLDB_t< Ty >::Clear ()
{
    m_fTrs = false;
    m_fW2T = false;
    m_fB2T = false;
    m_fW2S = false;
    m_fW2B = false;
    m_fB2W = false;
    m_fWT2B = false;
    m_fB2WT = false;
    m_fTagDict = false;
    m_fW2H = false;
    m_fW2HAlt = false;
    m_fPosDict = false;
    m_fW2TP = false;
    m_fW2TPL = false;
    m_fW2TPR = false;
    m_fWbd = false;
    m_fNormDict = false;
    m_fConcat = false;
    m_fEmission = false;
    m_fGlobal = false;
    m_fT2P = false;
    m_fTT2P = false;
    m_fTTT2P = false;
    m_fW2V = false;

    m_trs.Clear ();
    m_w2t.Clear ();
    m_b2t.Clear ();
    m_w2s.Clear ();
    m_w2b.Clear ();
    m_b2w.Clear ();
    m_wt2b.Clear ();
    m_b2wt.Clear ();
    m_tag_dict.Clear ();
    m_w2h.Clear ();
    m_w2h_alt.Clear ();
    m_pos_dict.Clear ();
    m_w2tp.Clear ();
    m_w2tpl.Clear ();
    m_w2tpr.Clear ();
    m_wbd.Clear ();
    m_norm_dict.Clear();
    m_concat.Clear();
    m_emission.Clear();
    m_global.Clear ();
    m_t2p.Clear ();
    m_tt2p.Clear ();
    m_ttt2p.Clear ();
    m_w2v.Clear ();
}


template < class Ty >
void FAMorphLDB_t< Ty >::Init ()
{
    const int * pValues;
    int Size;

    pValues = NULL;
    Size = m_Conf.Get (FAFsmConst::FUNC_TRS, &pValues);
    m_trs.Init (pValues, Size);
    m_fTrs = 0 < Size;

    pValues = NULL;
    Size = m_Conf.Get (FAFsmConst::FUNC_W2T, &pValues);
    m_w2t.Initialize (this, pValues, Size);
    m_fW2T = 0 < Size;

    pValues = NULL;
    Size = m_Conf.Get (FAFsmConst::FUNC_B2T, &pValues);
    m_b2t.Initialize (this, pValues, Size);
    m_fB2T = 0 < Size;

    pValues = NULL;
    Size = m_Conf.Get (FAFsmConst::FUNC_TAG_DICT, &pValues);
    m_tag_dict.Init (pValues, Size);
    m_fTagDict = 0 < Size;

    pValues = NULL;
    Size = m_Conf.Get (FAFsmConst::FUNC_POS_DICT, &pValues);
    m_pos_dict.Init (pValues, Size);
    m_fPosDict = 0 < Size;

    pValues = NULL;
    Size = m_Conf.Get (FAFsmConst::FUNC_W2H, &pValues);
    m_w2h.Init (pValues, Size);
    m_fW2H = 0 < Size;

    pValues = NULL;
    Size = m_Conf.Get (FAFsmConst::FUNC_W2H_ALT, &pValues);
    m_w2h_alt.Init (pValues, Size);
    m_fW2HAlt = 0 < Size;

    pValues = NULL;
    Size = m_Conf.Get (FAFsmConst::FUNC_W2S, &pValues);
    m_w2s.Init (pValues, Size);
    m_fW2S = 0 < Size;

    pValues = NULL;
    Size = m_Conf.Get (FAFsmConst::FUNC_W2B, &pValues);
    m_w2b.Initialize (this, pValues, Size);
    m_fW2B = 0 < Size;

    pValues = NULL;
    Size = m_Conf.Get (FAFsmConst::FUNC_B2W, &pValues);
    m_b2w.Initialize (this, pValues, Size);
    m_fB2W = 0 < Size;

    pValues = NULL;
    Size = m_Conf.Get (FAFsmConst::FUNC_WT2B, &pValues);
    m_wt2b.Initialize (this, pValues, Size);
    m_fWT2B = 0 < Size;

    pValues = NULL;
    Size = m_Conf.Get (FAFsmConst::FUNC_B2WT, &pValues);
    m_b2wt.Initialize (this, pValues, Size);
    m_fB2WT = 0 < Size;

    pValues = NULL;
    Size = m_Conf.Get (FAFsmConst::FUNC_W2TP, &pValues);
    m_w2tp.Initialize (this, pValues, Size);
    m_fW2TP = 0 < Size;

    pValues = NULL;
    Size = m_Conf.Get (FAFsmConst::FUNC_W2TPL, &pValues);
    m_w2tpl.Initialize (this, pValues, Size);
    m_fW2TPL = 0 < Size;

    pValues = NULL;
    Size = m_Conf.Get (FAFsmConst::FUNC_W2TPR, &pValues);
    m_w2tpr.Initialize (this, pValues, Size);
    m_fW2TPR = 0 < Size;

    pValues = NULL;
    Size = m_Conf.Get (FAFsmConst::FUNC_WBD, &pValues);
    m_wbd.Initialize (this, pValues, Size);
    m_fWbd = 0 < Size;

    pValues = NULL;
    Size = m_Conf.Get (FAFsmConst::FUNC_NORM_DICT, &pValues);
    m_norm_dict.Init (pValues, Size);
    m_fNormDict = 0 < Size;

    pValues = NULL;
    Size = m_Conf.Get (FAFsmConst::FUNC_NORM_RULES, &pValues);
    m_concat.Init (pValues, Size);
    m_fConcat = 0 < Size;

    pValues = NULL;
    Size = m_Conf.Get (FAFsmConst::FUNC_EMIT, &pValues);
    m_emission.Init (pValues, Size);
    m_fEmission = 0 < Size;

    pValues = NULL;
    Size = m_Conf.Get (FAFsmConst::FUNC_GLOBAL, &pValues);
    m_global.Init (pValues, Size);
    m_fGlobal = 0 < Size;

    pValues = NULL;
    Size = m_Conf.Get (FAFsmConst::FUNC_T2P, &pValues);
    m_t2p.Init (pValues, Size);
    m_fT2P = 0 < Size;

    pValues = NULL;
    Size = m_Conf.Get (FAFsmConst::FUNC_TT2P, &pValues);
    m_tt2p.Init (pValues, Size);
    m_fTT2P = 0 < Size;

    pValues = NULL;
    Size = m_Conf.Get (FAFsmConst::FUNC_TTT2P, &pValues);
    m_ttt2p.Init (pValues, Size);
    m_fTTT2P = 0 < Size;

    pValues = NULL;
    Size = m_Conf.Get (FAFsmConst::FUNC_W2V, &pValues);
    m_w2v.Initialize (this, pValues, Size);
    m_fW2V = 0 < Size;
}


template < class Ty >
void FAMorphLDB_t< Ty >::SetImage (const unsigned char * pImgDump)
{
    m_trs.SetLDB (this);
    m_w2s.SetLDB (this);
    m_tag_dict.SetLDB (this);
    m_w2h.SetLDB (this);
    m_w2h_alt.SetLDB (this);
    m_pos_dict.SetLDB (this);
    m_norm_dict.SetLDB (this);
    m_concat.SetLDB (this);
    m_emission.SetLDB (this);
    m_global.SetLDB (this);
    m_t2p.SetLDB (this);
    m_tt2p.SetLDB (this);
    m_ttt2p.SetLDB (this);

    FAMorphLDB_t< Ty >::Clear ();

    FALDB::SetImage (pImgDump);

    FAMorphLDB_t< Ty >::Init ();
}


template < class Ty >
const FATransformCA_t < Ty > * FAMorphLDB_t< Ty >::GetInTr () const
{
    if (!m_fTrs) {
        return NULL;
    }
    return m_trs.GetInTr ();
}

template < class Ty >
const FATransformCA_t < Ty > * FAMorphLDB_t< Ty >::GetOutTr () const
{
    if (!m_fTrs) {
        return NULL;
    }
    return m_trs.GetOutTr ();
}

template < class Ty >
const FAWgConfKeeper * FAMorphLDB_t< Ty >::GetW2TConf () const
{
    if (!m_fW2T) {
        return NULL;
    }
    return & m_w2t;
}

template < class Ty >
const FAWgConfKeeper * FAMorphLDB_t< Ty >::GetB2TConf () const
{
    if (!m_fB2T) {
        return NULL;
    }
    return & m_b2t;
}

template < class Ty >
const FADictConfKeeper * FAMorphLDB_t< Ty >::GetTagDictConf () const
{
    if (!m_fTagDict) {
        return NULL;
    }
    return & m_tag_dict;
}

template < class Ty >
const FADictConfKeeper * FAMorphLDB_t< Ty >::GetPosDictConf () const
{
    if (!m_fPosDict) {
        return NULL;
    }
    return & m_pos_dict;
}

template < class Ty >
const FAHyphConfKeeper * FAMorphLDB_t< Ty >::GetW2HConf () const
{
    if (!m_fW2H) {
        return NULL;
    }
    return & m_w2h;
}

template < class Ty >
const FAHyphConfKeeper * FAMorphLDB_t< Ty >::GetW2HAltConf () const
{
    if (!m_fW2HAlt) {
        return NULL;
    }
    return & m_w2h_alt;
}

template < class Ty >
const FAW2SConfKeeper * FAMorphLDB_t< Ty >::GetW2SConf () const
{
    if (!m_fW2S) {
        return NULL;
    }
    return & m_w2s;
}

template < class Ty >
const FAWftConfKeeper * FAMorphLDB_t< Ty >::GetW2BConf () const
{
    if (!m_fW2B) {
        return NULL;
    }
    return & m_w2b;
}

template < class Ty >
const FAWftConfKeeper * FAMorphLDB_t< Ty >::GetB2WConf () const
{
    if (!m_fB2W) {
        return NULL;
    }
    return & m_b2w;
}

template < class Ty >
const FAWftConfKeeper * FAMorphLDB_t< Ty >::GetWT2BConf () const
{
    if (!m_fWT2B) {
        return NULL;
    }
    return & m_wt2b;
}

template < class Ty >
const FAWftConfKeeper * FAMorphLDB_t< Ty >::GetB2WTConf () const
{
    if (!m_fB2WT) {
        return NULL;
    }
    return & m_b2wt;
}

template < class Ty >
const FAWgConfKeeper * FAMorphLDB_t< Ty >::GetW2TPConf () const
{
    if (!m_fW2TP) {
        return NULL;
    }
    return & m_w2tp;
}

template < class Ty >
const FAWgConfKeeper * FAMorphLDB_t< Ty >::GetW2TPLConf () const
{
    if (!m_fW2TPL) {
        return NULL;
    }
    return & m_w2tpl;
}

template < class Ty >
const FAWgConfKeeper * FAMorphLDB_t< Ty >::GetW2TPRConf () const
{
    if (!m_fW2TPR) {
        return NULL;
    }
    return & m_w2tpr;
}

template < class Ty >
const FAWbdConfKeeper * FAMorphLDB_t< Ty >::GetWbdConf () const
{
    if (!m_fWbd) {
        return NULL;
    }
    return & m_wbd;
}

template < class Ty >
const FADictConfKeeper * FAMorphLDB_t< Ty >::GetNormDictConf () const
{
    if (!m_fNormDict) {
        return NULL;
    }
    return & m_norm_dict;
}

template < class Ty >
const FADictConfKeeper * FAMorphLDB_t< Ty >::GetConcatDictConf () const
{
    if (!m_fConcat) {
        return NULL;
    }
    return & m_concat;
}

template < class Ty >
const FADictConfKeeper * FAMorphLDB_t< Ty >::GetEmissionDictConf () const
{
    if (!m_fEmission) {
        return NULL;
    }
    return & m_emission;
}

template < class Ty >
const FAGlobalConfKeeper * FAMorphLDB_t< Ty >::GetGlobalConf () const
{
    if (!m_fGlobal) {
        return NULL;
    }
    return & m_global;
}

template < class Ty >
const FATsConfKeeper * FAMorphLDB_t< Ty >::GetT2PConf () const
{
    if (!m_fT2P) {
        return NULL;
    }
    return & m_t2p;
}

template < class Ty >
const FATsConfKeeper * FAMorphLDB_t< Ty >::GetTT2PConf () const
{
    if (!m_fTT2P) {
        return NULL;
    }
    return & m_tt2p;
}

template < class Ty >
const FATsConfKeeper * FAMorphLDB_t< Ty >::GetTTT2PConf () const
{
    if (!m_fTTT2P) {
        return NULL;
    }
    return & m_ttt2p;
}

template < class Ty >
const FAWftConfKeeper * FAMorphLDB_t< Ty >::GetW2VConf () const
{
    if (!m_fW2V) {
        return NULL;
    }
    return & m_w2v;
}

}

#endif
