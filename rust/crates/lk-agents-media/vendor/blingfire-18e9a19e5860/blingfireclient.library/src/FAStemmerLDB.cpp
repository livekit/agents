/**
 * Copyright (c) Microsoft Corporation. All rights reserved.
 * Licensed under the MIT License.
 */


#include "blingfire-client_src_pch.h"
#include "FAConfig.h"
#include "FAFsmConst.h"
#include "FAStemmerLDB.h"

namespace BlingFire
{

FAStemmerLDB::FAStemmerLDB () :
    FALDB (),
    m_fW2B (false),
    m_fB2W (false),
    m_fWT2B (false),
    m_fB2WT (false),
    m_fW2P (false)
{}


FAStemmerLDB::~FAStemmerLDB ()
{
    FAStemmerLDB::Clear ();
}


void FAStemmerLDB::Clear ()
{
    m_fW2B = false;
    m_fB2W = false;
    m_fWT2B = false;
    m_fB2WT = false;
    m_fW2P = false;

    m_w2b.Clear ();
    m_b2w.Clear ();
    m_wt2b.Clear ();
    m_b2wt.Clear ();
    m_w2p.Clear ();
}



void FAStemmerLDB::Init ()
{
    const int * pValues;
    int Size;

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
    Size = m_Conf.Get (FAFsmConst::FUNC_W2P, &pValues);
    m_w2p.Initialize (this, pValues, Size);
    m_fW2P = 0 < Size;
}



void FAStemmerLDB::SetImage (const unsigned char * pImgDump)
{
    Clear ();

    FALDB::SetImage (pImgDump);

    if (NULL != pImgDump) {
        Init ();
    }
}



const FAWftConfKeeper * FAStemmerLDB::GetW2BConf () const
{
    if (!m_fW2B) {
        return NULL;
    }
    return & m_w2b;
}


const FAWftConfKeeper * FAStemmerLDB::GetB2WConf () const
{
    if (!m_fB2W) {
        return NULL;
    }
    return & m_b2w;
}


const FAWftConfKeeper * FAStemmerLDB::GetWT2BConf () const
{
    if (!m_fWT2B) {
        return NULL;
    }
    return & m_wt2b;
}


const FAWftConfKeeper * FAStemmerLDB::GetB2WTConf () const
{
    if (!m_fB2WT) {
        return NULL;
    }
    return & m_b2wt;
}


const FAW2PConfKeeper * FAStemmerLDB::GetW2PConf () const
{
    if (!m_fW2P) {
        return NULL;
    }
    return & m_w2p;
}

}
