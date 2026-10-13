/**
 * Copyright (c) Microsoft Corporation. All rights reserved.
 * Licensed under the MIT License.
 */

#include "blingfire-client_src_pch.h"
#include "FAConfig.h"
#include "FAWREConf_pack.h"
#include "FAFsmConst.h"
#include "FAArray_pack.h"
#include "FAMultiMap_pack.h"
#include "FAState2Ow_pack_triv.h"
#include "FAState2Ows_pack_triv.h"
#include "FAMealyDfa_pack_triv.h"
#include "FARSDfa_pack_triv.h"

namespace BlingFire
{

FAWREConf_pack::FAWREConf_pack () :
    m_WreType (0),
    m_TokenType (0),
    m_TagOwBase (0),
    m_pTxtDfa (NULL),
    m_pTxtOws (NULL),
    m_pDctArr (NULL),
    m_pDfa1 (NULL),
    m_pDfa2 (NULL),
    m_pState2Ows (NULL),
    m_pSigma1 (NULL),
    m_pSigma2 (NULL),
    m_pTrBr (NULL)
{}


FAWREConf_pack::~FAWREConf_pack ()
{
    FAWREConf_pack::Clear ();
}


void FAWREConf_pack::Clear ()
{
    if (m_pTxtDfa) {
        delete m_pTxtDfa;
        m_pTxtDfa = NULL;
    }
    if (m_pTxtOws) {
        delete m_pTxtOws;
        m_pTxtOws = NULL;
    }
    if (m_pDctArr) {
        delete m_pDctArr;
        m_pDctArr = NULL;
    }
    if (m_pDfa1) {
        delete m_pDfa1;
        m_pDfa1 = NULL;
    }
    if (m_pDfa2) {
        delete m_pDfa2;
        m_pDfa2 = NULL;
    }
    if (m_pState2Ows) {
        delete m_pState2Ows;
        m_pState2Ows = NULL;
    }
    if (m_pSigma1) {
        delete m_pSigma1;
        m_pSigma1 = NULL;
    }
    if (m_pSigma2) {
        delete m_pSigma2;
        m_pSigma2 = NULL;
    }
    if (m_pTrBr) {
        delete m_pTrBr;
        m_pTrBr = NULL;
    }
}


void FAWREConf_pack::SetImage (const unsigned char * pImage)
{
    FAWREConf_pack::Clear ();

    if (!pImage) {
        return;
    }

    const int * pConf = (const int *) pImage ;

    /// setup constants
    m_WreType = pConf [FAFsmConst::WRE_CONF_WRE_TYPE];
    LogAssert (FAFsmConst::WRE_TYPE_RS <= m_WreType && \
        FAFsmConst::WRE_TYPE_COUNT > m_WreType);

    m_TokenType = pConf [FAFsmConst::WRE_CONF_TOKEN_TYPE];
    LogAssert ((FAFsmConst::WRE_TT_TEXT | FAFsmConst::WRE_TT_TAGS | \
        FAFsmConst::WRE_TT_DCTS) & m_TokenType);

    m_TagOwBase = pConf [FAFsmConst::WRE_CONF_TAG_OW_BASE];

    /// setup text digitizer, if any
    if (0 < pConf [FAFsmConst::WRE_CONF_TXT_DIG]) {

        const unsigned char * pImg = \
            pImage + pConf [FAFsmConst::WRE_CONF_TXT_DIG];

        DebugLogAssert (NULL == m_pTxtDfa);
        m_pTxtDfa = NEW FARSDfa_pack_triv;
        m_pTxtDfa->SetImage (pImg);

        DebugLogAssert (NULL == m_pTxtOws);
        m_pTxtOws = NEW FAState2Ow_pack_triv;
        m_pTxtOws->SetImage (pImg);
    }

    /// setup dict digitizer, if any
    if (0 < pConf [FAFsmConst::WRE_CONF_DCT_DIG]) {

        const unsigned char * pImg = \
            pImage + pConf [FAFsmConst::WRE_CONF_DCT_DIG];

        DebugLogAssert (NULL == m_pDctArr);
        m_pDctArr = NEW FAArray_pack;
        m_pDctArr->SetImage (pImg);
    }

    /// setup rules1 automaton, if any
    if (0 < pConf [FAFsmConst::WRE_CONF_FSM1]) {

        const unsigned char * pImg = \
            pImage + pConf [FAFsmConst::WRE_CONF_FSM1];

        DebugLogAssert (NULL == m_pDfa1);
        m_pDfa1 = NEW FARSDfa_pack_triv;
        m_pDfa1->SetImage (pImg);

        if (FAFsmConst::WRE_TYPE_MOORE == m_WreType) {

            DebugLogAssert (NULL == m_pState2Ows);
            m_pState2Ows = NEW FAState2Ows_pack_triv;
            m_pState2Ows->SetImage (pImg);

        } else if (FAFsmConst::WRE_TYPE_MEALY == m_WreType) {

            DebugLogAssert (NULL == m_pSigma1);
            m_pSigma1 = NEW FAMealyDfa_pack_triv;
            m_pSigma1->SetImage (pImg);
        }
    }

    /// setup rules2 automaton, if any
    if (0 < pConf [FAFsmConst::WRE_CONF_FSM2]) {

        const unsigned char * pImg = \
            pImage + pConf [FAFsmConst::WRE_CONF_FSM2];

        DebugLogAssert (NULL == m_pDfa2);
        m_pDfa2 = NEW FARSDfa_pack_triv;
        m_pDfa2->SetImage (pImg);

        DebugLogAssert (NULL == m_pSigma2);
        m_pSigma2 = NEW FAMealyDfa_pack_triv;
        m_pSigma2->SetImage (pImg);
    }

    /// setup trbr map, if any
    if (0 < pConf [FAFsmConst::WRE_CONF_TRBR]) {

        const unsigned char * pImg = \
            pImage + pConf [FAFsmConst::WRE_CONF_TRBR];

        DebugLogAssert (NULL == m_pTrBr);
        m_pTrBr = NEW FAMultiMap_pack;
        m_pTrBr->SetImage (pImg);
    }
}


const int FAWREConf_pack::GetType () const
{
    return m_WreType;
}

const int FAWREConf_pack::GetTokenType () const
{
    return m_TokenType;
}

const int FAWREConf_pack::GetTagOwBase () const
{
    return m_TagOwBase;
}

const FARSDfaCA * FAWREConf_pack::GetTxtDigDfa () const
{
    return m_pTxtDfa;
}

const FAState2OwCA * FAWREConf_pack::GetTxtDigOws () const
{
    return m_pTxtOws;
}

const FAArrayCA * FAWREConf_pack::GetDictDig () const
{
    return m_pDctArr;
}

const FARSDfaCA * FAWREConf_pack::GetDfa1 () const
{
    return m_pDfa1;
}

const FARSDfaCA * FAWREConf_pack::GetDfa2 () const
{
    return m_pDfa2;
}

const FAState2OwsCA * FAWREConf_pack::GetState2Ows () const
{
    return m_pState2Ows;
}

const FAMealyDfaCA * FAWREConf_pack::GetSigma1 () const
{
    return m_pSigma1;
}

const FAMealyDfaCA * FAWREConf_pack::GetSigma2 () const
{
    return m_pSigma2;
}

const FAMultiMapCA * FAWREConf_pack::GetTrBrMap () const
{
    return m_pTrBr;
}

}
