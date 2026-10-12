/**
 * Copyright (c) Microsoft Corporation. All rights reserved.
 * Licensed under the MIT License.
 */

#ifndef _FA_WRECONF_PACK_H_
#define _FA_WRECONF_PACK_H_

#include "FAConfig.h"
#include "FAWREConfCA.h"
#include "FASetImageA.h"

namespace BlingFire
{

class FAArray_pack;
class FAMultiMap_pack;
class FAState2Ow_pack_triv;
class FAState2Ows_pack_triv;
class FAMealyDfa_pack_triv;
class FARSDfa_pack_triv;

///
/// An container for the compiled WRE.
///

class FAWREConf_pack : public FASetImageA,
                       public FAWREConfCA {

public:
    FAWREConf_pack ();
    virtual ~FAWREConf_pack ();

public:
    void SetImage (const unsigned char * pImage);

public:
    const int GetType () const;
    const int GetTokenType () const;
    const int GetTagOwBase () const;
    const FARSDfaCA * GetTxtDigDfa () const;
    const FAState2OwCA * GetTxtDigOws () const;
    const FAArrayCA * GetDictDig () const;
    const FARSDfaCA * GetDfa1 () const;
    const FARSDfaCA * GetDfa2 () const;
    const FAState2OwsCA * GetState2Ows () const;
    const FAMealyDfaCA * GetSigma1 () const;
    const FAMealyDfaCA * GetSigma2 () const;
    const FAMultiMapCA * GetTrBrMap () const;

private:
    void Clear ();

private:
    int m_WreType;
    int m_TokenType;
    int m_TagOwBase;

    FARSDfa_pack_triv * m_pTxtDfa;
    FAState2Ow_pack_triv * m_pTxtOws;
    FAArray_pack * m_pDctArr;
    FARSDfa_pack_triv * m_pDfa1;
    FARSDfa_pack_triv * m_pDfa2;
    FAState2Ows_pack_triv * m_pState2Ows;
    FAMealyDfa_pack_triv * m_pSigma1;
    FAMealyDfa_pack_triv * m_pSigma2;
    FAMultiMap_pack * m_pTrBr;

};

}

#endif
