/**
 * Copyright (c) Microsoft Corporation. All rights reserved.
 * Licensed under the MIT License.
 */

#ifndef _FA_TRSCONFKEEPER_T_H_
#define _FA_TRSCONFKEEPER_T_H_

#include "FAConfig.h"
#include "FATransform_hyph_redup_t.h"
#include "FATransform_hyph_redup_rev_t.h"
#include "FATransform_prefix_t.h"
#include "FATransform_prefix_rev_t.h"
#include "FATransform_capital_t.h"
#include "FATransform_capital_rev_t.h"
#include "FATransform_cascade_t.h"
#include "FARSDfa_pack_triv.h"
#include "FASecurity.h"

namespace BlingFire
{

template < class Ty >
class FATrsConfKeeper_t {

public:
    FATrsConfKeeper_t ();
    virtual ~FATrsConfKeeper_t ();

public:
    /// this LDB will be used to get the data from
    void SetLDB (const FALDB * pLDB);
    /// initialization vector
    void Init (const int * pValues, const int Size);
    /// returns object into the initial state
    void Clear ();

public:
    const FATransformCA_t < Ty > * GetInTr () const;
    const FATransformCA_t < Ty > * GetOutTr () const;
    const bool GetIgnoreCase () const;

private:
    const FATransformCA_t < Ty > * SelectTr (const int TrType);

private:
    /// input LDB
    const FALDB * m_pLDB;
    /// transformations objects, if any
    FATransform_hyph_redup_t < Ty > * m_pTr_hyph_redup;
    FATransform_hyph_redup_rev_t < Ty > * m_pTr_hyph_redup_rev;
    FATransform_prefix_t < Ty > * m_pTr_prefix;
    FATransform_prefix_rev_t < Ty > * m_pTr_prefix_rev;
    FATransform_capital_t < Ty > * m_pTr_capital;
    FATransform_capital_rev_t < Ty > * m_pTr_capital_rev;
    FATransform_cascade_t < Ty > * m_pInTrCascade;
    FATransform_cascade_t < Ty > * m_pOutTrCascade;
    /// const interface pointer to input and output transformations
    const FATransformCA_t < Ty > * m_pInTrA;
    const FATransformCA_t < Ty > * m_pOutTrA;
    /// automaton of prefixes for prefix transformation, if any
    FARSDfa_pack_triv * m_pPrefDfa;
    /// a global ignore case flag
    bool m_ignore_case;
};


template < class Ty >
FATrsConfKeeper_t< Ty >::FATrsConfKeeper_t () : 
    m_pLDB (NULL),
    m_pTr_hyph_redup (NULL),
    m_pTr_hyph_redup_rev (NULL),
    m_pTr_prefix (NULL),
    m_pTr_prefix_rev (NULL),
    m_pTr_capital (NULL),
    m_pTr_capital_rev (NULL),
    m_pInTrCascade (NULL),
    m_pOutTrCascade (NULL),
    m_pInTrA (NULL),
    m_pOutTrA (NULL),
    m_pPrefDfa (NULL),
    m_ignore_case (false)
{}


template < class Ty >
FATrsConfKeeper_t< Ty >::~FATrsConfKeeper_t ()
{
    FATrsConfKeeper_t< Ty >::Clear ();
}


template < class Ty >
void FATrsConfKeeper_t< Ty >::Clear ()
{
    m_ignore_case = false;

    m_pInTrA = NULL;
    m_pOutTrA = NULL;

    if (m_pPrefDfa) {
        delete m_pPrefDfa;
        m_pPrefDfa = NULL;
    }
    if (m_pTr_hyph_redup) {
        delete m_pTr_hyph_redup;
        m_pTr_hyph_redup = NULL;
    }
    if (m_pTr_hyph_redup_rev) {
        delete m_pTr_hyph_redup_rev;
        m_pTr_hyph_redup_rev = NULL;
    }
    if (m_pTr_prefix) {
        delete m_pTr_prefix;
        m_pTr_prefix = NULL;
    }
    if (m_pTr_prefix_rev) {
        delete m_pTr_prefix_rev;
        m_pTr_prefix_rev = NULL;
    }
    if (m_pTr_capital) {
        delete m_pTr_capital;
        m_pTr_capital = NULL;
    }
    if (m_pTr_capital_rev) {
        delete m_pTr_capital_rev;
        m_pTr_capital_rev = NULL;
    }
    if (m_pInTrCascade) {
        delete m_pInTrCascade;
        m_pInTrCascade = NULL;
    }
    if (m_pOutTrCascade) {
        delete m_pOutTrCascade;
        m_pOutTrCascade = NULL;
    }
}


template < class Ty >
void FATrsConfKeeper_t< Ty >::SetLDB (const FALDB * pLDB)
{
    m_pLDB = pLDB;
}


template < class Ty >
const FATransformCA_t < Ty > * FATrsConfKeeper_t< Ty >::
    SelectTr (const int TrType)
{
    const FATransformCA_t < Ty > * pSelectedTr = NULL;

    switch (TrType) {

    case FAFsmConst::TR_HYPH_REDUP:
    {
        if (!m_pTr_hyph_redup) {
            m_pTr_hyph_redup = new FATransform_hyph_redup_t < Ty >;
            LogAssert (m_pTr_hyph_redup);
        }
        pSelectedTr = m_pTr_hyph_redup;
        break;
    }
    case FAFsmConst::TR_HYPH_REDUP_REV:
    {
        if (!m_pTr_hyph_redup_rev) {
            m_pTr_hyph_redup_rev = new FATransform_hyph_redup_rev_t < Ty >;
            LogAssert (m_pTr_hyph_redup_rev);
        }
        pSelectedTr = m_pTr_hyph_redup_rev;
        break;
    }
    case FAFsmConst::TR_PREFIX:
    {
        if (!m_pTr_prefix) {
            m_pTr_prefix = new FATransform_prefix_t < Ty >;
            LogAssert (m_pTr_prefix);
        }
        pSelectedTr = m_pTr_prefix;
        break;
    }
    case FAFsmConst::TR_PREFIX_REV:
    {
        if (!m_pTr_prefix_rev) {
            m_pTr_prefix_rev = new FATransform_prefix_rev_t < Ty >;
            LogAssert (m_pTr_prefix_rev);
        }
        pSelectedTr = m_pTr_prefix_rev;
        break;
    }
    case FAFsmConst::TR_UCF:
    {
        if (!m_pTr_capital) {
            m_pTr_capital = new FATransform_capital_t < Ty >;
            LogAssert (m_pTr_capital);
        }
        pSelectedTr = m_pTr_capital;
        break;
    }
    case FAFsmConst::TR_UCF_REV:
    {
        if (!m_pTr_capital_rev) {
            m_pTr_capital_rev = new FATransform_capital_rev_t < Ty >;
            LogAssert (m_pTr_capital_rev);
        }
        pSelectedTr = m_pTr_capital_rev;
        break;
    }
    default:
        // unknown transformaion type
        LogAssert (false);

    } // of switch (TrType) ...

    return pSelectedTr;
}


template < class Ty >
void FATrsConfKeeper_t< Ty >::Init (const int * pValues, const int Size)
{
    LogAssert (m_pLDB && (pValues || 0 >= Size));

    int RedupDelim = -1;
    int PrefDelim = -1;
    int UcfDelim = -1;

    m_pInTrA = NULL;
    m_pOutTrA = NULL;

    m_ignore_case = false;

    for (int i = 0; i < Size; ++i) {

        const int Param = pValues [i];

        switch (Param) {

        case FAFsmConst::PARAM_IGNORE_CASE:
        {
            m_ignore_case = true;
            break;
        }
        case FAFsmConst::PARAM_REDUP_DELIM:
        {
            RedupDelim = pValues [++i];
            break;
        }
        case FAFsmConst::PARAM_PREF_DELIM:
        {
            PrefDelim = pValues [++i];
            break;
        }
        case FAFsmConst::PARAM_UCF_DELIM:
        {
            UcfDelim = pValues [++i];
            break;
        }
        case FAFsmConst::PARAM_PREF_FSM:
        {
            const int DumpNum = pValues [++i];
            const unsigned char * pDump = m_pLDB->GetDump (DumpNum);
            LogAssert (pDump);

            if (!m_pPrefDfa) {
                m_pPrefDfa = new FARSDfa_pack_triv;
                LogAssert (m_pPrefDfa);
            }
            m_pPrefDfa->SetImage (pDump);

            break;
        }
        case FAFsmConst::PARAM_IN_TR:
        {
            const int TrType = pValues [++i];

            if (!m_pInTrA) {

                m_pInTrA = SelectTr (TrType);
                DebugLogAssert (m_pInTrA);

            } else {

                // see whether super-position has not been created yet
                if (!m_pInTrCascade) {

                    m_pInTrCascade = new FATransform_cascade_t < Ty > ();
                    LogAssert (m_pInTrCascade);

                    m_pInTrCascade->AddTransformation (m_pInTrA);
                    m_pInTrA = m_pInTrCascade;
                }

                const FATransformCA_t < Ty > * pTr = SelectTr (TrType);
                DebugLogAssert (pTr);
                m_pInTrCascade->AddTransformation (pTr);

            } // of if (!m_pInTrA)
            break;
        }
        case FAFsmConst::PARAM_OUT_TR:
        {
            const int TrType = pValues [++i];

            if (!m_pOutTrA) {

                m_pOutTrA = SelectTr (TrType);
                DebugLogAssert (m_pOutTrA);

            } else {

                // see whether super-position has not been created yet
                if (!m_pOutTrCascade) {

                    m_pOutTrCascade = new FATransform_cascade_t < Ty > ();
                    LogAssert (m_pOutTrCascade);

                    m_pOutTrCascade->AddTransformation (m_pOutTrA);
                    m_pOutTrA = m_pOutTrCascade;
                }

                const FATransformCA_t < Ty > * pTr = SelectTr (TrType);
                DebugLogAssert (pTr);
                m_pOutTrCascade->AddTransformation (pTr);

            } // of if (!m_pOutTrA) ...
            break;
        }
        default:
            // unknown parameter
            LogAssert (false);
        } // of switch ...

    } // of for (int i = 0; ...

    if (-1 != RedupDelim) {
        if (m_pTr_hyph_redup) {
            m_pTr_hyph_redup->SetDelim (Ty (RedupDelim));
        }
        if (m_pTr_hyph_redup_rev) {
            m_pTr_hyph_redup_rev->SetDelim (Ty (RedupDelim));
        }
    }
    if (-1 != PrefDelim) {
        if (m_pTr_prefix) {
            m_pTr_prefix->SetDelim (Ty (PrefDelim));
        }
        if (m_pTr_prefix_rev) {
            m_pTr_prefix_rev->SetDelim (Ty (PrefDelim));
        }
    }
    if (-1 != UcfDelim) {
        if (m_pTr_capital) {
            m_pTr_capital->SetDelim (Ty (UcfDelim));
        }
        if (m_pTr_capital_rev) {
            m_pTr_capital_rev->SetDelim (Ty (UcfDelim));
        }
    }
    if (m_pPrefDfa){
        if (m_pTr_prefix) {
            m_pTr_prefix->SetRsDfa (m_pPrefDfa);
        }
    }
}


template < class Ty >
const FATransformCA_t < Ty > * FATrsConfKeeper_t< Ty >::GetInTr () const
{
    return m_pInTrA;
}


template < class Ty >
const FATransformCA_t < Ty > * FATrsConfKeeper_t< Ty >::GetOutTr () const
{
    return m_pOutTrA;
}


template < class Ty >
const bool FATrsConfKeeper_t< Ty >::GetIgnoreCase () const
{
    return m_ignore_case;
}

}

#endif
