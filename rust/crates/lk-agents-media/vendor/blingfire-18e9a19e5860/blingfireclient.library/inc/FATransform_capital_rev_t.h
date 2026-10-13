/**
 * Copyright (c) Microsoft Corporation. All rights reserved.
 * Licensed under the MIT License.
 */


#ifndef _FA_TRANSFORM_CAPITAL_REV_T_H_
#define _FA_TRANSFORM_CAPITAL_REV_T_H_

#include "FAConfig.h"
#include "FATransformCA_t.h"
#include "FAUtf32Utils.h"
#include "FASecurity.h"

namespace BlingFire
{

///
/// This is a reverse Upper-Case-First (UCF) transformation.
///
/// Works as follows:
/// "apfel\x02" -> "Apfel"
/// "aPFEL\x02" -> "APFEL"
/// "Apfel" -> -1
/// where 0x02 is a selected delimiter
///
/// Note:
/// 1. If the transformation is not applicable then Process method return -1.
/// 2. if pIn == pOut then transformation is made in-place.
/// 3. Language independent but works only with Latin-1 and Unicode.
///

template < class Ty >
class FATransform_capital_rev_t : public FATransformCA_t < Ty > {

public:
    FATransform_capital_rev_t ();

public:
    /// sets up delimiter value
    void SetDelim (const Ty Delim);

    /// makes transformation
    /// Note: this function has to be *within* the class definition, because otherwise compiler makes an error
    ///   and generates some warnings about "unreferenced local function has been removed".
    const int Process (
            const Ty * pIn,
            const int InCount,
            __out_ecount(MaxOutSize) Ty * pOut,
            const int MaxOutSize
        ) const
    {
        // as __in_range(0, FALimits::MaxWordSize) const int InCount does not work
        __analysis_assume (0 <= InCount && FALimits::MaxWordSize >= InCount && pIn);
        DebugLogAssert (0 <= InCount && FALimits::MaxWordSize >= InCount && pIn);

        if (1 >= InCount) {
            return -1;
        }

        // check whether reversed UCF is applicable
        if (m_Delim != pIn [InCount - 1]) {
            return -1;
        }

        if (InCount - 1 <= MaxOutSize) {

            const int InSym = *pIn;
            const int OutSym = FAUtf32ToUpper (InSym);
            DebugLogAssert (InSym != OutSym);

            // write down the first symbol
            pOut [0] = (Ty) OutSym;

            // copy the rest if needed
            if (pIn != pOut) {
                memcpy (pOut + 1, pIn + 1, sizeof (Ty) * (InCount - 2));
            }
        }

        return InCount - 1;
    }

private:
    // delimiter
    Ty m_Delim;
    // constants
    enum { 
        DefDelim = 2,
    };
};


template < class Ty >
FATransform_capital_rev_t< Ty >::
    FATransform_capital_rev_t () :
        m_Delim (DefDelim)
{
}


template < class Ty >
void FATransform_capital_rev_t< Ty >::
    SetDelim (const Ty Delim)
{
    m_Delim = Delim;
}

}

#endif
