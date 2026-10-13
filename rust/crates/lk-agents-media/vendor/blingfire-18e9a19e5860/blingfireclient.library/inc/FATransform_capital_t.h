/**
 * Copyright (c) Microsoft Corporation. All rights reserved.
 * Licensed under the MIT License.
 */


#ifndef _FA_TRANSFORM_CAPITAL_T_H_
#define _FA_TRANSFORM_CAPITAL_T_H_

#include "FAConfig.h"
#include "FATransformCA_t.h"
#include "FAUtf32Utils.h"
#include "FASecurity.h"

namespace BlingFire
{

///
/// This is transformation detects if the first letter is capital, then it
/// makes it lower case and adds a special marker at the end.
///
/// Works as follows:
/// "Apfel" -> "apfel\x02"
/// "APFEL" -> "aPFEL\x02"
/// "apfel" -> -1
/// where 0x02 is a selected delimiter
///
/// Note:
/// 1. If the transformation is not applicable then Process method return -1.
/// 2. if pIn == pOut then transformation is made in-place.
/// 3. Language independent but works only with Latin-1 and Unicode.
///

template < class Ty >
class FATransform_capital_t : public FATransformCA_t < Ty > {

public:
    FATransform_capital_t ();

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

        if (0 == InCount) {

            return -1;
        }

        // check whether input is capitalized
        const int InSym = *pIn;
        const int OutSym = FAUtf32ToLower (InSym);

        if (InSym == OutSym) {

            return -1;

        } else {

            // check whether there is enough of a buffer
            if (MaxOutSize >= InCount + 1) {

                // write down the first symbol
                pOut [0] = (Ty) OutSym;
                // copy the rest if needed
                if (pIn != pOut) {
                    for (int i = 1; i < InCount; ++i) {
                        pOut [i] = pIn [i];
                    }
                }
                // write down the delimiter
                pOut [InCount] = m_Delim;
            }

            return InCount + 1;

        } // of if (InSym == OutSym) ...
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
FATransform_capital_t< Ty >::
    FATransform_capital_t () :
        m_Delim (DefDelim)
{
}


template < class Ty >
void FATransform_capital_t< Ty >::
    SetDelim (const Ty Delim)
{
    m_Delim = Delim;
}

}

#endif
