/*                                                                           *
 * Microsoft VCBU Internal C++ YACC Parser v1.00.4101, Sundeep Bhatia        *
 *                                                                           *
 * Modified by Sergei Olonichev, sergeio                                     *
 *                                                                           *
 * Note: Do not delete this file, it was not generated.                      *
 *                                                                           */

#ifndef _FA_REGEXPPARSER_MSYACC_H_
#define _FA_REGEXPPARSER_MSYACC_H_

#include "FASecurity.h"

#ifndef YYAPI_PACKAGE
# define YYAPI_TOKENNAME		yychar			//name used for return value of yylex	
# define YYAPI_TOKENTYPE		int				//type of the token
# define YYAPI_TOKENEME(t)		(t)				//the value of the token that the parser should see
# define YYAPI_TOKENNONE		-1				//the representation when there is no token
# define YYAPI_TOKENSTR(t)		(itoa(t))		//string representation of the token
# define YYAPI_VALUENAME		yylval			//the name of the value of the token
# define YYAPI_VALUETYPE		YYSTYPE			//the type of the value of the token (if null, then the value is derivable from the token itself)
# define YYAPI_VALUEOF(v)		(v)				//how to get the value of the token
# define YYAPI_CALLAFTERYYLEX(t)				//
# define YYAPI_PACKAGE							//package is in use
#endif	// YYAPI_PACKAGE


#ifndef YYMAXDEPTH
#define YYMAXDEPTH 7500
#endif
#ifndef YYFARDATA
#define	YYFARDATA
#endif
#if ! defined YYSTATIC
#define	YYSTATIC
#endif
#if ! defined YYCONST
#define	YYCONST
#endif
#ifndef	YYACT
#define	YYACT yyact
#endif
#ifndef	YYPACT
#define	YYPACT yypact
#endif
#ifndef	YYPGO
#define	YYPGO yypgo
#endif
#ifndef	YYR1
#define	YYR1 yyr1
#endif
#ifndef	YYR2
#define	YYR2 yyr2
#endif
#ifndef	YYCHK
#define	YYCHK yychk
#endif
#ifndef	YYDEF
#define	YYDEF yydef
#endif
#ifndef	YYV
#define	YYV yyv
#endif
#ifndef	YYS
#define	YYS yys
#endif
#ifndef	YYLOCAL
#define	YYLOCAL
#endif
#ifndef YYR_T
#define	YYR_T int
#endif
typedef	YYR_T yyr_t;
#ifndef YYEXIND_T
#define	YYEXIND_T unsigned int
#endif
typedef	YYEXIND_T yyexind_t;
#ifndef YYOPTTIME
#define	YYOPTTIME 0
#endif
#ifndef YYLEX
# define YYLEX yylex
#endif
//provided for compatibility with YACC
#define yyerrok ClearErrRecoveryState()
#define yyclearin YYAPI_TOKENNAME = YYAPI_TOKENNONE

namespace BlingFire
{


class FAToken;
class FARegexpTree;


class FAParserStateKeeper_msyacc {

public:

    FAParserStateKeeper_msyacc ()
    {
        ResetState ();
    }

protected:

    void ResetState ()
    {
        YYAPI_TOKENNAME = YYAPI_TOKENNONE;
        yynerrs = yyerrflag = yystate = 0;
// disable warning for array index out of range        
#pragma warning ( push )
#pragma warning ( disable : 6201 )
        yyps = &yys[-1];
        yypv = &yyv[-1];
#pragma warning ( pop )
#ifdef YYDEBUG
        yydebug = 0;
#endif
    }

protected:

    typedef union {
        int node_id;
        const FAToken* value;
    } YYSTYPE;

    // current input token
    YYAPI_TOKENTYPE YYAPI_TOKENNAME;
#ifdef YYAPI_VALUETYPE					
    YYAPI_VALUETYPE YYAPI_VALUENAME;
#endif
    YYSTYPE yyval;
    int yynerrs;
    int yyerrflag;

    short yystate, *yyps;
    YYSTYPE *yypv;
    YYSTYPE yyv [YYMAXDEPTH];
    short yys [YYMAXDEPTH];

#ifdef YYDEBUG
    int yydebug;
#endif

};




class FARegexpParser_msyacc : public FAParserStateKeeper_msyacc {

public: 

    FARegexpParser_msyacc ();
    ~FARegexpParser_msyacc ();

public:
    /* make processing */
    void Process ();
    /* sets up an input container  */
    void SetTokens (const FAToken * pTokens, const int TokensCount);
    /* sets up regular expression text */
    void SetRegexp (const char * pRegexp, const int RegexpLen);
    /* sets up an output container  */
    void SetTree (FARegexpTree * pTree);
    /* returns false, if the parsing was successful */
    const bool GetStatus () const;
    /* returns index of the last accepted token */
    const int GetTokenIdx () const;

private:

    int Parse ();
    int NoOfErrors () const;
    int ErrRecoveryState () const;
    void ClearErrRecoveryState ();
    YYAPI_TOKENTYPE GetCurrentToken () const;
    void SetCurrentToken (YYAPI_TOKENTYPE newToken);
    int yylex ();
    void Clear ();

#ifdef YYDUMP
    void DumpYYS ();
    void DumpYYV ();
#endif
#ifdef YYDEBUG
    const char* yytokenstr (int token);
    void Trace (const char *message, const char *tokname, short state = 0);
    void Trace (const char *message, int state, short tostate = 0, short token = 0);
#endif
    void yyerror (const char *szMsg);

private:
    /* output tree */
    FARegexpTree * m_pTree;
    /* input tokens */
    const FAToken * m_pTokens;
    /* the number of tokens */
    int m_token_count;
    /* currently processing token */
    int  m_token_pos;
    /* true, if something went wrong  */
    bool m_parse_failed;
    /* regexp text and length */
    const char * m_pRegexp;
    int m_RegexpLen;

    static YYCONST short yyexca [];
#if YYOPTTIME
    static YYCONST yyexind_t yyexcaind [];
#endif
    static YYCONST short yyact [];
    static YYCONST short yypact [];
    static YYCONST short yypgo [];
    static YYCONST yyr_t yyr1 [];
    static YYCONST yyr_t yyr2 [];
    static YYCONST short yychk [];
    static YYCONST short yydef [];
#ifdef YYRECOVER
    static YYCONST short yyrecover [];
#endif

};

}

#endif
