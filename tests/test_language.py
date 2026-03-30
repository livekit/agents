from livekit.agents.language import LanguageCode
from livekit.agents.stt import SpeechData


class TestSpeechDataCoercion:
    def test_str_coerced_to_language(self):
        sd = SpeechData(language="en-US", text="hello")
        assert isinstance(sd.language, LanguageCode)
        assert sd.language == "en-US"

    def test_language_name_normalized(self):
        sd = SpeechData(language="english", text="hello")
        assert isinstance(sd.language, LanguageCode)
        assert sd.language == "en"

    def test_language_instance_passthrough(self):
        lang = LanguageCode("fr")
        sd = SpeechData(language=lang, text="bonjour")
        assert sd.language is lang

    def test_empty_string(self):
        sd = SpeechData(language="", text="")
        assert isinstance(sd.language, LanguageCode)
        assert sd.language == ""

    def test_properties_accessible(self):
        sd = SpeechData(language="en-US", text="hello")
        assert sd.language.language == "en"
        assert sd.language.region == "US"
        assert sd.language.to_language_name() == "english"


class TestLanguageNormalization:
    def test_language_name(self):
        assert LanguageCode("english") == "en"
        assert LanguageCode("french") == "fr"
        assert LanguageCode("German") == "de"
        assert LanguageCode("SPANISH") == "es"

    def test_iso639_3(self):
        assert LanguageCode("eng") == "en"
        assert LanguageCode("fra") == "fr"
        assert LanguageCode("deu") == "de"
        assert LanguageCode("zho") == "zh"

    def test_iso639_1_passthrough(self):
        assert LanguageCode("en") == "en"
        assert LanguageCode("fr") == "fr"
        assert LanguageCode("zh") == "zh"

    def test_bcp47_passthrough(self):
        assert LanguageCode("en-US") == "en-US"
        assert LanguageCode("zh-CN") == "zh-CN"
        assert LanguageCode("pt-BR") == "pt-BR"

    def test_bcp47_casing(self):
        assert LanguageCode("en-us") == "en-US"
        assert LanguageCode("EN-US") == "en-US"
        assert LanguageCode("zh-cn") == "zh-CN"

    def test_bcp47_underscore(self):
        assert LanguageCode("en_us") == "en-US"
        assert LanguageCode("pt_br") == "pt-BR"

    def test_bcp47_script(self):
        assert LanguageCode("zh-hans-cn") == "zh-Hans-CN"
        assert LanguageCode("zh-hant-tw") == "zh-Hant-TW"

    def test_bcp47_iso639_3_subtag_preserved(self):
        # ISO 639-3 subtags in compound BCP-47 are preserved for API round-tripping
        # (e.g. Google STT expects "cmn-Hans-CN", not "zh-Hans-CN")
        assert LanguageCode("cmn-Hans-CN") == "cmn-Hans-CN"
        assert LanguageCode("cmn-Hant-TW") == "cmn-Hant-TW"
        assert LanguageCode("cmn-cn") == "cmn-CN"
        # But .language resolves to ISO 639-1
        assert LanguageCode("cmn-Hans-CN").language == "zh"
        assert LanguageCode("cmn-Hans-CN").to_language_name() == "chinese"

    def test_bcp47_iso639_3_no_iso1_subtag(self):
        # ISO 639-3 codes with no ISO 639-1 equivalent stay as-is
        assert LanguageCode("yue-Hant-HK") == "yue-Hant-HK"
        assert LanguageCode("yue-Hant-HK").language == "yue"

    def test_unknown_passthrough(self):
        assert LanguageCode("multi") == "multi"
        assert LanguageCode("auto") == "auto"

    def test_iso639_3_no_iso1(self):
        # These ISO 639-3 codes have no ISO 639-1 equivalent
        assert LanguageCode("ast") == "ast"
        assert LanguageCode("ceb") == "ceb"
        assert LanguageCode("fil") == "fil"

    def test_whitespace_stripped(self):
        assert LanguageCode("  en  ") == "en"
        assert LanguageCode(" english ") == "en"


class TestLanguageProperties:
    def test_language_property(self):
        assert LanguageCode("en-US").language == "en"
        assert LanguageCode("en").language == "en"
        assert LanguageCode("zh-Hans-CN").language == "zh"
        assert LanguageCode("cmn-Hans-CN").language == "zh"
        assert LanguageCode("cmn").language == "zh"

    def test_region_property(self):
        assert LanguageCode("en-US").region == "US"
        assert LanguageCode("pt-BR").region == "BR"
        assert LanguageCode("zh-Hans-CN").region == "CN"
        assert LanguageCode("cmn-Hans-CN").region == "CN"
        assert LanguageCode("en").region is None
        assert LanguageCode("zh").region is None

    def test_iso_property(self):
        assert LanguageCode("en-US").iso == "en-US"
        assert LanguageCode("cmn-Hans-CN").iso == "zh-CN"
        assert LanguageCode("cmn-Hant-TW").iso == "zh-TW"
        assert LanguageCode("zh-Hans-CN").iso == "zh-CN"
        assert LanguageCode("en").iso == "en"
        assert LanguageCode("cmn").iso == "zh"

    def test_to_language_name(self):
        assert LanguageCode("en").to_language_name() == "english"
        assert LanguageCode("fr").to_language_name() == "french"
        assert LanguageCode("en-US").to_language_name() == "english"
        assert LanguageCode("cmn-Hans-CN").to_language_name() == "chinese"
        assert LanguageCode("multi").to_language_name() is None


class TestLanguageStrCompat:
    def test_is_str(self):
        lang = LanguageCode("en")
        assert isinstance(lang, str)
        assert lang == "en"

    def test_string_ops(self):
        lang = LanguageCode("en-US")
        assert lang.upper() == "EN-US"
        assert lang.startswith("en")
        assert "US" in lang

    def test_hashable(self):
        # Can be used as dict key / set member
        d = {LanguageCode("en"): "English"}
        assert d["en"] == "English"
        assert d[LanguageCode("english")] == "English"

    def test_in_set(self):
        s = {LanguageCode("en"), LanguageCode("fr")}
        assert "en" in s
        assert LanguageCode("english") in s
