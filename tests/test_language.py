from livekit.agents.language import Language
from livekit.agents.stt import SpeechData


class TestSpeechDataCoercion:
    def test_str_coerced_to_language(self):
        sd = SpeechData(language="en-US", text="hello")
        assert isinstance(sd.language, Language)
        assert sd.language == "en-US"

    def test_language_name_normalized(self):
        sd = SpeechData(language="english", text="hello")
        assert isinstance(sd.language, Language)
        assert sd.language == "en"

    def test_language_instance_passthrough(self):
        lang = Language("fr")
        sd = SpeechData(language=lang, text="bonjour")
        assert sd.language is lang

    def test_empty_string(self):
        sd = SpeechData(language="", text="")
        assert isinstance(sd.language, Language)
        assert sd.language == ""

    def test_properties_accessible(self):
        sd = SpeechData(language="en-US", text="hello")
        assert sd.language.language == "en"
        assert sd.language.region == "US"
        assert sd.language.to_language_name() == "english"


class TestLanguageNormalization:
    def test_language_name(self):
        assert Language("english") == "en"
        assert Language("french") == "fr"
        assert Language("German") == "de"
        assert Language("SPANISH") == "es"

    def test_iso639_3(self):
        assert Language("eng") == "en"
        assert Language("fra") == "fr"
        assert Language("deu") == "de"
        assert Language("zho") == "zh"

    def test_iso639_1_passthrough(self):
        assert Language("en") == "en"
        assert Language("fr") == "fr"
        assert Language("zh") == "zh"

    def test_bcp47_passthrough(self):
        assert Language("en-US") == "en-US"
        assert Language("zh-CN") == "zh-CN"
        assert Language("pt-BR") == "pt-BR"

    def test_bcp47_casing(self):
        assert Language("en-us") == "en-US"
        assert Language("EN-US") == "en-US"
        assert Language("zh-cn") == "zh-CN"

    def test_bcp47_underscore(self):
        assert Language("en_us") == "en-US"
        assert Language("pt_br") == "pt-BR"

    def test_bcp47_script(self):
        assert Language("zh-hans-cn") == "zh-Hans-CN"
        assert Language("zh-hant-tw") == "zh-Hant-TW"

    def test_bcp47_iso639_3_subtag_preserved(self):
        # ISO 639-3 subtags in compound BCP-47 are preserved for API round-tripping
        # (e.g. Google STT expects "cmn-Hans-CN", not "zh-Hans-CN")
        assert Language("cmn-Hans-CN") == "cmn-Hans-CN"
        assert Language("cmn-Hant-TW") == "cmn-Hant-TW"
        assert Language("cmn-cn") == "cmn-CN"
        # But .language resolves to ISO 639-1
        assert Language("cmn-Hans-CN").language == "zh"
        assert Language("cmn-Hans-CN").to_language_name() == "chinese"

    def test_bcp47_iso639_3_no_iso1_subtag(self):
        # ISO 639-3 codes with no ISO 639-1 equivalent stay as-is
        assert Language("yue-Hant-HK") == "yue-Hant-HK"
        assert Language("yue-Hant-HK").language == "yue"

    def test_unknown_passthrough(self):
        assert Language("multi") == "multi"
        assert Language("auto") == "auto"

    def test_iso639_3_no_iso1(self):
        # These ISO 639-3 codes have no ISO 639-1 equivalent
        assert Language("ast") == "ast"
        assert Language("ceb") == "ceb"
        assert Language("fil") == "fil"

    def test_whitespace_stripped(self):
        assert Language("  en  ") == "en"
        assert Language(" english ") == "en"


class TestLanguageProperties:
    def test_language_property(self):
        assert Language("en-US").language == "en"
        assert Language("en").language == "en"
        assert Language("zh-Hans-CN").language == "zh"
        assert Language("cmn-Hans-CN").language == "zh"
        assert Language("cmn").language == "zh"

    def test_region_property(self):
        assert Language("en-US").region == "US"
        assert Language("pt-BR").region == "BR"
        assert Language("zh-Hans-CN").region == "CN"
        assert Language("cmn-Hans-CN").region == "CN"
        assert Language("en").region is None
        assert Language("zh").region is None

    def test_iso_property(self):
        assert Language("en-US").iso == "en-US"
        assert Language("cmn-Hans-CN").iso == "zh-CN"
        assert Language("cmn-Hant-TW").iso == "zh-TW"
        assert Language("zh-Hans-CN").iso == "zh-CN"
        assert Language("en").iso == "en"
        assert Language("cmn").iso == "zh"

    def test_to_language_name(self):
        assert Language("en").to_language_name() == "english"
        assert Language("fr").to_language_name() == "french"
        assert Language("en-US").to_language_name() == "english"
        assert Language("cmn-Hans-CN").to_language_name() == "chinese"
        assert Language("multi").to_language_name() is None


class TestLanguageStrCompat:
    def test_is_str(self):
        lang = Language("en")
        assert isinstance(lang, str)
        assert lang == "en"

    def test_string_ops(self):
        lang = Language("en-US")
        assert lang.upper() == "EN-US"
        assert lang.startswith("en")
        assert "US" in lang

    def test_hashable(self):
        # Can be used as dict key / set member
        d = {Language("en"): "English"}
        assert d["en"] == "English"
        assert d[Language("english")] == "English"

    def test_in_set(self):
        s = {Language("en"), Language("fr")}
        assert "en" in s
        assert Language("english") in s
