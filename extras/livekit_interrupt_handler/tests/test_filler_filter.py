# test_filler_filter.py
import pytest
import asyncio
from filler_filter import FillerFilter, ASRResult

@pytest.mark.asyncio
async def test_suppress_filler_while_speaking():
    ff = FillerFilter(ignored_words=['uh','umm','hmm'])
    hit = {"suppress": 0}
    ff.on_suppress = lambda asr: hit.__setitem__('suppress', hit['suppress'] + 1)
    ff.set_agent_speaking(True)
    await ff.handle_asr(ASRResult(text='uh', confidence=0.9, is_final=True))
    assert hit['suppress'] == 1

@pytest.mark.asyncio
async def test_forward_stop_word():
    ff = FillerFilter()
    hit = {"forward": 0}
    ff.on_forward = lambda asr: hit.__setitem__('forward', hit['forward'] + 1)
    ff.set_agent_speaking(True)
    await ff.handle_asr(ASRResult(text='stop', confidence=0.95, is_final=True))
    assert hit['forward'] == 1

@pytest.mark.asyncio
async def test_register_when_quiet():
    ff = FillerFilter()
    hit = {"register": 0}
    ff.on_register = lambda asr: hit.__setitem__('register', hit['register'] + 1)
    ff.set_agent_speaking(False)
    await ff.handle_asr(ASRResult(text='umm', confidence=0.95, is_final=True))
    assert hit['register'] == 1
