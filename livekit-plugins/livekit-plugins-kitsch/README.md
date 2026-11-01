# Kitsch plugin for LiveKit Agents

Kitsch의 streaming TTS와 Speech to Text를 LiveKit Agents에서 사용한다. Kitsch workspace API key가 필요하며 사용량은 Kitsch 계정에 청구한다.

## 설치와 인증

```bash
uv add "livekit-agents[kitsch]"
```

`KITSCH_API_KEY`를 환경변수로 설정하거나 constructor의 `api_key`로 전달한다. 음성은 인증된 `GET https://api.kitschlabs.com/v1/voices`에서 접근 가능한 voice_id를 선택한다.

```python
from livekit.agents import AgentSession
from livekit.plugins import kitsch

session = AgentSession(
    stt=kitsch.STT(language="ko"),
    tts=kitsch.TTS(voice_id="minjun", language_code="ko"),
    # llm=...,
    # vad=...,
)
```

## TTS

`synthesize(text)`는 HTTP PCM stream을 사용하고 `stream()`은 LLM text 조각을 WebSocket으로 전송한다. 출력은 24 kHz mono PCM16 AudioFrame이다.

한 stream은 한 발화다. `flush()` 또는 `end_input()`으로 마무리하고 다음 발화는 새 stream을 사용한다. WebSocket의 final을 확인한 뒤 정상 완료한다.

## STT

`recognize()`는 mono AudioFrame 또는 frame 목록을 WAV로 전송한다. `stream()`은 입력을 24 kHz mono로 resample하고 partial 미리보기와 final transcript를 제공한다. `INTERIM_TRANSCRIPT`는 이전 미리보기를 교체하며 `FINAL_TRANSCRIPT`가 최종문이다. provisional text를 이어 붙이거나 별도 delta와 합산하지 않는다.

server VAD를 기본으로 사용한다. manual `flush()`는 해당 commit의 final을 기다리고 `end_input()`은 마지막 final을 수신한 뒤 연결을 닫는다. 정확한 음성 경계 timestamp는 제공하지 않으며 빠른 interruption에는 별도 VAD를 사용한다. 서비스의 세션 제한에 도달하면 연결이 종료될 수 있다.

## 오류와 정리

유료 요청은 자동 retry/reconnect/replay하지 않는다. timeout·interrupt 후에도 이미 실행된 작업은 청구될 수 있다. engine 전에 stream을 먼저 `aclose()`하거나 `async with`를 사용한다. 기본 요청·final 대기는 120초이며 `APIConnectOptions(timeout=...)`로 변경한다.

`base_url`은 `/v1` 없는 HTTPS origin이다. HTTP는 loopback 테스트만 허용하고 redirect는 거절한다. 플러그인 자체는 key·원문·audio를 로그에 남기지 않으며 LiveKit recording/telemetry는 애플리케이션이 관리한다.
