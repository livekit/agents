// Offline reference only. Links the pinned libwebrtc distribution; no product dependency.
#include <algorithm>
#include <cstdint>
#include <iterator>
#include <fstream>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

#include "api/audio/audio_frame.h"
#include "api/audio_codecs/L16/audio_decoder_L16.h"
#include "api/audio_codecs/audio_decoder_factory_template.h"
#include "api/environment/environment_factory.h"
#include "api/neteq/default_neteq_factory.h"
#include "api/neteq/neteq.h"
#include "api/rtp_headers.h"
#include "api/rtp_packet_info.h"
#include "system_wrappers/include/clock.h"

struct Chunk { int64_t arrival_us; size_t start; size_t count; };

int main(int argc, char** argv) {
  if (argc != 6) {
    std::cerr << "neteq_reference input48.pcm arrivals.csv output48.pcm duration_ms minimum_delay_ms\n";
    return 2;
  }
  std::ifstream pcm_file(argv[1], std::ios::binary);
  std::ifstream schedule(argv[2]);
  std::ofstream output(argv[3], std::ios::binary);
  if (!pcm_file || !schedule || !output) return 3;
  std::vector<uint8_t> pcm((std::istreambuf_iterator<char>(pcm_file)), {});
  std::vector<Chunk> chunks;
  std::string line;
  std::getline(schedule, line);
  while (std::getline(schedule, line)) {
    for (auto& ch : line) if (ch == ',') ch = ' ';
    std::istringstream row(line);
    Chunk chunk;
    if (!(row >> chunk.arrival_us >> chunk.start >> chunk.count)) return 4;
    // Schedule positions are mono 24 kHz source samples. The reference uses 48 kHz L16.
    chunk.start *= 2;
    chunk.count *= 2;
    if ((chunk.start + chunk.count) * 2 > pcm.size()) return 5;
    chunks.push_back(chunk);
  }
  webrtc::SimulatedClock clock(0);
  auto env = webrtc::CreateEnvironment(&clock);
  webrtc::NetEq::Config config;
  config.sample_rate_hz = 48000;
  config.min_delay_ms = std::stoi(argv[5]);
  auto neteq = webrtc::DefaultNetEqFactory().Create(
      env, config, webrtc::CreateAudioDecoderFactory<webrtc::AudioDecoderL16>());
  if (!neteq->RegisterPayloadType(93, webrtc::SdpAudioFormat("L16", 48000, 1))) return 6;
  size_t next = 0;
  uint16_t sequence = 0;
  int64_t now_us = 0;
  for (int64_t tick = 0; tick < std::stoll(argv[4]) * 1000; tick += 10000) {
    while (next < chunks.size() && chunks[next].arrival_us <= tick) {
      const auto& chunk = chunks[next++];
      if (chunk.arrival_us < now_us) return 7;
      clock.AdvanceTimeMicroseconds(chunk.arrival_us - now_us);
      now_us = chunk.arrival_us;
      // L16 packets of <=20 ms all retain the enclosing WebSocket chunk's arrival timestamp.
      for (size_t offset = 0; offset < chunk.count; offset += 960) {
        size_t count = std::min(size_t(960), chunk.count - offset);
        std::vector<uint8_t> payload(count * 2);
        for (size_t i = 0; i < count; ++i) {
          size_t at = (chunk.start + offset + i) * 2;
          payload[2 * i] = pcm[at + 1];
          payload[2 * i + 1] = pcm[at];
        }
        webrtc::RTPHeader header;
        header.payloadType = 93;
        header.sequenceNumber = sequence++;
        header.timestamp = static_cast<uint32_t>(chunk.start + offset);
        header.ssrc = 1;
        if (neteq->InsertPacket(header, payload,
            webrtc::RtpPacketInfo(header, webrtc::Timestamp::Micros(now_us))) != 0) return 8;
      }
    }
    clock.AdvanceTimeMicroseconds(tick - now_us);
    now_us = tick;
    webrtc::AudioFrame frame;
    if (neteq->GetAudio(&frame) != 0 || frame.samples_per_channel_ != 480) return 9;
    for (size_t i = 0; i < 480; ++i) {
      auto sample = static_cast<uint16_t>(frame.data()[i]);
      output.put(static_cast<char>(sample & 255));
      output.put(static_cast<char>(sample >> 8));
    }
  }
  auto stats = neteq->GetLifetimeStatistics();
  std::cout << "concealed_samples=" << stats.concealed_samples
            << " accelerate_samples=" << stats.removed_samples_for_acceleration
            << " expand_samples=" << stats.inserted_samples_for_deceleration << '\n';
  return output ? 0 : 10;
}
