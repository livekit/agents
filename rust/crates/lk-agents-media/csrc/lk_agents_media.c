/*
 * The whole ffmpeg surface, as a flat C API.
 *
 * WHY A SHIM RATHER THAN extern "C" ON ffmpeg ITSELF
 *
 * §10 asks for hand-written bindings rather than bindgen over all of ffmpeg, and it is right
 * about the size -- the surface really is one demux+decode loop and one encode+mux loop. But
 * hand-writing the *declarations* means hand-transcribing AVFrame, AVPacket, AVFormatContext,
 * AVStream and AVCodecParameters, and a wrong offset in any of those is a silent read of the
 * wrong memory, not a link error. AVFrame alone changed shape in 6.0 (ch_layout) and again in
 * 7.0.
 *
 * So the ffmpeg-facing side is C, where the real headers define the layouts, and the Rust side
 * declares only the eleven functions below -- which is smaller than the transcription would have
 * been and cannot go silently wrong. build.rs compiles this against the vendored tree's headers.
 *
 * Both objects call BACK into Rust for their bytes: the decoder pulls through `read_cb` (which is
 * `StreamBuffer::read`, blocking, which is what a demuxer needs) and the encoder pushes through
 * `write_cb`. Neither owns a buffer of its own.
 */
#include <stdint.h>
#include <string.h>

#include <libavcodec/avcodec.h>
#include <libavformat/avformat.h>
#include <libavutil/audio_fifo.h>
#include <libavutil/channel_layout.h>
#include <libavutil/opt.h>
#include <libswresample/swresample.h>

#define AVIO_BUF 4096

static void set_err(char *err, int errlen, const char *op, int rc) {
    if (!err || errlen <= 0) return;
    char detail[AV_ERROR_MAX_STRING_SIZE] = {0};
    if (av_strerror(rc, detail, sizeof(detail)) < 0)
        snprintf(detail, sizeof(detail), "error %d", rc);
    snprintf(err, (size_t)errlen, "%s: %s", op, detail);
}

static void set_msg(char *err, int errlen, const char *msg) {
    if (!err || errlen <= 0) return;
    snprintf(err, (size_t)errlen, "%s", msg);
}

/* ------------------------------------------------------------------ decode */

typedef int (*lk_read_fn)(void *opaque, uint8_t *buf, int size);
typedef void (*lk_pcm_fn)(void *opaque, const int16_t *pcm, int n_samples);

/* The trampoline's payload: the Rust callback plus its own opaque, owned by the decoder so the
 * AVIOContext's `opaque` cannot outlive it. */
struct LkReader {
    lk_read_fn read;
    void *opaque;
};

struct LkDecoder;
void lk_dec_free(struct LkDecoder *d);

struct LkDecoder {
    struct LkReader reader;
    AVFormatContext *fmt;
    AVIOContext *avio;
    AVCodecContext *dec;
    SwrContext *swr;
    AVPacket *pkt;
    AVFrame *frame;
    int stream_index;
    int out_rate;
    int out_channels;
    int src_rate;
    int src_channels;
    uint8_t *out_buf;
    int out_cap_samples; /* per channel */
};

/* AVIO read, straight through to Rust. Returning AVERROR_EOF rather than 0 at end of stream is
 * required from ffmpeg 6.0 on: a 0 is now "try again", and the demuxer spins on it forever. */
static int lk_reader_trampoline(void *opaque, uint8_t *buf, int size) {
    struct LkReader *r = (struct LkReader *)opaque;
    int n = r->read(r->opaque, buf, size);
    if (n <= 0) return AVERROR_EOF;
    return n;
}

struct LkDecoder *lk_dec_open(const char *format_short_name, /* NULL = probe */
                              int out_rate, int out_channels,
                              lk_read_fn read_cb, void *read_opaque,
                              char *err, int errlen) {
    int rc;
    uint8_t *buf;
    const AVInputFormat *ifmt = NULL;
    AVDictionary *opts = NULL;
    AVCodecParameters *par;
    const AVCodec *codec;

    struct LkDecoder *d = av_mallocz(sizeof(*d));
    if (!d) { set_msg(err, errlen, "out of memory"); return NULL; }
    d->stream_index = -1;
    d->out_rate = out_rate;
    d->out_channels = out_channels;
    d->reader.read = read_cb;
    d->reader.opaque = read_opaque;

    buf = av_malloc(AVIO_BUF);
    if (!buf) { av_free(d); set_msg(err, errlen, "out of memory"); return NULL; }
    d->avio = avio_alloc_context(buf, AVIO_BUF, 0, &d->reader, lk_reader_trampoline, NULL, NULL);
    if (!d->avio) { av_free(buf); av_free(d); set_msg(err, errlen, "avio_alloc_context failed"); return NULL; }

    d->fmt = avformat_alloc_context();
    if (!d->fmt) { set_msg(err, errlen, "avformat_alloc_context failed"); goto fail; }
    d->fmt->pb = d->avio;
    d->fmt->flags |= AVFMT_FLAG_CUSTOM_IO | AVFMT_FLAG_FLUSH_PACKETS;

    if (format_short_name && format_short_name[0]) {
        ifmt = av_find_input_format(format_short_name);
        if (!ifmt) { set_msg(err, errlen, "unknown container short name"); goto fail; }
    }

    /* The reference's low-latency dictionary, verbatim (decoder.py:_decode_loop). Without it the
     * demuxer buffers up to 5 s before it will emit the first packet, which is the difference
     * between a 200 ms and a 5 s time-to-first-audio. */
    av_dict_set(&opts, "probesize", "32", 0);
    av_dict_set(&opts, "analyzeduration", "0", 0);
    av_dict_set(&opts, "fflags", "flush_packets", 0);
    av_dict_set(&opts, "flags", "low_delay", 0);
    av_dict_set(&opts, "max_delay", "0", 0);
    av_dict_set(&opts, "avioflags", "direct", 0);

    rc = avformat_open_input(&d->fmt, NULL, ifmt, &opts);
    av_dict_free(&opts);
    if (rc < 0) { set_err(err, errlen, "avformat_open_input", rc); d->fmt = NULL; goto fail; }

    rc = avformat_find_stream_info(d->fmt, NULL);
    if (rc < 0) { set_err(err, errlen, "avformat_find_stream_info", rc); goto fail; }

    for (unsigned i = 0; i < d->fmt->nb_streams; i++) {
        if (d->fmt->streams[i]->codecpar->codec_type == AVMEDIA_TYPE_AUDIO) {
            d->stream_index = (int)i;
            break;
        }
    }
    if (d->stream_index < 0) { set_msg(err, errlen, "no audio stream found"); goto fail; }

    par = d->fmt->streams[d->stream_index]->codecpar;
    codec = avcodec_find_decoder(par->codec_id);
    if (!codec) { set_msg(err, errlen, "no decoder for this codec in this build"); goto fail; }

    d->dec = avcodec_alloc_context3(codec);
    if (!d->dec) { set_msg(err, errlen, "avcodec_alloc_context3 failed"); goto fail; }
    rc = avcodec_parameters_to_context(d->dec, par);
    if (rc < 0) { set_err(err, errlen, "avcodec_parameters_to_context", rc); goto fail; }
    d->dec->pkt_timebase = d->fmt->streams[d->stream_index]->time_base;
    rc = avcodec_open2(d->dec, codec, NULL);
    if (rc < 0) { set_err(err, errlen, "avcodec_open2", rc); goto fail; }

    d->src_rate = d->dec->sample_rate;
    d->src_channels = d->dec->ch_layout.nb_channels;

    d->pkt = av_packet_alloc();
    d->frame = av_frame_alloc();
    if (!d->pkt || !d->frame) { set_msg(err, errlen, "out of memory"); goto fail; }
    return d;

fail:
    lk_dec_free(d);
    return NULL;
}

int lk_dec_src_rate(const struct LkDecoder *d) { return d ? d->src_rate : 0; }
int lk_dec_src_channels(const struct LkDecoder *d) { return d ? d->src_channels : 0; }

/* av.AudioResampler(format="s16", layout=..., rate=...). Built lazily because the first decoded
 * frame is what tells us the real input format -- codecpar may say nothing before then. */
static int ensure_swr(struct LkDecoder *d, const AVFrame *f, char *err, int errlen) {
    if (d->swr) return 0;
    AVChannelLayout out_layout;
    av_channel_layout_default(&out_layout, d->out_channels);
    int rc = swr_alloc_set_opts2(&d->swr, &out_layout, AV_SAMPLE_FMT_S16, d->out_rate,
                                 &f->ch_layout, (enum AVSampleFormat)f->format,
                                 f->sample_rate ? f->sample_rate : d->src_rate, 0, NULL);
    av_channel_layout_uninit(&out_layout);
    if (rc < 0) { set_err(err, errlen, "swr_alloc_set_opts2", rc); return rc; }
    rc = swr_init(d->swr);
    if (rc < 0) { set_err(err, errlen, "swr_init", rc); return rc; }
    return 0;
}

static int emit(struct LkDecoder *d, const AVFrame *f, lk_pcm_fn on_pcm, void *opaque,
                char *err, int errlen) {
    int in_rate = f ? (f->sample_rate ? f->sample_rate : d->src_rate) : d->src_rate;
    int in_n = f ? f->nb_samples : 0;
    int64_t want = swr_get_delay(d->swr, in_rate) + in_n;
    int out_n = (int)av_rescale_rnd(want, d->out_rate, in_rate ? in_rate : d->out_rate, AV_ROUND_UP);
    if (out_n <= 0) return 0;

    if (out_n > d->out_cap_samples) {
        av_freep(&d->out_buf);
        int rc = av_samples_alloc(&d->out_buf, NULL, d->out_channels, out_n, AV_SAMPLE_FMT_S16, 0);
        if (rc < 0) { set_err(err, errlen, "av_samples_alloc", rc); d->out_cap_samples = 0; return rc; }
        d->out_cap_samples = out_n;
    }
    int got = swr_convert(d->swr, &d->out_buf, out_n,
                          f ? (const uint8_t **)f->extended_data : NULL, in_n);
    if (got < 0) { set_err(err, errlen, "swr_convert", got); return got; }
    if (got > 0) on_pcm(opaque, (const int16_t *)d->out_buf, got * d->out_channels);
    return 0;
}

/* Runs to end of input. Blocks inside read_cb, so the caller runs this on its own thread --
 * exactly the shape of the reference's `_decode_loop`. */
int lk_dec_run(struct LkDecoder *d, lk_pcm_fn on_pcm, void *opaque, char *err, int errlen) {
    int rc;
    for (;;) {
        rc = av_read_frame(d->fmt, d->pkt);
        if (rc == AVERROR_EOF) break;
        if (rc < 0) { set_err(err, errlen, "av_read_frame", rc); return rc; }
        if (d->pkt->stream_index != d->stream_index) { av_packet_unref(d->pkt); continue; }

        rc = avcodec_send_packet(d->dec, d->pkt);
        av_packet_unref(d->pkt);
        if (rc < 0 && rc != AVERROR(EAGAIN)) { set_err(err, errlen, "avcodec_send_packet", rc); return rc; }

        for (;;) {
            rc = avcodec_receive_frame(d->dec, d->frame);
            if (rc == AVERROR(EAGAIN) || rc == AVERROR_EOF) break;
            if (rc < 0) { set_err(err, errlen, "avcodec_receive_frame", rc); return rc; }
            if (ensure_swr(d, d->frame, err, errlen) < 0) { av_frame_unref(d->frame); return -1; }
            rc = emit(d, d->frame, on_pcm, opaque, err, errlen);
            av_frame_unref(d->frame);
            if (rc < 0) return rc;
        }
    }

    /* Drain the decoder, then the resampler. Skipping either clips the last frames of every
     * stream -- a few tens of milliseconds, which reads as the agent cutting itself off. */
    rc = avcodec_send_packet(d->dec, NULL);
    if (rc < 0 && rc != AVERROR_EOF) { set_err(err, errlen, "avcodec_send_packet(flush)", rc); return rc; }
    for (;;) {
        rc = avcodec_receive_frame(d->dec, d->frame);
        if (rc == AVERROR(EAGAIN) || rc == AVERROR_EOF) break;
        if (rc < 0) { set_err(err, errlen, "avcodec_receive_frame(flush)", rc); return rc; }
        if (ensure_swr(d, d->frame, err, errlen) < 0) { av_frame_unref(d->frame); return -1; }
        rc = emit(d, d->frame, on_pcm, opaque, err, errlen);
        av_frame_unref(d->frame);
        if (rc < 0) return rc;
    }
    if (d->swr) {
        rc = emit(d, NULL, on_pcm, opaque, err, errlen);
        if (rc < 0) return rc;
    }
    return 0;
}

void lk_dec_free(struct LkDecoder *d) {
    if (!d) return;
    if (d->frame) av_frame_free(&d->frame);
    if (d->pkt) av_packet_free(&d->pkt);
    if (d->swr) swr_free(&d->swr);
    if (d->dec) avcodec_free_context(&d->dec);
    if (d->fmt) {
        /* avformat_close_input frees the context but NOT a custom pb's buffer. */
        avformat_close_input(&d->fmt);
    }
    if (d->avio) {
        av_freep(&d->avio->buffer);
        avio_context_free(&d->avio);
    }
    if (d->out_buf) av_freep(&d->out_buf);
    av_free(d);
}

/* ------------------------------------------------------------------ encode */

typedef void (*lk_write_fn)(void *opaque, const uint8_t *bytes, int len);

struct LkWriter {
    lk_write_fn write;
    void *opaque;
};

struct LkEncoder;
void lk_enc_free(struct LkEncoder *e);

struct LkEncoder {
    struct LkWriter writer;
    AVFormatContext *oc;
    AVIOContext *avio;
    AVCodecContext *enc;
    AVStream *st;
    AVFrame *frame;
    AVPacket *pkt;
    int64_t pts;
    int channels;
};

static int lk_writer_trampoline(void *opaque, const uint8_t *buf, int size) {
    struct LkWriter *w = (struct LkWriter *)opaque;
    w->write(w->opaque, buf, size);
    return size;
}

struct LkEncoder *lk_enc_open(int rate, int channels, int bitrate,
                              lk_write_fn write_cb, void *write_opaque,
                              char *err, int errlen) {
    int rc;
    uint8_t *buf;
    const AVCodec *codec;

    struct LkEncoder *e = av_mallocz(sizeof(*e));
    if (!e) { set_msg(err, errlen, "out of memory"); return NULL; }
    e->channels = channels;
    e->writer.write = write_cb;
    e->writer.opaque = write_opaque;

    /* §10 pins --enable-libopus precisely so this lookup cannot fall back to ffmpeg's native
     * (experimental) opus encoder, which is why the reference's FFMPEG_STRICT_LEVEL hack (#5579)
     * does not port. If it is missing, the build is wrong -- say so rather than degrade. */
    codec = avcodec_find_encoder_by_name("libopus");
    if (!codec) { set_msg(err, errlen, "libopus encoder missing -- built without --enable-libopus"); goto fail; }

    e->enc = avcodec_alloc_context3(codec);
    if (!e->enc) { set_msg(err, errlen, "avcodec_alloc_context3 failed"); goto fail; }
    e->enc->sample_rate = rate;
    e->enc->sample_fmt = AV_SAMPLE_FMT_S16;
    e->enc->bit_rate = bitrate;
    av_channel_layout_default(&e->enc->ch_layout, channels);
    e->enc->time_base = (AVRational){1, rate};

    rc = avformat_alloc_output_context2(&e->oc, NULL, "ogg", NULL);
    if (rc < 0) { set_err(err, errlen, "avformat_alloc_output_context2", rc); goto fail; }
    /* The ogg muxer is AVFMT_GLOBALHEADER: the opus head goes in extradata, not in a packet. */
    if (e->oc->oformat->flags & AVFMT_GLOBALHEADER)
        e->enc->flags |= AV_CODEC_FLAG_GLOBAL_HEADER;

    rc = avcodec_open2(e->enc, codec, NULL);
    if (rc < 0) { set_err(err, errlen, "avcodec_open2(libopus)", rc); goto fail; }

    e->st = avformat_new_stream(e->oc, NULL);
    if (!e->st) { set_msg(err, errlen, "avformat_new_stream failed"); goto fail; }
    rc = avcodec_parameters_from_context(e->st->codecpar, e->enc);
    if (rc < 0) { set_err(err, errlen, "avcodec_parameters_from_context", rc); goto fail; }
    e->st->time_base = e->enc->time_base;

    buf = av_malloc(AVIO_BUF);
    if (!buf) { set_msg(err, errlen, "out of memory"); goto fail; }
    e->avio = avio_alloc_context(buf, AVIO_BUF, 1, &e->writer, NULL, lk_writer_trampoline, NULL);
    if (!e->avio) { av_free(buf); set_msg(err, errlen, "avio_alloc_context failed"); goto fail; }
    e->oc->pb = e->avio;
    e->oc->flags |= AVFMT_FLAG_CUSTOM_IO;

    rc = avformat_write_header(e->oc, NULL);
    if (rc < 0) { set_err(err, errlen, "avformat_write_header", rc); goto fail; }

    e->frame = av_frame_alloc();
    e->pkt = av_packet_alloc();
    if (!e->frame || !e->pkt) { set_msg(err, errlen, "out of memory"); goto fail; }
    e->frame->format = AV_SAMPLE_FMT_S16;
    e->frame->nb_samples = e->enc->frame_size;
    av_channel_layout_copy(&e->frame->ch_layout, &e->enc->ch_layout);
    e->frame->sample_rate = rate;
    rc = av_frame_get_buffer(e->frame, 0);
    if (rc < 0) { set_err(err, errlen, "av_frame_get_buffer", rc); goto fail; }
    return e;

fail:
    lk_enc_free(e);
    return NULL;
}

/* Samples per channel that libopus wants per packet -- 960 at 48 kHz (20 ms). The caller buffers
 * to this; doing it in Rust keeps one copy out of the path. */
int lk_enc_frame_size(const struct LkEncoder *e) { return e ? e->enc->frame_size : 0; }

static int drain(struct LkEncoder *e, char *err, int errlen) {
    for (;;) {
        int rc = avcodec_receive_packet(e->enc, e->pkt);
        if (rc == AVERROR(EAGAIN) || rc == AVERROR_EOF) return 0;
        if (rc < 0) { set_err(err, errlen, "avcodec_receive_packet", rc); return rc; }
        av_packet_rescale_ts(e->pkt, e->enc->time_base, e->st->time_base);
        e->pkt->stream_index = e->st->index;
        rc = av_interleaved_write_frame(e->oc, e->pkt);
        av_packet_unref(e->pkt);
        if (rc < 0) { set_err(err, errlen, "av_interleaved_write_frame", rc); return rc; }
        /* Push whatever the muxer handed AVIO out to the caller now. The ogg muxer still batches
         * a whole page (`page_duration`, 1 s by default, which is what PyAV produces too), but
         * without this the 4 KB AVIO buffer adds a second layer of batching on top -- and the
         * recorder's writer loop asks for bytes every 2.5 s, so it would see them a page late. */
        avio_flush(e->oc->pb);
    }
}

/* Exactly `lk_enc_frame_size` samples per channel. */
int lk_enc_push(struct LkEncoder *e, const int16_t *pcm, char *err, int errlen) {
    int rc = av_frame_make_writable(e->frame);
    if (rc < 0) { set_err(err, errlen, "av_frame_make_writable", rc); return rc; }
    memcpy(e->frame->data[0], pcm,
           (size_t)e->enc->frame_size * (size_t)e->channels * sizeof(int16_t));
    e->frame->pts = e->pts;
    e->pts += e->enc->frame_size;
    rc = avcodec_send_frame(e->enc, e->frame);
    if (rc < 0) { set_err(err, errlen, "avcodec_send_frame", rc); return rc; }
    return drain(e, err, errlen);
}

int lk_enc_finish(struct LkEncoder *e, char *err, int errlen) {
    int rc = avcodec_send_frame(e->enc, NULL);
    if (rc < 0 && rc != AVERROR_EOF) { set_err(err, errlen, "avcodec_send_frame(flush)", rc); return rc; }
    rc = drain(e, err, errlen);
    if (rc < 0) return rc;
    rc = av_write_trailer(e->oc);
    if (rc < 0) { set_err(err, errlen, "av_write_trailer", rc); return rc; }
    avio_flush(e->oc->pb);
    return 0;
}

void lk_enc_free(struct LkEncoder *e) {
    if (!e) return;
    if (e->frame) av_frame_free(&e->frame);
    if (e->pkt) av_packet_free(&e->pkt);
    if (e->enc) avcodec_free_context(&e->enc);
    if (e->oc) avformat_free_context(e->oc);
    if (e->avio) {
        av_freep(&e->avio->buffer);
        avio_context_free(&e->avio);
    }
    av_free(e);
}

/* The version this shim was written against, so a mismatched prebuilt fails loudly. */
unsigned lk_avcodec_version(void) { return avcodec_version(); }
unsigned lk_avformat_version(void) { return avformat_version(); }
