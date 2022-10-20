#include <va/va.h>
#include <va/va_vpp.h>
#include <va/va_drm.h>
#include <va/va_drmcommon.h>
struct AVCodecInternal {
    int is_copy;
    int last_audio_frame;
    int pad_samples;
    struct AVBufferRef *pool;
    void *thread_ctx;
    struct AVPacket *in_pkt;
    struct AVBSFContext *bsf;
    struct AVPacket *last_pkt_props;
    struct AVFifo *pkt_props;
    uint8_t *byte_buffer;
    unsigned int byte_buffer_size;
    int intra_only_flag;
    void *frame_thread_encoder;
    struct AVFrame *in_frame;
    struct AVFrame *recon_frame;
    int needs_close;
    int skip_samples;
    void *hwaccel_priv_data;
	// ..
};
struct VAAPIDecodeContext {
    VAConfigID            va_config;
    VAContextID           va_context;
    struct AVHWDeviceContext    *device;
    /*struct AVVAAPIDeviceContext{*/VADisplay/*..}*/ *hwctx;
    struct AVHWFramesContext    *frames;
    struct AVVAAPIFramesContext *hwfc;
    /*enum AVPixelFormat    surface_format;
    int                   surface_count;
    VASurfaceAttrib       pixel_format_attribute;*/
};