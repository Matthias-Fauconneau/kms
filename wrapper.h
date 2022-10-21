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
    /*struct AVPacket *in_pkt;
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
    void *hwaccel_priv_data;*/
	// ..
};
#define __SIZEOF_PTHREAD_MUTEX_T 40
typedef union { char __size[__SIZEOF_PTHREAD_MUTEX_T]; long int __align; } pthread_mutex_t;
#define __SIZEOF_PTHREAD_COND_T 48
typedef union { char __size[__SIZEOF_PTHREAD_COND_T]; __extension__ long long int __align; } pthread_cond_t;
struct FrameThreadContext {
    struct PerThreadContext *threads;     ///< The contexts for each thread.
    struct PerThreadContext *prev_thread; ///< The last thread submit_packet() was called on.

    unsigned    pthread_init_cnt;  ///< Number of successfully initialized mutexes/conditions
    pthread_mutex_t buffer_mutex;  ///< Mutex used to protect get/release_buffer().
    /**
     * This lock is used for ensuring threads run in serial when hwaccel
     * is used.
     */
    pthread_mutex_t hwaccel_mutex;
    pthread_mutex_t async_mutex;
    pthread_cond_t async_cond;
    int async_lock;

    int next_decoding;             ///< The next context to submit a packet to.
    int next_finished;             ///< The next context to return output from.

    int delaying;                  /**<
                                    * Set for the first N packets, where N is the number of threads.
                                    * While it is set, ff_thread_en/decode_frame won't return any results.
                                    */

    /* hwaccel state is temporarily stored here in order to transfer its ownership
     * to the next decoding thread without the need for extra synchronization */
    const struct AVHWAccel *stash_hwaccel;
    void            *stash_hwaccel_context;
    void            *stash_hwaccel_priv;
};
struct VAAPIDecodeContext {
    VAConfigID            va_config;
    VAContextID           va_context;
    /*struct AVHWDeviceContext*/void    *device;
    VADisplay *hwctx; // ..
    struct AVHWFramesContext    *frames;
    struct AVVAAPIFramesContext *hwfc;
    // ..
};