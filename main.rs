#![feature(core_intrinsics)]
pub fn main() -> Result<(), Box<dyn std::error::Error>> {
    use drm::{Device, control::{self, Device as _, *}, buffer::DrmFourcc};
    struct Card(std::fs::File);
    impl std::os::unix::io::AsRawFd for Card { fn as_raw_fd(&self) -> std::os::unix::io::RawFd { self.0.as_raw_fd() } }
    impl Device for Card {}
    impl control::Device for Card {}
    let card = Card(std::fs::OpenOptions::new().read(true).write(true).open("/dev/dri/card0").unwrap());
    card.set_client_capability(drm::ClientCapability::UniversalPlanes, true).unwrap();
    card.set_client_capability(drm::ClientCapability::Atomic, true).unwrap();
    let res = card.resource_handles().expect("Could not load normal resource ids.");
    let coninfo: Vec<connector::Info> = res.connectors().iter().flat_map(|con| card.get_connector(*con, true)).collect();
    let crtcinfo: Vec<crtc::Info> = res.crtcs().iter().flat_map(|crtc| card.get_crtc(*crtc)).collect();
    let con = coninfo.iter().find(|&i| i.state() == connector::State::Connected).unwrap();
    let crtc = crtcinfo[0];
    let &mode = con.modes().get(0).expect("No modes found on connector");
    let (width, height) = mode.size();
    let (width, height) = (width.into(), height.into());
    let fmt = DrmFourcc::Xrgb2101010;
    let /*mut*/ db = card.create_dumb_buffer((width, height), fmt, 32).unwrap();
    let fb = card.add_framebuffer(&db, None, 0).unwrap();
    let planes = card.plane_handles().unwrap();
    let (better_planes, compatible_planes): (Vec<control::plane::Handle>, Vec<control::plane::Handle>) = planes.iter()
        .filter(|&&plane| card.get_plane(plane).map(|plane_info| { let compatible_crtcs = res.filter_crtcs(plane_info.possible_crtcs()); compatible_crtcs.contains(&crtc.handle()) }).unwrap_or(false))
        .partition(|&&plane| {
            if let Ok(props) = card.get_properties(plane) {
                let (ids, vals) = props.as_props_and_values();
                for (&id, &val) in ids.iter().zip(vals.iter()) {
                    if let Ok(info) = card.get_property(id) { if info.name().to_str().map(|x| x == "type").unwrap_or(false) { return val == (drm::control::PlaneType::Primary as u32).into(); } }
                }
            }
            false
        });
    let plane = *better_planes.get(0).unwrap_or(&compatible_planes[0]);
    let mut atomic_req = atomic::AtomicModeReq::new();
    fn find_prop_id<T: ResourceHandle>(card: &Card, handle: T, name: &'static str) -> Option<property::Handle> {
        let props = card.get_properties(handle).unwrap();
        let (ids, _vals) = props.as_props_and_values();
        ids.iter().find(|&id| { let info = card.get_property(*id).unwrap(); info.name().to_str().map(|x| x == name).unwrap_or(false) }).cloned()
    }
    atomic_req.add_property(con.handle(), find_prop_id(&card, con.handle(), "CRTC_ID").unwrap(), property::Value::CRTC(Some(crtc.handle())));
    let blob = card.create_property_blob(&mode).unwrap();
    atomic_req.add_property(crtc.handle(), find_prop_id(&card, crtc.handle(), "MODE_ID").unwrap(), blob);
    atomic_req.add_property(crtc.handle(), find_prop_id(&card, crtc.handle(), "ACTIVE").unwrap(), property::Value::Boolean(true));
    atomic_req.add_property(plane, find_prop_id(&card, plane, "FB_ID").unwrap(), property::Value::Framebuffer(Some(fb)));
    atomic_req.add_property(plane, find_prop_id(&card, plane, "CRTC_ID").expect("Could not get CRTC_ID"), property::Value::CRTC(Some(crtc.handle())));
    atomic_req.add_property(plane, find_prop_id(&card, plane, "SRC_X").expect("Could not get SRC_X"), property::Value::UnsignedRange(0));
    atomic_req.add_property(plane, find_prop_id(&card, plane, "SRC_Y").expect("Could not get SRC_Y"), property::Value::UnsignedRange(0));
    atomic_req.add_property(plane, find_prop_id(&card, plane, "SRC_W").expect("Could not get SRC_W"), property::Value::UnsignedRange((mode.size().0 as u64) << 16));
    atomic_req.add_property(plane, find_prop_id(&card, plane, "SRC_H").expect("Could not get SRC_H"), property::Value::UnsignedRange((mode.size().1 as u64) << 16));
    atomic_req.add_property(plane, find_prop_id(&card, plane, "CRTC_X").expect("Could not get CRTC_X"), property::Value::SignedRange(0));
    atomic_req.add_property(plane, find_prop_id(&card, plane, "CRTC_Y").expect("Could not get CRTC_Y"), property::Value::SignedRange(0));
    atomic_req.add_property(plane, find_prop_id(&card, plane, "CRTC_W").expect("Could not get CRTC_W"), property::Value::UnsignedRange(mode.size().0 as u64));
    atomic_req.add_property(plane, find_prop_id(&card, plane, "CRTC_H").expect("Could not get CRTC_H"), property::Value::UnsignedRange(mode.size().1 as u64));

    #[repr(C)] struct xy { x: u16, y: u16 }
    fn xy(x: f64, y: f64) -> xy { xy{x: f64::round(x/0.00002) as u16, y: f64::round(y/0.00002) as u16} }
    const PQ : u8 = 2;
    #[repr(C)]
    struct hdr_output_metadata { // Bindgen fails to parse this
        metadata_type: u32,
        // Infoframe
        eot: u8,
        static_metadata_descriptor_id: u8,
        display_primaries: [xy; 3],
        white_point: xy,
        max_display_mastering_luminance: u16,
        min_display_mastering_luminance: u16,
        max_content_light_level: u16,
        max_frame_average_light_level: u16
    }
    let hdr_output_metadata = card.create_property_blob(&hdr_output_metadata{
        metadata_type: 0, // HDMI_STATIC_METADATA_TYPE1
        eot: PQ,
        static_metadata_descriptor_id: 0,
        //display_primaries: [xy(0.6835, 0.3154), xy(0.1962, 0.7333), xy(0.1416, 0.0449)],
        display_primaries: [xy(0.68, 0.32), xy(0.265, 0.69), xy(0.15, 0.06)],
        //white_point: xy(0.3105, 0.3232),
        white_point: xy(0.3127, 0.3290),
        //min_display_mastering_luminance: 1250, //0.125 / 0.0001,
        min_display_mastering_luminance: 1, //0.100 / 0.0001,
        //max_display_mastering_luminance: 508,
        max_display_mastering_luminance: 1000,
        //max_content_light_level: 246,
        max_content_light_level: 192,
        //max_frame_average_light_level: 92
        max_frame_average_light_level: 156,
    }).unwrap();
    atomic_req.add_property(con.handle(), find_prop_id(&card, con.handle(), "HDR_OUTPUT_METADATA").expect("Could not get HDR_OUTPUT_METADATA"), hdr_output_metadata);
    //atomic_req.add_property(con.handle(), find_prop_id(&card, con.handle(), "max bpc").expect("Could not get max bpc"), property::Value::UnsignedRange(10));
    let id = find_prop_id(&card, con.handle(), "Colorspace").expect("Could not get Colorspace");
    let property = card.get_property(id).unwrap();
    let control::property::ValueType::Enum(enum_values) = property.value_type() else {panic!()};
    let (_,enum_values) = enum_values.values();
    atomic_req.add_property(con.handle(), id, property::Value::Enum(Some( enum_values.iter().find(|enum_value| enum_value.name().to_bytes() == b"BT2020_RGB").unwrap())));
    //atomic_req.add_property(con.handle(), id, property::Value::Enum(Some( enum_values.iter().find(|enum_value| enum_value.name().to_bytes() == b"Default").unwrap())));
    if card.atomic_commit(AtomicCommitFlags::ALLOW_MODESET, atomic_req).is_ok() { println!("Mode set!"); } else { println!("No mode set"); }

    use ffmpeg::{*, ffi::*, util::{frame::video::Video}};
    ffmpeg::init().unwrap();
    let path = std::env::args().skip(1).next().unwrap_or(std::env::var("HOME")?+"/input.mkv");
    let mut context = format::input(&path)?;
    let video = context.streams().best(media::Type::Video).unwrap();
    let video_index = video.index();
    let video = codec::context::Context::from_parameters(video.parameters())?;
    let mut video = video.decoder().video()?;

    #[track_caller] fn check(status: std::ffi::c_int) { if status!=0 { panic!("{}", ffmpeg::Error::from(status)); } }
    let hw_device_context = unsafe{&mut *av_hwdevice_ctx_alloc(AVHWDeviceType::AV_HWDEVICE_TYPE_VAAPI)};
    let av_hw_device_context = unsafe{&mut *(hw_device_context.data as *mut AVHWDeviceContext)};
    // av_hwdevice_ctx_create = device_create, av_hwdevice_ctx_init
    // vaapi_device_create
    av_hw_device_context.user_opaque = unsafe{av_mallocz(std::mem::size_of::<std::ffi::c_int>())};
    extern "C" fn vaapi_device_free(_: *mut AVHWDeviceContext) { unimplemented!() }
    av_hw_device_context.free        = Some(vaapi_device_free);
    *unsafe{&mut *(av_hw_device_context.user_opaque as *mut std::ffi::c_int)} = card.0.as_raw_fd();
    extern "C" fn error(_user_context: *mut std::ffi::c_void, message: *const std::ffi::c_char) { panic!("{:?}", unsafe{std::ffi::CStr::from_ptr(message)}) }
    extern "C" fn info(_user_context: *mut std::ffi::c_void, message: *const std::ffi::c_char) { println!("{:?}", unsafe{std::ffi::CStr::from_ptr(message)}); }
    use std::os::unix::io::AsRawFd;
    let va = unsafe{va::va_display_drm::vaGetDisplayDRM(card.0.as_raw_fd())};
    #[repr(C)] struct AVVAAPIDeviceContext { display: va::VADisplay, driver_quirks: std::ffi::c_uint }
    let av_va_api_device_context = unsafe{&mut *(av_hw_device_context.hwctx as *mut AVVAAPIDeviceContext)};
    av_va_api_device_context.display = va;
    unsafe{va::vaSetErrorCallback(va, Some(error), std::ptr::null_mut())};
    unsafe{va::vaSetInfoCallback(va,  Some(info),  std::ptr::null_mut())};
    let (mut major, mut minor) = (0,0);
    check(unsafe{va::vaInitialize(va, &mut major, &mut minor)});
    check(unsafe{av_hwdevice_ctx_init(hw_device_context)});

    // decode_init -> ff_decode_get_hw_frames_ctx -> avcodec_get_hw_frames_parameters -> frame_params -> vaapi_decode_make_config
    //unsafe{avcodec_get_hw_frames_parameters(decoder.as_mut_ptr(), hw_device_context, AVPixelFormat::AV_PIX_FMT_VAAPI, &mut (*decoder.as_mut_ptr()).hw_frames_ctx)};
    let frame_context = unsafe{&mut *av_hwframe_ctx_alloc(hw_device_context)};

    let frames_context = unsafe{&mut *(frame_context.data as *mut AVHWFramesContext)};
    frames_context.format = AVPixelFormat::AV_PIX_FMT_VAAPI;
    //type AVVAAPIHWConfig = va::VAConfigID;
    //let config = unsafe{av_hwdevice_hwconfig_alloc(hw_device_context) as *mut AVVAAPIHWConfig};
    //let mut hwconfig = unsafe{&mut *(av_hwdevice_hwconfig_alloc(hw_device_context) as *mut AVVAAPIHWConfig)};
    let mut profiles = Vec::with_capacity(unsafe{va::vaMaxNumProfiles(va)} as usize);
    let mut len = profiles.capacity() as i32;
    vacheck(unsafe{va::vaQueryConfigProfiles(va, profiles.as_mut_ptr(), &mut len)});
    unsafe{profiles.set_len(len as usize)};
    let (profile, entrypoint) = profiles.into_iter().find_map(|profile| (profile /* != va::va_display::VAProfile_VAProfileNone*/== va::va_display::VAProfile_VAProfileHEVCMain10).then(||{
        let mut entrypoints = Vec::with_capacity(unsafe{va::vaMaxNumEntrypoints(va)} as usize);
        let mut len = entrypoints.capacity() as i32;
        vacheck(unsafe{va::vaQueryConfigEntrypoints(va, profile, entrypoints.as_mut_ptr(), &mut len)});
        unsafe{entrypoints.set_len(len as usize)};
        (profile, entrypoints.into_iter().find(|&entrypoint| entrypoint == va::va_display::VAEntrypoint_VAEntrypointVLD).unwrap())
    })).unwrap();
    let mut config = 0; // va::VAConfigID
    vacheck(unsafe{va::vaCreateConfig(va, profile, entrypoint, std::ptr::null_mut(), 0, &mut config)});
    /*let mut attributes = Vec::with_capacity({let mut capacity = 0; check(unsafe{va::vaQuerySurfaceAttributes(va, config, std::ptr::null_mut(), &mut capacity)}); capacity as usize});
    let mut len = attributes.capacity() as u32;
    vacheck(unsafe{va::vaQuerySurfaceAttributes(va, config, attributes.as_mut_ptr(), &mut len)});
    unsafe{attributes.set_len(len as usize)};*/

    //check(unsafe{av_hwframe_ctx_init(frame_context)});

    let mut render_targets = [0; 1]; //VASurfaceID
    #[track_caller] fn vacheck(status: va::va_str::VAStatus) { if status!=0 { panic!("{:?}", unsafe{std::ffi::CStr::from_ptr(va::vaErrorStr(status))}); } }
    println!("{:?}",(video.width(), video.height()));
    vacheck(unsafe{va::vaCreateSurfaces(va, va::VA_RT_FORMAT_YUV420_10, video.width(), video.height(), render_targets.as_mut_ptr(), render_targets.len() as _, std::ptr::null_mut(), 0/*attributes.as_mut_ptr(), attributes.len() as u32*/)});

    extern "C" fn init(_: *mut AVCodecContext) -> std::ffi::c_int { unimplemented!(); }
    /*let mut context = 0;
    vacheck(unsafe{va::vaCreateContext(va, config, video.width() as _, video.height() as _, va::VA_PROGRESSIVE as _, render_targets.as_mut_ptr(),
                          render_targets.len() as _, &mut context)});*/

    /*let frames_constraints = unsafe{*av_hwdevice_get_hwframe_constraints(hw_device_context, (&*hwconfig as *const AVVAAPIHWConfig).cast())};
    frames_context.sw_format = *std::iter::successors(Some(frames_constraints.valid_sw_formats), |p| Some(unsafe{p.add(1)})).filter_map(|p| std::ptr::NonNull::new(p).map(|p| unsafe{p.as_ref()}))
        .find(|&&format| format==AVPixelFormat::AV_PIX_FMT_NV12).unwrap();
    av_hwframe_constraints_free(&frames_constraints);*/
    frames_context.sw_format = AVPixelFormat::AV_PIX_FMT_NV12;
    (frames_context.width, frames_context.height) = /*(1,1)*/ (128, 128);

    let frame = unsafe{av_frame_alloc()};
    check(unsafe{av_hwframe_get_buffer(frame_context, frame, 0)});

    unsafe {*video.as_mut_ptr()}.hw_device_ctx = hw_device_context;
    unsafe extern "C" fn get_format(_s: *mut AVCodecContext, _fmt: *const AVPixelFormat) -> AVPixelFormat { unimplemented!(); }
    unsafe {*video.as_mut_ptr()}.get_format = Some(get_format);
    log::set_level(log::level::Level::Trace);
    //unsafe{std::intrinsics::breakpoint()};

    /*  HW_CONFIG_HWACCEL(1, 1, 1, VAAPI,        VAAPI,        ff_ ## codec ## _vaapi_hwaccel)
    &(const AVCodecHWConfigInternal) { \
        .public          = { \
            .pix_fmt     = AV_PIX_FMT_ ## format, \
            .methods     = (device ? AV_CODEC_HW_CONFIG_METHOD_HW_DEVICE_CTX : 0) | \
                           (frames ? AV_CODEC_HW_CONFIG_METHOD_HW_FRAMES_CTX : 0) | \
                           (ad_hoc ? AV_CODEC_HW_CONFIG_METHOD_AD_HOC        : 0),  \
            .device_type = AV_HWDEVICE_TYPE_ ## device_type_, \
        }, \
        .hwaccel         = &name, \
    }*/
    //assert!(unsafe {*decoder.as_mut_ptr()}.hwaccel != std::ptr::null());
    const HWACCEL_CAP_ASYNC_SAFE : std::ffi::c_int = 1<<0;
    extern "C" fn start_frame(ctx: *mut AVCodecContext, _: *const u8, _: u32) -> std::ffi::c_int {
        unimplemented!();
        /*const HEVCContext        *h = avctx->priv_data;
        VAAPIDecodePictureHEVC *pic = h->ref->hwaccel_picture_private;
        const HEVCSPS          *sps = h->ps.sps;
        const HEVCPPS          *pps = h->ps.pps;

        const ScalingList *scaling_list = NULL;
        int pic_param_size, err, i;

        VAPictureParameterBufferHEVC *pic_param = (VAPictureParameterBufferHEVC *)&pic->pic_param;

        pic->pic.output_surface = ff_vaapi_get_surface_id(h->ref->frame);

        *pic_param = (VAPictureParameterBufferHEVC) {
        .pic_width_in_luma_samples                    = sps->width,
        .pic_height_in_luma_samples                   = sps->height,
        .log2_min_luma_coding_block_size_minus3       = sps->log2_min_cb_size - 3,
        .sps_max_dec_pic_buffering_minus1             = sps->temporal_layer[sps->max_sub_layers - 1].max_dec_pic_buffering - 1,
        .log2_diff_max_min_luma_coding_block_size     = sps->log2_diff_max_min_coding_block_size,
        .log2_min_transform_block_size_minus2         = sps->log2_min_tb_size - 2,
        .log2_diff_max_min_transform_block_size       = sps->log2_max_trafo_size  - sps->log2_min_tb_size,
        .max_transform_hierarchy_depth_inter          = sps->max_transform_hierarchy_depth_inter,
        .max_transform_hierarchy_depth_intra          = sps->max_transform_hierarchy_depth_intra,
        .num_short_term_ref_pic_sets                  = sps->nb_st_rps,
        .num_long_term_ref_pic_sps                    = sps->num_long_term_ref_pics_sps,
        .num_ref_idx_l0_default_active_minus1         = pps->num_ref_idx_l0_default_active - 1,
        .num_ref_idx_l1_default_active_minus1         = pps->num_ref_idx_l1_default_active - 1,
        .init_qp_minus26                              = pps->pic_init_qp_minus26,
        .pps_cb_qp_offset                             = pps->cb_qp_offset,
        .pps_cr_qp_offset                             = pps->cr_qp_offset,
        .pcm_sample_bit_depth_luma_minus1             = sps->pcm.bit_depth - 1,
        .pcm_sample_bit_depth_chroma_minus1           = sps->pcm.bit_depth_chroma - 1,
        .log2_min_pcm_luma_coding_block_size_minus3   = sps->pcm.log2_min_pcm_cb_size - 3,
        .log2_diff_max_min_pcm_luma_coding_block_size = sps->pcm.log2_max_pcm_cb_size - sps->pcm.log2_min_pcm_cb_size,
        .diff_cu_qp_delta_depth                       = pps->diff_cu_qp_delta_depth,
        .pps_beta_offset_div2                         = pps->beta_offset / 2,
        .pps_tc_offset_div2                           = pps->tc_offset / 2,
        .log2_parallel_merge_level_minus2             = pps->log2_parallel_merge_level - 2,
        .bit_depth_luma_minus8                        = sps->bit_depth - 8,
        .bit_depth_chroma_minus8                      = sps->bit_depth - 8,
        .log2_max_pic_order_cnt_lsb_minus4            = sps->log2_max_poc_lsb - 4,
        .num_extra_slice_header_bits                  = pps->num_extra_slice_header_bits,
        .pic_fields.bits = {
        .chroma_format_idc                          = sps->chroma_format_idc,
        .tiles_enabled_flag                         = pps->tiles_enabled_flag,
        .separate_colour_plane_flag                 = sps->separate_colour_plane_flag,
        .pcm_enabled_flag                           = sps->pcm_enabled_flag,
        .scaling_list_enabled_flag                  = sps->scaling_list_enable_flag,
        .transform_skip_enabled_flag                = pps->transform_skip_enabled_flag,
        .amp_enabled_flag                           = sps->amp_enabled_flag,
        .strong_intra_smoothing_enabled_flag        = sps->sps_strong_intra_smoothing_enable_flag,
        .sign_data_hiding_enabled_flag              = pps->sign_data_hiding_flag,
        .constrained_intra_pred_flag                = pps->constrained_intra_pred_flag,
        .cu_qp_delta_enabled_flag                   = pps->cu_qp_delta_enabled_flag,
        .weighted_pred_flag                         = pps->weighted_pred_flag,
        .weighted_bipred_flag                       = pps->weighted_bipred_flag,
        .transquant_bypass_enabled_flag             = pps->transquant_bypass_enable_flag,
        .entropy_coding_sync_enabled_flag           = pps->entropy_coding_sync_enabled_flag,
        .pps_loop_filter_across_slices_enabled_flag = pps->seq_loop_filter_across_slices_enabled_flag,
        .loop_filter_across_tiles_enabled_flag      = pps->loop_filter_across_tiles_enabled_flag,
        .pcm_loop_filter_disabled_flag              = sps->pcm.loop_filter_disable_flag,
        },
        .slice_parsing_fields.bits = {
        .lists_modification_present_flag             = pps->lists_modification_present_flag,
        .long_term_ref_pics_present_flag             = sps->long_term_ref_pics_present_flag,
        .sps_temporal_mvp_enabled_flag               = sps->sps_temporal_mvp_enabled_flag,
        .cabac_init_present_flag                     = pps->cabac_init_present_flag,
        .output_flag_present_flag                    = pps->output_flag_present_flag,
        .dependent_slice_segments_enabled_flag       = pps->dependent_slice_segments_enabled_flag,
        .pps_slice_chroma_qp_offsets_present_flag    = pps->pic_slice_level_chroma_qp_offsets_present_flag,
        .sample_adaptive_offset_enabled_flag         = sps->sao_enabled,
        .deblocking_filter_override_enabled_flag     = pps->deblocking_filter_override_enabled_flag,
        .pps_disable_deblocking_filter_flag          = pps->disable_dbf,
        .slice_segment_header_extension_present_flag = pps->slice_header_extension_present_flag,
        .RapPicFlag                                  = IS_IRAP(h),
        .IdrPicFlag                                  = IS_IDR(h),
        .IntraPicFlag                                = IS_IRAP(h),
        },
        };

        fill_vaapi_pic(&pic_param->CurrPic, h->ref, 0);
        fill_vaapi_reference_frames(h, pic_param);

        if (pps->tiles_enabled_flag) {
        pic_param->num_tile_columns_minus1 = pps->num_tile_columns - 1;
        pic_param->num_tile_rows_minus1    = pps->num_tile_rows - 1;

        for (i = 0; i < pps->num_tile_columns; i++)
        pic_param->column_width_minus1[i] = pps->column_width[i] - 1;

        for (i = 0; i < pps->num_tile_rows; i++)
        pic_param->row_height_minus1[i] = pps->row_height[i] - 1;
        }

        if (h->sh.short_term_ref_pic_set_sps_flag == 0 && h->sh.short_term_rps) {
        pic_param->st_rps_bits = h->sh.short_term_ref_pic_set_size;
        } else {
        pic_param->st_rps_bits = 0;
        }

        #if VA_CHECK_VERSION(1, 2, 0)
        if (avctx->profile == FF_PROFILE_HEVC_REXT) {
        pic->pic_param.rext = (VAPictureParameterBufferHEVCRext) {
        .range_extension_pic_fields.bits  = {
        .transform_skip_rotation_enabled_flag       = sps->transform_skip_rotation_enabled_flag,
        .transform_skip_context_enabled_flag        = sps->transform_skip_context_enabled_flag,
        .implicit_rdpcm_enabled_flag                = sps->implicit_rdpcm_enabled_flag,
        .explicit_rdpcm_enabled_flag                = sps->explicit_rdpcm_enabled_flag,
        .extended_precision_processing_flag         = sps->extended_precision_processing_flag,
        .intra_smoothing_disabled_flag              = sps->intra_smoothing_disabled_flag,
        .high_precision_offsets_enabled_flag        = sps->high_precision_offsets_enabled_flag,
        .persistent_rice_adaptation_enabled_flag    = sps->persistent_rice_adaptation_enabled_flag,
        .cabac_bypass_alignment_enabled_flag        = sps->cabac_bypass_alignment_enabled_flag,
        .cross_component_prediction_enabled_flag    = pps->cross_component_prediction_enabled_flag,
        .chroma_qp_offset_list_enabled_flag         = pps->chroma_qp_offset_list_enabled_flag,
        },
        .diff_cu_chroma_qp_offset_depth                 = pps->diff_cu_chroma_qp_offset_depth,
        .chroma_qp_offset_list_len_minus1               = pps->chroma_qp_offset_list_len_minus1,
        .log2_sao_offset_scale_luma                     = pps->log2_sao_offset_scale_luma,
        .log2_sao_offset_scale_chroma                   = pps->log2_sao_offset_scale_chroma,
        .log2_max_transform_skip_block_size_minus2      = pps->log2_max_transform_skip_block_size - 2,
        };

        for (i = 0; i < 6; i++)
        pic->pic_param.rext.cb_qp_offset_list[i]        = pps->cb_qp_offset_list[i];
        for (i = 0; i < 6; i++)
        pic->pic_param.rext.cr_qp_offset_list[i]        = pps->cr_qp_offset_list[i];
        }
        #endif
        pic_param_size = avctx->profile == FF_PROFILE_HEVC_REXT ?
        sizeof(pic->pic_param) : sizeof(VAPictureParameterBufferHEVC);

        err = ff_vaapi_decode_make_param_buffer(avctx, &pic->pic,
                        VAPictureParameterBufferType,
                        &pic->pic_param, pic_param_size);
        if (err < 0)
        goto fail;

        if (pps->scaling_list_data_present_flag)
        scaling_list = &pps->scaling_list;
        else if (sps->scaling_list_enable_flag)
        scaling_list = &sps->scaling_list;

        if (scaling_list) {
        VAIQMatrixBufferHEVC iq_matrix;
        int j;

        for (i = 0; i < 6; i++) {
        for (j = 0; j < 16; j++)
        iq_matrix.ScalingList4x4[i][j] = scaling_list->sl[0][i][j];
        for (j = 0; j < 64; j++) {
        iq_matrix.ScalingList8x8[i][j]   = scaling_list->sl[1][i][j];
        iq_matrix.ScalingList16x16[i][j] = scaling_list->sl[2][i][j];
        if (i < 2)
        iq_matrix.ScalingList32x32[i][j] = scaling_list->sl[3][i * 3][j];
        }
        iq_matrix.ScalingListDC16x16[i] = scaling_list->sl_dc[0][i];
        if (i < 2)
        iq_matrix.ScalingListDC32x32[i] = scaling_list->sl_dc[1][i * 3];
        }

        err = ff_vaapi_decode_make_param_buffer(avctx, &pic->pic,
                            VAIQMatrixBufferType,
                            &iq_matrix, sizeof(iq_matrix));
        if (err < 0)
        goto fail;
        }

        return 0;

        fail:
        ff_vaapi_decode_cancel(avctx, &pic->pic);
        return err;*/
    }

    extern "C" fn end_frame(ctx: *mut AVCodecContext) -> std::ffi::c_int {
        unimplemented!();
        /*const HEVCContext        *h = avctx->priv_data;
        VAAPIDecodePictureHEVC *pic = h->ref->hwaccel_picture_private;
        VASliceParameterBufferHEVC *last_slice_param = (VASliceParameterBufferHEVC *)&pic->last_slice_param;
        int ret;

        int slice_param_size = avctx->profile == FF_PROFILE_HEVC_REXT ?
        sizeof(pic->last_slice_param) : sizeof(VASliceParameterBufferHEVC);

        if (pic->last_size) {
        last_slice_param->LongSliceFlags.fields.LastSliceOfPic = 1;
        ret = ff_vaapi_decode_make_slice_buffer(avctx, &pic->pic,
                            &pic->last_slice_param, slice_param_size,
                            pic->last_buffer, pic->last_size);
        if (ret < 0)
        goto fail;
        }


        ret = ff_vaapi_decode_issue(avctx, &pic->pic);
        if (ret < 0)
        goto fail;

        return 0;*/
    }

    extern "C" fn decode_slice(ctx: *const AVCodecContext, _buffer: *const u8, _len: u32) -> std::ffi::c_int {
        unimplemented!()
        /*const HEVCContext        *h = avctx->priv_data;
        const SliceHeader       *sh = &h->sh;
        VAAPIDecodePictureHEVC *pic = h->ref->hwaccel_picture_private;
        VASliceParameterBufferHEVC *last_slice_param = (VASliceParameterBufferHEVC *)&pic->last_slice_param;

        int slice_param_size = avctx->profile == FF_PROFILE_HEVC_REXT ?
        sizeof(pic->last_slice_param) : sizeof(VASliceParameterBufferHEVC);

        int nb_list = (sh->slice_type == HEVC_SLICE_B) ?
        2 : (sh->slice_type == HEVC_SLICE_I ? 0 : 1);

        int err, i, list_idx;

        if (!sh->first_slice_in_pic_flag) {
        err = ff_vaapi_decode_make_slice_buffer(avctx, &pic->pic,
                            &pic->last_slice_param, slice_param_size,
                            pic->last_buffer, pic->last_size);
        pic->last_buffer = NULL;
        pic->last_size   = 0;
        if (err) {
        ff_vaapi_decode_cancel(avctx, &pic->pic);
        return err;
        }
        }

        *last_slice_param = (VASliceParameterBufferHEVC) {
        .slice_data_size               = size,
        .slice_data_offset             = 0,
        .slice_data_flag               = VA_SLICE_DATA_FLAG_ALL,
        /* Add 1 to the bits count here to account for the byte_alignment bit, which
        * always is at least one bit and not accounted for otherwise. */
        .slice_data_byte_offset        = (get_bits_count(&h->HEVClc->gb) + 1 + 7) / 8,
        .slice_segment_address         = sh->slice_segment_addr,
        .slice_qp_delta                = sh->slice_qp_delta,
        .slice_cb_qp_offset            = sh->slice_cb_qp_offset,
        .slice_cr_qp_offset            = sh->slice_cr_qp_offset,
        .slice_beta_offset_div2        = sh->beta_offset / 2,
        .slice_tc_offset_div2          = sh->tc_offset / 2,
        .collocated_ref_idx            = sh->slice_temporal_mvp_enabled_flag ? sh->collocated_ref_idx : 0xFF,
        .five_minus_max_num_merge_cand = sh->slice_type == HEVC_SLICE_I ? 0 : 5 - sh->max_num_merge_cand,
        .num_ref_idx_l0_active_minus1  = sh->nb_refs[L0] ? sh->nb_refs[L0] - 1 : 0,
        .num_ref_idx_l1_active_minus1  = sh->nb_refs[L1] ? sh->nb_refs[L1] - 1 : 0,

        .LongSliceFlags.fields = {
        .dependent_slice_segment_flag                 = sh->dependent_slice_segment_flag,
        .slice_type                                   = sh->slice_type,
        .color_plane_id                               = sh->colour_plane_id,
        .mvd_l1_zero_flag                             = sh->mvd_l1_zero_flag,
        .cabac_init_flag                              = sh->cabac_init_flag,
        .slice_temporal_mvp_enabled_flag              = sh->slice_temporal_mvp_enabled_flag,
        .slice_deblocking_filter_disabled_flag        = sh->disable_deblocking_filter_flag,
        .collocated_from_l0_flag                      = sh->collocated_list == L0 ? 1 : 0,
        .slice_loop_filter_across_slices_enabled_flag = sh->slice_loop_filter_across_slices_enabled_flag,
        .slice_sao_luma_flag                          = sh->slice_sample_adaptive_offset_flag[0],
        .slice_sao_chroma_flag                        = sh->slice_sample_adaptive_offset_flag[1],
        },
        };

        memset(last_slice_param->RefPicList, 0xFF, sizeof(last_slice_param->RefPicList));

        for (list_idx = 0; list_idx < nb_list; list_idx++) {
        RefPicList *rpl = &h->ref->refPicList[list_idx];

        for (i = 0; i < rpl->nb_refs; i++)
        last_slice_param->RefPicList[list_idx][i] = get_ref_pic_index(h, rpl->ref[i]);
        }

        fill_pred_weight_table(avctx, h, sh, last_slice_param);

        #if VA_CHECK_VERSION(1, 2, 0)
        if (avctx->profile == FF_PROFILE_HEVC_REXT) {
        pic->last_slice_param.rext = (VASliceParameterBufferHEVCRext) {
        .slice_ext_flags.bits = {
        .cu_chroma_qp_offset_enabled_flag = sh->cu_chroma_qp_offset_enabled_flag,
        },
        };
        for (i = 0; i < 15 && i < sh->nb_refs[L0]; i++) {
        pic->last_slice_param.rext.luma_offset_l0[i] = sh->luma_offset_l0[i];
        pic->last_slice_param.rext.ChromaOffsetL0[i][0] = sh->chroma_offset_l0[i][0];
        pic->last_slice_param.rext.ChromaOffsetL0[i][1] = sh->chroma_offset_l0[i][1];
        }

        for (i = 0; i < 15 && i < sh->nb_refs[L0]; i++) {
        pic->last_slice_param.rext.luma_offset_l0[i] = sh->luma_offset_l0[i];
        pic->last_slice_param.rext.ChromaOffsetL0[i][0] = sh->chroma_offset_l0[i][0];
        pic->last_slice_param.rext.ChromaOffsetL0[i][1] = sh->chroma_offset_l0[i][1];
        }

        if (sh->slice_type == HEVC_SLICE_B) {
        for (i = 0; i < 15 && i < sh->nb_refs[L1]; i++) {
        pic->last_slice_param.rext.luma_offset_l1[i] = sh->luma_offset_l1[i];
        pic->last_slice_param.rext.ChromaOffsetL1[i][0] = sh->chroma_offset_l1[i][0];
        pic->last_slice_param.rext.ChromaOffsetL1[i][1] = sh->chroma_offset_l1[i][1];
        }
        }
        }
        #endif

        pic->last_buffer = buffer;
        pic->last_size   = size;

        return 0;*/
    }


    extern "C" fn frame_params(_ctx: *const AVCodecContext, _hw_frames_ctx: *const AVBufferRef) -> std::ffi::c_int { unimplemented!() }
    extern "C" fn uninit(_: *mut AVCodecContext) -> std::ffi::c_int { unimplemented!(); }

    struct VAAPIDecodePictureHEVC {
        /*#if VA_CHECK_VERSION(1, 2, 0)
            VAPictureParameterBufferHEVCExtension pic_param;
            VASliceParameterBufferHEVCExtension last_slice_param;
        #else
            VAPictureParameterBufferHEVC pic_param;
            VASliceParameterBufferHEVC last_slice_param;
        #endif
            const uint8_t *last_buffer;
            size_t         last_size;
            VAAPIDecodePicture pic;*/
    }
    struct VAAPIDecodeContext {
        /*VAConfigID            va_config;
        VAContextID           va_context;

        AVHWDeviceContext    *device;
        AVVAAPIDeviceContext *hwctx;

        AVHWFramesContext    *frames;
        AVVAAPIFramesContext *hwfc;

        enum AVPixelFormat    surface_format;
        int                   surface_count;

        VASurfaceAttrib       pixel_format_attribute;*/
    };
    /*unsafe{&mut *unsafe {*video.as_mut_ptr()}.hwaccel} = AVHWAccel{
        name: std::ffi::CStr::from_bytes_with_nul(b"hevc_vaapi\0").unwrap().as_ptr(),
        type_: AVMediaType::AVMEDIA_TYPE_VIDEO,
        id: AVCodecID::AV_CODEC_ID_HEVC,
        pix_fmt:  AVPixelFormat::AV_PIX_FMT_VAAPI,
        start_frame, end_frame, decode_slice,
        frame_priv_data_size: std::mem::size_of::<VAAPIDecodePictureHEVC>(),
        init, uninit, frame_params,
        priv_data_size: std::mem::size_of::<VAAPIDecodeContext>(),
        caps_internal: HWACCEL_CAP_ASYNC_SAFE,
        alloc_frame: None, // Enc
        capabilities: 0, // Enc
    };*/

    for (stream, packet) in context.packets() {
        if stream.index() == video_index {
            video.send_packet(&packet).unwrap();
            loop {
                let mut frame = Video::empty();
                //let mut video = unsafe{Video::wrap(frame)};
                if video.receive_frame(&mut frame).is_err() { break; }
                let frame = unsafe{&*frame.as_ptr()};
                assert!(frame.hw_frames_ctx != std::ptr::null_mut());
                //let format = unsafe{std::intrinsics::transmute::<_,AVPixelFormat>(frame.format)};
                //assert!(format == AVPixelFormat::AV_PIX_FMT_YUV420P10LE, "{format:?}");
                /*let mut map = card.map_dumb_buffer(&mut db).expect("Could not map dumbbuffer");
                let map = bytemuck::cast_slice_mut(map.as_mut());
                for y in 0..video.height() {
                    #[allow(non_snake_case)] for x in 0..video.width() {
                        let plane = |index| unsafe {
                            std::slice::from_raw_parts(
                                (*video.as_ptr()).data[index] as *const u16,
                                video.stride(index) * video.plane_height(index) as usize / std::mem::size_of::<u16>(),
                            )
                        };
                        let Y = plane(0)[y as usize*video.stride(0)/2+x as usize];
                        let Cb = plane(1)[(y/2) as usize*video.stride(1)/2+(x/2) as usize] as i16 - 512; // FIXME: bilinear
                        let Cr = plane(2)[(y/2) as usize*video.stride(2)/2+(x/2) as usize] as i16 - 512; // FIXME: bilinear
                        let Y = Y as f32;
                        let Cr = Cr as f32;
                        let Cb = Cb as f32;
                        const A : f32 = 0.2627;
                        const B : f32 = 0.6780;
                        const C : f32 = 1.-A-B;
                        const D : f32 = 2.*(A+B);
                        const E : f32 = 2.*(1.-A);
                        let r = Y + E * Cr;
                        let g = Y - (A * E / B) * Cr - (C * D / B) * Cb;
                        let b = Y + D * Cb;
                        let r = r as u32;
                        let g = g as u32;
                        let b = b as u32;
                        map[(y*width +x) as usize] = b | g<<10 | r<<20;
                    }
                }
                if card.page_flip(crtc.handle(), fb, PageFlipFlags::empty(), None).is_ok() { println!("flip!"); } else { println!("no flip"); }*/
            }
        }
    }
    //card.destroy_framebuffer(fb).unwrap(); card.destroy_dumb_buffer(db).unwrap();*/
    Ok(())
}
