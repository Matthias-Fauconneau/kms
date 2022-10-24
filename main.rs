#![allow(incomplete_features)]#![feature(int_log,generic_arg_infer,generic_const_exprs,array_zip,unchecked_math,array_methods,iter_array_chunks,array_windows,array_chunks,new_uninit,closure_track_caller)]
fn from_iter_or_else<T, const N: usize>(iter: impl IntoIterator<Item=T>, f: impl Fn() -> T+Copy) -> [T; N] { let mut iter = iter.into_iter(); [(); N].map(|_| iter.next().unwrap_or_else(f)) }
fn from_iter_or<T: Copy, const N: usize>(iter: impl IntoIterator<Item=T>, v: T) -> [T; N] { from_iter_or_else(iter, || v) }
#[track_caller] fn from_iter<T, const N: usize>(iter: impl IntoIterator<Item=T>) -> [T; N] { from_iter_or_else(iter, #[track_caller] || unreachable!()) }
fn from_iter_or_default<T: Default, const N: usize>(iter: impl IntoIterator<Item=T>) -> [T; N] { from_iter_or_else(iter, || Default::default()) }
fn array<T: Default, const N: usize>(len: usize, mut f: impl FnMut()->T) -> [T; N] { from_iter_or_default((0..len).map(|_| f())) }

fn bytes_of<T>(value: &T) -> &[u8] { unsafe{std::slice::from_raw_parts(value as *const _ as *const u8, std::mem::size_of::<T>())} }

mod hevc;

mod va {
    #![allow(dead_code,non_camel_case_types,non_upper_case_globals,improper_ctypes,non_snake_case)]
    include!(concat!(env!("OUT_DIR"), "/va.rs"));
}

pub fn main() -> Result<(), Box<dyn std::error::Error>> {
    let path = std::env::args().skip(1).next().unwrap_or(std::env::var("HOME")?+"/input.mkv");

    let av = if true {
        mod av {
            #![allow(dead_code,non_camel_case_types,non_upper_case_globals,improper_ctypes,non_snake_case)]
            include!(concat!(env!("OUT_DIR"), "/av.rs"));
        }
        use av::*;

        #[track_caller] fn check(status: std::ffi::c_int) -> std::ffi::c_int { if status!=0 { let mut s=[0;64]; unsafe{av_strerror(status,s.as_mut_ptr(),s.len());} panic!("{}", unsafe{std::ffi::CStr::from_ptr(s.as_ptr())}.to_str().unwrap()); } else { status } }
        let mut context = std::ptr::null_mut();
        let path = std::ffi::CString::new(path.clone()).unwrap();
        check(unsafe{avformat_open_input(&mut context, path.as_ptr(), std::ptr::null_mut(), std::ptr::null_mut())});
        check(unsafe{avformat_find_stream_info(context, std::ptr::null_mut())});
        let mut decoder = std::ptr::null();
        let video_stream = check(unsafe{av_find_best_stream(context, AVMediaType_AVMEDIA_TYPE_VIDEO, -1, -1, &mut decoder as *mut _, 0)});
        assert!(decoder != std::ptr::null_mut());
        let video = &mut unsafe{&**(*context).streams.offset(video_stream as isize)};
        let decoder_context = unsafe{avcodec_alloc_context3(decoder)};
        check(unsafe{avcodec_parameters_to_context(decoder_context, video.codecpar)});
        extern "C" fn get_format(_s: *mut AVCodecContext, _fmt: *const AVPixelFormat) -> AVPixelFormat { AVPixelFormat_AV_PIX_FMT_VAAPI  }
        unsafe{&mut *decoder_context}.get_format  = Some(get_format);
        let mut hw_device_context = std::ptr::null_mut();
        //unsafe{av_log_set_level(AV_LOG_TRACE as _)};
        check(unsafe{av_hwdevice_ctx_create(&mut hw_device_context, AVHWDeviceType_AV_HWDEVICE_TYPE_VAAPI, std::ptr::null(), std::ptr::null_mut(), 0)});
        assert!(hw_device_context != std::ptr::null_mut());
        unsafe{&mut *decoder_context}.hw_device_ctx = unsafe{av_buffer_ref(hw_device_context)};
        check(unsafe{avcodec_open2(decoder_context, decoder, std::ptr::null_mut())});

        let packet = unsafe{av_packet_alloc()};
        let mut av = Vec::new();
        while av.len()<187 && unsafe{av_read_frame(context, packet)} >= 0 {
            if video_stream == unsafe{&*packet}.stream_index {
                check(unsafe{avcodec_send_packet(decoder_context, packet)});
                assert!(unsafe{&mut *decoder_context}.hwaccel != std::ptr::null());
                let frame = unsafe{av_frame_alloc()};
                let status = unsafe{avcodec_receive_frame(decoder_context, frame)};
                if status == -libc::EAGAIN { continue; }
                check(status);
                #[repr(C)] struct VAAPIDecodeTraceBuffer {
                    r#type: u32,
                    len: usize,
                    data: *const u8,
                }
                extern "C" { static mut vaapi_decode_trace_buffers : [VAAPIDecodeTraceBuffer; 64]; }
                let buffers = unsafe{&vaapi_decode_trace_buffers}.into_iter().map(|b| (!b.data.is_null()).then(||(b.r#type, unsafe{std::slice::from_raw_parts(b.data, b.len)}))).fuse().flatten().collect::<Box<_>>();
                println!("buffers {}", buffers.len());
                unsafe{vaapi_decode_trace_buffers = [(); _].map(|_|VAAPIDecodeTraceBuffer{r#type: 0, len: 0, data: std::ptr::null()})}; // Leaks

                /*let frame = {
                    let sw = unsafe{&mut *av_frame_alloc()};
                    sw.format = AVPixelFormat_AV_PIX_FMT_P010LE;
                    check(unsafe{av_hwframe_transfer_data(sw, frame, 0)});
                    assert_eq!(sw.format, AVPixelFormat_AV_PIX_FMT_P010LE);
                    sw
                };
                let mut buffer = unsafe{Box::new_uninit_slice(av_image_get_buffer_size(frame.format, frame.width, frame.height, 1) as _).assume_init()};
                assert_eq!(unsafe{av_image_copy_to_buffer(buffer.as_mut_ptr(), buffer.len() as _, frame.data.as_ptr() as *const *const _, frame.linesize.as_ptr(), frame.format, frame.width, frame.height, 1)}, buffer.len() as i32);*/
                av.append(&mut buffers.into_vec());
            }
            unsafe{av_packet_unref(packet)};
        }
        av
    } else {Vec::new()};
    let mut av = std::collections::VecDeque::from(av);

    let input = unsafe{memmap::Mmap::map(&std::fs::File::open(&path)?)}?;

    struct Card(std::fs::File);
    let card = Card(std::fs::OpenOptions::new().read(true).write(true).open("/dev/dri/card0").unwrap());
    impl std::os::unix::io::AsRawFd for Card { fn as_raw_fd(&self) -> std::os::unix::io::RawFd { self.0.as_raw_fd() } }
    use std::os::unix::io::AsRawFd;
    use drm::Device;
    impl Device for Card {}
    let va = unsafe{va::vaGetDisplayDRM(card.0.as_raw_fd())};
    extern "C" fn error(_user_context: *mut std::ffi::c_void, message: *const std::ffi::c_char) { panic!("{:?}", unsafe{std::ffi::CStr::from_ptr(message)}) }
    unsafe{va::vaSetErrorCallback(va, Some(error), std::ptr::null_mut())};
    extern "C" fn info(_user_context: *mut std::ffi::c_void, _message: *const std::ffi::c_char) { /*println!("{:?}", unsafe{std::ffi::CStr::from_ptr(message)});*/ }
    unsafe{va::vaSetInfoCallback(va,  Some(info),  std::ptr::null_mut())};
    let (mut major, mut minor) = (0,0);
    #[track_caller] fn check(status: va::VAStatus) { if status!=0 { panic!("{:?}", unsafe{std::ffi::CStr::from_ptr(va::vaErrorStr(status))}); } }
    check(unsafe{va::vaInitialize(va, &mut major, &mut minor)});
    let mut config = 0; // va::VAConfigID
    check(unsafe{va::vaCreateConfig(va, va::VAProfile_VAProfileHEVCMain10, va::VAEntrypoint_VAEntrypointVLD, std::ptr::null_mut(), 0, &mut config)});

    #[derive(Default)] struct Frame {
        id: va::VASurfaceID,
        poc: Option<u32>,
    }
    struct Context {
        frames: [Frame; 20],
        context: va::VAContextID,
        current_id: Option<va::VASurfaceID>,
    }
    let mut context = None;
    let mut buffers = Vec::new();
    let mut buffer_count = 0;
    use hevc::*;
    HEVC::parse(&*input, |Slice{pps, sps, unit, slice_header, escaped_data,slice_data_byte_offset,dependent_slice_segment,slice_segment_address}, last_slice_of_picture: bool| {
        use itertools::Itertools;
        let context = context.get_or_insert_with(|| {
            let mut ids = [0; _];
            check(unsafe{va::vaCreateSurfaces(va, va::VA_RT_FORMAT_YUV420_10, sps.width as _, sps.height as _, ids.as_mut_ptr(), ids.len() as _, std::ptr::null_mut(), 0)});
            ids.reverse(); // to match AV
            //ids = [19,18,17,16,15,14,13,12,11,10,9,8,7,6,5,4,3,2,1,0].map(|i| ids[i]); // to match AV
            Context{context: {
                let mut context = 0;
                check(unsafe{va::vaCreateContext(va, config, sps.width as _, sps.height as _, va::VA_PROGRESSIVE as _, ids.as_ptr() as *mut _, ids.len() as _, &mut context)});
                context
            }, frames: ids.map(|id| Frame{id, poc: None}), current_id: None}
        });
        let frames = &mut context.frames;
        let reference = slice_header.reference.as_ref();

        let mut render = {let context = context.context; let buffers = &mut buffers; let av = &mut av; let buffer_count = &mut buffer_count; move |r#type, data:&[u8]| {
            *buffer_count += 1;
            if let Some(av) = av.pop_front() {
                assert_eq!(r#type, av.0);
                let (a, b) = (data, av.1);
                if a != b {
                    println!("{}", r#type);
                    for chunk in a.iter().zip(b).array_chunks::<64>() {
                        let (a,b) = chunk.iter().map(|(a,b)| { let f = if a==b { "\x1b[0m" } else { "\x1b[31m" }.to_string(); (f.clone()+&format!("{a:02x}"), f+&format!("{b:02x}")) }).unzip::<_,_,Vec<_>,Vec<_>>();
                        println!("+ {}", &a.array_chunks::<16>().map(|x| x.concat()+" ").format(""));
                        println!("- {}", &b.array_chunks::<16>().map(|x| x.concat()+" ").format(""));
                    }
                    if r#type != 0 /*|| a[1..]!=b[1..]*/ {
                        if r#type == 0 {
                            assert!(a.len() == std::mem::size_of::<va::VAPictureParameterBufferHEVC>());
                            assert!(a.len() == b.len());
                            let a = unsafe{&*(a.as_ptr() as *const va::VAPictureParameterBufferHEVC)};
                            let b = unsafe{&*(b.as_ptr() as *const va::VAPictureParameterBufferHEVC)};
                            //macro_rules field { ($field:id) => { assert_eq!(a.$field, b.$field); } };
                            impl std::cmp::PartialEq for va::VAPictureHEVC { fn eq(&self, b: &Self) -> bool { self.picture_id == b.picture_id && self.pic_order_cnt == b.pic_order_cnt && self.flags == b.flags } }
                            assert_eq!(a.CurrPic, b.CurrPic);
                            for (a,b) in a.ReferenceFrames.iter().zip(b.ReferenceFrames) { assert_eq!(a,&b); }
                        }
                        panic!();
                    }
                }
            }
            let mut buffer = 0;
            check(unsafe{va::vaCreateBuffer(va, context, r#type, data.len() as _, 1, data.as_ptr() as *const _ as *mut _, &mut buffer)});
            buffers.push(buffer);
            //trace_buffers.push((r#type, data.to_owned().into_boxed_slice()));
        }};

        let current_id = *context.current_id.get_or_insert_with(|| {
            let current_poc = reference.map(|r| r.poc).unwrap_or(0);
            println!("POC {}", current_poc);
            println!("DPB [{}]", frames.iter().filter_map(|f| f.poc).format(" "));
            reference.map(|r| println!("refs [{};{}]", r.short_term_pictures.iter().map(|p| (current_poc as i32+p.delta_poc as i32) as u32).format(" "),r.long_term_pictures.iter().map(|p| p.poc).format(" ")));

            let current_id = frames.iter().find(|f| f.poc.is_none()).unwrap().id;

            /*for Frame{poc: frame_poc,..} in frames.iter_mut() {
                *frame_poc = frame_poc.filter(|&frame_poc| reference.map(|r| r.short_term_pictures.iter().any(|p| ((current_poc as i32+p.delta_poc as i32) as u32) == frame_poc) || r.long_term_pictures.iter().any(|p| p.poc == frame_poc)).unwrap_or(false));
            }*/for Frame{poc: frame_poc,id,..} in frames.iter_mut() {
                *frame_poc = frame_poc.filter(|&frame_poc| reference.map(|r| r.short_term_pictures.iter().any(|p| ((current_poc as i32+p.delta_poc as i32) as u32) == frame_poc) || r.long_term_pictures.iter().any(|p| p.poc == frame_poc)).unwrap_or(false) || *id == current_id);
            }

            #[allow(non_snake_case)] let ReferenceFrames = from_iter_or(frames.iter().filter_map(|frame| {
                let frame_poc = frame.poc?;
                Some(va::VAPictureHEVC{
                    picture_id: frame.id,
                    pic_order_cnt: frame_poc as i32,
                    flags: reference.map(|r|
                        r.short_term_pictures.iter().filter_map(|&hevc::ShortTermReferencePicture{delta_poc, used}| used.then(|| (current_poc as i32+delta_poc as i32) as u32)).find(|&poc| poc == frame_poc).map(|poc|
                            match poc.cmp(&current_poc) {
                                std::cmp::Ordering::Less => va::VA_PICTURE_HEVC_RPS_ST_CURR_BEFORE,
                                std::cmp::Ordering::Greater => va::VA_PICTURE_HEVC_RPS_ST_CURR_AFTER,
                                _ => unreachable!()
                            }
                        ).unwrap_or(0) |
                        if r.long_term_pictures.iter().any(|&hevc::LongTermReferencePicture{poc, used}| used && poc == frame_poc) { va::VA_PICTURE_HEVC_RPS_LT_CURR } else {0}
                    ).unwrap_or(0),
                    va_reserved:[0;_]
                })
            }), va::VAPictureHEVC{picture_id:0xFFFFFFFF,pic_order_cnt:0,flags:va::VA_PICTURE_HEVC_INVALID,va_reserved:[0;_]}); // before setting current.poc

            /*let current = frames.iter_mut().find(|f| f.poc.is_none()).unwrap();
            assert!(current.poc.is_none());
            current.poc = Some(current_poc); // after setting ReferenceFrames*/
            frames.iter_mut().find(|f| f.id == current_id).unwrap().poc = Some(current_poc); // after setting ReferenceFrames

            render(va::VABufferType_VAPictureParameterBufferType, bytes_of(&va::VAPictureParameterBufferHEVC{
                CurrPic: va::VAPictureHEVC {
                    picture_id: current_id,
                    pic_order_cnt: current_poc as i32,
                    flags: 0,
                    va_reserved: [0; 4],
                },
                ReferenceFrames,
                pic_width_in_luma_samples: sps.width,
                pic_height_in_luma_samples: sps.height,
                pic_fields: va::_VAPictureParameterBufferHEVC__bindgen_ty_1{bits: va::_VAPictureParameterBufferHEVC__bindgen_ty_1__bindgen_ty_1{_bitfield_align_1: [], _bitfield_1:
                    va::_VAPictureParameterBufferHEVC__bindgen_ty_1__bindgen_ty_1::new_bitfield_1(
                        sps.chroma_format_idc as _,
                        sps.separate_color_plane as _,
                        sps.pulse_code_modulation.is_some() as _,
                        sps.scaling_list.is_some() as _,
                        pps.transform_skip as _,
                        sps.asymmetric_motion_partitioning as _,
                        sps.strong_intra_smoothing as _,
                        pps.sign_data_hiding as _,
                        pps.constrained_intra_prediction as _,
                        pps.diff_cu_qp_delta_depth.is_some() as _,
                        pps.weighted_prediction as _,
                        pps.weighted_biprediction as _,
                        pps.transquant_bypass as _,
                        pps.tiles.1.is_some() as _,
                        pps.tiles.0/*entropy_coding_sync*/ as _,
                        pps.loop_filter_across_slices as _,
                        pps.tiles.1.as_ref().map(|t| t.loop_filter_across_tiles).unwrap_or(/*false*/true) as _,
                        sps.pulse_code_modulation.as_ref().map(|pcm| pcm.loop_filter_disable).unwrap_or(false) as _,
                        false as _, //NoPicReordering
                        false as _, //NoBiPred
                        0, //Reserved
                    )
                }},
                log2_min_luma_coding_block_size_minus3: sps.log2_min_coding_block_size - 3,
                sps_max_dec_pic_buffering_minus1: sps.layer_ordering.last().unwrap().max_dec_pic_buffering - 1,
                log2_diff_max_min_luma_coding_block_size: sps.log2_diff_max_min_coding_block_size,
                log2_min_transform_block_size_minus2: sps.log2_min_transform_block_size - 2,
                log2_diff_max_min_transform_block_size: sps.log2_diff_max_min_transform_block_size,
                max_transform_hierarchy_depth_inter: sps.max_transform_hierarchy_depth_inter,
                max_transform_hierarchy_depth_intra: sps.max_transform_hierarchy_depth_intra,
                num_short_term_ref_pic_sets: sps.short_term_reference_picture_sets.len() as u8,
                num_long_term_ref_pic_sps: sps.long_term_reference_picture_set.as_ref().map(|s| s.len()).unwrap_or(0) as u8,
                num_ref_idx_l0_default_active_minus1: pps.num_ref_idx_default_active[0] - 1,
                num_ref_idx_l1_default_active_minus1: pps.num_ref_idx_default_active[1] - 1,
                init_qp_minus26: pps.init_qp_minus26,
                pcm_sample_bit_depth_luma_minus1: sps.pulse_code_modulation.as_ref().map(|p| p.bit_depth - 1).unwrap_or(/*0*/0xFF),
                pcm_sample_bit_depth_chroma_minus1: sps.pulse_code_modulation.as_ref().map(|p| p.bit_depth_chroma - 1).unwrap_or(/*0*/0xFF),
                log2_min_pcm_luma_coding_block_size_minus3: sps.pulse_code_modulation.as_ref().map(|p| p.log2_min_coding_block_size - 3).unwrap_or(/*0*/0xFD),
                log2_diff_max_min_pcm_luma_coding_block_size: sps.pulse_code_modulation.as_ref().map(|p| p.log2_diff_max_min_coding_block_size).unwrap_or(0),
                diff_cu_qp_delta_depth: pps.diff_cu_qp_delta_depth.unwrap_or(0),
                pps_cb_qp_offset: pps.cb_qp_offset,
                pps_cr_qp_offset: pps.cr_qp_offset,
                pps_beta_offset_div2: pps.deblocking_filter.as_ref().map(|f| f.1.as_ref().map(|f| f.beta_offset / 2)).flatten().unwrap_or(0),
                pps_tc_offset_div2: pps.deblocking_filter.as_ref().map(|f| f.1.as_ref().map(|f| f.tc_offset / 2)).flatten().unwrap_or(0),
                log2_parallel_merge_level_minus2: pps.log2_parallel_merge_level - 2,
                bit_depth_luma_minus8: sps.bit_depth - 8,
                bit_depth_chroma_minus8: sps.bit_depth - 8,
                log2_max_pic_order_cnt_lsb_minus4: sps.log2_max_poc_lsb - 4,
                num_extra_slice_header_bits: pps.num_extra_slice_header_bits,
                slice_parsing_fields: va::_VAPictureParameterBufferHEVC__bindgen_ty_2{bits: va::_VAPictureParameterBufferHEVC__bindgen_ty_2__bindgen_ty_1{_bitfield_align_1: [], _bitfield_1:
                    va::_VAPictureParameterBufferHEVC__bindgen_ty_2__bindgen_ty_1::new_bitfield_1(
                        pps.lists_modification as _,
                        sps.long_term_reference_picture_set.as_ref().map(|s| !s.is_empty()).unwrap_or(false) as _,
                        sps.temporal_motion_vector_predictor as _,
                        pps.cabac_init as _,
                        pps.output as _,
                        pps.dependent_slice_segments as _,
                        pps.slice_chroma_qp_offsets as _,
                        sps.sample_adaptive_offset as _,
                        /*override:*/ pps.deblocking_filter.as_ref().map(|f| f.0).unwrap_or(false) as _,
                        /*disable:*/ pps.deblocking_filter.as_ref().map(|f| f.1.is_none()).unwrap_or(false) as _,
                        pps.slice_header_extension as _,
                        hevc::Intra_Random_Access_Picture(unit) as _,
                        hevc::Instantaneous_Decoder_Refresh(unit) as _,
                        hevc::Intra_Random_Access_Picture(unit) as _,
                        0
                    ),
                }},
                num_tile_columns_minus1: pps.tiles.1.as_ref().map(|t| t.columns.len() - 1).unwrap_or(0) as u8,
                num_tile_rows_minus1: pps.tiles.1.as_ref().map(|t| t.rows.len() - 1).unwrap_or(0) as u8,
                column_width_minus1: pps.tiles.1.as_ref().map(|t| from_iter(t.columns.into_iter().map(|w| w-1))).unwrap_or_default(),
                row_height_minus1: pps.tiles.1.as_ref().map(|t| from_iter(t.rows.into_iter().map(|h| h-1))).unwrap_or_default(),
                st_rps_bits: reference.map(|r| r.short_term_picture_set_encoded_bits_len_skip).flatten().unwrap_or(0),
                va_reserved: [0; _]
            }));
            if let Some(_scaling_list) = pps.scaling_list.as_ref().or(sps.scaling_list.as_ref()) {
                unimplemented!();
                /*let mut buffer = 0;
                check(unsafe{va::vaCreateBuffer(va, context, va::VABufferType_VAIQMatrixBufferType, std::mem::size_of::<va::VAIQMatrixBufferHEVC> as u32, 1, &mut va::VAIQMatrixBufferHEVC{
                    ScalingList4x4: scaling_list.x4,
                    ScalingList8x8: scaling_list.x8,
                    ScalingListDC16x16: scaling_list.x16.map(|x| x.0),
                    ScalingList16x16: scaling_list.x16.map(|x| x.1),
                    ScalingListDC32x32: scaling_list.x32.map(|x| x.0),
                    ScalingList32x32: scaling_list.x32.map(|x| x.1),
                    va_reserved: [0; _],
                } as *const _ as *mut _, &mut buffer)});
                check(unsafe{va::vaRenderPicture(va, context, &buffer as *const _ as *mut _, 1)});*/
            }
            //*current.id
            current_id
        });
        let prediction_weights = slice_header.inter.as_ref().map(|s| s.prediction_weights.clone()).flatten().unwrap_or_default();
        let sh=&slice_header;
        render(va::VABufferType_VASliceParameterBufferType, bytes_of(&va::VASliceParameterBufferHEVC{
            slice_data_size: escaped_data.len() as u32,
            slice_data_offset: 0,
            slice_data_flag: va::VA_SLICE_DATA_FLAG_ALL,
            slice_data_byte_offset: slice_data_byte_offset as u32,
            slice_segment_address: slice_segment_address.unwrap_or(0) as u32,
            RefPicList: slice_header.reference.as_ref().map(|r| {
                pub fn list<T>(iter: impl std::iter::IntoIterator<Item=T>) -> Box<[T]> { iter.into_iter().collect() }
                pub fn sort_by<T>(mut s: Box<[T]>, f: impl Fn(&T,&T)->std::cmp::Ordering) -> Box<[T]> { s.sort_by(f); s }
                pub fn map<T,U>(iter: impl std::iter::IntoIterator<Item=T>, f: impl Fn(T)->U) -> Box<[U]> { list(iter.into_iter().map(f)) }
                let ref frames = frames.iter().filter(|frame| frame.poc.map(|poc| poc != r.poc).unwrap_or(false)).map(|frame| frame.poc.unwrap()).collect::<Box<_>>();
                fn index(frames: &[u32], ref_poc: u32) -> usize { frames.iter().position(|&poc| poc == ref_poc).unwrap() }
                fn stps(frames: &[u32], r: &hevc::SHReference, f: impl Fn(std::cmp::Ordering)->std::cmp::Ordering) -> Box<[u8]> {
                    let current_poc = r.poc;
                    map(&*sort_by(
                        list(r.short_term_pictures.into_iter().filter_map(|&hevc::ShortTermReferencePicture{delta_poc, used}|(used && f(delta_poc.cmp(&0)) == std::cmp::Ordering::Greater).then(|| delta_poc))),
                        |a,b| f(a.cmp(b))
                    ), |delta_poc| index(frames, (current_poc as i32+*delta_poc as i32) as u32) as u8)
                }
                let ref st_after = stps(frames, r, |o| o);
                let ref st_before = stps(frames, r, |o| o.reverse());
                let ref lt = map(sort_by(list(r.long_term_pictures.into_iter().filter_map(|&hevc::LongTermReferencePicture{poc, used}| used.then(|| poc))), Ord::cmp).into_iter(), |&poc| index(frames, poc) as u8);
                let list = |list:[_;3]| list.map(|e:&Box<[_]>| e.into_iter()).into_iter().flatten().copied().collect::<Box<_>>();
                let rpl = MayB{p: list([st_before, st_after, lt]), b: matches!(sh.slice_type, SliceType::B).then(|| list([st_after, st_before, lt]))};
                let inter = slice_header.inter.as_ref().unwrap();
                assert!(inter.list_entry_lx.is_none());
                let rpl = if let Some(list_entry_lx) = slice_header.inter.as_ref().unwrap().list_entry_lx.as_ref() { rpl.zip(list_entry_lx).map(|(rpl, list_entry_lx)| list_entry_lx.as_ref().map(|list| map(list.into_iter(), |i| rpl[*i as usize])).unwrap_or(rpl)) } else { rpl.zip(&inter.active_references).map(|(rpl, active_references)| rpl.into_vec().into_iter().take(*active_references as usize).collect()) };
                let rpl = rpl.map(|list| from_iter_or(list.into_vec().into_iter(), 0xFF));
                [rpl.p, rpl.b.unwrap_or([0xFF; _])]
            }).unwrap_or([[0xFF; _]; _]),
            padding0: [0; 2],
            LongSliceFlags: va::_VASliceParameterBufferHEVC__bindgen_ty_1{fields: va::_VASliceParameterBufferHEVC__bindgen_ty_1__bindgen_ty_1{_bitfield_align_1: [], _bitfield_1: va::_VASliceParameterBufferHEVC__bindgen_ty_1__bindgen_ty_1::new_bitfield_1(
                last_slice_of_picture as _,
                dependent_slice_segment as _,
                sh.slice_type as u32,
                sh.color_plane_id as u32,
                sh.sample_adaptive_offset.luma as _,
                sh.sample_adaptive_offset.chroma.unwrap_or(false) as _,
                sh.inter.as_ref().map(|i| i.mvd_l1_zero).unwrap_or(false) as _,
                sh.inter.as_ref().map(|i| i.cabac_init).unwrap_or(false) as _,
                sh.reference.as_ref().map(|i| i.temporal_motion_vector_predictor).unwrap_or(false) as _,
                sh.deblocking_filter.is_none() as _,
                sh.inter.as_ref().map(|i| i.collocated_list.map(|(collocated,_)| !collocated)).flatten().unwrap_or(true) as _,
                sh.loop_filter_across_slices as _,
                /*reserved*/ 0
            )}},
            collocated_ref_idx: sh.inter.as_ref().map(|i| i.collocated_list.map(|(_,index)| index.unwrap_or(0))).flatten().unwrap_or(0xFF),
            num_ref_idx_l0_active_minus1: sh.inter.as_ref().map(|s| s.active_references.p-1).unwrap_or(0) as u8,
            num_ref_idx_l1_active_minus1: sh.inter.as_ref().map(|s| s.active_references.b.map(|len| len-1)).flatten().unwrap_or(0) as u8,
            slice_qp_delta: sh.qp_delta,
            slice_cb_qp_offset: sh.qp_offsets.map(|o| o.0).unwrap_or(0),
            slice_cr_qp_offset: sh.qp_offsets.map(|o| o.1).unwrap_or(0),
            slice_beta_offset_div2: sh.deblocking_filter.as_ref().map(|f| f.beta_offset / 2).unwrap_or(0),
            slice_tc_offset_div2: sh.deblocking_filter.as_ref().map(|f| f.tc_offset / 2).unwrap_or(0),
            luma_log2_weight_denom: prediction_weights.luma.log2_denom_weight,
            delta_chroma_log2_weight_denom: prediction_weights.chroma.as_ref().map(|c| c.log2_denom_weight as i8 - prediction_weights.luma.log2_denom_weight as i8).unwrap_or(0),
            delta_luma_weight_l0: prediction_weights.luma.pb.p.weight,
            luma_offset_l0: prediction_weights.luma.pb.p.offset,
            delta_chroma_weight_l0: prediction_weights.chroma.clone().unwrap_or_default().pb.p.weight,
            ChromaOffsetL0: prediction_weights.chroma.clone().unwrap_or_default().pb.p.offset,
            delta_luma_weight_l1: prediction_weights.luma.pb.b.unwrap_or_default().weight,
            luma_offset_l1: prediction_weights.luma.pb.b.unwrap_or_default().offset,
            delta_chroma_weight_l1: prediction_weights.chroma.clone().unwrap_or_default().pb.b.unwrap_or_default().weight,
            ChromaOffsetL1: prediction_weights.chroma.unwrap_or_default().pb.b.unwrap_or_default().offset,
            five_minus_max_num_merge_cand: sh.inter.as_ref().map(|s| 5 - s.max_num_merge_cand).unwrap_or(0),
            padding1: 0,
            num_entry_point_offsets: 0,
            entry_offset_to_subset_array: 0,
            slice_data_num_emu_prevn_bytes: 0,
            padding2: 0,
            va_reserved: [0; _],
            ..Default::default() // Helps zero initialize padding holes (but unreliable)
        }));
        render(va::VABufferType_VASliceDataBufferType, escaped_data);
        if last_slice_of_picture {
            {
                let context = context.context;
                check(unsafe{va::vaBeginPicture(va, context, current_id)});
                check(unsafe{va::vaRenderPicture(va, context, buffers.as_ptr() as *const _ as *mut _, buffers.len() as _)}); buffers.clear(); println!("buffer_count {buffer_count}");
                check(unsafe{va::vaEndPicture(va, context)});
            }

            /*check(unsafe{va::vaSyncSurface(va, current_id)});
            let ref format = va::VAImageFormat{fourcc: va::VA_FOURCC_P010, byte_order: va::VA_LSB_FIRST, bits_per_pixel: 24, depth: 0, red_mask: 0, green_mask: 0, blue_mask: 0, alpha_mask: 0, va_reserved: [0; _]};
            let mut image = va::VAImage{image_id: va::VA_INVALID_ID, ..Default::default()};
            let size = xy{x: sps.width as u32, y: sps.height as u32};
            check(unsafe{va::vaCreateImage(va, format as *const _ as *mut _, size.x as _, size.y as _, &mut image)});
            check(unsafe{va::vaGetImage(va, current_id, 0, 0, size.x, size.y, image.image_id)});
            let mut address : *const u16 = std::ptr::null();
            check(unsafe{va::vaMapBuffer(va, image.buf, &mut address as *mut * const _ as *mut *mut _)});*/
            //let Y = unsafe{std::slice::from_raw_parts(address, (size.y*size.x) as usize)};
            /*let b : &[u16] = bytemuck::cast_slice(&av.as_ref().unwrap()[0..((size.x*size.y)*2)as usize]);
            for y in 0..size.y { for x in 0..size.x { assert_eq!(a[(y*size.x+x) as usize]>>6, b[(y*size.x+x) as usize]>>6, "{:?}", (x, y)); } }*/

            use image::Image;
            struct Widget<'t> (Image<&'t [u16]>);
            impl ui::Widget for Widget<'_> { fn paint(&mut self, target: &mut ui::Target, _: ui::size, _: ui::int2) -> ui::Result<()> {
                use {vector::xy, std::cmp::min};
                for y in 0..min(self.0.size.y, target.size.y) { for x in 0..min(self.0.size.x, target.size.x) { target[xy{x,y}] = ((((self.0[xy{x,y}]>>6)+0x80)/0x101) as u8).into(); }}
                Ok(())
            } }
            //ui::run(&mut Widget(Image::new(size, Y))).unwrap();
            context.current_id = None; // Signals that next slice is a new picture
            //av.pop_front();
        }
    });

    /*use drm::{Device, control::{self, Device as _, dumbbuffer::DumbBuffer, *}, buffer::{DrmFourcc, Handle, Buffer, PlanarBuffer}};
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
    struct Buffer2<'t>(&'t DumbBuffer);
    impl PlanarBuffer for Buffer2<'_> {
        fn size(&self) -> (u32, u32) { self.0.size() }
        fn format(&self) -> DrmFourcc { self.0.format() }
        fn pitches(&self) -> [u32; 4] { [self.0.pitch(),0,0,0] }
        fn handles(&self) -> [Option<Handle>; 4] { [Some(self.0.handle()),None,None,None] }
        fn offsets(&self) -> [u32; 4] { [0; 4] }
    }
    let fb = card.add_planar_framebuffer(&Buffer2(&db), &[None; 4], 0).unwrap();
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
    atomic_req.add_property(plane, find_prop_id(&card, plane, "CRTC_H").expect("Could not get CRTC_H"), property::Value::UnsignedRange(mode.size().1 as u64));*/

    Ok(())
}
