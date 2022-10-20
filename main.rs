pub fn main() -> Result<(), Box<dyn std::error::Error>> {
    use drm::{Device, control::{self, Device as _, dumbbuffer::DumbBuffer, *}, buffer::{DrmFourcc, Handle, Buffer, PlanarBuffer}};
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
    atomic_req.add_property(plane, find_prop_id(&card, plane, "CRTC_H").expect("Could not get CRTC_H"), property::Value::UnsignedRange(mode.size().1 as u64));

    use ffmpeg::*;
    unsafe {avdevice_register_all()};
    let path = std::env::args().skip(1).next().unwrap_or(std::env::var("HOME")?+"/input.mkv");
    #[track_caller] fn check(status: std::ffi::c_int) -> std::ffi::c_int { if status!=0 { let mut s=[0;AV_ERROR_MAX_STRING_SIZE]; unsafe{av_strerror(status,s.as_mut_ptr(),s.len());} panic!("{}", unsafe{std::ffi::CStr::from_ptr(s.as_ptr())}.to_str().unwrap()); } else { status } }
    let mut context = std::ptr::null_mut();
    let path = std::ffi::CString::new(path).unwrap();
    //unsafe{av_log_set_level(AV_LOG_TRACE)};
    check(unsafe{avformat_open_input(&mut context, path.as_ptr(), std::ptr::null_mut(), std::ptr::null_mut())});
    check(unsafe{avformat_find_stream_info(context, std::ptr::null_mut())});
    let mut decoder = std::ptr::null_mut();
    let video_stream = check(unsafe{av_find_best_stream(context, AVMediaType::AVMEDIA_TYPE_VIDEO, -1, -1, &mut decoder, 0)});
    assert!(decoder != std::ptr::null_mut());
    let video = &mut unsafe{&**(*context).streams.offset(video_stream as isize)};
    let decoder_context = unsafe{avcodec_alloc_context3(decoder)};
    check(unsafe{avcodec_parameters_to_context(decoder_context, video.codecpar)});
    extern "C" fn get_format(_s: *mut AVCodecContext, _fmt: *const AVPixelFormat) -> AVPixelFormat { AVPixelFormat::AV_PIX_FMT_VAAPI_VLD  }
    unsafe{&mut *decoder_context}.get_format  = Some(get_format);
    let mut hw_device_context = std::ptr::null_mut();
    check(unsafe{av_hwdevice_ctx_create(&mut hw_device_context, AVHWDeviceType::AV_HWDEVICE_TYPE_VAAPI, std::ptr::null(), std::ptr::null_mut(), 0)});
    assert!(hw_device_context != std::ptr::null_mut());
    unsafe{&mut *decoder_context}.hw_device_ctx = unsafe{av_buffer_ref(hw_device_context)};
    assert!(unsafe{&mut *decoder_context}.hw_device_ctx != std::ptr::null_mut());
    check(unsafe{avcodec_open2(decoder_context, decoder, std::ptr::null_mut())});

    let packet = unsafe{av_packet_alloc()};
    while unsafe{av_read_frame(context, packet)} >= 0 {
        if video_stream == unsafe{&*packet}.stream_index {
            check(unsafe{avcodec_send_packet(decoder_context, packet)});
            assert!(unsafe{&mut *decoder_context}.hwaccel != std::ptr::null());
            let mut frame = unsafe{av_frame_alloc()};
            let status = unsafe{avcodec_receive_frame(decoder_context, frame)};
            if status == -EAGAIN { continue; }
            check(status);
            let surface = unsafe{&*frame}.data[3] as u32; // VASurfaceID

            mod va {
                #![allow(dead_code,non_camel_case_types,non_upper_case_globals,improper_ctypes,non_snake_case)]
                include!(concat!(env!("OUT_DIR"), "/bindings.rs"));
            }
            use va::*;
            #[track_caller] fn check(status: VAStatus) { if status!=0 { panic!("{:?}", unsafe{std::ffi::CStr::from_ptr(va::vaErrorStr(status))}); } }

            let (va, context) = {
                let hwaccel_priv_data = unsafe{&*((&*decoder_context).internal as *const va::AVCodecInternal)}.hwaccel_priv_data as *const VAAPIDecodeContext;
                assert!(!hwaccel_priv_data.is_null());
                let hwaccel_priv_data = unsafe{&*hwaccel_priv_data};
                dbg!(unsafe{*hwaccel_priv_data.hwctx}, hwaccel_priv_data.va_context)
            };

            let mut pipeline = VA_INVALID_ID;
            check(unsafe{vaCreateBuffer(va, context, VABufferType_VAProcPipelineParameterBufferType, std::mem::size_of::<VAProcPipelineParameterBuffer>() as _, 1, &VAProcPipelineParameterBuffer{surface,
                ..Default::default()
            } as *const _ as *mut _, &mut pipeline as *mut _)});
            let mut rgb = VA_INVALID_SURFACE;
            check(unsafe{va::vaCreateSurfaces(va, va::VA_RT_FORMAT_YUV420_10, width as _, height as _, &mut rgb as *mut _, 1, std::ptr::null_mut(), 0)});
            check(unsafe{vaBeginPicture(va, context, rgb)});
            check(unsafe{vaRenderPicture(va, context, &pipeline as *const _ as *mut _, 1)});
            check(unsafe{vaEndPicture(va, context)});

            let mut descriptor = va::VADRMPRIMESurfaceDescriptor::default();
            check(unsafe{va::vaExportSurfaceHandle(va, rgb, va::VA_SURFACE_ATTRIB_MEM_TYPE_DRM_PRIME_2, va::VA_EXPORT_SURFACE_READ_ONLY | va::VA_EXPORT_SURFACE_SEPARATE_LAYERS, &mut descriptor as *mut _ as *mut _)});

            unsafe{av_frame_free(&mut frame as *mut _)};
            //if card.page_flip(crtc.handle(), fb, PageFlipFlags::empty(), None).is_ok() { println!("flip!"); } else { println!("no flip"); }
        }
        unsafe{av_packet_unref(packet)};
    }
    //card.destroy_framebuffer(fb).unwrap(); card.destroy_dumb_buffer(db).unwrap();*/
    Ok(())
}
