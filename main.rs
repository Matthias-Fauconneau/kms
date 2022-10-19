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

    use ffmpeg::*;
    let path = std::env::args().skip(1).next().unwrap_or(std::env::var("HOME")?+"/input.mkv");
    #[track_caller] fn check(status: std::ffi::c_int) -> std::ffi::c_int { if status!=0 { let mut s=[0;AV_ERROR_MAX_STRING_SIZE]; unsafe{av_strerror(status,s.as_mut_ptr(),s.len());} panic!("{}", unsafe{std::ffi::CStr::from_ptr(s.as_ptr())}.to_str().unwrap()); } else { status } }
    let mut context = std::ptr::null_mut();
    let path = std::ffi::CString::new(path).unwrap();
    check(unsafe{avformat_open_input(&mut context, path.as_ptr(), std::ptr::null_mut(), std::ptr::null_mut())});
    check(unsafe{avformat_find_stream_info(context, std::ptr::null_mut())});
    let mut decoder = std::ptr::null_mut();
    let video_stream = check(unsafe{av_find_best_stream(context, AVMediaType::AVMEDIA_TYPE_VIDEO, -1, -1, &mut decoder, 0)});
    for i in 0.. {
        let config = &unsafe{*avcodec_get_hw_config(decoder, i)};
        if (config.methods & (AV_CODEC_HW_CONFIG_METHOD_HW_DEVICE_CTX as i32) != 0) && config.device_type == AVHWDeviceType::AV_HWDEVICE_TYPE_VAAPI { panic!("{:?}", config.pix_fmt); }
    }
    let video = &mut unsafe{**(*context).streams.offset(video_stream as isize)};
    let decoder_context = unsafe{avcodec_alloc_context3(decoder)};
    check(unsafe{avcodec_parameters_to_context(decoder_context, video.codecpar)});
    unsafe extern "C" fn get_format(_s: *mut AVCodecContext, _fmt: *const AVPixelFormat) -> AVPixelFormat { unimplemented!(); }
    unsafe{*decoder_context}.get_format  = Some(get_format);
    let mut hw_device_context = std::ptr::null_mut();
    check(unsafe{av_hwdevice_ctx_create(&mut hw_device_context, AVHWDeviceType::AV_HWDEVICE_TYPE_VAAPI, std::ptr::null(), std::ptr::null_mut(), 0)});
    unsafe{*decoder_context}.hw_device_ctx = unsafe{av_buffer_ref(hw_device_context)};
    check(unsafe{avcodec_open2(decoder_context, decoder, std::ptr::null_mut())});

    let packet = unsafe{av_packet_alloc()};
    while unsafe{av_read_frame(context, packet)} >= 0 {
        if video_stream == unsafe{*packet}.stream_index {
            check(unsafe{avcodec_send_packet(decoder_context, packet)});
            let frame = unsafe{av_frame_alloc()};
            check(unsafe{avcodec_receive_frame(decoder_context, frame)});
            /*av_hwframe_transfer_data(sw_frame, frame, 0)
            size = av_image_get_buffer_size(frame.format, frame.width, frame.height, 1);
            buffer = av_malloc(size);
            av_image_copy_to_buffer(buffer, size, frame.data, frame.linesize, frame.format, frame.width, frame.height, 1);
            let mut map = card.map_dumb_buffer(&mut db).expect("Could not map dumbbuffer");
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
        unsafe{av_packet_unref(packet)};
    }
    //card.destroy_framebuffer(fb).unwrap(); card.destroy_dumb_buffer(db).unwrap();*/
    Ok(())
}
