use drm::{Device, control::{self, Device as _, *}, buffer::DrmFourcc};
struct Card(std::fs::File);
impl std::os::unix::io::AsRawFd for Card { fn as_raw_fd(&self) -> std::os::unix::io::RawFd { self.0.as_raw_fd() } }
impl Device for Card {}
impl control::Device for Card {}

fn find_prop_id<T: ResourceHandle>(
    card: &Card,
    handle: T,
    name: &'static str,
) -> Option<property::Handle> {
    let props = card
        .get_properties(handle)
        .expect("Could not get props of connector");
    let (ids, _vals) = props.as_props_and_values();
    ids.iter()
        .find(|&id| {
            let info = card.get_property(*id).unwrap();
            info.name().to_str().map(|x| x == name).unwrap_or(false)
        })
        .cloned()
}

pub fn main() {
    let card = Card(std::fs::OpenOptions::new().read(true).write(true).open("/dev/dri/card0").unwrap());

    card.set_client_capability(drm::ClientCapability::UniversalPlanes, true)
        .expect("Unable to request UniversalPlanes capability");
    card.set_client_capability(drm::ClientCapability::Atomic, true)
        .expect("Unable to request Atomic capability");

    // Load the information.
    let res = card
        .resource_handles()
        .expect("Could not load normal resource ids.");
    let coninfo: Vec<connector::Info> = res
        .connectors()
        .iter()
        .flat_map(|con| card.get_connector(*con, true))
        .collect();
    let crtcinfo: Vec<crtc::Info> = res
        .crtcs()
        .iter()
        .flat_map(|crtc| card.get_crtc(*crtc))
        .collect();

    // Filter each connector until we find one that's connected.
    let con = coninfo
        .iter()
        .find(|&i| i.state() == connector::State::Connected)
        .expect("No connected connectors");

    // Get the first (usually best) mode
    let &mode = con.modes().get(1).expect("No modes found on connector");

    let (width, height) = mode.size();
    let (width, height) = (width.into(), height.into());

    // Find a crtc and FB
    let crtc = crtcinfo.get(0).expect("No crtcs found");

    // Select the pixel format
    //let fmt = DrmFourcc::Xrgb8888;
    let fmt = DrmFourcc::Xrgb2101010;

    // Create a DB
    // If buffer resolution is above display resolution, a ENOSPC (not enough GPU memory) error may
    // occur
    let mut db = card
        .create_dumb_buffer((width, height), fmt, 32)
        .expect("Could not create dumb buffer");

    // Map it and grey it out.
    {
        let mut map = card.map_dumb_buffer(&mut db).expect("Could not map dumbbuffer");
        let map = bytemuck::cast_slice_mut(map.as_mut());
        let mut draw_gradient = |y0,y1,r,g,b| for x in 0..width {
            let f = x as f32/width as f32;
            let r = (1024.*f*r) as u32; // TODO: PQ
            let g = (1024.*f*g) as u32;
            let b = (1024.*f*b) as u32;
	    assert!(r < 1024 && g < 1024 && b<1024,"{:?}",(r,g,b));
	    let rgb = b | g<<10 | r<<20;
            //let rgb = b&0xFF | (b>>8)<<14;// | g<<10 | r<<20;
	    //let rgb = b>>2 | (g>>2)<<8 | (r>>2)<<16;
            for y in y0..y1 { map[(y*width+x) as usize] = rgb; }
        };
        draw_gradient(0,height/4,1.,0.,0.);
        draw_gradient(1*height/4,2*height/4,0.,1.,0.);
        draw_gradient(2*height/4,3*height/4,0.,0.,1.);
        draw_gradient(3*height/4,height,1.,1.,1.);
    }
    // Create an FB:
    let fb = card
        .add_framebuffer(&db, None, 0)
        .expect("Could not create FB");

    let planes = card.plane_handles().expect("Could not list planes");
    let (better_planes, compatible_planes): (
        Vec<control::plane::Handle>,
        Vec<control::plane::Handle>,
    ) = planes
        .iter()
        .filter(|&&plane| {
            card.get_plane(plane)
                .map(|plane_info| {
                    let compatible_crtcs = res.filter_crtcs(plane_info.possible_crtcs());
                    compatible_crtcs.contains(&crtc.handle())
                })
                .unwrap_or(false)
        })
        .partition(|&&plane| {
            if let Ok(props) = card.get_properties(plane) {
                let (ids, vals) = props.as_props_and_values();
                for (&id, &val) in ids.iter().zip(vals.iter()) {
                    if let Ok(info) = card.get_property(id) {
                        if info.name().to_str().map(|x| x == "type").unwrap_or(false) {
                            return val == (drm::control::PlaneType::Primary as u32).into();
                        }
                    }
                }
            }
            false
        });
    let plane = *better_planes.get(0).unwrap_or(&compatible_planes[0]);

    println!("{:#?}", mode);
    println!("{:#?}", fb);
    println!("{:#?}", db);
    println!("{:#?}", plane);

    let mut atomic_req = atomic::AtomicModeReq::new();
    atomic_req.add_property(
        con.handle(),
        find_prop_id(&card, con.handle(), "CRTC_ID").expect("Could not get CRTC_ID"),
        property::Value::CRTC(Some(crtc.handle())),
    );
    let blob = card
        .create_property_blob(&mode)
        .expect("Failed to create blob");
    atomic_req.add_property(
        crtc.handle(),
        find_prop_id(&card, crtc.handle(), "MODE_ID").expect("Could not get MODE_ID"),
        blob,
    );
    atomic_req.add_property(
        crtc.handle(),
        find_prop_id(&card, crtc.handle(), "ACTIVE").expect("Could not get ACTIVE"),
        property::Value::Boolean(true),
    );
    atomic_req.add_property(
        plane,
        find_prop_id(&card, plane, "FB_ID").expect("Could not get FB_ID"),
        property::Value::Framebuffer(Some(fb)),
    );
    atomic_req.add_property(
        plane,
        find_prop_id(&card, plane, "CRTC_ID").expect("Could not get CRTC_ID"),
        property::Value::CRTC(Some(crtc.handle())),
    );
    atomic_req.add_property(
        plane,
        find_prop_id(&card, plane, "SRC_X").expect("Could not get SRC_X"),
        property::Value::UnsignedRange(0),
    );
    atomic_req.add_property(
        plane,
        find_prop_id(&card, plane, "SRC_Y").expect("Could not get SRC_Y"),
        property::Value::UnsignedRange(0),
    );
    atomic_req.add_property(
        plane,
        find_prop_id(&card, plane, "SRC_W").expect("Could not get SRC_W"),
        property::Value::UnsignedRange((mode.size().0 as u64) << 16),
    );
    atomic_req.add_property(
        plane,
        find_prop_id(&card, plane, "SRC_H").expect("Could not get SRC_H"),
        property::Value::UnsignedRange((mode.size().1 as u64) << 16),
    );
    atomic_req.add_property(
        plane,
        find_prop_id(&card, plane, "CRTC_X").expect("Could not get CRTC_X"),
        property::Value::SignedRange(0),
    );
    atomic_req.add_property(
        plane,
        find_prop_id(&card, plane, "CRTC_Y").expect("Could not get CRTC_Y"),
        property::Value::SignedRange(0),
    );
    atomic_req.add_property(
        plane,
        find_prop_id(&card, plane, "CRTC_W").expect("Could not get CRTC_W"),
        property::Value::UnsignedRange(mode.size().0 as u64),
    );
    atomic_req.add_property(
        plane,
        find_prop_id(&card, plane, "CRTC_H").expect("Could not get CRTC_H"),
        property::Value::UnsignedRange(mode.size().1 as u64),
    );

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
        display_primaries: [xy(0.6835, 0.3154), xy(0.1962, 0.7333), xy(0.1416, 0.0449)],
        white_point: xy(0.3105, 0.3232),
        min_display_mastering_luminance: 1250, //0.125 / 0.0001,
        max_display_mastering_luminance: 508,
        max_content_light_level: 508,
        max_frame_average_light_level: 508
	})
    .expect("Failed to create blob");
    atomic_req.add_property(con.handle(), find_prop_id(&card, con.handle(), "HDR_OUTPUT_METADATA").expect("Could not get HDR_OUTPUT_METADATA"), hdr_output_metadata);
    atomic_req.add_property(con.handle(), find_prop_id(&card, con.handle(), "max bpc").expect("Could not get max bpc"), property::Value::UnsignedRange(10));
    card.atomic_commit(AtomicCommitFlags::ALLOW_MODESET, atomic_req).expect("Failed to set mode");
    ::std::thread::sleep(std::time::Duration::from_millis(16000));
    card.destroy_framebuffer(fb).unwrap();
    card.destroy_dumb_buffer(db).unwrap();
}
