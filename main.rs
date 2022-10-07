#![feature(int_log,unchecked_math,generic_arg_infer)]
#![allow(dead_code,unreachable_code,unused_variables)]

use nom::combinator::iterator;
use nom::number::complete::be_u32;
fn unit(input: &[u8]) -> nom::IResult<&[u8], &[u8]> { let (rest, length) = be_u32(input)?; let length = length as usize; Ok((&rest[length..], &rest[..length])) }

pub const AUD: u8 = 35;
pub const SEI_PREFIX: u8 = 39;
pub const VPS: u8 = 32;
pub const SPS: u8 = 33;
pub const PPS: u8 = 34;
pub const IDR_W_RADL: u8 = 19;
pub const IDR_N_LP: u8 = 20;
pub const TRAIL_R: u8 = 1;
pub const TRAIL_N: u8 = 0;

#[allow(non_snake_case)] fn IRAP(unit_type: u8) -> bool { 16 <= unit_type && unit_type <= 23 }
#[allow(non_snake_case)] fn IDR(unit_type: u8) -> bool { matches!(unit_type, IDR_W_RADL|IDR_N_LP) }

mod bit; use bit::BitReader;

fn profile_tier_level(s: &mut BitReader, max_layers: usize) {
    let _profile_space = s.bits(2);
    let _tier = s.bit();
    let profile = s.bits(5);
    enum Profile { Main = 1, Main10 }
    assert!(profile == Profile::Main10 as _);
    let _profile_compatibility = s.u32();
    let _progressive_source = s.bit();
    let _interlaced_source = s.bit();
    let _non_packed_constraint = s.bit();
    let _frame_only_constraint = s.bit();
    /*let check_profile = |id: u8| -> bool { profile == id || profile_compatibility&(1<<(31-id)) != 0 };
    if (check_profile(4) || check_profile(5) || check_profile(6) || check_profile(7) || check_profile(8) || check_profile(9) || check_profile(10)) {
        let _max_12bit = s.bit();
        let _max_10bit = s.bit();
        let _max_8bit = s.bit();
        let _max_422chroma = s.bit();
        let _max_420chroma = s.bit();
        let _max_monochrome = s.bit();
        let _intra = s.bit();
        let _one_picture_only = s.bit();
        let _lower_bit_rate = s.bit();
        s.bits(34);*/
    s.bits(44);
    let _level_idc = s.u8();
    if max_layers > 1 {
        let layers : [_; 8] = std::array::from_fn(|_| (s.bit(), s.bit()));
        println!("{layers:?}");
        let _layers = layers.map(|(profile, level)| {
            if profile { s.bits(44); s.bits(43); }
            if level { s.u8(); }
        });
    }
}
#[derive(Default,Clone,Copy)] struct LayerOrdering { max_dec_pic_buffering: u8, num_reorder_pics: u8, max_latency_increase: u8 }
fn layer_ordering(s: &mut BitReader, max_layers: usize) -> Box<[LayerOrdering]> {
    (0 .. (if s.bit() { max_layers } else { 1 })).map(|_| LayerOrdering{
        max_dec_pic_buffering: 1 + s.ue() as u8,
        num_reorder_pics: s.ue() as u8,
        max_latency_increase: {let mut v = s.ue(); if v > 0 { v -= 1; } v} as u8,
    }).collect()
}

fn hrd_parameters(s: &mut BitReader, common_inf: bool, max_layers: usize) {
    let mut nal_params = false;
    let mut vcl_params = false;
    let mut subpic_params = false;
    if common_inf {
        nal_params = s.bit();
        vcl_params = s.bit();
        if nal_params || vcl_params {
            subpic_params = s.bit();
            if subpic_params {
                s.u8(); // tick_divisor_minus2
                s.bits(5); // du_cpb_removal_delay_increment_length_minus1
                s.bit(); // sub_pic_cpb_params_in_pic_timing_sei
                s.bits(5); // dpb_output_delay_du_length_minus1
            }
            s.bits(4); // bit_rate_scale
            s.bits(4); // cpb_size_scale
            if subpic_params {
                s.bits(4); // cpb_size_du_scale
            }
            s.bits(5); // initial_cpb_removal_delay_length_minus1
            s.bits(5); // au_cpb_removal_delay_length_minus1
            s.bits(5); // dpb_output_delay_length_minus1
        }
    }
    for _ in 0..max_layers {
        let fixed_rate = if s.bit() { true } else { s.bit() };
        let low_delay = if fixed_rate { s.ue(); false } else { s.bit() };
        let nb_cpb = if !low_delay { s.ue() + 1 } else { 1 } as usize;
        fn sub_layer_hrd_parameter(s: &mut BitReader, nb_cpb: usize, subpic_params: bool) {
            for _ in 0..nb_cpb {
                s.ue(); // bit_rate_value_minus1
                s.ue(); // cpb_size_value_minus1
                if subpic_params {
                    s.ue(); // cpb_size_du_value_minus1
                    s.ue(); // bit_rate_du_value_minus1
                }
                s.bit(); // cbr
            }
        }
        if nal_params { sub_layer_hrd_parameter(s, nb_cpb, subpic_params); }
        if vcl_params { sub_layer_hrd_parameter(s, nb_cpb, subpic_params); }
    }
}

fn scaling_list(s: &mut BitReader) {
    if s.bit() {
        for i in 0..4 {
            let matrix_size = [6,6,6,2][i];
            for _j in 0..matrix_size {
                if !s.bit() {
                    /*prediction_matrix_id_delta[i][j] =*/ s.ue();
                } else {
                    if i >= 2 { /*dc_coef_minus8[i][j] =*/ s.se(); }
                    /*delta_coef[i][j] =*/ for _ in 0..[16,64,64,64][i] { s.se(); }
                }
            }
        }
    } else { Default::default() }
}

struct VPS {}
#[derive(Clone,Debug)] struct ShortTermReferencePicture { delta_poc: i8, used: bool }
struct LongTermReferencePicture { poc_lsb_sps: u8, used: bool } // _by_curr_pic_sps
struct PulseCodeModulation { bit_depth: u8, bit_depth_chroma: u8, log2_min_coding_block_size: u8, log2_diff_max_min_coding_block_size: u8, loop_filter_disable: bool }

struct SPS {
    separate_colour_plane: bool,
    chroma_format_idc: u8,
    width: u16,
    height: u16,
    bit_depth: u8,
    log2_max_poc_lsb: u8,
    layer_ordering: Box<[LayerOrdering]>, // 8
    log2_min_coding_block_size: u8,
    log2_diff_max_min_coding_block_size: u8,
    log2_min_transform_block_size: u8,
    log2_diff_max_min_transform_block_size: u8,
    max_transform_hierarchy_depth_inter: u8,
    max_transform_hierarchy_depth_intra: u8,
    scaling_list: Option<()>,
    asymmetric_motion_partitioning: bool,
    sample_adaptive_offset: bool,
    pulse_code_modulation: Option<PulseCodeModulation>,
    short_term_reference_picture_sets: Box<[Box<[ShortTermReferencePicture]>]>,
    long_term_reference_picture_set: Box<[LongTermReferencePicture]>,
    temporal_motion_vector_predictor: bool,
    strong_intra_smoothing: bool,
}

struct DeblockingFilter { beta_offset: i8, tc_offset: i8 }
struct Tiles {columns: Box<[u16]>, rows: Box<[u16]>, loop_filter_across_tiles: bool}

struct PPS {
    sps: usize,
    dependent_slice_segments: bool,
    output: bool,
    num_extra_slice_header_bits: u8,
    sign_data_hiding: bool,
    cabac_init: bool,
    num_ref_idx_l0_default_active: u8,
    num_ref_idx_l1_default_active: u8,
    pic_init_qp_minus26: i8,
    constrained_intra_prediction: bool,
    transform_skip: bool,
    diff_cu_qp_delta_depth: Option<u8>,
    cb_qp_offset: i8,
    cr_qp_offset: i8,
    pic_slice_chroma_qp_offsets: bool,
    weighted_prediction: bool,
    weighted_biprediction: bool,
    transquant_bypass: bool,
    tiles: (/*entropy_coding_sync*/bool, Option<Tiles>),
    loop_filter_across_slices: bool,
    deblocking_filter: Option<(/*deblocking_filter_override*/bool, Option<DeblockingFilter>)>,
    scaling_list: Option<()>,
    lists_modification: bool,
    log2_parallel_merge_level: u8,
    slice_header_extension: bool,
    pps_extension: bool
}

fn decode_short_term_reference_picture_set(s: &mut BitReader, sets: &[Box<[ShortTermReferencePicture]>]) -> Box<[ShortTermReferencePicture]> {
    if !sets.is_empty() && /*predict*/s.bit() {
        //let ref _reference = sets[sets.len()-1-(if is_last { s.ue() as usize } else { 0 })];
        let ref _set = &sets[sets.len()-1-s.ue() as usize];
        let _delta = if s.bit() { -1 } else { 1 } * (s.ue()+1) as i8;
        //let mut parse = |(reference,_)| { let used = s.bit(); if used || s.bit() { Some((reference+delta, used)) } else { None } };
        unimplemented!()//set.iter().filter_map(parse).chain(std::iter::once(parse((0,false)))).collect()
    } else {
        let negative = s.ue() as usize;
        let positive = s.ue() as usize;
        let mut set = Vec::with_capacity(negative+positive);
        use std::iter::successors; type P = ShortTermReferencePicture;
        set.extend(successors(Some(P{delta_poc:0,used:false}), |P{delta_poc,..}| Some(P{delta_poc: delta_poc -1 -s.ue() as i8, used: s.bit()})).take(negative));
        set.extend(successors(Some(P{delta_poc:0,used:false}), |P{delta_poc,..}| Some(P{delta_poc: delta_poc+1+s.ue() as i8, used: s.bit()})).take(positive));
        set.into_boxed_slice()
    }
}

fn main() -> std::result::Result<(), Box<dyn std::error::Error>> {
    let path = std::env::args().skip(1).next().unwrap_or(std::env::var("HOME")?+"/input.mkv");
    let input = unsafe{memmap::Mmap::map(&std::fs::File::open(path)?)}?;
    let input = &*input;
    let mut demuxer = matroska::demuxer::MkvDemuxer::new();
    let (input, ()) = demuxer.parse_until_tracks(input).unwrap();
    let tracks = demuxer.tracks.unwrap();
    let video = &tracks.tracks[0];
    assert!(video.codec_id == "V_MPEGH/ISO/HEVC");
    let mut vps : [_; 16] = Default::default();
    let mut sps : [_; 16] = Default::default();
    let mut pps = (|len|{let mut v = Vec::new(); v.resize_with(len, || None); v})(64); //: [_; 64] = Default::default();
    let mut poc_tid0 = 0;

    struct Card(std::fs::File);
    let card = Card(std::fs::OpenOptions::new().read(true).write(true).open("/dev/dri/card0").unwrap());
    impl std::os::unix::io::AsRawFd for Card { fn as_raw_fd(&self) -> std::os::unix::io::RawFd { self.0.as_raw_fd() } }
    use std::os::unix::io::AsRawFd;
    use va::va_display_drm::*;
    let va = unsafe{vaGetDisplayDRM(card.0.as_raw_fd())};
    #[repr(C)] struct AVVAAPIDeviceContext { display: va::VADisplay, driver_quirks: std::ffi::c_uint }
    extern "C" fn error(_user_context: *mut std::ffi::c_void, message: *const std::ffi::c_char) { panic!("{:?}", unsafe{std::ffi::CStr::from_ptr(message)}) }
    unsafe{va::vaSetErrorCallback(va, Some(error), std::ptr::null_mut())};
    extern "C" fn info(_user_context: *mut std::ffi::c_void, message: *const std::ffi::c_char) { println!("{:?}", unsafe{std::ffi::CStr::from_ptr(message)}); }
    unsafe{va::vaSetInfoCallback(va,  Some(info),  std::ptr::null_mut())};
    let (mut major, mut minor) = (0,0);
    #[track_caller] fn check(status: va::va_str::VAStatus) { if status!=0 { panic!("{:?}", unsafe{std::ffi::CStr::from_ptr(va::vaErrorStr(status))}); } }
    check(unsafe{va::vaInitialize(va, &mut major, &mut minor)});
    let mut profiles = Vec::with_capacity(unsafe{va::vaMaxNumProfiles(va)} as usize);
    let mut len = profiles.capacity() as i32;
    check(unsafe{va::vaQueryConfigProfiles(va, profiles.as_mut_ptr(), &mut len)});
    unsafe{profiles.set_len(len as usize)};
    let (profile, entrypoint) = profiles.into_iter().find_map(|profile| (profile /* != VAProfile_VAProfileNone*/== VAProfile_VAProfileHEVCMain10).then(||{
        let mut entrypoints = Vec::with_capacity(unsafe{va::vaMaxNumEntrypoints(va)} as usize);
        let mut len = entrypoints.capacity() as i32;
        check(unsafe{va::vaQueryConfigEntrypoints(va, profile, entrypoints.as_mut_ptr(), &mut len)});
        unsafe{entrypoints.set_len(len as usize)};
        (profile, entrypoints.into_iter().find(|&entrypoint| entrypoint == VAEntrypoint_VAEntrypointVLD).unwrap())
    })).unwrap();
    let mut config = 0; // va::VAConfigID
    check(unsafe{va::vaCreateConfig(va, profile, entrypoint, std::ptr::null_mut(), 0, &mut config)});
    let mut surfaces = [0; 1];
    let mut context = 0;

    let mut parse_nal = |data| {
                    pub fn clear_start_code_emulation_prevention_3_byte(data: &[u8]) -> Vec<u8> {
                        data.iter()
                            .enumerate()
                            .filter_map(|(index, value)| {
                                if index > 2
                                    && index < data.len() - 2
                                    && data[index - 2] == 0
                                    && data[index - 1] == 0
                                    && data[index] == 3
                                {
                                    None
                                } else {
                                    Some(*value)
                                }
                            })
                            .collect::<Vec<u8>>()
                    }
                    let data = clear_start_code_emulation_prevention_3_byte(data);
                    let ref mut s = BitReader::new(&data);
                    assert!(s.bit() == false); //forbidden_zero_bit
                    //let _ref_idc = s.bits(2);
                    let unit_type = s.bits(/*5*/6) as u8;
                    let temporal_id = if !false/*matches!(unit_type, EOS_NUT|EOB_NUT)*/ {
                        let _layer_id = s.bits(6);
                        s.bits(3) - 1
                    } else { 0 };
                    match unit_type {
                        SEI_PREFIX => {
                            let mut decode = ||{
                                let mut value = 0;
                                while {
                                    let byte = s.u8();
                                    value += byte;
                                    byte == 0xFF
                                }{}
                                value
                            };
                            const BUFFERING_PERIOD : u8 = 0;
                            const PICTURE_TIMING : u8 = 1;
                            const USER_DATA_REGISTERED_ITU_T_T35 : u8 = 4;
                            const USER_DATA_UNREGISTERED : u8 = 5;
                            const ACTIVE_PARAMETER_SETS : u8 = 129;
                            const MASTERING_DISPLAY_INFO : u8 = 137;
                            const CONTENT_LIGHT_LEVEL_INFO : u8 = 144;
                            let sei_type = decode();
                            let _size = decode();
                            //println!("{sei_type} {size} {:}", data);
                            match sei_type {
                                ACTIVE_PARAMETER_SETS => {
                                    let _active_video_parameter_set_id = s.bits(4);
                                    let _self_contained_cvs = s.bit();
                                    let _no_parameter_set_update = s.bit();
                                    let num_sps_ids = s.ue()+1;
                                    let active_seq_parameter_set_id = s.ue();
                                    println!("active parameter sets: {_active_video_parameter_set_id} {_self_contained_cvs} {_no_parameter_set_update} {num_sps_ids} {active_seq_parameter_set_id}");
                                }
                                USER_DATA_UNREGISTERED => {},
                                MASTERING_DISPLAY_INFO => {},
                                CONTENT_LIGHT_LEVEL_INFO => {},
                                BUFFERING_PERIOD => {},
                                PICTURE_TIMING => {},
                                USER_DATA_REGISTERED_ITU_T_T35 => {} // likely Dynamic HDR+
                                _ => panic!("SEI {sei_type:}"),
                            }
                        }
                        AUD => println!("AUD {data:?}"),
                        VPS => {
                            println!("VPS");
                            let id = s.bits(4) as usize;
                            assert!(id == 0);
                            assert!(s.bits(2) == 3);
                            let _max_sup_layers = s.bits(6) + 1;
                            let max_layers = (s.bits(3) + 1) as usize;
                            let _temporal_id_nesting = s.bit();
                            assert!(s.u16() == 0xFFFF);
                            profile_tier_level(s, max_layers);
                            let max_layer_id = s.bits(6);
                            let num_layer_sets = s.ue() + 1;
                            for _ in 1..num_layer_sets {
                                for _ in 0..=max_layer_id {
                                    s.bit(); // layer_id_included[i][j]
                                }
                            }
                            if s.bit() {
                                let _num_units_in_tick = s.u32();
                                let _time_scale = s.u32();
                                if s.bit() { let _num_ticks_poc_diff_one = s.ue() + 1; }
                                (0..s.ue()).map(|i| { let _hrd_layer_set_index = s.ue(); let bit = s.bit(); hrd_parameters(s, i > 0 && bit, max_layers) }).count();
                            }
                            s.bit(); // vps_extension
                            vps[id] = Some(VPS{});
                        }
                        SPS => {
                            let vps = s.bits(4) as usize;
                            assert!(vps == 0);
                            let max_layers = s.bits(3) as usize + 1;
                            assert!(max_layers <= 7);
                            let _temporal_id_nesting = s.bit();
                            profile_tier_level(s, max_layers);
                            let id = s.ue() as usize;
                            assert!(id < sps.len(), "{id}");
                            let chroma_format_idc = s.ue() as u8;
                            let separate_colour_plane = if chroma_format_idc == 3 { s.bit() } else { false };
                            let chroma_format_idc = if separate_colour_plane { 0 } else { chroma_format_idc };
                            let width = s.ue() as u16;
                            let height = s.ue() as u16;

                            check(unsafe{va::vaCreateSurfaces(va, va::VA_RT_FORMAT_YUV420_10, width as _, height as _, surfaces.as_mut_ptr(), surfaces.len() as _, std::ptr::null_mut(), 0)});
                            check(unsafe{va::vaCreateContext(va, config, width as _, height as _, va::VA_PROGRESSIVE as _, surfaces.as_mut_ptr(), surfaces.len() as _, &mut context)});

                            if s.bit() { let (_left, _right, _top, _bottom) = (s.ue(), s.ue(), s.ue(), s.ue()); }
                            let bit_depth = 8 + s.ue() as u8;
                            assert!(bit_depth == 10);
                            let _bit_depth_chroma = 8 + s.ue() as u8;
                            let log2_max_poc_lsb = 4 + s.ue() as u8;
                            let layer_ordering = layer_ordering(s, max_layers);
                            let log2_min_coding_block_size = 3 + s.ue() as u8;
                            let log2_diff_max_min_coding_block_size = s.ue() as u8;
                            let log2_min_transform_block_size = 2 + s.ue() as u8;
                            let log2_diff_max_min_transform_block_size = s.ue() as u8;
                            let max_transform_hierarchy_depth_inter = s.ue() as u8;
                            let max_transform_hierarchy_depth_intra = s.ue() as u8;
                            let scaling_list = s.bit().then(|| scaling_list(s));
                            let asymmetric_motion_partitioning = s.bit();
                            let sample_adaptive_offset = s.bit();
                            let pulse_code_modulation = s.bit().then(|| PulseCodeModulation{
                                bit_depth: 1 + s.bits(4) as u8,
                                bit_depth_chroma: 1 + s.bits(4) as u8,
                                log2_min_coding_block_size: 3 + s.ue() as u8,
                                log2_diff_max_min_coding_block_size: s.ue() as u8,
                                loop_filter_disable: s.bit()
                            });
                            let short_term_reference_picture_sets = {
                                let mut sets = Vec::<Box<[ShortTermReferencePicture]>>::with_capacity(s.ue() as usize);
                                for i in 0..sets.capacity() {
                                    let set = decode_short_term_reference_picture_set(s, &sets);
                                    sets.push(set);
                                }
                                sets.into_boxed_slice()
                            };
                            println!("SPS {id} {short_term_reference_picture_sets:?}");
                            let long_term_reference_picture_set = if s.bit() { (0..s.ue()).map(|_| LongTermReferencePicture{poc_lsb_sps: s.bits(log2_max_poc_lsb) as u8, used: s.bit()}).collect() } else { Default::default() };
                            let temporal_motion_vector_predictor = s.bit();
                            let strong_intra_smoothing = s.bit();
                            if s.bit() { // VUI
                                if s.bit() { // SAR
                                    let sar = s.u8();
                                    if sar == 0xFF { let (_num, _den) = (s.u16(), s.u16()); }
                                }
                                if s.bit() { let _overscan_appropriate = s.bit(); }
                                if s.bit() {
                                    let _format = s.bits(3);
                                    let _full_range = s.bit();
                                    if s.bit() {
                                        let _colour_primaries = s.u8();
                                        let _transfer_characteristic = s.u8();
                                        let _matrix_coeffs = s.u8();
                                    }
                                }
                                if s.bit() { //chroma_sample_loc_type
                                    let (_top, _bottom) = (s.ue(), s.ue());
                                }
                                let _neutral_chroma_indication = s.bit();
                                let _field_seq = s.bit();
                                let _frame_field_info_present = s.bit();
                                if s.bit() { let (_left, _right, _top, _bottom) = (s.ue(), s.ue(), s.ue(), s.ue()); } //default_display_window
                                if s.bit() { // timing_info
                                    let _num_units_in_tick = s.u32();
                                    let _time_scale = s.u32();
                                    if s.bit() { let _num_ticks_poc_diff_one = s.ue() + 1; }
                                    if s.bit() { let _hrd_parameters = hrd_parameters(s, true, max_layers); }
                                }
                                if s.bit() {
                                    let _tiles_fixed_structure = s.bit();
                                    let _motion_vectors_over_pic_boundaries = s.bit();
                                    let _restricted_ref_pic_lists = s.bit();
                                    let _min_spatial_segmentation_idc = s.ue();
                                    let _max_bytes_per_pic_denom = s.ue();
                                    let _max_bits_per_min_cu_denom = s.ue();
                                    let _log2_max_mv_length_horizontal = s.ue();
                                    let _log2_max_mv_length_vertical = s.ue();
                                }
                            }
                            let _sps_extension = s.bit();
                            sps[id] = Some(SPS{separate_colour_plane, chroma_format_idc, width, height, bit_depth, log2_max_poc_lsb, layer_ordering,
                                log2_min_coding_block_size, log2_diff_max_min_coding_block_size,
                                log2_min_transform_block_size, log2_diff_max_min_transform_block_size, max_transform_hierarchy_depth_inter, max_transform_hierarchy_depth_intra,
                                scaling_list,
                                asymmetric_motion_partitioning,
                                sample_adaptive_offset,
                                pulse_code_modulation,
                                short_term_reference_picture_sets, long_term_reference_picture_set,
                                temporal_motion_vector_predictor,
                                strong_intra_smoothing});
                        },
                        PPS => {
                            let id = s.ue() as usize;
                            pps[id] = Some(PPS{
                                sps: s.ue() as usize,
                                dependent_slice_segments: s.bit(),
                                output: s.bit(),
                                num_extra_slice_header_bits: s.bits(3) as u8,
                                sign_data_hiding: s.bit(),
                                cabac_init: s.bit(),
                                num_ref_idx_l0_default_active: 1 + s.ue() as u8,
                                num_ref_idx_l1_default_active: 1 + s.ue() as u8,
                                pic_init_qp_minus26: s.se() as i8,
                                constrained_intra_prediction: s.bit(),
                                transform_skip: s.bit(),
                                diff_cu_qp_delta_depth: s.bit().then(|| s.ue() as u8),
                                cb_qp_offset: s.se() as i8,
                                cr_qp_offset: s.se() as i8,
                                pic_slice_chroma_qp_offsets: s.bit(),
                                weighted_prediction: s.bit(),
                                weighted_biprediction: s.bit(),
                                transquant_bypass: s.bit(),
                                tiles: {
                                    let tiles = s.bit();
                                    let entropy_coding_sync = s.bit();
                                    (entropy_coding_sync, tiles.then(|| {
                                        let columns = 1 + s.ue();
                                        let rows = 1 + s.ue();
                                        let (columns, rows) = if !s.bit() { // not uniform spacing
                                            unimplemented!();//let _column_widths = (0.._num_tile_columns-1).map(|_| s.ue() + 1).collect();
                                            //let _row_heights = (0.._num_tile_rows-1).map(|_| s.ue() + 1).collect();
                                        } else { unimplemented!(); };
                                        let loop_filter_across_tiles = s.bit();
                                        Tiles{columns, rows, loop_filter_across_tiles}
                                    }))
                                },
                                loop_filter_across_slices: s.bit(),
                                deblocking_filter: s.bit().then(|| {
                                    let deblocking_filter_override = s.bit();
                                    (deblocking_filter_override, (!s.bit()).then(|| DeblockingFilter{ // not disable dbf
                                        beta_offset: 2 * s.se() as i8,
                                        tc_offset: 2 * s.se() as i8
                                    }))
                                }),
                                scaling_list: s.bit().then(|| scaling_list(s)),
                                lists_modification: s.bit(),
                                log2_parallel_merge_level: 2 + s.ue() as u8,
                                slice_header_extension: s.bit(),
                                pps_extension: s.bit(),
                            });
                        }
                        IDR_N_LP|TRAIL_R|TRAIL_N => {
                            println!("{}", match unit_type {IDR_N_LP=>"IDR_N_LP",TRAIL_R=>"TRAIL_R",TRAIL_N=>"TRAIL_N",_=>unreachable!()});
                            let first_slice_in_pic = s.bit();
                            if IRAP(unit_type) { let _no_output_of_prior_pics = s.bit(); }
                            let ref pps = pps[s.ue() as usize].as_ref().unwrap();
                            let ref sps = sps[pps.sps].as_ref().unwrap_or_else(|| panic!("{}", pps.sps));
                            let dependent_slice_segment = if !first_slice_in_pic {
                                let dependent_slice_segment = pps.dependent_slice_segments && s.bit();
                                let pic_size = sps.width * sps.height;
                                let slice_address_length = ((pic_size-1)<<1).ilog2() as u8;
                                assert!(slice_address_length < 24);
                                let _slice_segment_address = s.bits(slice_address_length);
                                dependent_slice_segment
                            } else { false };
                            if !dependent_slice_segment {
                                s.advance(pps.num_extra_slice_header_bits);
                                let _slice_type = s.ue();
                                let _output = pps.output && s.bit();
                                let _separate_colour_plane = if sps.separate_colour_plane { s.bits(2) } else { 0 };
                                let output_picture_number = if !IDR(unit_type) {
                                    let poc_lsb = s.bits(sps.log2_max_poc_lsb);
                                    let max_poc_lsb = 1 << sps.log2_max_poc_lsb;
                                    let prev_poc_lsb = poc_tid0 % max_poc_lsb;
                                    let prev_poc_msb = poc_tid0 - prev_poc_lsb;
                                    let poc_msb = if poc_lsb < prev_poc_lsb && prev_poc_lsb - poc_lsb >= max_poc_lsb / 2 {
                                        prev_poc_msb + max_poc_lsb
                                    } else if poc_lsb > prev_poc_lsb && poc_lsb - prev_poc_lsb > max_poc_lsb / 2 {
                                        prev_poc_msb - max_poc_lsb
                                    } else {
                                        prev_poc_msb
                                    };
                                    (if /*matches!(unit_type,BLA_W_RADL|BLA_W_LP|BLA_N_LP)*/false { 0 } else { poc_msb }) + poc_lsb
                                } else { 0 };
                                if temporal_id == 0 && !matches!(unit_type, TRAIL_N/*|TSA_N|STSA_N|RADL_N|RASL_N|RADL_R|RASL_R*/) { poc_tid0 = output_picture_number; }
                                let (set, strps_encoded_bits_len_skip) = if !IDR(unit_type) {
                                    if !s.bit() { let start = s.available()+1; (Some(decode_short_term_reference_picture_set(s, &sps.short_term_reference_picture_sets)), Some((start - s.available()) as u32)) }
                                    else {
                                        let set = if sps.short_term_reference_picture_sets.len()>1 { s.bits((sps.short_term_reference_picture_sets.len()-1<<1).ilog2() as u8) as usize } else { 0 };
                                        (Some(sps.short_term_reference_picture_sets[set].clone()), None)
                                    }
                                } else { (None, None) };
                                if first_slice_in_pic {
                                    let mut buffer = 0; //VABufferID
                                    check(unsafe{va::vaCreateBuffer(va, context, VABufferType_VAPictureParameterBufferType, std::mem::size_of::<va::VAPictureParameterBufferHEVC> as u32, 1, &mut va::VAPictureParameterBufferHEVC{
                                        pic_width_in_luma_samples: sps.width,
                                        pic_height_in_luma_samples: sps.height,
                                        log2_min_luma_coding_block_size_minus3: sps.log2_min_coding_block_size - 3,
                                        sps_max_dec_pic_buffering_minus1: sps.layer_ordering.last().unwrap().max_dec_pic_buffering - 1,
                                        log2_diff_max_min_luma_coding_block_size: sps.log2_diff_max_min_coding_block_size,
                                        log2_min_transform_block_size_minus2: sps.log2_min_transform_block_size - 2,
                                        log2_diff_max_min_transform_block_size: sps.log2_diff_max_min_transform_block_size,
                                        max_transform_hierarchy_depth_inter: sps.max_transform_hierarchy_depth_inter,
                                        max_transform_hierarchy_depth_intra: sps.max_transform_hierarchy_depth_intra,
                                        num_short_term_ref_pic_sets: sps.short_term_reference_picture_sets.len() as u8,
                                        num_long_term_ref_pic_sps: sps.long_term_reference_picture_set.len() as u8,
                                        num_ref_idx_l0_default_active_minus1: pps.num_ref_idx_l0_default_active - 1,
                                        num_ref_idx_l1_default_active_minus1: pps.num_ref_idx_l1_default_active - 1,
                                        init_qp_minus26: pps.pic_init_qp_minus26,
                                        pps_cb_qp_offset: pps.cb_qp_offset,
                                        pps_cr_qp_offset: pps.cr_qp_offset,
                                        pcm_sample_bit_depth_luma_minus1: sps.pulse_code_modulation.as_ref().map(|p| p.bit_depth - 1).unwrap_or(0),
                                        pcm_sample_bit_depth_chroma_minus1: sps.pulse_code_modulation.as_ref().map(|p| p.bit_depth_chroma - 1).unwrap_or(0),
                                        log2_min_pcm_luma_coding_block_size_minus3: sps.pulse_code_modulation.as_ref().map(|p| p.log2_min_coding_block_size - 3).unwrap_or(0),
                                        log2_diff_max_min_pcm_luma_coding_block_size: sps.pulse_code_modulation.as_ref().map(|p| p.log2_diff_max_min_coding_block_size).unwrap_or(0),
                                        diff_cu_qp_delta_depth: pps.diff_cu_qp_delta_depth.unwrap_or(0),
                                        pps_beta_offset_div2: pps.deblocking_filter.as_ref().map(|f| f.1.as_ref().map(|f| f.beta_offset / 2)).flatten().unwrap_or(0),
                                        pps_tc_offset_div2: pps.deblocking_filter.as_ref().map(|f| f.1.as_ref().map(|f| f.tc_offset / 2)).flatten().unwrap_or(0),
                                        log2_parallel_merge_level_minus2: pps.log2_parallel_merge_level - 2,
                                        bit_depth_luma_minus8: sps.bit_depth - 8,
                                        bit_depth_chroma_minus8: sps.bit_depth - 8,
                                        log2_max_pic_order_cnt_lsb_minus4: sps.log2_max_poc_lsb - 4,
                                        num_extra_slice_header_bits: pps.num_extra_slice_header_bits,
                                        pic_fields: va::_VAPictureParameterBufferHEVC__bindgen_ty_1{
                                            bits: va::_VAPictureParameterBufferHEVC__bindgen_ty_1__bindgen_ty_1{
                                                _bitfield_align_1: [],
                                                _bitfield_1: va::_VAPictureParameterBufferHEVC__bindgen_ty_1__bindgen_ty_1::new_bitfield_1(
                                                    sps.chroma_format_idc as _,
                                                    sps.separate_colour_plane as _,
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
                                                    pps.tiles.1.as_ref().map(|t| t.loop_filter_across_tiles).unwrap_or(false) as _,
                                                    sps.pulse_code_modulation.as_ref().map(|pcm| pcm.loop_filter_disable).unwrap_or(false) as _,
                                                    false as _, //NoPicReordering
                                                    false as _, //NoBiPred
                                                    0, //Reserved
                                                )
                                            }
                                        },
                                        slice_parsing_fields: va::_VAPictureParameterBufferHEVC__bindgen_ty_2{
                                            bits: va::_VAPictureParameterBufferHEVC__bindgen_ty_2__bindgen_ty_1{
                                                _bitfield_align_1: [],
                                                _bitfield_1: va::_VAPictureParameterBufferHEVC__bindgen_ty_2__bindgen_ty_1::new_bitfield_1(
                                                    pps.lists_modification as _,
                                                    !sps.long_term_reference_picture_set.is_empty() as _,
                                                    sps.temporal_motion_vector_predictor as _,
                                                    pps.cabac_init as _,
                                                    pps.output as _,
                                                    pps.dependent_slice_segments as _,
                                                    pps.pic_slice_chroma_qp_offsets as _,
                                                    sps.sample_adaptive_offset as _,
                                                    /*override:*/ pps.deblocking_filter.as_ref().map(|f| f.0).unwrap_or(false) as _,
                                                    /*disable:*/ pps.deblocking_filter.as_ref().map(|f| f.1.is_none()).unwrap_or(false) as _,
                                                    pps.slice_header_extension as _,
                                                    IRAP(unit_type) as _,
                                                    IDR(unit_type) as _,
                                                    IRAP(unit_type) as _,
                                                    0
                                                ),
                                            }
                                        },
                                        CurrPic: va::VAPictureHEVC {
                                            picture_id: surfaces[0],
                                            pic_order_cnt: 0, //TODO
                                            flags: 0, //SHORT|LONG _REF
                                            va_reserved: [0; 4],
                                        },
                                        ReferenceFrames: [va::VAPictureHEVC {
                                            picture_id: 0,
                                            pic_order_cnt: 0, //TODO
                                            flags: 0, //PREV|NEXT, SHORT|LONG _REF
                                            va_reserved: [0; 4],
                                        }; _],
                                        num_tile_columns_minus1: pps.tiles.1.as_ref().map(|t| t.columns.len() - 1).unwrap_or(0) as u8,
                                        num_tile_rows_minus1: pps.tiles.1.as_ref().map(|t| t.rows.len() - 1).unwrap_or(0) as u8,
                                        //column_width_minus1: pps.tiles.1.map(|t| { let iter=t.columns.into_iter().map(|w| w-1).chain(std::iter::repeat(0)); [_;_].map(|_| iter.next().unwrap()) }).unwrap_or_default(),
                                        column_width_minus1: pps.tiles.1.as_ref().map(|t| t.columns.into_iter().map(|w| w-1).chain(std::iter::repeat(0)).take(19).collect::<Vec<_>>().try_into().unwrap()).unwrap_or_default(),
                                        row_height_minus1: pps.tiles.1.as_ref().map(|t| t.rows.into_iter().map(|h| h-1).chain(std::iter::repeat(0)).take(15).collect::<Vec<_>>().try_into().unwrap()).unwrap_or_default(),
                                        st_rps_bits: strps_encoded_bits_len_skip.unwrap_or(0),
                                        va_reserved: [0; _]
                                    } as *mut _ as *mut std::ffi::c_void, &mut buffer)});
                                } // first_slice
                            }
                        }
                        _ => panic!("Unit {unit_type:?}"),
                    };
                };
    use nom::{multi::{length_count, length_value}, sequence::pair, combinator::map, number::complete::{u8,be_u16}};
    length_count::<_,_,_,nom::error::Error<_>,_,_>(u8, pair(map(u8, |t| t&0x3f), length_count(be_u16, length_value(be_u16, |nal| Ok((nal,parse_nal(nal)))))))(&video.codec_private.as_ref().unwrap()[22..]).unwrap();
    for element in &mut iterator(input, matroska::elements::segment_element) { use  matroska::elements::SegmentElement::*; match element {
        Void(_) => {},
        Cluster(cluster) => for data_block in cluster.simple_block {
            let (data, block) = matroska::elements::simple_block(data_block).unwrap();
            if block.track_number == video.track_number {
                for data in &mut iterator(data, unit) {
                    parse_nal(data);
                }
            }
        },
        Cues(_) => {},
        Chapters(_) => {},
        _ => panic!("{element:?}")
    }}
    Ok(())
}
