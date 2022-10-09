#![allow(incomplete_features)]#![feature(int_log,unchecked_math,generic_arg_infer, generic_const_exprs, array_methods, array_zip)]
#![allow(dead_code,unreachable_code,unused_variables)]

fn from_iter_or_else<T, const N: usize>(iter: impl IntoIterator<Item=T>, f: impl Fn() -> T+Copy) -> [T; N] { let mut iter = iter.into_iter(); [(); N].map(|_| iter.next().unwrap_or_else(f)) }
fn from_iter_or<T: Copy, const N: usize>(iter: impl IntoIterator<Item=T>, v: T) -> [T; N] { from_iter_or_else(iter, || v) }
fn from_iter<T: Default, const N: usize>(iter: impl IntoIterator<Item=T>) -> [T; N] { from_iter_or_else(iter, || Default::default()) }
fn array<T: Default, const N: usize>(len: usize, mut f: impl FnMut()->T) -> [T; N] { from_iter((0..len).map(|_| f())) }

fn ceil_log2(x: usize) -> u8 { ((x-1)<<1).ilog2() as u8 }

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

#[allow(non_snake_case)] fn Intra_Random_Access_Picture(unit_type: u8) -> bool { 16 <= unit_type && unit_type <= 23 }
#[allow(non_snake_case)] fn Instantaneous_Decoder_Refresh(unit_type: u8) -> bool { matches!(unit_type, IDR_W_RADL|IDR_N_LP) }

mod bit; use bit::BitReader;

#[derive(num_derive::FromPrimitive)] enum SliceType { B, P, I }
use num_traits::FromPrimitive;

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

struct ScalingList {
    pub x4: [[u8; 16]; 6],
    pub x8: [[u8; 64]; 6],
    pub x16: [(u8, [u8; 64]); 6],
    pub x32: [(u8, [u8; 64]); 2],
}

fn scaling_list(s: &mut BitReader) -> ScalingList {
    let mut x4 = [(0, [16; _]); _];
    let mut x8 = [(0, [64; _]); _];
    let mut x16 = [(16, [64; _]); _];
    let mut x32 = [(16, [64; _]); _];
    if s.bit() {
        fn decompress<const DC: bool, const N: usize, const M: usize>(s: &mut BitReader, matrices: &mut [(u8, [u8; N*N]); M]) {
            for matrix_id in 0..M {
                matrices[matrix_id] = if !s.bit() { matrices[matrix_id - s.ue() as usize] } else {
                    let dc = 8 + if DC { s.se() } else { 0 } as u8;
                    let diagonally_ordered_coefficients = std::iter::successors(Some(dc), |&last| Some((last as i64 + 0x100 + s.se()) as u8)).take(N*N).collect::<Box<_>>();
                    let diagonal_scan = (0..N).map(|i| (0..=i).map(move |j| (i-j,j))).flatten() .chain( (1..N).map(|j| (0..N-j).rev().map(move |i| (i-j,j))).flatten() ).map(|(i,j)| i*N+j);
                    let mut matrix = [0; {N*N}]; for (diagonal, raster) in diagonal_scan.enumerate() { matrix[raster] = diagonally_ordered_coefficients[diagonal]; }
                    (dc, matrix)
                }
            }
        }
        decompress::<false,4,_>(s, &mut x4);
        decompress::<false,8,_>(s, &mut x8);
        decompress::<true,8,_>(s, &mut x16);
        decompress::<true,8,_>(s, &mut x32);
    }
    ScalingList{x4: x4.map(|m| m.1), x8: x8.map(|m| m.1), x16, x32}
}

struct VPS {}

struct PulseCodeModulation { bit_depth: u8, bit_depth_chroma: u8, log2_min_coding_block_size: u8, log2_diff_max_min_coding_block_size: u8, loop_filter_disable: bool }
#[derive(Clone,Debug)] struct ShortTermReferencePicture { delta_poc: i8, used: bool }
#[derive(Clone)] struct LongTermReferencePicture { poc: u8, used: bool }

struct SPS {
    separate_color_plane: bool,
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
    scaling_list: Option<ScalingList>,
    asymmetric_motion_partitioning: bool,
    sample_adaptive_offset: bool,
    pulse_code_modulation: Option<PulseCodeModulation>,
    short_term_reference_picture_sets: Box<[Box<[ShortTermReferencePicture]>]>,
    long_term_reference_picture_set: Option<Box<[LongTermReferencePicture]>>,
    temporal_motion_vector_predictor: bool,
    strong_intra_smoothing: bool,
}

#[derive(Clone)] struct DeblockingFilter { beta_offset: i8, tc_offset: i8 }
struct Tiles {columns: Box<[u16]>, rows: Box<[u16]>, loop_filter_across_tiles: bool}

struct PPS {
    sps: usize,
    dependent_slice_segments: bool,
    output: bool,
    num_extra_slice_header_bits: u8,
    sign_data_hiding: bool,
    cabac_init: bool,
    num_ref_idx_l0_default_active: usize,
    num_ref_idx_l1_default_active: usize,
    init_qp_minus26: i8,
    constrained_intra_prediction: bool,
    transform_skip: bool,
    diff_cu_qp_delta_depth: Option<u8>,
    cb_qp_offset: i8,
    cr_qp_offset: i8,
    slice_chroma_qp_offsets: bool,
    weighted_prediction: bool,
    weighted_biprediction: bool,
    transquant_bypass: bool,
    tiles: (/*entropy_coding_sync*/bool, Option<Tiles>),
    loop_filter_across_slices: bool,
    deblocking_filter: Option<(/*deblocking_filter_override*/bool, Option<DeblockingFilter>)>,
    scaling_list: Option<ScalingList>,
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
        let mut v = Vec::with_capacity(negative+positive);
        /*use std::iter::repeat_with;*/ type P = ShortTermReferencePicture;
        //v.extend({let mut a=0; repeat_with(|| { a=a-1-s.ue() as i8; P{delta_poc: a, used: s.bit()}}).take(negative)});
        {let mut a = 0; for _ in 0..negative { a=a-1-s.ue() as i8; v.push(P{delta_poc: a, used: s.bit()}); }}
        //v.extend(successors(Some(P{delta_poc:0,used:false}), |P{delta_poc,..}| Some(P{delta_poc: delta_poc+1+s.ue() as i8, used: s.bit()})).take(positive));
        {let mut a = 0; for _ in 0..positive { a=a+1+s.ue() as i8; v.push(P{delta_poc: a, used: s.bit()}); }}
        v.into_boxed_slice()
    }
}

struct SHReference {
    poc: u8,
    short_term_pictures: Box<[ShortTermReferencePicture]>,
    short_term_picture_set_encoded_bits_len_skip: Option<u32>,
    long_term_pictures: Box<[LongTermReferencePicture]>,
    temporal_motion_vector_predictor: bool
}

#[derive(Clone,Copy,Default)] struct MayB<T> { p: T, b: Option<T> }
impl<T> MayB<T> {
    fn as_ref(&self) -> MayB<&T> { MayB{p: &self.p, b: self.b.as_ref()} }
    fn map<U>(self, mut f: impl FnMut(T)->U) -> MayB<U> { MayB{p: f(self.p), b: self.b.map(f)} }
}
#[derive(Clone,Default,Debug)] struct LumaChroma<L, C=L> { luma: L, chroma: Option<C> }
#[derive(Clone,Copy,Default,Debug)] struct WeightOffset<W, O> { weight: W, offset: O }
type WeightOffsets<W, O, const N: usize> = WeightOffset<[W; N],[O; N]>;
#[derive(Clone,Default)] struct Tables<T, W=u8> { log2_denom_weight: W, pb: MayB<T> }
type PredictionWeights<const N: usize> = LumaChroma<Tables<WeightOffsets<i8,i8,N>>, Tables<WeightOffsets<[i8;2],[i8;2],N>>>;
struct SHInter {
    active_references: MayB<usize>,
    list_entry_lx: Option<MayB<Box<[u8]>>>,
    mvd_l1_zero: bool,
    cabac_init: bool,
    collocated_list: Option<(bool, Option<u8>)>,
    prediction_weights: Option<PredictionWeights<15>>,
    max_num_merge_cand: u8
}

struct SliceHeader {
    slice_type: SliceType,
    output: bool,
    color_plane_id: u8,
    reference: Option<SHReference>,
    sample_adaptive_offset: LumaChroma<bool>,
    inter: Option<SHInter>,
    qp_delta: i8,
    qp_offsets: Option<(i8, i8)>,
    //cu_chroma_qp_offset: bool,
    deblocking_filter: Option<DeblockingFilter>,
    loop_filter_across_slices: bool,
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
    let mut sh = None;
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
    #[derive(Default)] struct Frame {
        id: VASurfaceID,
        poc: Option<u8>,
    }
    let mut frames = None;
    let mut context = 0;

    let ref mut parse_nal = |data: &[u8]| {
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
                layer_ordering(s, max_layers);
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
                    (0..s.ue()).map(|i| { let _hrd_layer_set_index = s.ue(); let common_inf = i>0 && s.bit(); hrd_parameters(s, common_inf, max_layers) }).count();
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
                let separate_color_plane = if chroma_format_idc == 3 { s.bit() } else { false };
                let chroma_format_idc = if separate_color_plane { 0 } else { chroma_format_idc };
                let width = s.ue() as u16;
                let height = s.ue() as u16;

                let mut ids = [0; 16];
                check(unsafe{va::vaCreateSurfaces(va, va::VA_RT_FORMAT_YUV420_10, width as _, height as _, ids.as_mut_ptr(), ids.len() as _, std::ptr::null_mut(), 0)});
                check(unsafe{va::vaCreateContext(va, config, width as _, height as _, va::VA_PROGRESSIVE as _, ids.as_ptr() as *mut _, ids.len() as _, &mut context)});
                frames = Some(ids.map(|id| Frame{id, poc: None}));

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
                let long_term_reference_picture_set = s.bit().then(|| (0..s.ue()).map(|_| LongTermReferencePicture{poc: s.bits(log2_max_poc_lsb) as u8, used: s.bit()}).collect());
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
                            let _color_primaries = s.u8();
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
                sps[id] = Some(SPS{separate_color_plane, chroma_format_idc, width, height, bit_depth, log2_max_poc_lsb, layer_ordering,
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
                    num_ref_idx_l0_default_active: 1 + s.ue() as usize,
                    num_ref_idx_l1_default_active: 1 + s.ue() as usize,
                    init_qp_minus26: s.se() as i8,
                    constrained_intra_prediction: s.bit(),
                    transform_skip: s.bit(),
                    diff_cu_qp_delta_depth: s.bit().then(|| s.ue() as u8),
                    cb_qp_offset: s.se() as i8,
                    cr_qp_offset: s.se() as i8,
                    slice_chroma_qp_offsets: s.bit(),
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
                let first_slice = s.bit();
                if Intra_Random_Access_Picture(unit_type) { let _no_output_of_prior_pics = s.bit(); }
                let ref pps = pps[s.ue() as usize].as_ref().unwrap();
                let ref sps = sps[pps.sps].as_ref().unwrap_or_else(|| panic!("{}", pps.sps));
                let (dependent_slice_segment, slice_segment_address) = if !first_slice {
                    let dependent_slice_segment = pps.dependent_slice_segments && s.bit();
                    (dependent_slice_segment, Some(s.bits(ceil_log2((sps.width * sps.height) as usize))))
                } else { (false, None) };
                if !dependent_slice_segment {
                    s.advance(pps.num_extra_slice_header_bits);
                    let slice_type = SliceType::from_u64(s.ue()).unwrap();
                    let output = pps.output && s.bit();
                    let color_plane_id = if sps.separate_color_plane { s.bits(2) } else { 0 } as u8;
                    let reference = (!Instantaneous_Decoder_Refresh(unit_type)).then(|| {
                        let poc_lsb = s.bits(sps.log2_max_poc_lsb) as u8;
                        /*println!("{}", sps.log2_max_poc_lsb);
                        let max_poc_lsb = 1 << sps.log2_max_poc_lsb;
                        let prev_poc_lsb = poc_tid0 % max_poc_lsb;
                        let prev_poc_msb = poc_tid0 - prev_poc_lsb;
                        let poc_msb = if poc_lsb < prev_poc_lsb && prev_poc_lsb - poc_lsb >= max_poc_lsb / 2 {
                            prev_poc_msb + max_poc_lsb
                        } else if poc_lsb > prev_poc_lsb && poc_lsb - prev_poc_lsb > max_poc_lsb / 2 {
                            prev_poc_msb - max_poc_lsb
                        } else {
                            prev_poc_msb
                        };*/
                        let poc = (/*if /*matches!(unit_type,BLA_W_RADL|BLA_W_LP|BLA_N_LP)*/false { 0 } else { poc_msb } +*/ poc_lsb) as u8;
                        let (short_term_picture_set, short_term_picture_set_encoded_bits_len_skip) = if !s.bit() { let start = s.available()+1; (decode_short_term_reference_picture_set(s, &sps.short_term_reference_picture_sets), Some((start - s.available()) as u32)) }
                        else {
                            let set = if sps.short_term_reference_picture_sets.len()>1 { s.bits(ceil_log2(sps.short_term_reference_picture_sets.len()-1<<1)) as usize } else { 0 };
                            (sps.short_term_reference_picture_sets[set].clone(), None)
                        };
                        SHReference{
                            poc, short_term_pictures: short_term_picture_set, short_term_picture_set_encoded_bits_len_skip,
                            long_term_pictures: sps.long_term_reference_picture_set.as_ref().map(|set| {
                                let sequence = if !set.is_empty() { s.ue() } else { 0 };
                                let slice = s.ue();
                                let mut v = Vec::with_capacity((sequence+slice) as usize);
                                let poc = |s: &mut BitReader, msb: &mut u8| -> u8 { if !s.bit() { 0 } else {
                                    *msb += s.ue() as u8;
                                    poc - *msb << sps.log2_max_poc_lsb - poc_lsb
                                }};
                                v.extend((0..sequence).scan(0, |msb, _| { let mut p = set[if set.len() > 1 { s.bits(ceil_log2(set.len())) } else { 0 } as usize].clone(); p.poc += poc(s, msb); Some(p)}));
                                v.extend((0..slice).scan(0, |msb, _| { let mut p = LongTermReferencePicture{poc: s.bits(sps.log2_max_poc_lsb) as u8, used: s.bit()}; p.poc += poc(s, msb); Some(p)}));
                                v.into_boxed_slice()
                            }).unwrap_or_default(),
                            temporal_motion_vector_predictor: sps.temporal_motion_vector_predictor && s.bit()
                        }
                    });
                    let frames = frames.as_mut().unwrap();
                    if temporal_id == 0 && !matches!(unit_type, TRAIL_N/*|TSA_N|STSA_N|RADL_N|RASL_N|RADL_R|RASL_R*/) { poc_tid0 = reference.as_ref().map(|r| r.poc).unwrap_or(0); }
                    let chroma = sps.chroma_format_idc>0;
                    let sample_adaptive_offset = sps.sample_adaptive_offset.then(|| LumaChroma{luma: s.bit(), chroma: chroma.then(|| s.bit())}).unwrap_or(LumaChroma{luma: false, chroma: chroma.then(|| false)});
                    let inter = matches!(slice_type, SliceType::P | SliceType::B).then(|| {
                        let b = matches!(slice_type, SliceType::B);
                        let active_references =
                            if s.bit() { MayB{p: 1 + s.ue() as usize, b: b.then(|| 1 + s.ue() as usize)} }
                            else { MayB{p:pps.num_ref_idx_l0_default_active, b: b.then_some(pps.num_ref_idx_l1_default_active)} };
                        let reference = reference.as_ref().unwrap();
                        let references_len = reference.short_term_pictures.iter().filter(|p| p.used).count() + reference.long_term_pictures.iter().filter(|p| p.used).count();
                        SHInter{
                            active_references,
                            list_entry_lx: (pps.lists_modification && references_len > 1).then(|| active_references.map(|len| (0..len).map(|_| s.bits(ceil_log2(references_len)) as u8).collect::<Box<_>>())),
                            mvd_l1_zero: b && s.bit(),
                            cabac_init: pps.cabac_init && s.bit(),
                            collocated_list: reference.temporal_motion_vector_predictor.then(|| {
                                let collocated_list = b && !s.bit();
                                (collocated_list, (if collocated_list { active_references.b.unwrap() } else { active_references.p } > 1).then(|| s.ue() as u8))
                            }),
                            prediction_weights: ((matches!(slice_type, SliceType::P) && pps.weighted_prediction) || (b && pps.weighted_biprediction)).then(|| {
                                let log2_denom_weight_luma = s.ue() as u8;
                                let log2_denom_weight_chroma = chroma.then(|| (log2_denom_weight_luma as i64 + s.se()) as u8);
                                let ref pb = active_references.map(|active_references| {
                                    array(active_references, || s.bit()).zip(chroma.then(|| array(active_references, || s.bit())).map(|a| a.map(Some)).unwrap_or_default()).map(|(l,c)| LumaChroma{
                                        luma: l.then(|| WeightOffset{weight: s.se() as i8, offset: s.se() as i8}).unwrap_or_default(),
                                        chroma: c.map(|c| c.then(|| [(); 2].map(|_| { let w = s.se(); WeightOffset{weight: w as i8, offset: s.se() as i8}})).unwrap_or_default())
                                    })
                                });
                                LumaChroma {
                                    luma: Tables{ log2_denom_weight: log2_denom_weight_luma, pb: pb.as_ref().map(|t| WeightOffset{weight: t.each_ref().map(|lc| lc.luma.weight), offset: t.each_ref().map(|lc| lc.luma.offset)}) },
                                    chroma: chroma.then(|| Tables{ log2_denom_weight: log2_denom_weight_luma, pb: pb.as_ref().map(|t| WeightOffset{weight: t.each_ref().map(|lc| lc.chroma.unwrap().map(|w| w.weight)), offset: t.each_ref().map(|lc| lc.chroma.unwrap().map(|w| w.offset))})})
                                }
                            }),
                            max_num_merge_cand: 5 - s.ue() as u8
                        }
                    }); // inter
                    let qp_delta = s.se() as i8;
                    let qp_offsets = pps.slice_chroma_qp_offsets.then(|| (s.se() as i8, s.se() as i8));
                    //cu_chroma_qp_offsets: pps.chroma_qp_offset_list && s.bit(),
                    let deblocking_filter = pps.deblocking_filter.as_ref().map(|(r#override, pps)| if *r#override && s.bit() { (!s.bit()).then(|| DeblockingFilter{beta_offset: 2 * s.se() as i8, tc_offset: 2 * s.se() as i8}) } else { pps.clone() }).flatten();
                    let loop_filter_across_slices = if pps.loop_filter_across_slices && (sample_adaptive_offset.luma || sample_adaptive_offset.chroma.unwrap_or(false) || deblocking_filter.is_some()) { s.bit() } else { pps.loop_filter_across_slices };
                    sh = Some(SliceHeader{slice_type, output, color_plane_id, reference, sample_adaptive_offset, inter, qp_delta, qp_offsets,/*cu_chroma_qp_offsets,*/deblocking_filter, loop_filter_across_slices});

                    if first_slice {
                        let sh = sh.as_ref().unwrap();
                        let reference = sh.reference.as_ref();
                        let current_poc = reference.map(|r| r.poc).unwrap_or(0);
                        use itertools::Itertools;
                        println!("POC {}", current_poc);
                        println!("DPB [{}]", frames.iter().filter_map(|f| f.poc).format(" "));
                        reference.map(|r| println!("refs [{}]", r.short_term_pictures.iter().map(|p| (current_poc as i8+p.delta_poc) as u8).chain(r.long_term_pictures.iter().map(|p| p.poc)).format(" ")));
                        for Frame{poc: frame_poc,..} in frames.iter_mut() {
                            *frame_poc = frame_poc.filter(|&frame_poc| reference.map(|r| r.short_term_pictures.iter().any(|p| ((current_poc as i8+p.delta_poc) as u8) == frame_poc) || r.long_term_pictures.iter().any(|p| p.poc == frame_poc)).unwrap_or(false));
                        }
                        let ref mut current = frames.iter_mut().find(|f| f.poc.is_none()).unwrap();
                        current.poc = Some(current_poc);

                        let mut buffer = 0; //VABufferID
                        check(unsafe{va::vaCreateBuffer(va, context, VABufferType_VAPictureParameterBufferType, std::mem::size_of::<va::VAPictureParameterBufferHEVC>() as std::ffi::c_uint, 1,
                            &mut va::VAPictureParameterBufferHEVC{
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
                                num_long_term_ref_pic_sps: sps.long_term_reference_picture_set.as_ref().map(|s| s.len()).unwrap_or(0) as u8,
                                num_ref_idx_l0_default_active_minus1: pps.num_ref_idx_l0_default_active as u8 - 1,
                                num_ref_idx_l1_default_active_minus1: pps.num_ref_idx_l1_default_active as u8 - 1,
                                init_qp_minus26: pps.init_qp_minus26,
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
                                        pps.tiles.1.as_ref().map(|t| t.loop_filter_across_tiles).unwrap_or(false) as _,
                                        sps.pulse_code_modulation.as_ref().map(|pcm| pcm.loop_filter_disable).unwrap_or(false) as _,
                                        false as _, //NoPicReordering
                                        false as _, //NoBiPred
                                        0, //Reserved
                                    )
                                }},
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
                                        Intra_Random_Access_Picture(unit_type) as _,
                                        Instantaneous_Decoder_Refresh(unit_type) as _,
                                        Intra_Random_Access_Picture(unit_type) as _,
                                        0
                                    ),
                                }},
                                CurrPic: va::VAPictureHEVC {
                                    picture_id: current.id,
                                    pic_order_cnt: current_poc as i32,
                                    flags: 0,
                                    va_reserved: [0; 4],
                                },
                                ReferenceFrames: from_iter_or(frames.iter().filter_map(|frame| {
                                    let frame_poc = frame.poc?;
                                    Some(va::VAPictureHEVC{
                                        picture_id: frame.id,
                                        pic_order_cnt: frame_poc as i32,
                                        flags: reference.map(|r|
                                            r.short_term_pictures.iter().filter_map(|&ShortTermReferencePicture{delta_poc, used}| used.then(|| (current_poc as i8+delta_poc) as u8)).find(|&poc| poc == frame_poc).map(|poc|
                                                match poc.cmp(&current_poc) {
                                                    std::cmp::Ordering::Less => VA_PICTURE_HEVC_RPS_ST_CURR_BEFORE,
                                                    std::cmp::Ordering::Greater => VA_PICTURE_HEVC_RPS_ST_CURR_AFTER,
                                                    _ => unreachable!()
                                                }
                                            ).unwrap_or(0) |
                                            if r.long_term_pictures.iter().any(|&LongTermReferencePicture{poc, used}| used && poc == frame_poc) { VA_PICTURE_HEVC_RPS_LT_CURR } else {0}
                                        ).unwrap_or(0),
                                        va_reserved:[0;_]
                                    })
                                }), va::VAPictureHEVC{picture_id:0,pic_order_cnt:0,flags:0,va_reserved:[0;_]}),
                                num_tile_columns_minus1: pps.tiles.1.as_ref().map(|t| t.columns.len() - 1).unwrap_or(0) as u8,
                                num_tile_rows_minus1: pps.tiles.1.as_ref().map(|t| t.rows.len() - 1).unwrap_or(0) as u8,
                                column_width_minus1: pps.tiles.1.as_ref().map(|t| from_iter(t.columns.into_iter().map(|w| w-1))).unwrap_or_default(),
                                row_height_minus1: pps.tiles.1.as_ref().map(|t| from_iter(t.rows.into_iter().map(|h| h-1))).unwrap_or_default(),
                                st_rps_bits: reference.map(|r| r.short_term_picture_set_encoded_bits_len_skip).flatten().unwrap_or(0),
                                va_reserved: [0; _]
                            } as *const _ as *mut _, &mut buffer)
                        });
                        if let Some(scaling_list) = pps.scaling_list.as_ref().or(sps.scaling_list.as_ref()) {
                            check(unsafe{va::vaCreateBuffer(va, context, VABufferType_VAIQMatrixBufferType, std::mem::size_of::<va::VAIQMatrixBufferHEVC> as u32, 1, &mut VAIQMatrixBufferHEVC{
                                ScalingList4x4: scaling_list.x4,
                                ScalingList8x8: scaling_list.x8,
                                ScalingListDC16x16: scaling_list.x16.map(|x| x.0),
                                ScalingList16x16: scaling_list.x16.map(|x| x.1),
                                ScalingListDC32x32: scaling_list.x32.map(|x| x.0),
                                ScalingList32x32: scaling_list.x32.map(|x| x.1),
                                va_reserved: [0; _],
                            } as *const _ as *mut _, &mut buffer)});
                        }
                    } // first_slice
                } // !dependent_slice_segment
                let sh = sh.as_ref().unwrap();
                let mut buffer = 0;
                let prediction_weights = sh.inter.as_ref().map(|s| s.prediction_weights.clone()/*.map(|LumaChroma{luma: l, chroma: c}| LumaChroma{
                    luma: Tables{ log2_denom_weight: l.log2_denom_weight, pb: l.pb.map(|t| WeightOffset{weight: t.map(|w| w.weight - (1<<l.log2_denom_weight)), offset: t.map(|w| w.offset)})) },
                    chroma: c.map(|c| Tables{ log2_denom_weight: c.log2_denom_weight as i8 - l.log2_denom_weight as i8, pb: c.pb.map(|t| t.map(|p| p.map(|w| WeightOffset{weight: w.weight - (1<<c.log2_denom_weight), offset: w.offset})))}),
                })*/).flatten().unwrap_or_default();
                check(unsafe{vaCreateBuffer(va, context, VABufferType_VASliceParameterBufferType, std::mem::size_of::<VASliceParameterBufferHEVC>() as u32, 1, &mut VASliceParameterBufferHEVC{
                    slice_data_size: data.len() as u32,
                    slice_data_offset: 0,
                    slice_data_flag: VA_SLICE_DATA_FLAG_ALL,
                    slice_data_byte_offset: (s.bits_offset() + 1 + 7) as u32 / 8, // Add 1 to the bits count here to account for the byte_alignment bit, which always is at least one bit and not accounted for otherwise
                    slice_segment_address: slice_segment_address.unwrap_or(0) as u32,
                    slice_qp_delta: sh.qp_delta,
                    slice_cb_qp_offset: sh.qp_offsets.map(|o| o.0).unwrap_or(0),
                    slice_cr_qp_offset: sh.qp_offsets.map(|o| o.1).unwrap_or(0),
                    slice_beta_offset_div2: sh.deblocking_filter.as_ref().map(|f| f.beta_offset / 2).unwrap_or(0),
                    slice_tc_offset_div2: sh.deblocking_filter.as_ref().map(|f| f.tc_offset / 2).unwrap_or(0),
                    collocated_ref_idx: sh.inter.as_ref().map(|i| i.collocated_list.map(|(_,index)| index.unwrap_or(0))).flatten().unwrap_or(0xFF),
                    five_minus_max_num_merge_cand: sh.inter.as_ref().map(|s| 5 - s.max_num_merge_cand).unwrap_or(0),
                    num_ref_idx_l0_active_minus1: sh.inter.as_ref().map(|s| s.active_references.p-1).unwrap_or(0) as u8,
                    num_ref_idx_l1_active_minus1: sh.inter.as_ref().map(|s| s.active_references.b.map(|len| len-1)).flatten().unwrap_or(0) as u8,
                    LongSliceFlags: _VASliceParameterBufferHEVC__bindgen_ty_1{fields: _VASliceParameterBufferHEVC__bindgen_ty_1__bindgen_ty_1{_bitfield_align_1: [], _bitfield_1: _VASliceParameterBufferHEVC__bindgen_ty_1__bindgen_ty_1::new_bitfield_1(
                        /*LastSliceOfPic*/0,
                        dependent_slice_segment as _,
                        sh.slice_type as u32,
                        sh.color_plane_id as u32,
                        sh.sample_adaptive_offset.luma as _,
                        sh.sample_adaptive_offset.chroma.unwrap_or(false) as _,
                        sh.inter.as_ref().map(|i| i.mvd_l1_zero).unwrap_or(false) as _,
                        sh.inter.as_ref().map(|i| i.cabac_init).unwrap_or(false) as _,
                        sh.reference.as_ref().map(|i| i.temporal_motion_vector_predictor).unwrap_or(false) as _,
                        sh.deblocking_filter.is_none() as _,
                        sh.inter.as_ref().map(|i| i.collocated_list.map(|(collocated,_)| !collocated)).flatten().unwrap_or(false) as _,
                        sh.loop_filter_across_slices as _,
                        /*reserved*/ 0
                    )}},
                    luma_log2_weight_denom: prediction_weights.luma.log2_denom_weight,
                    delta_chroma_log2_weight_denom: prediction_weights.chroma.as_ref().map(|c| c.log2_denom_weight as i8 - prediction_weights.luma.log2_denom_weight as i8).unwrap_or(0),
                    delta_luma_weight_l0: prediction_weights.luma.pb.p.weight,
                    delta_chroma_weight_l0: prediction_weights.chroma.clone().unwrap_or_default().pb.p.weight,
                    luma_offset_l0: prediction_weights.luma.pb.p.offset,
                    ChromaOffsetL0: prediction_weights.chroma.clone().unwrap_or_default().pb.p.offset,
                    delta_luma_weight_l1: prediction_weights.luma.pb.b.unwrap_or_default().weight,
                    delta_chroma_weight_l1: prediction_weights.chroma.clone().unwrap_or_default().pb.b.unwrap_or_default().weight,
                    luma_offset_l1: prediction_weights.luma.pb.b.unwrap_or_default().offset,
                    ChromaOffsetL1: prediction_weights.chroma.unwrap_or_default().pb.b.unwrap_or_default().offset,
                    RefPicList: sh.reference.as_ref().map(|r| {
                        fn index(frames: &[Frame], ref_poc: u8) -> u8 { frames.iter().filter_map(|frame| frame.poc).position(|poc| poc == ref_poc).unwrap_or_else(|| panic!("Missing {ref_poc}")) as u8 }
                        pub fn list<T>(iter: impl std::iter::IntoIterator<Item=T>) -> Box<[T]> { iter.into_iter().collect() }
                        pub fn sort_by<T>(mut s: Box<[T]>, f: impl Fn(&T,&T)->std::cmp::Ordering) -> Box<[T]> { s.sort_by(f); s }
                        pub fn map<T,U>(iter: impl std::iter::IntoIterator<Item=T>, f: impl Fn(T)->U) -> Box<[U]> { list(iter.into_iter().map(f)) }
                        fn stps(frames: &[Frame], r: &SHReference, f: impl Fn(std::cmp::Ordering)->std::cmp::Ordering) -> Box<[u8]> {
                            let current_poc = r.poc;
                            map(&*sort_by(
                                list(r.short_term_pictures.into_iter().filter_map(|&ShortTermReferencePicture{delta_poc, used}|(used && f(delta_poc.cmp(&0)) == std::cmp::Ordering::Greater).then(|| delta_poc))),
                                |a,b| f(a.cmp(b))
                            ), |delta_poc| index(frames, (current_poc as i8+delta_poc) as u8))
                        }
                        let frames = frames.as_ref().unwrap();
                        let stps_after = stps(frames, r, |o| o);
                        let stps_before = stps(frames, r, |o| o.reverse());
                        let ltps = map(sort_by(list(r.long_term_pictures.into_iter().filter_map(|&LongTermReferencePicture{poc, used}| used.then(|| poc))), Ord::cmp).into_iter(), |&poc| index(frames, poc));
                        [ from_iter([&*stps_before, &*stps_after, &*ltps].into_iter().flatten().copied()), from_iter([&*stps_after, &*stps_before, &*ltps].into_iter().flatten().copied())]
                    }).unwrap_or_default(),
                    num_entry_point_offsets: 0,
                    entry_offset_to_subset_array: 0,
                    slice_data_num_emu_prevn_bytes: 0,
                    va_reserved: [0; _]
                } as *const _ as *mut _, &mut buffer)});
                let mut buffer = 0;
                check(unsafe{va::vaCreateBuffer(va, context, VABufferType_VASliceDataBufferType, data.len() as _, 1, data.as_ptr() as *const std::ffi::c_void as *mut _, &mut buffer)});
            }
            _ => panic!("Unit {unit_type:?}"),
        };
    };
    //use nom::{multi::{length_count, length_value}, sequence::pair, combinator::map, number::complete::{u8,be_u16}};
    //length_count::<_,_,_,nom::error::Error<_>,_,_>(u8, pair(map(u8, |t| t&0x3f), length_count(be_u16, length_value(be_u16, |nal:&[u8]| Ok((nal,parse_nal(nal)))))))(&video.codec_private.as_ref().unwrap()[22..]).unwrap();
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
