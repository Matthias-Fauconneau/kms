#![feature(int_log,unchecked_math)]
#![allow(dead_code)]

use nom::combinator::iterator;
use nom::number::complete::be_u32;
fn unit(input: &[u8]) -> nom::IResult<&[u8], &[u8]> { let (rest, length) = be_u32(input)?; let length = length as usize; Ok((&rest[length..], &rest[..length])) }

struct BitReader<'t> {
    pub word: u64,
    ptr: *const u8,
    end: *const u8,
    count: u32,
    phantom: ::core::marker::PhantomData<&'t [u8]>,
}
impl<'t> BitReader<'t> {
    fn new(data: &'t [u8]) -> Self {
        Self {
            ptr: data.as_ptr(),
            end: data.as_ptr_range().end,
            word: 0,
            count: 0,
            phantom: ::core::marker::PhantomData,
        }
    }
    unsafe fn refill(&mut self) {
        self.word |= core::ptr::read_unaligned(self.ptr as *const u64).to_be() >> self.count;
        self.ptr = self.ptr.add(7);
        self.ptr = self.ptr.sub((self.count as usize >> 3) & 7);
        self.count |= 56;
    }
    fn peek(&self, count: u32) -> u64 { unsafe { self.word.unchecked_shr(64 - count as u64) } }
    fn consume(&mut self, count: u32) { self.word <<= count; self.count -= count; }
    #[track_caller] fn bits(&mut self, count: u32) -> u64 {
        if count > self.count { unsafe { self.refill(); } }
        let result = self.peek(count);
        self.consume(count);
        result
    }
    fn bit(&mut self) -> bool { self.bits(1) != 0 }
    fn u8(&mut self) -> u8 { self.bits(8) as u8 }
    fn u16(&mut self) -> u16 { self.bits(16) as u16 }
    fn u32(&mut self) -> u32 { self.bits(32) as u32 }
    fn ue(&mut self) -> u64 { // Exp-Golomb
        unsafe { self.refill(); }
        let count = self.word.leading_zeros();
        self.consume(count);
        self.bits(1+count) - 1
    }
    fn se(&mut self) -> i64 {
        let v = self.ue() as i64;
        let sign = -(v & 1);
        ((v >> 1) ^ sign) - sign
    }
    fn available(&self, len: usize) -> bool { self.count as usize + (self.end as usize-self.ptr as usize)*8 > len }
}

fn profile_tier_level(s: &mut BitReader, max_layers: usize) {
    assert!(s.available(88));
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
    println!("{max_layers} {_level_idc}");
    if max_layers > 1 {
        let layers : [_; 8] = std::array::from_fn(|_| (s.bit(), s.bit()));
        println!("{layers:?}");
        let _layers = layers.map(|(profile, level)| {
            if profile { s.bits(44); s.bits(43); }
            if level { s.u8(); }
        });
    }
}
#[derive(Default,Clone,Copy)] struct LayerOrdering { max_dec_pic_buffering: u64, num_reorder_pics: u64, max_latency_increase: u64 }
fn layer_ordering(s: &mut BitReader, max_layers: usize) -> [LayerOrdering; 8] {
    let mut layer_ordering = [LayerOrdering::default(); 8];
    for i in 0 .. if s.bit() { max_layers } else { 1 } {
        layer_ordering[i] = LayerOrdering{
            max_dec_pic_buffering: s.ue() + 1,
            num_reorder_pics: s.ue(),
            max_latency_increase: {let mut v = s.ue(); if v > 0 { v -= 1; } v},
        };
    }
    layer_ordering
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
                    /*pred_matrix_id_delta[i][j] =*/ s.ue();
                } else {
                    if i >= 2 { /*dc_coef_minus8[i][j] =*/ s.se(); }
                    /*delta_coef[i][j] =*/ for _ in 0..[16,64,64,64][i] { s.se(); }
                }
            }
        }
    } else { Default::default() }
}

pub const AUD: u8 = 35;
pub const SEI_PREFIX: u8 = 39;
pub const VPS: u8 = 32;
pub const SPS: u8 = 33;
pub const PPS: u8 = 34;
pub const IDR_N_LP: u8 = 20;
pub const TRAIL_R: u8 = 1;
pub const TRAIL_N: u8 = 0;

struct VPS {}
struct SPS {
    separate_colour_plane: bool,
    width: u32,
    height: u32,
    log2_max_poc_lsb: u32,
}
struct PPS {
    sps: usize,
    dependent_slice_segments: bool,
    output: bool,
    num_extra_slice_header_bits: u32,
    sign_data_hiding: bool,
    cabac_init: bool,
    num_ref_idx_l0_default_active: u64,
    num_ref_idx_l1_default_active: u64,
    pic_init_qp_minus26: i64,
    constrained_intra_pred: bool,
    transform_skip: bool,
    cu_qp_delta_depth: u64,
    cb_qp_offset: i64,
    cr_qp_offset: i64,
    pic_slice_level_chroma_qp_offsets: bool,
    weighted_pred: bool,
    weighted_bipred: bool,
    transquant_bypass: bool,
    tiles: (),
    seq_loop_filter_across_slices: bool,
    deblocking_filter: (),
    scaling_list: (),
    lists_modification: bool,
    log2_parallel_merge_level: u64,
    slice_header_extension: bool,
    pps_extension: bool
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
                            println!("SPS {id}");
                            assert!(id < sps.len(), "{id}");
                            let chroma_format_idc = s.ue();
                            let separate_colour_plane = if chroma_format_idc == 3 { s.bit() } else { false };
                            let _chroma_format_idc = if separate_colour_plane { 0 } else { chroma_format_idc };
                            let width = s.ue() as u32;
                            let height = s.ue() as u32;
                            if s.bit() { let (_left, _right, _top, _bottom) = (s.ue(), s.ue(), s.ue(), s.ue()); }
                            let _bit_depth = s.ue() + 8;
                            let _bit_depth_chroma = s.ue() + 8;
                            let log2_max_poc_lsb = s.ue() as u32 + 4;
                            layer_ordering(s, max_layers);
                            let _log2_min_cb_size = s.ue() + 3;
                            let _log2_diff_max_min_coding_block_size = s.ue();
                            let _log2_min_tb_size = s.ue() + 2;
                            let _log2_diff_max_min_transform_block_size = s.ue();
                            let _max_transform_hierarchy_depth_inter = s.ue();
                            let _max_transform_hierarchy_depth_intra = s.ue();
                            if s.bit() { let _scaling_list = scaling_list(s); }
                            let _amp = s.bit();
                            let _sao = s.bit();
                            if s.bit() { // PCM
                                let _bit_depth = s.bits(4) + 1;
                                let _bit_depth_chroma = s.bits(4) + 1;
                                let pcm_log2_min_pcm_cb_size = 3 + s.ue();
                                let _pcm_log2_max_pcm_cb_size = pcm_log2_min_pcm_cb_size + s.ue();
                                let _pcm_loop_filter_disable = s.bit();
                            }
                            let rpss_len = s.ue() as usize;
                            let mut rpss = Vec::<Vec<(i8, bool)>>::new();
                            for i in 0..rpss_len {
                                let rps = if i > 0 && s.bit() {
                                    let ref _reference = rpss[i-1-(if i == rpss_len { s.ue() as usize } else { 0 })];
                                    let _delta = if s.bit() { -1 } else { 1 } * (s.ue()+1) as i8;
                                    //let mut parse = |(reference,_)| { let used = s.bit(); if used || s.bit() { Some((reference+delta, used)) } else { None } };
                                    unimplemented!()//reference.iter().filter_map(parse).chain(std::iter::once(parse((0,false)))).collect()
                                } else {
                                    let negative = s.ue() as usize;
                                    let positive = s.ue() as usize;
                                    use std::iter::successors;
                                    successors(Some((0,false)), |(p,_)| Some((p-1-s.ue() as i8, s.bit()))).take(negative).collect::<Vec<_>>().into_iter().chain(successors(Some((0,false)), |(p,_)| Some((p+1+s.ue() as i8, s.bit()))).take(positive)).collect()
                                };
                                rpss.push(rps);
                            }
                            if s.bit() { let _/*lt_ref_pic_poc_lsb_sps, used_by_curr_pic_lt_sps*/ = (0..s.ue()).map(|_| (s.bits(log2_max_poc_lsb), s.bit())); }
                            let _temporal_mvp = s.bit();
                            let _strong_intra_smoothing = s.bit();
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
                            sps[id] = Some(SPS{separate_colour_plane, width, height, log2_max_poc_lsb});
                        },
                        PPS => {
                            let id = s.ue() as usize;
                            pps[id] = Some(PPS{
                                sps: s.ue() as usize,
                                dependent_slice_segments: s.bit(),
                                output: s.bit(),
                                num_extra_slice_header_bits: s.bits(3) as u32,
                                sign_data_hiding: s.bit(),
                                cabac_init: s.bit(),
                                num_ref_idx_l0_default_active: s.ue() + 1,
                                num_ref_idx_l1_default_active: s.ue() + 1,
                                pic_init_qp_minus26: s.se(),
                                constrained_intra_pred: s.bit(),
                                transform_skip: s.bit(),
                                cu_qp_delta_depth: if s.bit() { s.ue() } else { 0 },
                                cb_qp_offset: s.se(),
                                cr_qp_offset: s.se(),
                                pic_slice_level_chroma_qp_offsets: s.bit(),
                                weighted_pred: s.bit(),
                                weighted_bipred: s.bit(),
                                transquant_bypass: s.bit(),
                                tiles: {
                                    let _tiles = s.bit();
                                    let _entropy_coding_sync = s.bit();
                                    if _tiles {
                                        let _num_tile_columns = s.ue() + 1;
                                        let _num_tile_rows = s.ue() + 1;
                                        if !s.bit() { // not uniform spacing
                                            unimplemented!();//let _column_widths = (0.._num_tile_columns-1).map(|_| s.ue() + 1).collect();
                                            //let _row_heights = (0.._num_tile_rows-1).map(|_| s.ue() + 1).collect();
                                        }
                                        let _loop_filter_across_tiles = s.bit();
                                    }
                                },
                                seq_loop_filter_across_slices: s.bit(),
                                deblocking_filter: if s.bit() {
                                    let _deblocking_filter_override_enabled = s.bit();
                                    if !s.bit() { // not disable dbf
                                        let _beta_offset = 2 * s.se();
                                        let _tc_offset = 2 * s.se();
                                    }
                                },
                                scaling_list: if s.bit() { let _scaling_list = scaling_list(s); },
                                lists_modification: s.bit(),
                                log2_parallel_merge_level: s.ue() + 2,
                                slice_header_extension: s.bit(),
                                pps_extension: s.bit(),
                            });
                            println!("PPS {id} {}", pps[id].as_ref().unwrap().sps);
                        }
                        IDR_N_LP|TRAIL_R|TRAIL_N => {
                            println!("{}", match unit_type {IDR_N_LP=>"IDR_N_LP",TRAIL_R=>"TRAIL_R",TRAIL_N=>"TRAIL_N",_=>unreachable!()});
                            let first_slice_in_pic = s.bit();
                            #[allow(non_snake_case)] fn IRAP(unit_type: u8) -> bool { 16 <= unit_type && unit_type <= 23 }
                            if IRAP(unit_type) { let _no_output_of_prior_pics = s.bit(); }
                            let ref pps = pps[s.ue() as usize].as_ref().unwrap();
                            let ref sps = sps[pps.sps].as_ref().unwrap_or_else(|| panic!("{}", pps.sps));
                            let dependent_slice_segment = if !first_slice_in_pic {
                                let dependent_slice_segment = pps.dependent_slice_segments && s.bit();
                                let pic_size = sps.width * sps.height;
                                let slice_address_length = ((pic_size-1)<<1).ilog2();
                                assert!(slice_address_length < 24);
                                let _slice_segment_address = s.bits(slice_address_length);
                                dependent_slice_segment
                            } else { false };
                            if !dependent_slice_segment {
                                let _slice_reserved_undetermined = s.bits(pps.num_extra_slice_header_bits);
                                let _slice_type = s.ue();
                                let _output = pps.output && s.bit();
                                let _separate_colour_plane = if sps.separate_colour_plane { s.bits(2) } else { 0 };
                                let output_picture_number = if !matches!(unit_type, /*IDR_W_RADL|*/IDR_N_LP) {
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
        _ => panic!("{element:?}")
    }}
    Ok(())
}
