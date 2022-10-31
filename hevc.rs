#![allow(dead_code)]
use crate::array;
fn ceil_log2(x: usize) -> u8 { ((x-1)<<1).ilog2() as u8 }

#[allow(non_camel_case_types)] #[repr(u8)] #[derive(Clone,Copy,num_derive::FromPrimitive,Debug)] pub enum NAL {
    TRAIL_N, TRAIL_R, TSA_N, TSA_R, STSA_N, STSA_R, RADL_N, RADL_R, RASL_N, RASL_R,
    BLA_W_LP/*Leading Picture*/ = 16, BLA_W_RADL/*Random Access Decodable Leading*/, BLA_N_LP, IDR_W_RADL, IDR_N_LP, CRA_NUT,
    VPS = 32, SPS, PPS, AUD,
	EOS_NUT, EOB_NUT, FD_NUT,
    SEI_PREFIX
}
#[allow(non_snake_case)] pub fn Intra_Random_Access_Picture(unit: NAL) -> bool { 16 <= unit as u8 && unit as u8 <= 23 }
#[allow(non_snake_case)] pub fn Instantaneous_Decoder_Refresh(unit: NAL) -> bool { use NAL::*; matches!(unit, IDR_W_RADL|IDR_N_LP) }

#[derive(Clone,Copy,num_derive::FromPrimitive,Debug)] pub enum SliceType { B, P, I }
use num_traits::FromPrimitive;

use crate::bit::Reader;

fn profile_tier_level(s: &mut Reader, max_layers: usize) {
    let _profile_space = s.bits(2);
    let _tier = s.bit();
    #[derive(num_derive::FromPrimitive)] pub enum Profile { Main = 1, Main10 }
    let profile = Profile::from_u32(s.bits(5)).unwrap();
    assert!(matches!(profile, Profile::Main10));
    let _profile_compatibility = s.u32();
    let _progressive_source = s.bit();
    let _interlaced_source = s.bit();
    let _non_packed_constraint = s.bit();
    let _frame_only_constraint = s.bit();
    s.skip(44);
    let _level_idc = s.u8();
    if max_layers > 1 {
        let layers : [_; 8] = std::array::from_fn(|_| (s.bit(), s.bit()));
        let _layers = layers.map(|(profile, level)| {
            if profile { s.skip(44); s.skip(43); }
            if level { s.u8(); }
        });
    }
}

#[derive(Default,Clone,Copy)] pub struct LayerOrdering { pub max_dec_pic_buffering: u8, #[allow(dead_code)] num_reorder_pics: u8, #[allow(dead_code)] max_latency_increase: u8 }
fn layer_ordering(s: &mut Reader, max_layers: usize) -> Box<[LayerOrdering]> {
    (0 .. (if s.bit() { max_layers } else { 1 })).map(|_| LayerOrdering{
        max_dec_pic_buffering: 1 + s.ue() as u8,
        num_reorder_pics: s.ue() as u8,
        max_latency_increase: {let mut v = s.ue(); if v > 0 { v -= 1; } v} as u8,
    }).collect()
}

fn hrd_parameters(s: &mut Reader, common_inf: bool, max_layers: usize) {
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
        fn sub_layer_hrd_parameter(s: &mut Reader, nb_cpb: usize, subpic_params: bool) {
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

pub struct ScalingList {
    pub x4: [[u8; 16]; 6],
    pub x8: [[u8; 64]; 6],
    pub x16: [(u8, [u8; 64]); 6],
    pub x32: [(u8, [u8; 64]); 2],
}

fn scaling_list(s: &mut Reader) -> ScalingList {
    let mut x4 = [(0, [16; _]); _];
    let mut x8 = [(0, [64; _]); _];
    let mut x16 = [(16, [64; _]); _];
    let mut x32 = [(16, [64; _]); _];
    if s.bit() {
        fn decompress<const DC: bool, const N: usize, const M: usize>(s: &mut Reader, matrices: &mut [(u8, [u8; N*N]); M]) {
            for matrix_id in 0..M {
                matrices[matrix_id] = if !s.bit() { matrices[matrix_id - s.ue() as usize] } else {
                    let dc = 8 + if DC { s.se() } else { 0 } as u8;
                    let diagonally_ordered_coefficients = std::iter::successors(Some(dc), |&last| Some((last as i32 + 0x100 + s.se()) as u8)).take(N*N).collect::<Box<_>>();
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

pub struct PulseCodeModulation { pub bit_depth: u8, pub bit_depth_chroma: u8, pub log2_min_coding_block_size: u8, pub log2_diff_max_min_coding_block_size: u8, pub loop_filter_disable: bool }
#[derive(Clone)] pub struct ShortTermReferencePicture { pub delta_poc: i8, pub used: bool }
impl std::fmt::Debug for ShortTermReferencePicture { fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result { write!(f, "{}", self.delta_poc) } }
#[derive(Clone)] pub struct LongTermReferencePicture { pub poc: u32, pub used: bool }

pub struct SPS {
    pub separate_color_plane: bool,
    pub chroma_format_idc: u8,
    pub width: u16,
    pub height: u16,
    pub bit_depth: u8,
    pub log2_max_poc_lsb: u8,
    pub layer_ordering: Box<[LayerOrdering]>, // 8
    pub log2_min_coding_block_size: u8,
    pub log2_diff_max_min_coding_block_size: u8,
    pub log2_min_transform_block_size: u8,
    pub log2_diff_max_min_transform_block_size: u8,
    pub max_transform_hierarchy_depth_inter: u8,
    pub max_transform_hierarchy_depth_intra: u8,
    pub scaling_list: Option<ScalingList>,
    pub asymmetric_motion_partitioning: bool,
    pub sample_adaptive_offset: bool,
    pub pulse_code_modulation: Option<PulseCodeModulation>,
    pub short_term_reference_picture_sets: Box<[Box<[ShortTermReferencePicture]>]>,
    pub long_term_reference_picture_set: Option<Box<[LongTermReferencePicture]>>,
    pub temporal_motion_vector_predictor: bool,
    pub strong_intra_smoothing: bool,
}

#[derive(Clone)] pub struct DeblockingFilter { pub beta_offset: i8, pub tc_offset: i8 }
pub struct Tiles {pub columns: Box<[u16]>, pub rows: Box<[u16]>, pub loop_filter_across_tiles: bool}

pub struct PPS {
    pub sps: usize,
    pub dependent_slice_segments: bool,
    pub output: bool,
    pub num_extra_slice_header_bits: u8,
    pub sign_data_hiding: bool,
    pub cabac_init: bool,
    pub num_ref_idx_default_active: [u8; 2],
    pub init_qp_minus26: i8,
    pub constrained_intra_prediction: bool,
    pub transform_skip: bool,
    pub diff_cu_qp_delta_depth: Option<u8>,
    pub cb_qp_offset: i8,
    pub cr_qp_offset: i8,
    pub slice_chroma_qp_offsets: bool,
    pub weighted_prediction: bool,
    pub weighted_biprediction: bool,
    pub transquant_bypass: bool,
    pub tiles: (/*entropy_coding_sync*/bool, Option<Tiles>),
    pub loop_filter_across_slices: bool,
    pub deblocking_filter: Option<(/*deblocking_filter_override*/bool, Option<DeblockingFilter>)>,
    pub scaling_list: Option<ScalingList>,
    pub lists_modification: bool,
    pub log2_parallel_merge_level: u8,
    pub slice_header_extension: bool,
    #[allow(dead_code)] pps_extension: ()//bool
}

fn decode_short_term_reference_picture_set(s: &mut Reader, sets: &[Box<[ShortTermReferencePicture]>], slice_header: bool) -> Box<[ShortTermReferencePicture]> {
    if !sets.is_empty() && /*predict*/s.bit() {
        let ref set = &sets[sets.len()-1-if slice_header {s.ue() as usize} else {0}];
        let delta = if s.bit() { -1 } else { 1 } * (s.exp_golomb_code()) as i8;
        let mut parse = |&ShortTermReferencePicture{delta_poc,..}| { let used = s.bit(); if used || s.bit() { Some(ShortTermReferencePicture{delta_poc:delta_poc+delta, used}) } else { None } };
		let mut set = set.iter().filter_map(&mut parse).collect::<Vec<_>>();
        set.extend(parse(&ShortTermReferencePicture{delta_poc:0,used:false}));
        let (mut negative, mut positive) = set.into_iter().partition::<Vec<_>,_>(|&ShortTermReferencePicture{delta_poc,..}| delta_poc<0);
		negative.sort_by_key(|&ShortTermReferencePicture{delta_poc,..}| -delta_poc);
		positive.sort_by_key(|&ShortTermReferencePicture{delta_poc,..}| delta_poc);
		[negative,positive].concat().into_boxed_slice()
    } else {
        let negative = s.ue() as usize;
        let positive = s.ue() as usize;
        let mut set = Vec::with_capacity(negative+positive);
        type P = ShortTermReferencePicture;
        {let mut a = 0; for _ in 0..negative { a=a-s.exp_golomb_code() as i8; set.push(P{delta_poc: a, used: s.bit()}); }}
        {let mut a = 0; for _ in 0..positive { a=a+s.exp_golomb_code() as i8; set.push(P{delta_poc: a, used: s.bit()}); }}
		set.into_boxed_slice()
    }
}

pub struct SHReference {
    pub poc: u32,
    pub short_term_pictures: Box<[ShortTermReferencePicture]>,
    pub short_term_picture_set_encoded_bits_len_skip: Option<u32>,
    pub long_term_pictures: Box<[LongTermReferencePicture]>,
    pub temporal_motion_vector_predictor: bool
}

#[derive(Clone,Copy,Default)] pub struct MayB<T> { pub p: T, pub b: Option<T> }
impl<T> MayB<T> {
    fn as_ref(&self) -> MayB<&T> { MayB{p: &self.p, b: self.b.as_ref()} }
    pub fn map<U>(self, mut f: impl FnMut(T)->U) -> MayB<U> { MayB{p: f(self.p), b: self.b.map(f)} }
	pub fn zip<'t, U>(self, o: &'t MayB<U>) -> MayB<(T,&'t U)> { MayB{p: (self.p, &o.p), b: self.b.map(|b| (b, o.b.as_ref().unwrap()))} }
}
#[derive(Clone,Default,Debug)] pub struct LumaChroma<L, C=L> { pub luma: L, pub chroma: Option<C> }
#[derive(Clone,Copy,Default,Debug)] pub struct WeightOffset<W, O> { pub weight: W, pub offset: O }
pub type WeightOffsets<W, O, const N: usize> = WeightOffset<[W; N],[O; N]>;
#[derive(Clone,Default)] pub struct Tables<T, W=u8> { pub log2_denom_weight: W, pub pb: MayB<T> }
pub type PredictionWeights<const N: usize> = LumaChroma<Tables<WeightOffsets<i8,i8,N>>, Tables<WeightOffsets<[i8;2],[i8;2],N>>>;
pub struct SHInter {
    pub active_references: MayB<u8>,
    pub list_entry_lx: Option<MayB<Option<Box<[u8]>>>>,
    pub mvd_l1_zero: bool,
    pub cabac_init: bool,
    pub collocated_list: Option<(bool, Option<u8>)>,
    pub prediction_weights: Option<PredictionWeights<15>>,
    pub max_num_merge_cand: u8
}

pub struct SliceHeader {
    pub slice_type: SliceType,
    #[allow(dead_code)] output: bool,
    pub color_plane_id: u8,
    pub reference: Option<SHReference>,
    pub sample_adaptive_offset: LumaChroma<bool>,
    pub inter: Option<SHInter>,
    pub qp_delta: i8,
    pub qp_offsets: Option<(i8, i8)>,
    pub deblocking_filter: Option<DeblockingFilter>,
    pub loop_filter_across_slices: bool,
}

pub struct HEVC {
	vps : [Option<VPS>; 16],
	pub sps : [Option<SPS>; 16],
	pub pps: [Option<PPS>; 64],
	pub slice_header: Option<SliceHeader>,
	poc_tid0: u32,
}
impl HEVC { pub fn new() -> Self { Self{vps: Default::default(), sps: Default::default(), pps: [();_].map(|_|None), slice_header: None, poc_tid0: 0} } }

pub struct Slice<'t> {
	pub pps: usize,
	pub unit: NAL,
	pub escaped_data: &'t [u8],
	pub slice_data_byte_offset: usize,
	pub dependent_slice_segment: bool,
	pub slice_segment_address: Option<usize>,
}

impl crate::Decoder for HEVC {
	type Output<'t> = Slice<'t>;
fn decode<'t>(&mut self, escaped_data: &'t [u8]) -> Option<Self::Output<'t>> {
	let data = escaped_data[0..2].iter().copied().chain(escaped_data.array_windows().filter_map(|&[a,b,c]| (!(a == 0 && b== 0 && c == 3)).then(|| c))).collect::<Vec<u8>>();
	let ref mut s = Reader::new(&data);
	assert!(s.bit() == false); //forbidden_zero_bit
	let unit = NAL::from_u32(s.bits(6)).unwrap();
	let temporal_id = if !matches!(unit, NAL::EOS_NUT|NAL::EOB_NUT) {
		let layer_id = s.bits(6);
		assert!(layer_id == 0);
		s.bits(3) - 1
	} else { 0 };
	match unit {
		NAL::SEI_PREFIX => {
			fn decode(s: &mut Reader) -> u32 {
				let mut value = 0;
				while {
					let byte = s.u8();
					value += byte as u32;
					byte == 0xFF
				}{}
				value
			}
			const BUFFERING_PERIOD : u8 = 0;
			const PICTURE_TIMING : u8 = 1;
			const USER_DATA_REGISTERED_ITU_T_T35 : u8 = 4;
			const USER_DATA_UNREGISTERED : u8 = 5;
			const ACTIVE_PARAMETER_SETS : u8 = 129;
			const MASTERING_DISPLAY_INFO : u8 = 137;
			const CONTENT_LIGHT_LEVEL_INFO : u8 = 144;
			let sei_type = s.u8();
			let _size = decode(s);
			match sei_type {
				ACTIVE_PARAMETER_SETS => {
					let _active_video_parameter_set_id = s.bits(4);
					let _self_contained_cvs = s.bit();
					let _no_parameter_set_update = s.bit();
					let _num_sps_ids = s.exp_golomb_code();
					let _active_seq_parameter_set_id = s.ue();
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
		NAL::AUD => {}
		NAL::VPS => {
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
			self.vps[id] = Some(VPS{});
		}
		NAL::SPS => {
			let vps = s.bits(4) as usize;
			assert!(vps == 0);
			let max_layers = s.bits(3) as usize + 1;
			assert!(max_layers <= 7);
			let _temporal_id_nesting = s.bit();
			profile_tier_level(s, max_layers);
			let id = s.ue() as usize;
			assert!(id < self.sps.len(), "{id}");
			let chroma_format_idc = s.ue() as u8;
			let separate_color_plane = if chroma_format_idc == 3 { s.bit() } else { false };
			let chroma_format_idc = if separate_color_plane { 0 } else { chroma_format_idc };
			let width = s.ue() as u16;
			let height = s.ue() as u16;
			assert!(width <= 3840 && height <= 2160);
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
				for _ in 0..sets.capacity() {
					let set = decode_short_term_reference_picture_set(s, &sets, false);
					sets.push(set);
				}
				sets.into_boxed_slice()
			};
			let long_term_reference_picture_set = s.bit().then(|| (0..s.ue()).map(|_| LongTermReferencePicture{poc: s.bits(log2_max_poc_lsb), used: s.bit()}).collect());
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
			self.sps[id] = Some(SPS{separate_color_plane, chroma_format_idc, width, height, bit_depth, log2_max_poc_lsb, layer_ordering,
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
		NAL::PPS => {
			let id = s.ue() as usize;
			self.pps[id] = Some(PPS{
				sps: s.ue() as usize,
				dependent_slice_segments: s.bit(),
				output: s.bit(),
				num_extra_slice_header_bits: s.bits(3) as u8,
				sign_data_hiding: s.bit(),
				cabac_init: s.bit(),
				num_ref_idx_default_active: [(); 2].map(|_| 1 + s.ue() as u8),
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
					(entropy_coding_sync, tiles.then(|| /*{
						let _columns = 1 + s.ue();
						let _rows = 1 + s.ue();
						let (columns, rows) = if !s.bit() { // not uniform spacing
							unimplemented!();//let _column_widths = (0.._num_tile_columns-1).map(|_| s.ue() + 1).collect();
							//let _row_heights = (0.._num_tile_rows-1).map(|_| s.ue() + 1).collect();
						} else { unimplemented!(); };
						let loop_filter_across_tiles = s.bit();
						Tiles{columns, rows, loop_filter_across_tiles}
					}*/unimplemented!()))
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
				pps_extension: {assert!(s.bit()==false)},
			});
		}
		NAL::IDR_W_RADL|NAL::IDR_N_LP|NAL::TRAIL_R|NAL::TRAIL_N => {
			let first_slice = s.bit();
			if Intra_Random_Access_Picture(unit) { let _no_output_of_prior_pics = s.bit(); }
			let pps_id = s.ue() as usize;
			let pps = self.pps[pps_id].as_ref().unwrap();
			let ref sps = self.sps[pps.sps].as_ref().unwrap();
			let (dependent_slice_segment, slice_segment_address) = if !first_slice {
				let dependent_slice_segment = pps.dependent_slice_segments && s.bit();
				let log2_ctb_size = sps.log2_min_coding_block_size + sps.log2_diff_max_min_coding_block_size;
				let ctb_size = 1<<log2_ctb_size;
				let [ctb_width, ctb_height] = [sps.width, sps.height].map(|size| (size +(ctb_size-1)) >> log2_ctb_size);
				(dependent_slice_segment, Some(s.bits(ceil_log2((ctb_width*ctb_height) as usize) as u8) as usize))
			} else { (false, None) };
			if !dependent_slice_segment {
				s.skip(pps.num_extra_slice_header_bits);
				let slice_type = SliceType::from_u32(s.ue()).unwrap();
				let output = pps.output && s.bit();
				let color_plane_id = if sps.separate_color_plane { s.bits(2) } else { 0 } as u8;
				let reference = (!Instantaneous_Decoder_Refresh(unit)).then(|| {
					let poc_lsb = s.bits(sps.log2_max_poc_lsb) as u16;
					let max_poc_lsb = (1 << sps.log2_max_poc_lsb) as u16;
					let prev_poc_lsb = (self.poc_tid0 % max_poc_lsb as u32) as u16;
					let prev_poc_msb = self.poc_tid0 - prev_poc_lsb as u32;
					let poc_msb = if poc_lsb < prev_poc_lsb && prev_poc_lsb - poc_lsb >= max_poc_lsb / 2 {
						prev_poc_msb + max_poc_lsb as u32
					} else if poc_lsb > prev_poc_lsb && poc_lsb - prev_poc_lsb > max_poc_lsb / 2 {
						prev_poc_msb - max_poc_lsb as u32
					} else {
						prev_poc_msb
					};
					let poc = if matches!(unit, NAL::BLA_W_RADL|NAL::BLA_W_LP|NAL::BLA_N_LP) { 0 } else { poc_msb } + poc_lsb as u32;
					if temporal_id == 0 && !matches!(unit,NAL::TRAIL_N|NAL::TSA_N|NAL::STSA_N|NAL::RADL_N|NAL::RASL_N|NAL::RADL_R|NAL::RASL_R) { self.poc_tid0 = poc; }
					let (short_term_picture_set, short_term_picture_set_encoded_bits_len_skip) = if !s.bit() { let start = s.available()+1; (decode_short_term_reference_picture_set(s, &sps.short_term_reference_picture_sets, true), Some((start - s.available()) as u32)) }
					else {
						let set = if sps.short_term_reference_picture_sets.len()>1 { s.bits(ceil_log2(sps.short_term_reference_picture_sets.len())) as usize } else { 0 };
						(sps.short_term_reference_picture_sets[set].clone(), None)
					};
					SHReference{
						poc, short_term_pictures: short_term_picture_set, short_term_picture_set_encoded_bits_len_skip,
						long_term_pictures: sps.long_term_reference_picture_set.as_ref().map(|set| {
							let sequence = if !set.is_empty() { s.ue() } else { 0 };
							let slice = s.ue();
							let mut v = Vec::with_capacity((sequence+slice) as usize);
							let poc = |s: &mut Reader, msb: &mut u16| -> u32 { if !s.bit() { 0 } else {
								*msb += s.ue() as u16;
								poc - ((*msb as u32) << sps.log2_max_poc_lsb) - poc_lsb as u32
							}};
							v.extend((0..sequence).scan(0, |msb, _| { let mut p = set[if set.len() > 1 { s.bits(ceil_log2(set.len())) } else { 0 } as usize].clone(); p.poc += poc(s, msb); Some(p)}));
							v.extend((0..slice).scan(0, |msb, _| { let mut p = LongTermReferencePicture{poc: s.bits(sps.log2_max_poc_lsb), used: s.bit()}; p.poc += poc(s, msb); Some(p)}));
							v.into_boxed_slice()
						}).unwrap_or_default(),
						temporal_motion_vector_predictor: sps.temporal_motion_vector_predictor && s.bit()
					}
				});
				if temporal_id == 0 && !matches!(unit, NAL::TRAIL_N|NAL::TSA_N|NAL::STSA_N|NAL::RADL_N|NAL::RASL_N|NAL::RADL_R|NAL::RASL_R) { self.poc_tid0 = reference.as_ref().map(|r| r.poc).unwrap_or(0); }
				let chroma = sps.chroma_format_idc>0;
				let sample_adaptive_offset = sps.sample_adaptive_offset.then(|| LumaChroma{luma: s.bit(), chroma: chroma.then(|| s.bit())}).unwrap_or(LumaChroma{luma: false, chroma: chroma.then(|| false)});
				let inter = matches!(slice_type, SliceType::P | SliceType::B).then(|| {
					let b = matches!(slice_type, SliceType::B);
					let active_references =
						if s.bit() { MayB{p: 1 + s.ue() as u8, b: b.then(|| 1 + s.ue() as u8)} }
						else { MayB{p:pps.num_ref_idx_default_active[0], b: b.then_some(pps.num_ref_idx_default_active[1])} };
					let reference = reference.as_ref().unwrap();
					let references_len = reference.short_term_pictures.iter().filter(|p| p.used).count() + reference.long_term_pictures.iter().filter(|p| p.used).count();
					SHInter{
						active_references,
						list_entry_lx: (pps.lists_modification && references_len > 1).then(|| active_references.map(|len| s.bit().then(|| (0..len).map(|_| s.bits(ceil_log2(references_len)) as u8).collect::<Box<_>>()))),
						mvd_l1_zero: b && s.bit(),
						cabac_init: pps.cabac_init && s.bit(),
						collocated_list: reference.temporal_motion_vector_predictor.then(|| {
							let collocated_list = b && !s.bit();
							(collocated_list, (if collocated_list { active_references.b.unwrap() } else { active_references.p } > 1).then(|| s.ue() as u8))
						}),
						prediction_weights: ((matches!(slice_type, SliceType::P) && pps.weighted_prediction) || (b && pps.weighted_biprediction)).then(|| {
							let log2_denom_weight_luma = s.ue() as u8;
							let log2_denom_weight_chroma = chroma.then(|| (log2_denom_weight_luma as i32 + s.se()) as u8);
							let ref pb = active_references.map(|active_references| {
								array(active_references as usize, || s.bit()).zip(chroma.then(|| array(active_references as usize, || s.bit())).map(|a| a.map(Some)).unwrap_or_default()).map(|(l,c)| LumaChroma{
									luma: l.then(|| WeightOffset{weight: s.se() as i8, offset: s.se() as i8}).unwrap_or_default(),
									chroma: c.map(|c| c.then(|| [(); 2].map(|_| { let w = s.se(); WeightOffset{weight: w as i8, offset: s.se() as i8}})).unwrap_or_default())
								})
							});
							LumaChroma {
								luma: Tables{ log2_denom_weight: log2_denom_weight_luma, pb: pb.as_ref().map(|t| WeightOffset{weight: t.each_ref().map(|lc| lc.luma.weight), offset: t.each_ref().map(|lc| lc.luma.offset)}) },
								chroma: chroma.then(|| Tables{ log2_denom_weight: log2_denom_weight_chroma.unwrap(), pb: pb.as_ref().map(|t| WeightOffset{weight: t.each_ref().map(|lc| lc.chroma.unwrap().map(|w| w.weight)), offset: t.each_ref().map(|lc| lc.chroma.unwrap().map(|w| w.offset))})})
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
				self.slice_header = Some(SliceHeader{slice_type, output, color_plane_id, reference, sample_adaptive_offset, inter, qp_delta, qp_offsets, deblocking_filter, loop_filter_across_slices});
			} // !dependent_slice_segment
			let slice_data_byte_offset = (s.bits_offset() + 1 + 7) / 8; // Add 1 to the bits count here to account for the byte_alignment bit, which always is at least one bit and not accounted for otherwise
			assert!(slice_data_byte_offset <= 36, "{slice_data_byte_offset}"); // Assumes no escape
			return Some(Slice{pps: pps_id, unit, escaped_data, slice_data_byte_offset, dependent_slice_segment, slice_segment_address});
		}
		_ => unimplemented!(),
	}
	None
}}