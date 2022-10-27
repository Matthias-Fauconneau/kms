fn from_iter_or<T: Copy, const N: usize>(iter: impl IntoIterator<Item=T>, v: T) -> [T; N] { crate::from_iter_or_else(iter, || v) }
fn from_iter<T, const N: usize>(iter: impl IntoIterator<Item=T>) -> [T; N] { crate::from_iter_or_else(iter, || unreachable!()) }
pub fn list<T>(iter: impl std::iter::IntoIterator<Item=T>) -> Box<[T]> { iter.into_iter().collect() }
pub fn sort_by<T>(mut s: Box<[T]>, f: impl Fn(&T,&T)->std::cmp::Ordering) -> Box<[T]> { s.sort_by(f); s }
pub fn map<T,U>(iter: impl std::iter::IntoIterator<Item=T>, f: impl Fn(T)->U) -> Box<[U]> { list(iter.into_iter().map(f)) }
fn bytes_of<T>(value: &T) -> &[u8] { unsafe{std::slice::from_raw_parts(value as *const _ as *const u8, std::mem::size_of::<T>())} }

pub mod va {
    #![allow(dead_code,non_camel_case_types,non_upper_case_globals,improper_ctypes,non_snake_case)]
    include!(concat!(env!("OUT_DIR"), "/va.rs"));
}
#[track_caller] pub fn check(status: va::VAStatus) { if status!=0 { panic!("{:?}", unsafe{std::ffi::CStr::from_ptr(va::vaErrorStr(status))}); } }

#[derive(Default)] struct Frame {
    id: va::VASurfaceID,
    poc: Option<u32>,
}
struct Sequence {
    frames: [Frame; 16],
    context: va::VAContextID,
    current_id: Option<va::VASurfaceID>,
    buffers: Vec<va::VABufferID>,
}
pub struct Decoder<'t> {
    #[allow(dead_code)] card: std::os::fd::BorrowedFd<'t>,
    pub va: va::VADisplay,
    config: va::VAConfigID,
    sequence: Option<Sequence>,
}
pub struct Image<'t> where Self:'t {
    va: va::VADisplay,
    id: va::VABufferID,
    image: image::Image<&'t [u16]>,
}
impl Drop for Image<'_> { fn drop(&mut self) { check(unsafe{va::vaUnmapBuffer(self.va, self.id)}); } }
impl<'t> std::ops::Deref for Image<'t> { type Target = image::Image<&'t [u16]>; fn deref(&self) -> &Self::Target { &self.image } }

use crate::video::*;
impl<'t> Decoder<'t> {
    pub fn new(card: &'t impl std::os::fd::AsFd) -> Self {
        let card = card.as_fd();
        use std::os::fd::AsRawFd;
        let va = unsafe{va::vaGetDisplayDRM(card.as_raw_fd())};
        extern "C" fn error(_user_context: *mut std::ffi::c_void, message: *const std::ffi::c_char) { panic!("{:?}", unsafe{std::ffi::CStr::from_ptr(message)}) }
        unsafe{va::vaSetErrorCallback(va, Some(error), std::ptr::null_mut())};
        extern "C" fn info(_user_context: *mut std::ffi::c_void, _message: *const std::ffi::c_char) { /*println!("{:?}", unsafe{std::ffi::CStr::from_ptr(_message)});*/ }
        unsafe{va::vaSetInfoCallback(va,  Some(info),  std::ptr::null_mut())};
        let (mut major, mut minor) = (0,0);
        check(unsafe{va::vaInitialize(va, &mut major, &mut minor)});
        let mut config = 0;
        check(unsafe{va::vaCreateConfig(va, va::VAProfile_VAProfileHEVCMain10, va::VAEntrypoint_VAEntrypointVLD, std::ptr::null_mut(), 0, &mut config)});
        Self{card, va, config, sequence: None}
    }
	pub fn slice<'y>(&mut self, hevc: &HEVC, Slice{pps, unit, escaped_data,slice_data_byte_offset,dependent_slice_segment,slice_segment_address}: Slice, last_slice_of_picture: bool) -> Option<Image<'y>> {
        let va = self.va;
        let pps = hevc.pps[pps].as_ref().unwrap();
        let sps = hevc.sps[pps.sps].as_ref().unwrap();
        let slice_header = hevc.slice_header.as_ref().unwrap();
        let sequence = self.sequence.get_or_insert_with(|| {
            let mut ids = [0; _];
            check(unsafe{va::vaCreateSurfaces(va, va::VA_RT_FORMAT_YUV420_10, sps.width as _, sps.height as _, ids.as_mut_ptr(), ids.len() as _, std::ptr::null_mut(), 0)});
            let mut context = 0;
            check(unsafe{va::vaCreateContext(va, self.config, sps.width as _, sps.height as _, va::VA_PROGRESSIVE as _, ids.as_ptr() as *mut _, ids.len() as _, &mut context)});
            Sequence{context, frames: ids.map(|id| Frame{id, poc: None}), current_id: None, buffers: Vec::new()}
        });
        let frames = &mut sequence.frames;
        let reference = slice_header.reference.as_ref();

        let mut render = {let context = sequence.context; let buffers = &mut sequence.buffers; move |r#type, data:&[u8]| {
            let mut buffer = 0;
            check(unsafe{va::vaCreateBuffer(va, context, r#type, data.len() as _, 1, data.as_ptr() as *const _ as *mut _, &mut buffer)});
            buffers.push(buffer);
        }};

        let current_id = *sequence.current_id.get_or_insert_with(|| {
            let current_poc = reference.map(|r| r.poc).unwrap_or(0);
            //println!("POC {}", current_poc);
            //println!("DPB [{}]", frames.iter().filter_map(|f| f.poc).format(" "));
            //reference.map(|r| println!("refs [{};{}]", r.short_term_pictures.iter().map(|p| (current_poc as i32+p.delta_poc as i32) as u32).format(" "),r.long_term_pictures.iter().map(|p| p.poc).format(" ")));

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
                        r.short_term_pictures.iter().filter_map(|&ShortTermReferencePicture{delta_poc, used}| used.then(|| (current_poc as i32+delta_poc as i32) as u32)).find(|&poc| poc == frame_poc).map(|poc|
                            match poc.cmp(&current_poc) {
                                std::cmp::Ordering::Less => va::VA_PICTURE_HEVC_RPS_ST_CURR_BEFORE,
                                std::cmp::Ordering::Greater => va::VA_PICTURE_HEVC_RPS_ST_CURR_AFTER,
                                _ => unreachable!()
                            }
                        ).unwrap_or(0) |
                        if r.long_term_pictures.iter().any(|&LongTermReferencePicture{poc, used}| used && poc == frame_poc) { va::VA_PICTURE_HEVC_RPS_LT_CURR } else {0}
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
                        Intra_Random_Access_Picture(unit) as _,
                        Instantaneous_Decoder_Refresh(unit) as _,
                        Intra_Random_Access_Picture(unit) as _,
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
        let sh = &slice_header;
        render(va::VABufferType_VASliceParameterBufferType, bytes_of(&va::VASliceParameterBufferHEVC{
            slice_data_size: escaped_data.len() as u32,
            slice_data_offset: 0,
            slice_data_flag: va::VA_SLICE_DATA_FLAG_ALL,
            slice_data_byte_offset: slice_data_byte_offset as u32,
            slice_segment_address: slice_segment_address.unwrap_or(0) as u32,
            RefPicList: slice_header.reference.as_ref().map(|r| {
                let ref frames = frames.iter().filter(|frame| frame.poc.map(|poc| poc != r.poc).unwrap_or(false)).map(|frame| frame.poc.unwrap()).collect::<Box<_>>();
                #[track_caller] fn index(frames: &[u32], ref_poc: u32) -> usize { frames.iter().position(|&poc| poc == ref_poc).expect(&format!("{ref_poc} {frames:?}")) }
                fn stps(frames: &[u32], r: &SHReference, f: impl Fn(std::cmp::Ordering)->std::cmp::Ordering) -> Box<[u8]> {
                    let current_poc = r.poc;
                    map(&*sort_by(
                        list(r.short_term_pictures.into_iter().filter_map(|&ShortTermReferencePicture{delta_poc, used}|(used && f(delta_poc.cmp(&0)) == std::cmp::Ordering::Greater).then(|| delta_poc))),
                        |a,b| f(a.cmp(b))
                    ), |delta_poc| index(frames, (current_poc as i32+*delta_poc as i32) as u32) as u8)
                }
                let ref st_after = stps(frames, r, |o| o);
                let ref st_before = stps(frames, r, |o| o.reverse());
                let ref lt = map(sort_by(list(r.long_term_pictures.into_iter().filter_map(|&LongTermReferencePicture{poc, used}| used.then(|| poc))), Ord::cmp).into_iter(), |&poc| index(frames, poc) as u8);
                let list = |list:[_;3]| list.map(|e:&Box<[_]>| e.into_iter()).into_iter().flatten().copied().collect::<Box<_>>();
                let rpl = MayB{p: list([st_before, st_after, lt]), b: matches!(sh.slice_type, SliceType::B).then(|| list([st_after, st_before, lt]))};
                let inter = slice_header.inter.as_ref().unwrap();
                assert!(inter.list_entry_lx.is_none());
                let rpl = if let Some(list_entry_lx) = slice_header.inter.as_ref().unwrap().list_entry_lx.as_ref() { rpl.zip(list_entry_lx).map(|(rpl, list_entry_lx)| list_entry_lx.as_ref().map(|list| map(list.into_iter(), |i| rpl[*i as usize])).unwrap_or(rpl)) } else { rpl.zip(&inter.active_references).map(|(rpl, active_references)| rpl.into_vec().into_iter().take(*active_references as usize).collect()) };
                let rpl = rpl.map(|list| from_iter_or(list.into_vec().into_iter(), 0xFF));
                [rpl.p, rpl.b.unwrap_or([0xFF; _])]
            }).unwrap_or([[0xFF; _]; _]),
            //padding0: [0; 2],
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
            //padding1: 0,
            num_entry_point_offsets: 0,
            entry_offset_to_subset_array: 0,
            slice_data_num_emu_prevn_bytes: 0,
            //padding2: 0,
            va_reserved: [0; _],
            ..Default::default() // Helps zero initialize padding holes (but unreliable)
        }));
        render(va::VABufferType_VASliceDataBufferType, escaped_data);
        last_slice_of_picture.then(|| {
            let context = sequence.context;
            check(unsafe{va::vaBeginPicture(va, context, current_id)});
            check(unsafe{va::vaRenderPicture(va, context, sequence.buffers.as_ptr() as *const _ as *mut _, sequence.buffers.len() as _)}); sequence.buffers.clear();
            check(unsafe{va::vaEndPicture(va, context)});
            check(unsafe{va::vaSyncSurface(va, current_id)});
            let ref format = va::VAImageFormat{fourcc: va::VA_FOURCC_P010, byte_order: va::VA_LSB_FIRST, bits_per_pixel: 24, depth: 0, red_mask: 0, green_mask: 0, blue_mask: 0, alpha_mask: 0, va_reserved: [0; _]};
            let mut image = va::VAImage{image_id: va::VA_INVALID_ID, ..Default::default()};
            let size = vector::xy{x: sps.width as u32, y: sps.height as u32};
            check(unsafe{va::vaCreateImage(va, format as *const _ as *mut _, size.x as _, size.y as _, &mut image)});
            check(unsafe{va::vaGetImage(va, current_id, 0, 0, size.x, size.y, image.image_id)});
            let mut address : *const u16 = std::ptr::null();
            check(unsafe{va::vaMapBuffer(va, image.buf, &mut address as *mut * const _ as *mut *mut _)});
            sequence.current_id = None; // Signals that next slice is a new picture
            Image{va, id: image.buf, image: image::Image::new(size, unsafe{std::slice::from_raw_parts(address, (size.y*size.x) as usize)})}
        })
    }
}

