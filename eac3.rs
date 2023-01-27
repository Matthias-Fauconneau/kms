pub struct EAC3;
impl EAC3 { pub fn new() -> Self { Self } }
impl crate::Decoder for EAC3 {
type Output<'t> = &'t [u8];
fn decode<'t>(&mut self, input: &'t [u8]) -> Option<Self::Output<'t>> {
	fn sync<const N: usize>(mut input: &[u8], needle: [u8; N]) -> &[u8] { while !input.starts_with(&needle) { (_, input) = input.split_first().unwrap(); } &input[needle.len()..] }
	let ref mut s = crate::bit::Reader::new(sync(input, [0xb,0x77]));
	enum Type { Independent }
	let frame_type = s.bits(2);
	let _substream_id = s.bits(3);
	let frame_size = (s.bits(11) + 1) << 1;
	let sr_code = s.bits(2);
	let sample_rate = [48000, 44100, 32000];
	let (blocks_len, sample_rate, _shift):(u32,_,_) = if sr_code == 3 {
		(6, sample_rate[s.bits(2) as usize]>>1, 1)
	} else {
		([1, 2, 3, 6][s.bits(2) as usize], sample_rate[sr_code as usize], 0)
	};
	use num_traits::FromPrimitive;
	#[derive(num_derive::FromPrimitive,Clone,Copy,Debug,PartialEq)] enum ChannelLayout { F1R1, F1, F2, F3, F2R, F3R, F2R2, F3R2 } use ChannelLayout::*;
	let channel_mode = ChannelLayout::from_u32(s.bits(3)).unwrap();
	assert!(!matches!(channel_mode, F1R1));
	let lfe = s.bit();
	let _bit_rate = 8 * frame_size * sample_rate / (blocks_len as u32 * 256);
	let downmix = [
		&[[2usize,7],[7,2]] as &[_],
		&[[4,4]],
		&[[2,7],[7,2]],
		&[[2,7],[5,5],[7,2]],
		&[[2,7],[7,2],[6,6]],
		&[[2,7],[5,5],[7,2],[8,8]],
		&[[2,7],[7,2],[6,7],[7,6]],
		&[[2,7],[5,5],[7,2],[6,7],[7,6]],
	];
	let full_bandwidth_channels = downmix[channel_mode as usize].len();
	let all_channels = full_bandwidth_channels + if lfe { 1 } else { 0 };
	s.skip(5);
	let _dialog_normalization = -(match s.bits(5) { 0 => 31, v => v} as i8);
    if s.bit() { let _heavy_dynamic_range = s.u8()*2; }
	assert!(frame_type == Type::Independent as _);
    let mix = if s.bit() { // mixing metadata
        assert!(!matches!(channel_mode, F1|F2));
        let preferred_downmix = s.bits(2);
        let _center_mix_level_ltrt = s.bits(3);
		let e = |x| 2f32.powf(x/4f32);
		let gain_levels = [e(2.), e(1.), e(0.), e(-1.), e(-2.), e(-3.), e(-4.), 0., e(-6.)];
		let center = matches!(channel_mode, F3|F3R|F3R2).then(|| gain_levels[s.bits(3) as usize]);
		let _surround_mix_level_ltrt = s.bits(3);
		let surround = matches!(channel_mode, F2R|F3R|F2R2|F3R2).then(|| gain_levels[s.bits(3) as usize]);
		assert!(lfe);
        if s.bit() { let _lfe_mix_level = s.bits(5); }
		assert!(frame_type == Type::Independent as _);
		if s.bit() { s.skip(6) }
        if s.bit() { s.skip(6) }
        match s.bits(2) {
			0 => {},
			1 => s.skip(5),
			2 => s.skip(12),
			3 => {let len = (s.bits(5)+2)*8; s.skip_long(len)},
			_ => unreachable!()
        }
		assert!(!matches!(channel_mode, F1));
        if s.bit() { if blocks_len == 1 { s.skip(5); } else { for _ in 0..blocks_len { if s.bit() { s.skip(5) } } } }
		let mut downmix = downmix[channel_mode as usize].iter().map(|&[l,r]| [gain_levels[l],gain_levels[r]]).collect::<ArrayVec<_,5>>();
		if let Some(level) = center { downmix[1] = [level; 2]; }
		if let Some(mut level) = surround {
			if let F2R|F3R = channel_mode { level *= e(-2.); }
			downmix[match channel_mode {F2R|F2R2=>F2, F3R|F3R2=>F3, _=>unreachable!()} as usize] = [level; 2];
		}
		fn transpose<T:Copy, const M: usize, const N: usize>(a: &ArrayVec<[T; N], M>) -> [ArrayVec<T, M>; N] { crate::from_iter((0..N).map(|n| map(a, |m| m[n]))) }
		let downmix = transpose(&downmix);
		downmix.map(|downmix| map(&downmix, |x| x / downmix.iter().sum::<f32>()));
	} else { unimplemented!() };
	if s.bit() {
		let _bitstream_mode = s.bits(3);
		s.skip(2);
		assert!(!matches!(channel_mode, F2));
		if s.bit() { s.skip(8) }
		s.skip(1);
	}
	assert!(frame_type == Type::Independent as _);
	if blocks_len < 6 { s.skip(1); }
	if s.bit() { let len = (1+s.bits(6))*8; s.skip_long(len); }
    let (exponent, aht) = if blocks_len == 6 { (s.bit(), s.bit()) } else { (true, false) };
    let snr_offset = s.bits(2);
    let transient = s.bit();
    let block_switch = s.bit();
    let dither = s.bit();
    let _bit_allocation = s.bit(); // or {slow_decay: slow_decay[2], fast_decay: fast_decay[1], slow_gain: slow_gain[1], db_per_bit: db_per_bit[2], floor: floor[7] }
    /*let slow_decay = [0x0f, 0x11, 0x13, 0x15];
	let fast_decay = [0x3f, 0x53, 0x67, 0x7b];
	let slow_gain = [0x540, 0x4d8, 0x478, 0x410];
	let db_per_bit = [0x000, 0x700, 0x900, 0xb00];
	let floor = [0x2f0, 0x2b0, 0x270, 0x230, 0x1f0, 0x170, 0x0f0, 0xf800];*/
    let _fast_gain = s.bit();
    let _dba = s.bit();
    let _skip = s.bit();
    let attenuation = s.bit();
	assert!(!matches!(channel_mode, F1));
	use arrayvec::ArrayVec;
	type Blocks<T> = ArrayVec::<T, 6>;
	fn map<T,U,const N: usize>(a : &ArrayVec::<T, N>, f: impl FnMut(&T)->U) -> ArrayVec::<U, N> { a.iter().map(f).collect() }
	fn try_map<T,U,const N: usize>(a : &ArrayVec::<T, N>, f: impl FnMut(&T)->U) -> ArrayVec::<U, N> { a.iter().try_map(f).collect() }

	let coupling_use : Blocks<_> = std::iter::successors(Some((true, s.bit())),|&(_,prev)| Some(if s.bit() { (true, s.bit()) } else { (false, prev) })).take(blocks_len as usize).collect();
	type Channels<T> = (Option<T>, ArrayVec::<T, 6>);
	fn channels<T>(coupling_channel: bool, channels: usize, mut f: impl FnMut()->T) -> Channels<T> {
		(coupling_channel.then(&mut f), (0..channels).map(|_| f()).collect())
	}
	fn map_channels<T,U>((coupling, channels):&Channels<T>, mut f: impl FnMut(&T)->U) -> Channels<U> { (coupling.as_ref().map(&mut f), map(channels, f))}
    fn zip<A,B>(a:&Channels<T>, b:&Channels<B>) -> Channels<(A,B)> { (a.0.zip(b.0), a.1.zip(b.1)) }

	let exponent : Blocks<_> = if exponent {
        map(&coupling_use, |&(_,coupling)| channels(coupling, full_bandwidth_channels, || s.bits(2)))
    } else {
		assert!(!matches!(channel_mode, F1));
		let frame_exponent_strategy_combinations = [
			[1,0,0,0,0,0],[1,0,0,0,0,3],[1,0,0,0,2,0],[1,0,0,0,3,3],
			[2,0,0,2,0,0],[2,0,0,2,0,3],[2,0,0,3,2,0],[2,0,0,3,3,3],[2,0,1,0,0,0],[2,0,2,0,0,3],[2,0,2,0,2,0],[2,0,2,0,3,3],[2,0,3,2,0,0],[2,0,3,2,0,3],[2,0,3,3,2,0],[2,0,3,3,3,3],
			[3,1,0,0,0,0],[3,1,0,0,0,3],[3,2,0,0,2,0],[3,2,0,0,3,3],[3,2,0,2,0,0],[3,2,0,2,0,3],[3,2,0,3,2,0],[3,2,0,3,3,3],
			[3,3,1,0,0,0],[3,3,2,0,0,3],[3,3,2,0,2,0],[3,3,2,0,3,3],[3,3,3,2,0,0],[3,3,3,2,0,3],[3,3,3,3,2,0],[3,3,3,3,3,3],
		];
		let combinations = channels(coupling_use.iter().any(|&(_,c)| c), full_bandwidth_channels, || frame_exponent_strategy_combinations[s.bits(5) as usize]);
		(0..blocks_len).map(|block| map_channels(&combinations, |combination| combination[block as usize])).collect()
    };
	let mut exponent = exponent.map(|block| map_channels(block, |e| if e==0 { None } else { Some(e-1) }));
    if lfe { for block in &mut exponent { block.1.push(s.bit().then(|| 0)) } }
	assert!(frame_type == Type::Independent as _);
    if blocks_len == 6 || s.bit() { s.skip(5 * full_bandwidth_channels as u8); }
    let _aht = aht.then(|| {
		let channels = (try_map(&exponent, |e| e.0).ok(), (0..all_channels).map(|channel| map(&exponent, |e| e.1[channel])).collect());
		map_channels(&channels, |e:&Blocks<_>| e.iter().all(|&e| e.is_some()) && s.bit())
	});
    if snr_offset == 0 { let _snr_offset = [(((s.bits(6)-15)<<4)+s.bits(2)) << 2; 7]; }
    let _ = transient.then(|| (0..full_bandwidth_channels).map(|_| if s.bit() { s.skip(18) }).collect::<ArrayVec<_,5>>());
    let _attenuation = attenuation.then(|| (0..full_bandwidth_channels).map(|_| s.bit().then(|| s.bits(5))).collect::<ArrayVec<_,5>>());
    if blocks_len > 1 && s.bit() { s.skip_long((blocks_len-1)*(4+(frame_size-2).ilog2())); }
	//lfe: {end_freq: 7, num_exp_groups: 2}
	let bool = |x| x.map(|same| match same { 0=>false, 1=>true });
	let mut spectral_extension_band_structure = bool([0,0,0,0,0,0,0,0,1,0,1,0,1,0,1,0,1]);
	struct SpectralExtension {
		channels: ArrayVec<bool, 6>,
		start: usize, end: usize,
		bands: ArrayVec<ArrayVec<(u8/*noise*/, u8/*signal*/), 22>, 6>
	}
	let mut spectral_extension = (0..full_bandwidth_channels).map(|_| None).collect();
	let mut coupling_band_structure = bool([0,0,0,0,0,0,0,0,1,0,1,1,0,1,1,1,1,1]);
	struct CouplingStrategy {
		channels: ArrayVec<bool, 6>,
		start: usize, end: usize,
		bands: ArrayVec<u8, 22>
	}
	let mut coupling_strategy = None;
	let mut coupling_coordinates = (0..full_bandwidth_channels).map(|_| None).collect();
	let mut range = 1.;
	let mut end_frequency = (0..full_bandwidth_channels).map(|_| 0).collect();
	for (block, (&(coupling_strategy_change, coupling_use), exponent)) in coupling_use.zip(exponent).iter().enumerate() {
		let block_switch = block_switch.then(|| (0..full_bandwidth_channels).map(|_| s.bit()));
		//let different_transforms = block_switch.group_by(PartialEq::eq).len() > 1;
		let dither = dither.then(|| (0..full_bandwidth_channels).map(|_| s.bit()).collect()); //or all
		if s.bit() {
			let f = |exponent_width, trailing_significand_width| {
				let exp = (s.bits(exponent_width)^(1<<(exponent_width-1))) as i32 - (1<<exponent_width);
				let significand = (1<<trailing_significand_width)+s.bits(trailing_significand_width);
				2f32.powi(exp-trailing_significand_width as i32) * significand as f32
			};
			range = f(3,5);
        }
		fn bands(s: &mut crate::bit::Reader, mut band_structure: &mut [bool], start: usize, end: usize) -> ArrayVec<u8, 22> {
			if s.bit() { for b in &mut band_structure[start+1..end] { *b = s.bit(); } }
			let bands = ArrayVec::try_from(&[1u8] as &[_]).unwrap();
			for subband in 1..end-start {
				if band_structure[start+subband] {
					*bands.last_mut().unwrap() += 1;
				} else {
					bands.push(1);
				}
			}
			bands
		}
		let frequency = |band:usize| { 25. + 12. * band as f32 };
		spectral_extension = if (block==0 || s.bit()) && s.bit() {
			let channels = (0..full_bandwidth_channels).map(|_| s.bit()).collect();
			let dst_start = s.bits(2);
			let start  = { let b = 2 + s.bits(3); if b > 7 { 2*b - 7 } else { b } } as usize;
			let end = { let b = 5 + s.bits(3); if b > 7 { 2*b - 7 } else { b } } as usize;
			let bands = bands(s, &mut spectral_extension_band_structure, start, end);
			SpectralExtension{channels, start, end,
				bands: spectral_extension.zip(channels).map(|spectral_extension, channel| channel.then(|| spectral_extension.filter(|_| !/*update*/s.bit()).unwrap_or(|| {
					let blend = s.bits(5) as f32/32.;
					let exp_bias = s.bits(2) * 3;
					bands.into_iter().scan(frequency(start), |bin, band| {
						let band = frequency(band as usize) as f32;
						let exp = s.bits(4);
						let trailing_significand = s.bits(2);
						let spx = if exp == 15 { trailing_significand << 1 } else { (1<<2)+trailing_significand  } as f32 * 2f32.powi(2 - exp as i32 - exp_bias as i32);
						let nratio = ((*bin + band/2.) / frequency(end) - blend).clamp(0., 1.);
						let nblend = f32::sqrt(3. * nratio);
						let sblend = f32::sqrt(1. - nratio);
						*bin += band;
						Some((spx*nblend, spx*sblend))
					}).collect()
				}))).collect()
			}
		} else { None };
		let bit_allocation_stages = 0;
		if coupling_use {
			if coupling_strategy_change {
				bit_allocation_stages = 3;
				assert!(!s.bit());
				assert!(channel_mode != F2);
				let channels = (0..full_bandwidth_channels).map(|_| s.bit()).collect();
				let start = s.bits(4) as usize;
				let end = spectral_extension.map(|(_, start)| start-1).unwrap_or_else(||3+s.bits(4));
				//freq[CPL] = freq(1+[start,end])
				let bands = bands(s, &mut coupling_band_structure, start, end);
				coupling_strategy = Some(CouplingStrategy{channels, start,end, bands});
			}
			let CouplingStrategy{channels, start, end, bands} = coupling_strategy.unwrap();
			(coupling_coordinates.zip(channels).map(|(coupling_bands, channel)| channel.then(|| coupling_bands.filter(|_| !/*update*/s.bit()).unwrap_or(|| {
                let exp_bias = s.bits(2) * 3;
				bands.into_iter().scan(frequency(start), |bin, band| {
					let band = frequency(band as usize) as f32;
					let exp = s.bits(4);
				    let trailing_significand = s.bits(4);
					Some((if exp == 15 { trailing_significand << 1 } else { (1<<4)+trailing_significand }) << (6-exp_bias+15-exp))
                })
			}))))
		}
		let bit_allocation_stages = map_channels(&exponent, |e| e.then(|| 3).unwrap_or(bit_allocation_stages));
		(end_frequency, num_exp_groups) =
			exponent.1.iter().zip(coupling_strategy.channels).zip(end_frequency).zip(spectral_extension).map(|(((exponent,coupling),end_frequency),spectral_extension)| {
			exponent.then(|| {
				let end_frequency =
					if coupling { frequency(1+coupling_strategy.start) }
					else if spectral_extension { frequency(spectral_extension.start) }
					else { s.bits(6) * 3 + 73 };
				let group_size = 3 << exponent;
				let num_exp_groups = (end_frequency + group_size-4) / group_size;
				(end_frequency, num_exp_groups)
			})
		}).unzip();
		let if (cpl_in_use && exp_strategy[blk][CPL_CH] != EXP_REUSE) {
			num_exp_groups[CPL_CH] = (end_freq[CPL_CH] - start_frequency[CPL_CH]) /
										(3 << (exp_strategy[blk][CPL_CH] - 1));
		}

		/* decode exponents for each channel */
		for (ch = !cpl_in_use; ch <= channels; ch++) {
			if (exp_strategy[blk][ch] != EXP_REUSE) {
				dexps[ch][0] = get_bits(gbc, 4) << !ch;
				if (decode_exponents(s, gbc, exp_strategy[blk][ch],
									num_exp_groups[ch], dexps[ch][0],
									&dexps[ch][start_frequency[ch]+!!ch])) {
					return AVERROR_INVALIDDATA;
				}
				if (ch != CPL_CH && ch != lfe_ch)
					skip_bits(gbc, 2); /* skip gainrng */
			}
		}

		/* bit allocation information */
		if (bit_allocation_syntax) {
			if (get_bits1(gbc)) {
				bit_alloc_params.slow_decay = ff_ac3_slow_decay_tab[get_bits(gbc, 2)] >> bit_alloc_params.sr_shift;
				bit_alloc_params.fast_decay = ff_ac3_fast_decay_tab[get_bits(gbc, 2)] >> bit_alloc_params.sr_shift;
				bit_alloc_params.slow_gain  = ff_ac3_slow_gain_tab[get_bits(gbc, 2)];
				bit_alloc_params.db_per_bit = ff_ac3_db_per_bit_tab[get_bits(gbc, 2)];
				bit_alloc_params.floor  = ff_ac3_floor_tab[get_bits(gbc, 3)];
				for (ch = !cpl_in_use; ch <= channels; ch++)
					bit_allocation_stages[ch] = FFMAX(bit_allocation_stages[ch], 2);
			} else if (!blk) {
				av_log(avctx, AV_LOG_ERROR, "new bit allocation info must "
					"be present in block 0\n");
				return AVERROR_INVALIDDATA;
			}
		}

		/* signal-to-noise ratio offsets and fast gains (signal-to-mask ratios) */
		if (!eac3 || !blk) {
			if (snr_offset_strategy && get_bits1(gbc)) {
				int snr = 0;
				int csnr;
				csnr = (get_bits(gbc, 6) - 15) << 4;
				for (i = ch = !cpl_in_use; ch <= channels; ch++) {
					/* snr offset */
					if (ch == i || snr_offset_strategy == 2)
						snr = (csnr + get_bits(gbc, 4)) << 2;
					/* run at least last bit allocation stage if snr offset changes */
					if (blk && snr_offset[ch] != snr) {
						bit_allocation_stages[ch] = FFMAX(bit_allocation_stages[ch], 1);
					}
					snr_offset[ch] = snr;

					/* fast gain (normal AC-3 only) */
					if (!eac3) {
						int prev = fast_gain[ch];
						fast_gain[ch] = ff_ac3_fast_gain_tab[get_bits(gbc, 3)];
						/* run last 2 bit allocation stages if fast gain changes */
						if (blk && prev != fast_gain[ch])
							bit_allocation_stages[ch] = FFMAX(bit_allocation_stages[ch], 2);
					}
				}
			} else if (!eac3 && !blk) {
				av_log(avctx, AV_LOG_ERROR, "new snr offsets must be present in block 0\n");
				return AVERROR_INVALIDDATA;
			}
		}

		/* fast gain (E-AC-3 only) */
		if (fast_gain_syntax && get_bits1(gbc)) {
			for (ch = !cpl_in_use; ch <= channels; ch++) {
				int prev = fast_gain[ch];
				fast_gain[ch] = ff_ac3_fast_gain_tab[get_bits(gbc, 3)];
				/* run last 2 bit allocation stages if fast gain changes */
				if (blk && prev != fast_gain[ch])
					bit_allocation_stages[ch] = FFMAX(bit_allocation_stages[ch], 2);
			}
		} else if (eac3 && !blk) {
			for (ch = !cpl_in_use; ch <= channels; ch++)
				fast_gain[ch] = ff_ac3_fast_gain_tab[4];
		}

		/* E-AC-3 to AC-3 converter SNR offset */
		if (frame_type == EAC3_FRAME_TYPE_INDEPENDENT && get_bits1(gbc)) {
			skip_bits(gbc, 10); // skip converter snr offset
		}

		/* coupling leak information */
		if (cpl_in_use) {
			if (first_cpl_leak || get_bits1(gbc)) {
				int fl = get_bits(gbc, 3);
				int sl = get_bits(gbc, 3);
				/* run last 2 bit allocation stages for coupling channel if
				coupling leak changes */
				if (blk && (fl != bit_alloc_params.cpl_fast_leak ||
					sl != bit_alloc_params.cpl_slow_leak)) {
					bit_allocation_stages[CPL_CH] = FFMAX(bit_allocation_stages[CPL_CH], 2);
				}
				bit_alloc_params.cpl_fast_leak = fl;
				bit_alloc_params.cpl_slow_leak = sl;
			} else if (!eac3 && !blk) {
				av_log(avctx, AV_LOG_ERROR, "new coupling leak info must "
					"be present in block 0\n");
				return AVERROR_INVALIDDATA;
			}
			first_cpl_leak = 0;
		}

		/* delta bit allocation information */
		if (dba_syntax && get_bits1(gbc)) {
			/* delta bit allocation exists (strategy) */
			for (ch = !cpl_in_use; ch <= fbw_channels; ch++) {
				dba_mode[ch] = get_bits(gbc, 2);
				if (dba_mode[ch] == DBA_RESERVED) {
					av_log(avctx, AV_LOG_ERROR, "delta bit allocation strategy reserved\n");
					return AVERROR_INVALIDDATA;
				}
				bit_allocation_stages[ch] = FFMAX(bit_allocation_stages[ch], 2);
			}
			/* channel delta offset, len and bit allocation */
			for (ch = !cpl_in_use; ch <= fbw_channels; ch++) {
				if (dba_mode[ch] == DBA_NEW) {
					dba_nsegs[ch] = get_bits(gbc, 3) + 1;
					for (seg = 0; seg < dba_nsegs[ch]; seg++) {
						dba_offsets[ch][seg] = get_bits(gbc, 5);
						dba_lengths[ch][seg] = get_bits(gbc, 4);
						dba_values[ch][seg]  = get_bits(gbc, 3);
					}
					/* run last 2 bit allocation stages if new dba values */
					bit_allocation_stages[ch] = FFMAX(bit_allocation_stages[ch], 2);
				}
			}
		} else if (blk == 0) {
			for (ch = 0; ch <= channels; ch++) {
				dba_mode[ch] = DBA_NONE;
			}
		}

		/* Bit allocation */
		for (ch = !cpl_in_use; ch <= channels; ch++) {
			if (bit_allocation_stages[ch] > 2) {
				/* Exponent mapping into PSD and PSD integration */
				ff_ac3_bit_alloc_calc_psd(dexps[ch],
										start_frequency[ch], end_freq[ch],
										psd[ch], band_psd[ch]);
			}
			if (bit_allocation_stages[ch] > 1) {
				/* Compute excitation function, Compute masking curve, and
				Apply delta bit allocation */
				if (ff_ac3_bit_alloc_calc_mask(&bit_alloc_params, band_psd[ch],
											start_frequency[ch],  end_freq[ch],
											fast_gain[ch],   (ch == lfe_ch),
											dba_mode[ch],    dba_nsegs[ch],
											dba_offsets[ch], dba_lengths[ch],
											dba_values[ch],  mask[ch])) {
					av_log(avctx, AV_LOG_ERROR, "error in bit allocation\n");
					return AVERROR_INVALIDDATA;
				}
			}
			if (bit_allocation_stages[ch] > 0) {
				/* Compute bit allocation */
				const uint8_t *bap_tab = channel_uses_aht[ch] ?
										ff_eac3_hebap_tab : ff_ac3_bap_tab;
				ac3dsp.bit_alloc_calc_bap(mask[ch], psd[ch],
										start_frequency[ch], end_freq[ch],
										snr_offset[ch],
										bit_alloc_params.floor,
										bap_tab, bap[ch]);
			}
		}

		/* unused dummy data */
		if (skip_syntax && get_bits1(gbc)) {
			int skipl = get_bits(gbc, 9);
			skip_bits_long(gbc, 8 * skipl);
		}

		/* unpack the transform coefficients
		this also uncouples channels if coupling is in use. */
		decode_transform_coeffs(s, blk);

		/* TODO: generate enhanced coupling coordinates and uncouple */

		/* recover coefficients if rematrixing is in use */
		if (channel_mode == AC3_CHMODE_STEREO)
			do_rematrixing(s);

		/* apply scaling to coefficients (headroom, dynrng) */
		for (ch = 1; ch <= channels; ch++) {
			int audio_channel = 0;
			INTFLOAT gain;
			if (channel_mode == AC3_CHMODE_DUALMONO && ch <= 2)
				audio_channel = 2-ch;
			if (heavy_compression && compression_exists[audio_channel])
				gain = heavy_dynamic_range[audio_channel];
			else
				gain = dynamic_range[audio_channel];

	#if USE_FIXED
			scale_coefs(transform_coeffs[ch], fixed_coeffs[ch], gain, 256);
	#else
			if (target_level != 0)
			gain = gain * level_gain[audio_channel];
			gain *= 1.0 / 4194304.0f;
			fmt_conv.int32_to_float_fmul_scalar(transform_coeffs[ch],
												fixed_coeffs[ch], gain, 256);
	#endif
		}

		/* apply spectral extension to high frequency bins */
		if (CONFIG_EAC3_DECODER && spx_in_use) {
			ff_eac3_apply_spectral_extension(s);
		}

		/* downmix and MDCT. order depends on whether block switching is used for
		any channel in this block. this is because coefficients for the long
		and short transforms cannot be mixed. */
		downmix_output = channels != out_channels &&
						!((output_mode & AC3_OUTPUT_LFEON) &&
						fbw_channels == out_channels);
		if (different_transforms) {
			/* the delay samples have already been downmixed, so we upmix the delay
			samples in order to reconstruct all channels before downmixing. */
			if (downmixed) {
				downmixed = 0;
				ac3_upmix_delay(s);
			}

			do_imdct(s, channels, offset);

			if (downmix_output) {
				ff_ac3dsp_downmix(&ac3dsp, outptr, downmix_coeffs,
								out_channels, fbw_channels, 256);
			}
		} else {
			if (downmix_output) {
				AC3_RENAME(ff_ac3dsp_downmix)(&ac3dsp, xcfptr + 1, downmix_coeffs,
											out_channels, fbw_channels, 256);
			}

			if (downmix_output && !downmixed) {
				downmixed = 1;
				AC3_RENAME(ff_ac3dsp_downmix)(&ac3dsp, dlyptr, downmix_coeffs,
											out_channels, fbw_channels, 128);
			}

			do_imdct(s, out_channels, offset);
		}
	}
	unimplemented!("")
}}