pub struct EAC3;
impl EAC3 { pub fn new() -> Self { Self } }
impl crate::Decoder for EAC3 {
type Output<'t> = &'t [u8];
fn decode<'t>(&mut self, input: &'t [u8]) -> Option<Self::Output<'t>> {
	fn sync<const N: usize>(mut input: &[u8], needle: [u8; N]) -> &[u8] { while !input.starts_with(&needle) { (_, input) = input.split_first().unwrap(); } &input[needle.len()..] }
	let mut s = crate::bit::Reader::new(sync(input, [0xb,0x77]));
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
    if s.bit() { let _heavy_dynamic_range = s.bits(8)*2; }
	assert!(frame_type == Type::Independent as _);
    let (center_mix_level, surround_mix_level) = if s.bit() { // mixing metadata
        assert!(!matches!(channel_mode, F1|F2));
        let _preferred_downmix = s.bits(2);
        let _center_mix_level_ltrt = s.bits(3);
		let center_mix_level = matches!(channel_mode, F3|F3R|F3R2).then(|| s.bits(3) as usize);
		let surround_mix_level_ltrt = s.bits(3); assert!(3 <= surround_mix_level_ltrt && surround_mix_level_ltrt <= 7);
		let surround_mix_level = matches!(channel_mode, F2R|F3R|F2R2|F3R2).then(|| s.bits(3) as usize);
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
		(center_mix_level, surround_mix_level)
	} else { Default::default() };
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
    let _block_switch = s.bit();
    let _dither = s.bit(); //or dither: [0, 1.., 0];
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

	let coupling : Blocks<_> = std::iter::successors(Some((true, s.bit())),|&(_,prev)| Some(if s.bit() { (true, s.bit()) } else { (false, prev) })).take(blocks_len as usize).map(|(_,x)| x).collect();
	type Channels<T> = (Option<T>, ArrayVec::<T, 6>);
	fn channels<T>(coupling_channel: bool, channels: usize, mut f: impl FnMut()->T) -> Channels<T> {
		(coupling_channel.then(&mut f), (0..channels).map(|_| f()).collect())
	}
	fn map_channels<T,U>((coupling, channels):&Channels<T>, mut f: impl FnMut(&T)->U) -> Channels<U> { (coupling.as_ref().map(&mut f), map(channels, f))}
    let mut exponent : Blocks<_> = if exponent {
        map(&coupling, |&coupling| channels(coupling, full_bandwidth_channels, || s.bits(2)))
    } else {
		assert!(!matches!(channel_mode, F1));
		let frame_exponent_strategy_combinations = [
			[1,0,0,0,0,0],[1,0,0,0,0,3],[1,0,0,0,2,0],[1,0,0,0,3,3],
			[2,0,0,2,0,0],[2,0,0,2,0,3],[2,0,0,3,2,0],[2,0,0,3,3,3],[2,0,1,0,0,0],[2,0,2,0,0,3],[2,0,2,0,2,0],[2,0,2,0,3,3],[2,0,3,2,0,0],[2,0,3,2,0,3],[2,0,3,3,2,0],[2,0,3,3,3,3],
			[3,1,0,0,0,0],[3,1,0,0,0,3],[3,2,0,0,2,0],[3,2,0,0,3,3],[3,2,0,2,0,0],[3,2,0,2,0,3],[3,2,0,3,2,0],[3,2,0,3,3,3],
			[3,3,1,0,0,0],[3,3,2,0,0,3],[3,3,2,0,2,0],[3,3,2,0,3,3],[3,3,3,2,0,0],[3,3,3,2,0,3],[3,3,3,3,2,0],[3,3,3,3,3,3],
		];
		let combinations = channels(coupling.iter().any(|&c| c), full_bandwidth_channels, || frame_exponent_strategy_combinations[s.bits(5) as usize]);
		(0..blocks_len).map(|block| map_channels(&combinations, |combination| combination[block as usize])).collect()
    };
    if lfe { for block in &mut exponent { block.1.push(if s.bit() {1} else {0}) } }
	assert!(frame_type == Type::Independent as _);
    if blocks_len == 6 || s.bit() { s.skip(5 * full_bandwidth_channels as u8); }
    let _aht = aht.then(|| {
		let channels = (Some(map(&exponent, |e| e.0)), (0..all_channels).map(|channel| map(&exponent, |e| Some(e.1[channel]))).collect());
		map_channels(&channels, |e:&Blocks<_>| e.iter().all(|&e| e.map(|e| e==0).unwrap_or(false)) && s.bit())
	});
    if snr_offset == 0 { let _snr_offset = [(((s.bits(6)-15)<<4)+s.bits(2)) << 2; 7]; }
    let _ = transient.then(|| (0..full_bandwidth_channels).map(|_| if s.bit() { s.skip(18) }).collect::<ArrayVec<_,5>>());
    let _attenuation = attenuation.then(|| (0..full_bandwidth_channels).map(|_| s.bit().then(|| s.bits(5))).collect::<ArrayVec<_,5>>());
    if blocks_len > 1 && s.bit() { s.skip_long((blocks_len-1)*(4+(frame_size-2).ilog2())); }
	//lfe: {end_freq: 7, num_exp_groups: 2}

	let e = |x| 2f32.powf(x/4f32);
	let gain_levels = [e(2.), e(1.), e(0.), e(-1.), e(-2.), e(-3.), e(-4.), 0., e(-6.)];
	let mut downmix = downmix[channel_mode as usize].iter().map(|&[l,r]| [gain_levels[l],gain_levels[r]]).collect::<ArrayVec<_,5>>();
    center_mix_level.map(|center_mix_level| downmix[1] = [gain_levels[center_mix_level]; 2]);
	surround_mix_level.map(|surround_mix_level| {
		let mut level = gain_levels[surround_mix_level];
		if let F2R|F3R = channel_mode { level *= e(-2.); }
		downmix[match channel_mode {F2R|F2R2=>F2, F3R|F3R2=>F3, _=>unreachable!()} as usize] = [level; 2];
	});
	fn transpose<T:Copy, const M: usize, const N: usize>(a: &ArrayVec<[T; N], M>) -> [ArrayVec<T, M>; N] { crate::from_iter((0..N).map(|n| map(a, |m| m[n]))) }
	let downmix = transpose(&downmix);
	let downmix = downmix.map(|downmix| map(&downmix, |x| x / downmix.iter().sum::<f32>()));
	unimplemented!("ac3_decode_frame")
}}