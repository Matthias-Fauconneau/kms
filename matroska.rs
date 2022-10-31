use nom::combinator::ParserIterator;
type Segments<'t> = ParserIterator<&'t [u8], matroska::ebml::Error<'t>, for<'a> fn(&'a [u8])->nom::IResult<&'a [u8], matroska::elements::SegmentElement<'a>, matroska::ebml::Error<'a>>>;
type Blocks<'t> = std::vec::IntoIter<matroska::elements::BlockGroup<'t>>;
type Units<'t> = ParserIterator<&'t [u8], nom::error::Error<&'t [u8]>, for<'a> fn(&'a [u8])->nom::IResult<&'a [u8], &'a [u8], nom::error::Error<&'a [u8]>>>;
pub struct State<'t> {
	video_track_number: u64,
	audio_track_number: u64,
	segments: Segments<'t>
}
pub enum Matroska<'t> {
	Start(State<'t>),
	NextSlice(State<'t>, Blocks<'t>, Units<'t>, &'t [u8]),
	NextFrame(State<'t>, Blocks<'t>),
	Done,
}
pub trait Video {
	type Slice<'t>;
	fn unit<'t>(&mut self, escaped_data: &'t [u8]) -> Option<Self::Slice<'t>>;
}
pub fn matroska<'t, V:Video>(input: &'t [u8], mut video: V) -> Result<(Matroska<'t>, V), nom::Err<matroska::ebml::Error>> {
	use nom::combinator::iterator;
	let mut demuxer = matroska::demuxer::MkvDemuxer::new();
    let (input, ()) = demuxer.parse_until_tracks(input)?;
    let tracks = demuxer.tracks.unwrap();
    let video_track = &tracks.tracks[0];
    assert!(video_track.codec_id == "V_MPEGH/ISO/HEVC");
	let audio = &tracks.tracks[1];
    assert!(audio.codec_id == "A_EAC3");
	let s = &video_track.codec_private.as_ref().unwrap()[22..];
	let (&count, mut s) = s.split_first().unwrap();
	for _ in 0..count { s = {
		let (_, s) = s.split_first().unwrap();
		let (count, s) = {let (v, s) = s.split_at(2); (u16::from_be(unsafe{*(v as *const _ as *const u16)}), s)};
		let mut s = s; for _ in 0..count { s = {
			let (length, s) = {let (v, s) = s.split_at(2); (u16::from_be(unsafe{*(v as *const _ as *const u16)}), s)};
			let (unit, s) = s.split_at(length as usize);
			assert!(video.unit(unit).is_none());
			s
		}}
		s
	}}
	let video_track_number = video_track.track_number;
	let audio_track_number = audio.track_number;
	let segments = iterator(input, matroska::elements::segment_element as _);
	use std::ops::GeneratorState as YieldedComplete;
	impl<'t, V:Video> std::ops::Generator<&mut V> for Matroska<'t> {
		type Yield = (V::Slice<'t>, bool);
		type Return = ();
		fn resume(mut self: std::pin::Pin<&mut Self>, video: &mut V) -> YieldedComplete<Self::Yield, Self::Return> {
			type YieldedComplete0<'t, V:Video> = YieldedComplete<(V::Slice<'t>, bool), ()>;
			use Matroska::*;
			fn unit(input: &[u8]) -> nom::IResult<&[u8], &[u8]> { let (rest, length) = nom::number::complete::be_u32(input)?; let length = length as usize; Ok((&rest[length..], &rest[..length])) }
			fn segments<'t: 'y, 's: 'y, 'y, V:Video>(video: &mut V, mut state: State<'t>) -> (Matroska<'t>, YieldedComplete0<'y, V>) {
				while let Some(segment) = (&mut state.segments).next() { use matroska::elements::SegmentElement::*; match segment {
					Void(_) => {},
					Cluster(cluster) => {
						let mut blocks = cluster.block.into_iter();
						while let Some(matroska::elements::BlockGroup{block,..}) = blocks.next() {
							let (block, matroska::elements::SimpleBlock{track_number,..}) = matroska::elements::simple_block(block).unwrap();
							if track_number == state.video_track_number {
								let mut units = iterator(block, unit as _);
								while let Some(unit) = (&mut units).next() {
									if let Some(slice) = video.unit(unit) {
										return if let Some(unit) = (&mut units).next() {
											(NextSlice(state, blocks, units, unit), YieldedComplete::Yielded((slice, false)))
										} else {
											(NextFrame(state, blocks), YieldedComplete::Yielded((slice, true)))
										}
									}
								}
							}
							else if track_number == state.audio_track_number {
								panic!("{:?}", block);
							}
						}
					}
					Cues(_) => {},
					Chapters(_) => {},
					_ => unreachable!()
				}}
				(Done, YieldedComplete::Complete(()))
			}
			let (point, yielded_complete) = match std::mem::replace(&mut *self, Done) {
				Start(state) => segments(video, state),
				NextSlice(state, blocks, mut units, unit) => {
					let Some(slice) = video.unit(unit) else {unreachable!()};
					if let Some(unit) = (&mut units).next() {
						(NextSlice(state, blocks, units, unit), YieldedComplete::Yielded((slice, false)))
					} else {
						(NextFrame(state, blocks), YieldedComplete::Yielded((slice, true)))
					}
				},
				NextFrame(state, mut blocks) => '_yield: {
					while let Some(matroska::elements::BlockGroup{block,..}) = blocks.next() {
						let (block, matroska::elements::SimpleBlock{track_number,..}) = matroska::elements::simple_block(block).unwrap();
						if track_number == state.video_track_number {
							let mut units = iterator(block, unit as _);
							while let Some(unit) = (&mut units).next() {
								if let Some(slice) = video.unit(unit) {
									break '_yield if let Some(unit) = (&mut units).next() {
										(NextSlice(state, blocks, units, unit), YieldedComplete::Yielded((slice, false)))
									} else {
										(NextFrame(state, blocks), YieldedComplete::Yielded((slice, false)))
									}
								}
							}
						}
					}
					segments(video, state)
				},
				Done => unreachable!()
			};
			*self = point;
			yielded_complete
		}
	}
	Ok((Matroska::Start(State{video_track_number, audio_track_number, segments}), video))
}
