pub trait Decoder {
	type Output<'t>;
	fn decode<'t>(&mut self, data: &'t [u8]) -> Option<Self::Output<'t>>;
}
impl<T:Decoder> Decoder for &mut T {
	type Output<'t> = T::Output<'t>;
	fn decode<'t>(&mut self, data: &'t [u8]) -> Option<Self::Output<'t>> { T::decode(self, data) }
}

pub enum Matroska<'t> {
	Start(Segments<'t>),
	NextVideoSlice(Segments<'t>, Blocks<'t>, Units<'t>, &'t [u8]),
	NextFrame(Segments<'t>, Blocks<'t>),
	Done,
}
impl Default for Matroska<'_> { fn default() -> Self { Self::Done } }
use Matroska::*;

use std::ops::GeneratorState as YieldedComplete;
pub enum Output<'t, V:Decoder, A:Decoder> {
	Video(V::Output<'t>, bool),
	Audio(A::Output<'t>),
}
impl<'t, V:Decoder, A:Decoder> std::ops::Generator<(&mut V, &mut A)> for Matroska<'t> {
	type Yield = Output<'t,V,A>;
	type Return = ();
	fn resume(mut self: std::pin::Pin<&mut Self>, ref mut state: (&mut V, &mut A)) -> YieldedComplete<Self::Yield, Self::Return> {
		replace_with(&mut *self, |point| match point {
			Start(segments) => self::segments(state, segments).break_value().unwrap(),
			NextVideoSlice(segments, blocks, units, unit) => self::unit(state, segments, blocks, units, unit).break_value().unwrap(),
			NextFrame(segments, blocks) => match self::blocks(state, segments, blocks) {
				Break(y) => y,
				Continue(segments) => self::segments(state, segments).break_value().unwrap()
			},
			Done => unreachable!()
		})
	}
}

use nom::combinator::{ParserIterator, iterator};
pub struct Segments<'t> {
	video_track_number: u64,
	audio_track_number: u64,
	segments: ParserIterator<&'t [u8], matroska::ebml::Error<'t>, for<'a> fn(&'a [u8])->nom::IResult<&'a [u8], matroska::elements::SegmentElement<'a>, matroska::ebml::Error<'a>>>,
}
type Blocks<'t> = std::vec::IntoIter<matroska::elements::BlockGroup<'t>>;
type Units<'t> = ParserIterator<&'t [u8], nom::error::Error<&'t [u8]>, for<'a> fn(&'a [u8])->nom::IResult<&'a [u8], &'a [u8], nom::error::Error<&'a [u8]>>>;

pub fn matroska<'t, V:Decoder>(input: &'t [u8], video: &mut V) -> Result<Matroska<'t>, nom::Err<matroska::ebml::Error<'t>>> {
	let mut demuxer = matroska::demuxer::MkvDemuxer::new();
    let (input, ()) = demuxer.parse_until_tracks(input)?;
    let tracks = demuxer.tracks.unwrap();
    let video_track_number = {
		let video_track = &tracks.tracks[0];
		assert!(video_track.codec_id == "V_MPEGH/ISO/HEVC");
		let s = &video_track.codec_private.as_ref().unwrap()[22..];
		let (&count, mut s) = s.split_first().unwrap();
		for _ in 0..count { s = {
			let (_, s) = s.split_first().unwrap();
			let (count, s) = {let (v, s) = s.split_at(2); (u16::from_be(unsafe{*(v as *const _ as *const u16)}), s)};
			let mut s = s; for _ in 0..count { s = {
				let (length, s) = {let (v, s) = s.split_at(2); (u16::from_be(unsafe{*(v as *const _ as *const u16)}), s)};
				let (unit, s) = s.split_at(length as usize);
				assert!(video.decode(unit).is_none());
				s
			}}
			s
		}}
		video_track.track_number
	};
	let audio_track_number = {
		let audio_track = &tracks.tracks[1];
		assert!(audio_track.codec_id == "A_EAC3");
		audio_track.track_number
	};
	let segments = iterator(input, matroska::elements::segment_element as _);
	Ok(Start(Segments{video_track_number, audio_track_number, segments}))
}

use std::ops::ControlFlow::*;
type ControlFlow<'t,'y, V, A, C> = std::ops::ControlFlow<(Matroska<'t>, YieldedComplete<Output<'y,V,A>, ()>), C>;

fn segments<'t: 'y, 'y, V:Decoder, A:Decoder>(state: &mut (&mut V, &mut A), mut segments: Segments<'t>) -> ControlFlow<'t, 'y, V, A, Segments<'t>> {
	while let Some(segment) = (&mut segments.segments).next() { use matroska::elements::SegmentElement::*; match segment {
		Void(_) => {},
		Cluster(cluster) => segments = blocks(state, segments, cluster.block.into_iter())?,
		Cues(_) => {},
		Chapters(_) => {},
		_ => unreachable!()
	}}
	Break((Done, YieldedComplete::Complete(())))
}
fn blocks<'t: 'y, 'y, V:Decoder, A:Decoder>(state: &mut (&mut V, &mut A), mut segments: Segments<'t>, mut blocks: Blocks<'t>) -> ControlFlow<'t, 'y, V, A, Segments<'t>> {
	while let Some(matroska::elements::BlockGroup{block,..}) = blocks.next() {
		let (block, matroska::elements::SimpleBlock{track_number,..}) = matroska::elements::simple_block(block).unwrap();
		if track_number == segments.video_track_number {
			fn unit(input: &[u8]) -> nom::IResult<&[u8], &[u8]> { let (rest, length) = nom::number::complete::be_u32(input)?; let length = length as usize; Ok((&rest[length..], &rest[..length])) }
			let mut units = iterator(block, unit as _);
			while let Some(unit) = (&mut units).next() { (segments, blocks, units) = self::unit(state, segments, blocks, units, unit)?; }
		}
		else if track_number == segments.audio_track_number {
			let (_, audio) = state;
			return Break((NextFrame(segments, blocks), YieldedComplete::Yielded(Output::Audio(audio.decode(block).unwrap()))));
		}
	}
	Continue(segments)
}
fn unit<'t: 'y, 'y, V:Decoder, A:Decoder>((video, _): &mut (&mut V, &mut A), segments: Segments<'t>, blocks: Blocks<'t>, mut units: Units<'t>, unit: &'t [u8]) -> ControlFlow<'t, 'y, V, A, (Segments<'t>, Blocks<'t>, Units<'t>)> {
	if let Some(slice) = video.decode(unit) {
		return if let Some(unit) = (&mut units).next() {
			Break((NextVideoSlice(segments, blocks, units, unit), YieldedComplete::Yielded(Output::Video(slice, false))))
		} else {
			Break((NextFrame(segments, blocks), YieldedComplete::Yielded(Output::Video(slice, true))))
		}
	}
	Continue((segments, blocks, units))
}

fn replace_with<T:Default, R>(state: &mut T, mut f: impl FnMut(T) -> (T,R)) -> R { let (new_state, value) = f(std::mem::take(&mut *state)); *state = new_state; value }