pub struct EAC3;
impl EAC3 { pub fn new() -> Self { Self } }
impl crate::Decoder for EAC3 {
	type Output<'t> = &'t [u8];
	fn decode<'t>(&mut self, input: &'t [u8]) -> Option<Self::Output<'t>> {
		fn sync<const N: usize>(mut input: &[u8], needle: [u8; N]) -> &[u8] { while !input.starts_with(&needle) { (_, input) = input.split_first().unwrap(); } &input[needle.len()..] }
		let mut s = crate::bit::Reader::new(sync(input, [0xb,0x77]));
		let _crc = s.u16();
		let sample_rate = s.bits(4);
		panic!("{sample_rate}");
		/*fn decode<'t>(_: &mut EAC3, input: &'t [u8]) -> nom::IResult<&'t [u8], <EAC3 as crate::Decoder>::Output<'t>> {
			use nom::{combinator::map, sequence::tuple, number::complete::be_u16, bits::{bits,complete::take}};
			#[derive(Debug)] struct Header { crc1: u16, sample_rate: u8, frame_size: u8};
			panic!("{:?}", map(tuple((/*crc1*/be_u16, bits::<_, _, nom::error::Error<(&[u8], usize)>, _, _>(tuple((take(4usize), take(8usize)))))), |(crc1, (sample_rate, frame_size))| Header{crc1, sample_rate, frame_size})(input)?)
			//use itertools::Itertools; panic!("{:x}", input.iter().format(" "));
		}
		Some(decode(self, input).unwrap().1)*/
	}
}