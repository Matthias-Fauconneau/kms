pub struct EAC3;
impl EAC3 { pub fn new() -> Self { Self } }
impl crate::Decoder for EAC3 {
	type Output<'t> = &'t [u8];
	fn decode<'t>(&mut self, data: &'t [u8]) -> Option<Self::Output<'t>> {
		fn sync<const N: usize>(mut data: &[u8], needle: [u8; N]) -> &[u8] { while !data.starts_with(&needle) { (_, data) = data.split_first().unwrap(); } &data[needle.len()..] }
		use itertools::Itertools;
		panic!("{:x}", sync(data, [0xb,0x77]).iter().format(" "));
	}
}