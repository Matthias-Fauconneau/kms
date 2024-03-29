#![allow(incomplete_features)]
#![feature(generic_arg_infer,generic_const_exprs,unchecked_math,int_log,array_windows,array_zip,array_methods,generator_trait,control_flow_enum,raw_ref_op)]
fn from_iter_or_else<T, const N: usize>(iter: impl IntoIterator<Item=T>, f: impl Fn() -> T+Copy) -> [T; N] {
    let mut iter = iter.into_iter();
    let a = [(); N].map(|_| iter.next().unwrap_or_else(f));
    assert!(iter.next().is_none());
    a
}
pub fn from_iter<T, const N: usize>(iter: impl IntoIterator<Item=T>) -> [T; N] { crate::from_iter_or_else(iter, || unreachable!()) }
fn from_iter_or_default<T: Default, const N: usize>(iter: impl IntoIterator<Item=T>) -> [T; N] { from_iter_or_else(iter, || Default::default()) }
fn array<T: Default, const N: usize>(len: usize, mut f: impl FnMut()->T) -> [T; N] { from_iter_or_default((0..len).map(|_| f())) }

mod matroska; use self::matroska::Decoder;
pub mod bit;
mod hevc;
mod eac3;
mod va;

struct Player<T>(T);
impl<T: FnMut()->va::DMABuf> ui::Widget for Player<T> { fn paint(&mut self, target: &mut ui::Target, _: ui::size, _: ui::int2) -> ui::Result<()> {
    *target = Some({let va::DMABuf{format,fd,modifiers,size}=(self.0)(); ui::widget::DMABuf{format,fd,modifiers,size}});
    Ok(())
} }

pub fn main() -> Result<(), Box<dyn std::error::Error>> {
    let path = std::env::args().skip(1).next().unwrap_or(std::env::var("HOME")?+"/input.mkv");
    let input = unsafe{memmap::Mmap::map(&std::fs::File::open(&path)?)}?;
    let mut hevc = hevc::HEVC::new();
    let mut eac3 = eac3::EAC3::new();
    let mut matroska = matroska::matroska(&*input, &mut hevc).unwrap();
    let device = ui::Device::new("/dev/dri/renderD128");
    let ref mut decoder = va::Decoder::new(&device);
    let mut slice = move |decoder:&mut va::Decoder| match std::ops::Generator::resume(std::pin::Pin::new(&mut matroska), (&mut hevc, &mut eac3)) {
        std::ops::GeneratorState::Yielded(matroska::Output::Video(slice,last)) => decoder.slice(&hevc, slice, last),
        std::ops::GeneratorState::Yielded(matroska::Output::Audio(_)) => unimplemented!(),
        std::ops::GeneratorState::Complete(()) => unimplemented!(),
    };
    let mut decoder = move || loop {
        while decoder.sequence.as_ref().filter(|s| s.decode_frame_id.is_some()).is_some() { slice(decoder); }
        while let Some(image) = decoder.next() { return image; }
        slice(decoder);
    };
    ui::run(&mut Player(&mut decoder), &mut |_| Ok(true))
}
