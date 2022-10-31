#![allow(incomplete_features)]#![feature(generic_arg_infer,generic_const_exprs,unchecked_math,int_log,array_windows,array_zip,array_methods,type_alias_impl_trait,generator_trait,control_flow_enum,try_trait_v2,try_blocks,closure_track_caller,raw_ref_op)]
fn from_iter_or_else<T, const N: usize>(iter: impl IntoIterator<Item=T>, f: impl Fn() -> T+Copy) -> [T; N] { let mut iter = iter.into_iter(); [(); N].map(|_| iter.next().unwrap_or_else(f)) }
pub fn from_iter<T, const N: usize>(iter: impl IntoIterator<Item=T>) -> [T; N] { crate::from_iter_or_else(iter, || unreachable!()) }
fn from_iter_or_default<T: Default, const N: usize>(iter: impl IntoIterator<Item=T>) -> [T; N] { from_iter_or_else(iter, || Default::default()) }
fn array<T: Default, const N: usize>(len: usize, mut f: impl FnMut()->T) -> [T; N] { from_iter_or_default((0..len).map(|_| f())) }

mod matroska; use self::matroska::Video;
mod hevc;
mod va;

struct Player<T>(T);
impl<T: FnMut()->va::DMABuf> ui::Widget for Player<T> { fn paint(&mut self, target: &mut ui::Target, _: ui::size, _: ui::int2) -> ui::Result<()> {
    *target = Some({let va::DMABuf{format,fd,modifiers,size}=(self.0)(); ui::widget::DMABuf{format,fd,modifiers,size}});
    Ok(())
} }

pub fn main() -> Result<(), Box<dyn std::error::Error>> {
    let path = std::env::args().skip(1).next().unwrap_or(std::env::var("HOME")?+"/input.mkv");
    let input = unsafe{memmap::Mmap::map(&std::fs::File::open(&path)?)}?;
    let (mut matroska, mut hevc) = matroska::matroska(&*input, hevc::HEVC::new()).unwrap();
    let device = ui::Device::new("/dev/dri/renderD128");
    let ref mut decoder = va::Decoder::new(&device);
    let mut slice = move |decoder:&mut va::Decoder| match std::ops::Generator::resume(std::pin::Pin::new(&mut matroska), &mut hevc) {
        std::ops::GeneratorState::Yielded((slice,last)) => decoder.slice(&hevc, slice, last),
        std::ops::GeneratorState::Complete(()) => unimplemented!(),
    };
    let mut decoder = move || loop {
        while decoder.sequence.as_ref().filter(|s| s.decode_frame_id.is_some()).is_some() { slice(decoder); }
        while let Some(image) = decoder.next() { return image; }
        slice(decoder);
    };
    ui::run(&mut Player(&mut decoder), &mut |_| Ok(true))
}
