//#![allow(incomplete_features)]#![feature(,generic_arg_infer,generic_const_exprs,,,iter_array_chunks,,array_chunks,new_uninit,closure_track_caller)]
#![allow(incomplete_features)]#![feature(generic_arg_infer,generic_const_exprs,unchecked_math,int_log,array_windows,array_zip,array_methods,type_alias_impl_trait,generator_trait,control_flow_enum,try_trait_v2,try_blocks)]
fn from_iter_or_else<T, const N: usize>(iter: impl IntoIterator<Item=T>, f: impl Fn() -> T+Copy) -> [T; N] { let mut iter = iter.into_iter(); [(); N].map(|_| iter.next().unwrap_or_else(f)) }
fn from_iter_or_default<T: Default, const N: usize>(iter: impl IntoIterator<Item=T>) -> [T; N] { from_iter_or_else(iter, || Default::default()) }
fn array<T: Default, const N: usize>(len: usize, mut f: impl FnMut()->T) -> [T; N] { from_iter_or_default((0..len).map(|_| f())) }

mod video;

struct Card(std::fs::File);
impl Card { fn new() -> Self { Self(std::fs::OpenOptions::new().read(true).write(true).open("/dev/dri/card0").unwrap()) } }
impl std::os::fd::AsFd for Card { fn as_fd(&self) -> std::os::fd::BorrowedFd { self.0.as_fd() } }
impl std::os::fd::AsRawFd for Card { fn as_raw_fd(&self) -> std::os::fd::RawFd { self.0.as_raw_fd() } }
use drm::Device;
impl Device for Card {}

mod va;

struct Player<T>(T);
impl<T: FnMut()->va::Image<'static>> ui::Widget for Player<T> { fn paint(&mut self, target: &mut ui::Target, _: ui::size, _: ui::int2) -> ui::Result<()> {
    let image = (self.0)();
    use {vector::xy, std::cmp::min};
    for y in 0..min(image.size.y, target.size.y) { for x in 0..min(image.size.x, target.size.x) { target[xy{x,y}] = ((image[xy{x,y}]>>(6+2)) as u8).into(); }}
    Ok(())
} }

pub fn main() -> Result<(), Box<dyn std::error::Error>> {
    let input = unsafe{memmap::Mmap::map(&std::fs::File::open(&std::env::args().skip(1).next().unwrap_or(std::env::var("HOME")?+"/input.mkv"))?)}?;
    let (mut matroska, mut hevc) = video::matroska(&*input).unwrap();
    let card = Card::new();
    let mut decoder = va::Decoder::new(&card);
    ui::run(&mut Player(move || loop { match std::ops::Generator::resume(std::pin::Pin::new(&mut matroska), &mut hevc) {
        std::ops::GeneratorState::Yielded((slice,last)) => if let Some(image) = decoder.slice(&hevc, slice, last) { return image; }
        std::ops::GeneratorState::Complete(()) => unimplemented!(),
    }}), &mut |_| Ok(true))
}
