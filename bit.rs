pub struct BitReader<'t> {
    pub word: u64,
    count: u8,
    ptr: *const u8,
    end: *const u8,
    begin: *const u8,
    phantom: ::core::marker::PhantomData<&'t [u8]>,
}
impl<'t> BitReader<'t> {
    pub fn new(data: &'t [u8]) -> Self {
        Self {
            word: 0,
            count: 0,
            ptr: data.as_ptr(),
            end: data.as_ptr_range().end,
            begin: data.as_ptr(),
            phantom: ::core::marker::PhantomData,
        }
    }
    unsafe fn refill(&mut self) {
        self.word |= core::ptr::read_unaligned(self.ptr as *const u64).to_be() >> self.count;
        self.ptr = self.ptr.add(7);
        self.ptr = self.ptr.sub((self.count as usize >> 3) & 7);
        self.count |= 56;
    }
    fn peek(&self, count: u8) -> u64 { unsafe { self.word.unchecked_shr(64 - count as u64) } }
    #[track_caller] pub fn advance(&mut self, count: u8) { self.word <<= count; assert!(count <= self.count, "{count} {}", self.count); self.count -= count; }
    #[track_caller] pub fn bits(&mut self, count: u8) -> u64 {
        if count > self.count { unsafe { self.refill(); } }
        let result = self.peek(count);
        self.advance(count);
        result
    }
    pub fn bit(&mut self) -> bool { self.bits(1) != 0 }
    pub fn u8(&mut self) -> u8 { self.bits(8) as u8 }
    pub fn u16(&mut self) -> u16 { self.bits(16) as u16 }
    pub fn u32(&mut self) -> u32 { self.bits(32) as u32 }
    #[track_caller] pub fn ue(&mut self) -> u64 { // Exp-Golomb
        unsafe { self.refill(); }
        let count = self.word.leading_zeros() as u8;
        self.advance(count);
        self.bits(1+count) - 1
    }
    pub fn se(&mut self) -> i64 {
        let v = self.ue() as i64;
        let sign = -(v & 1);
        ((v >> 1) ^ sign) - sign
    }
    pub fn available(&self) -> usize { self.count as usize + (self.end as usize-self.ptr as usize)*8 }
    pub fn bits_offset(&self) -> usize { (self.ptr as usize-self.begin as usize)*8 - self.count as usize }
}
