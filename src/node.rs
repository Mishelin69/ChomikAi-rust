use rand::{rngs::ThreadRng, Rng};

pub struct Node {

    pub w: Vec<f64>,
    pub n: usize,

}

impl Node {

    pub fn new(n: usize) -> Node {
        Node { w: Vec::with_capacity(n), n }
    }

    pub fn init(&mut self, rng: &mut ThreadRng) {
        for _ in 0_usize..self.n {
            self.w.push(rng.gen_range(0_f64..=1_f64));
        }
    }

}
