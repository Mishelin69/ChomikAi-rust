use rand::{rngs::ThreadRng, Rng};

use crate::node::Node;

pub struct Layer {

    nodes: Vec<Node>,
    n: usize, 

    ndc: usize, //node count 
    bias: Vec<f64>,

}

impl Layer {

    pub fn new(n: usize, x: usize) -> Layer {

        Layer { nodes: Vec::with_capacity(n), n, ndc: x, bias: Vec::with_capacity(n)}
        
    }

    pub fn init(&mut self, rng: &mut ThreadRng) {

       for x in 0_usize..self.n {
           self.nodes.push(Node::new(self.ndc));
           self.nodes[x].init(rng);
        }

       for _ in 0_usize..self.ndc {
           self.bias.push(rng.gen_range(0_f64..=1_f64));
        }

    }

    pub fn feed(&self, input: &Vec<f64>, out: &mut Vec<f64>) {

    }

}
