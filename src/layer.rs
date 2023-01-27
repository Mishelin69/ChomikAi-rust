use rand::{rngs::ThreadRng, Rng};

use crate::{node::Node, func::{FunctionPair, FnOption}};

pub struct Layer {

    nodes: Vec<Node>,
    n: usize, //weights am 

    ndc: usize, //node count 
    bias: Vec<f64>,

    fp: FunctionPair,

}

impl Layer {

    pub fn new(n: usize, x: usize) -> Layer {

        Layer { nodes: Vec::with_capacity(n), n, ndc: x, bias: Vec::with_capacity(x), fp: FunctionPair::new(FnOption::Sigmoid) }
        
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

    //feed the hungry chomik :sadge:
    pub fn feed(&self, input: &[f64], out: &mut Vec<f64>) {

        let mut off: usize = out.len();

        for i in 0_usize..self.ndc {

            out.push(self.bias[i]);
            for j in 0_usize..self.n {
                out[off] += input[j] * self.nodes[j].w[i];
            }

            out[off] = (self.fp.actv)(out[off]);
            off += 1;
        }

    }

}
