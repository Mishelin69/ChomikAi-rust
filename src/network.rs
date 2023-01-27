use rand::{rngs::ThreadRng, Rng};

use crate::layer::Layer;

pub struct Network<'a> {

    layers: Vec<Layer>,
    layers_total: usize,

    pub nodes_total: usize,

    pub shape: &'a [usize],
    pub shape_in: &'a usize,
    pub shape_out: &'a usize,

}

impl<'a> Network<'a> {

    pub fn new(network_shape: &'a [usize]) -> Network<'a> {
        Network { 
            layers: Vec::with_capacity(network_shape.len()), 
            layers_total: network_shape.len()-1,
            nodes_total: (network_shape.iter().sum::<usize>()) - network_shape[0],
            shape: network_shape, 
            shape_in: &network_shape[0], 
            shape_out: &network_shape[network_shape.len()-1],
        } 
    }

    pub fn init_self(&mut self) {

        let mut i: usize = 0;
        let mut rng = rand::thread_rng();

        while i <= self.layers_total-1 {
            self.layers.push(Layer::new(self.shape[i], self.shape[i+1]));
            self.layers[i].init(&mut rng);

            i += 1;
        }

    }

    pub fn feedforward(&self, input: &[f64], out: &mut Vec<f64>) {

        let elm: usize = input.len() / self.shape_in;

        for x in 0..elm-1 {

            let mut layers = self.layers.iter();
            layers.next().unwrap().feed(&input[x*self.shape_in..(x+1)*self.shape_in], out);

            while let Some(layer) = layers.next() {

                layer.feed(&input[x*self.shape_in..(x+1)*self.shape_in], out);

            }

        }

    }

    fn calc_delta(&self, actv: &[f64], crt: &[f64], out: &mut Vec<f64>) {

    }

    fn apply_change(&mut self, dlt: &Vec<f64>, lr: f64) {

    }

    pub fn learn(&mut self, inp: &Vec<f64>, correct: &Vec<f64>, n_epochs: usize, lr: f64) {

        let elm_am: usize = correct.len() / self.shape_out;

        for e in 0..n_epochs {
            
            //shuffle, calculate delta, apply change, repeat

        }

    }

    fn helper_shuffle_in(&self, arg1: &mut Vec<f64>, arg2: &mut Vec<f64>, rng: &mut ThreadRng) {

        let el_am: usize = arg1.len() / self.shape_in;

        for i in 0..el_am {
            
            let random_indx: usize = rng.gen_range(0..el_am-i) + i;
            
            for x in 0..*self.shape_in {
                arg1.swap(random_indx*self.shape_in + x, random_indx*self.shape_in + x);
            }

            for y in 0..*self.shape_out {
                arg2.swap(random_indx*self.shape_out + y, random_indx*self.shape_out + y);
            }

        }

    }

}
