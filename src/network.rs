use rand::{rngs::ThreadRng, Rng, thread_rng};

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

    pub fn feedforward(&self, input: &[f64], out: &mut Vec<f64>, rev: bool) {

        let elm: usize = input.len() / self.shape_in;

        for x in 0..elm {

            let mut layers = self.layers.iter();
            layers.next().unwrap().feed(&input[x*self.shape_in..(x+1)*self.shape_in], out);

            while let Some(layer) = layers.next() {
                layer.feed(&input[x*self.shape_in..(x+1)*self.shape_in], out);
            }
        }

        if rev {
            out.reverse();
        }
    }

    pub fn feedforward_lastonly(&self, input: &[f64], out: &mut Vec<f64>, rev: bool) -> Vec<f64> {

        let elm: usize = input.len() / self.shape_in;
        let mut last = Vec::new();

        for x in 0..elm {

            let mut layers = self.layers.iter();
            layers.next().unwrap().feed(&input[x*self.shape_in..(x+1)*self.shape_in], out);

            while let Some(layer) = layers.next() {
                layer.feed(&input[x*self.shape_in..(x+1)*self.shape_in], out);
            }

            for i in 0..*self.shape_out {
                last.push(out[out.len() - self.shape_out + i]);
            }
        }

        if rev {
            out.reverse();
        }

        last
    }

    fn calc_delta(&self, actv: &[f64], crt: &[f64], out: &mut Vec<f64>) {

        let mut off: usize = actv.len();
        let mut layers_reversed = self.layers.iter().rev();
        let last_layer = layers_reversed.next().unwrap();

        //calc error in output layer 
        for i in 0..*self.shape_out {
            out.push((crt[i] - actv[(off - self.shape_out) + i]) * last_layer.run_der(actv[i]));
        }

        off -= self.shape_out;
        let mut out_off: usize = 0;
        let mut last_layer_nodes = last_layer.ndc;

        //calc error in the rest, stop at input 
        while let Some(layer) = layers_reversed.next() {

            for i in 0..layer.ndc {

                let mut err: f64 = 0.0;
                for j in 0..last_layer_nodes {
                    err += out[out_off + j] * layer.nodes[i].w[j];
                }
                out.push(err * layer.run_der(actv[(off - layer.ndc) + i]));
            }

            out_off += layer.ndc; 
            off -= layer.ndc;
            last_layer_nodes = layer.ndc;
        }
    }

    fn apply_change(&mut self, input: &[f64], actv: &[f64], dlt: &Vec<f64>, lr: f64) {

        let mut layer_ref = self.layers.iter_mut().rev();
        let mut off: usize = actv.len() - self.shape_out;
        let mut dlt_off: usize = 0;

        for _ in 0..self.layers_total-1 {

            let layer = layer_ref.next().unwrap();
            for i in 0..layer.ndc {

                layer.bias[i] += dlt[dlt_off + i] * lr;

                for j in 0..layer.n {

                    layer.nodes[j].w[i] += actv[(off - layer.ndc) + j] * dlt[dlt_off + i] * lr;

                }
            }

            off -= layer.ndc;
            dlt_off += layer.ndc;
        }

        let last_layer = layer_ref.next().unwrap();

        for i in 0..last_layer.ndc  {

            last_layer.bias[i] += dlt[off + i] * lr;

            for j in 0..last_layer.n {

                last_layer.nodes[j].w[i] += input[j] * dlt[dlt_off + i] * lr;

            }
        }
    }

    pub fn learn(&mut self, inp: &mut Vec<f64>, correct: &mut Vec<f64>, n_epochs: usize, lr: f64) {

        let elm_am: usize = correct.len() / self.shape_out;
        let mut rng = thread_rng();
        let mut cur_pred: Vec<f64> = Vec::with_capacity(self.nodes_total);
        let mut out: Vec<f64> = Vec::with_capacity(self.nodes_total);

        for _ in 0..n_epochs {
            for i in 0..elm_am {

                //shuffle, calculate delta, apply change, repeat
                self.helper_shuffle_in(inp, correct, &mut rng);
                self.feedforward(&inp[(i*self.shape_in)..((i+1)*self.shape_in)], &mut cur_pred, false);

                self.calc_delta(
                    &cur_pred, 
                    &correct[(i*self.shape_out)..((i+1)*self.shape_out)], 
                    &mut out,
                );

                self.apply_change(&inp[(i*self.shape_in)..((i+1)*self.shape_in)], &cur_pred, &out, lr);

                cur_pred.clear(); //maybe it fixes the bugged out chomik idk :xdd:
                out.clear();
            }
        }

    }

    fn helper_shuffle_in(&self, arg1: &mut Vec<f64>, arg2: &mut Vec<f64>, rng: &mut ThreadRng) {

        let el_am: usize = arg1.len() / self.shape_in;
        for i in 0..el_am {

            let random_indx: usize = rng.gen_range(0..el_am-i) + i;

            for x in 0..*self.shape_in {
                arg1.swap(random_indx*self.shape_in + x, i*self.shape_in + x);
            }

            for y in 0..*self.shape_out {
                arg2.swap(random_indx*self.shape_out + y, i*self.shape_out + y);
            }

        }

    }

}
