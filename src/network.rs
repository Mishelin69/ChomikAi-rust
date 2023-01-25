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

        let mut off: usize = 0;

    }

}
