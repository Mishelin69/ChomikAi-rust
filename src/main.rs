use network::Network;

pub mod network;
pub mod layer;
pub mod node;
pub mod func;

fn main() {
    let mut network = Network::new(&[2, 2, 1]);
    network.init_self();

    let inp: [f64; 8] = [1.0, 1.0, 0.0, 0.0, 1.0, 0.0, 0.0, 1.0];
    let expc: [f64; 4] = [0.0, 0.0, 1.0, 1.0];

    let mut v: Vec<f64> = Vec::new();
    network.feedforward(&inp, &mut v);
}
