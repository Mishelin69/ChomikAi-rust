use network::Network;

pub mod network;
pub mod layer;
pub mod node;
pub mod func;

fn main() {
    let mut network = Network::new(&[2, 2, 1]);
    network.init_self();

    let mut inp = vec![1.0, 1.0, 0.0, 0.0, 1.0, 0.0, 0.0, 1.0];
    let mut expc = vec![0.0, 0.0, 1.0, 1.0];

    network.learn(&mut inp, &mut expc, 10, 0.1);
}
