use network::Network;

pub mod network;
pub mod layer;
pub mod node;

fn main() {
    let mut network = Network::new(&[2, 2, 1]);
    network.init_self();
}
