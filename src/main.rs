use std::sync::{RwLock,Arc};

use network::Network;

pub mod network;
pub mod layer;
pub mod node;
pub mod func;

fn main() {

    let mut network = Network::new(&[2, 2, 1]);
    network.init_self();

    let inp = vec![1.0, 1.0, 0.0, 0.0, 1.0, 0.0, 0.0, 1.0];
    let expc = vec![0.0, 0.0, 1.0, 1.0];
    let inp_stat = Arc::new(RwLock::new(inp.clone()));
    let expc_stat = Arc::new(RwLock::new(expc.clone()));

    network.multthrd_learn(4, inp_stat, expc_stat, 10000, 0.1, 1);

    let mut out: Vec<f64> = network.helper_init_actv(false);

    for i in 0..4_usize {

        network.feedforward(&inp[i*network.shape_in..(1+i)*network.shape_in], &mut out, 0);

        println!("EXP: {:?}", expc);
        println!("IN: {:?}", inp);
        println!("OUT: {:?}", out);

        //Network::clear_actv(&mut out); 

    }

}
