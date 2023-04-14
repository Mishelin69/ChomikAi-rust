use core::time;
use std::{sync::{Mutex, Arc, self, RwLockWriteGuard}, thread, time::Duration};

use rand::{rngs::ThreadRng, Rng, thread_rng};
use rust_thread_pool;

use crate::layer::Layer;

/*
<========================================================================================================>
TODO: 
  -> get rid of 2D arrays => reduce cache misses ✓
  -> rewrite `calc_delta` to work with the new system ✓
  -> implement multithreaded  learning
  -> maybe improve performance somewhere else ✓ (by half actually [>=])
<========================================================================================================>
*/

pub fn make_flat_copy(v: &[Vec<f64>], expc_s: usize) -> Vec<f64> {

    let mut cpy = Vec::with_capacity(expc_s);

    for x in v {
        for e in x {
            cpy.push(*e);
        }
    }
    cpy
}

struct ThreadDatSync<T> {
    pub dat: T,
}

impl<T> ThreadDatSync<T> {

    pub fn new(dat: T) -> ThreadDatSync<T> {
        ThreadDatSync { dat }
    }

}

#[derive(Clone)]
pub struct Network {

    layers: Vec<Layer>,
    layers_total: usize,

    pub nodes_total: usize,

    pub shape: Vec<usize>,
    pub shape_in: usize,
    pub shape_out: usize,

}

impl Network {

    ///Creates `not initialized` network with the giver parameters
    /// 
    ///`network shape` => should represent layers in the order left to right 
    pub fn new(network_shape: &[usize]) -> Network {
        Network { 
            layers: Vec::with_capacity(network_shape.len()), 
            layers_total: network_shape.len()-1,
            nodes_total: (network_shape.iter().sum::<usize>()) - network_shape[0],
            shape: network_shape.to_vec(), 
            shape_in: network_shape[0], 
            shape_out: network_shape[network_shape.len()-1],
        } 
    }

    ///Function initilizes network => creates nodes, biases, all is initialized with random valeus
    pub fn init_self(&mut self) {

        let mut i: usize = 0;
        let mut rng = rand::thread_rng();

        while i <= self.layers_total-1 {
            self.layers.push(Layer::new(self.shape[i], self.shape[i+1]));
            self.layers[i].init(&mut rng);

            i += 1;
        }

    }

    ///Does pass through of a network with the given input.
    ///`Note`: input in expected to be just one data block, not a whole bunch of them
    /// 
    ///`input` => input data for the network
    /// 
    ///`out` => activations are stored here
    /// 
    /// `Note`: no checks are done so random panics could happen resulting in a non-recoverable state
    pub fn feedforward(&self, input: &[f64], out: &mut Vec<f64>, nth: usize) {

        let elm: usize = input.len() / self.shape_in;
        let mut off: usize = nth*self.nodes_total;

        for _ in 0..elm {

            let mut layers = self.layers.iter();
            let mut last_layer = layers.next().unwrap();

            for i in 0_usize..last_layer.ndc {

                out[i] = last_layer.bias[i];
                for j in 0_usize..last_layer.n {
                    out[off+i] += input[j] * last_layer.nodes[j].w[i];
                }

                out[off+i] = last_layer.run_actv(out[off+i]);
            }

            off += last_layer.ndc;

            while let Some(layer) = layers.next() {

                for i in 0_usize..layer.ndc {

                    out[off+i] = layer.bias[i];
                    for j in 0_usize..layer.n {
                        out[off+i] += out[off-last_layer.ndc+j] * layer.nodes[j].w[i];
                    }

                    out[off+i] = layer.run_actv(out[off+i]);
                }

                off += layer.ndc;
                last_layer = &layer;
            }
        }
    }

    ///Calculates networks error on x dataset
    ///
    ///`actv` => x dataset pass through was done on
    /// 
    ///`crt` => expected output 
    /// 
    ///`out` => where output of this function is stored at
    /// 
    ///`nth` => n-th elm
    /// 
    ///`Note`: no checks are done so random panics could happen resulting in a non-recoverable state
    fn calc_delta(&self, actv: &Vec<f64>, crt: &[f64], out: &mut Vec<f64>, nth: usize) {

        let mut layers_reversed = self.layers.iter().rev();
        let mut last_layer = layers_reversed.next().unwrap();
        let mut off: usize = (nth*self.nodes_total) - last_layer.ndc;
        let mut test_start = 0;

        for i in 0..last_layer.ndc {
            out[test_start] = (crt[i] - actv[off + i]) * last_layer.run_der(actv[off + i]);
        }
        test_start += last_layer.ndc;

        let mut last_layer_nodes = last_layer.ndc;

        //calc error in the rest, stop at input 
        while let Some(layer) = layers_reversed.next() {

            off -= layer.ndc;
            for i in 0..layer.ndc {

                let mut err: f64 = 0.0;
                for j in 0..last_layer_nodes {
                    err += out[test_start - i - last_layer.ndc + j] * last_layer.nodes[i].w[j];
                }

                out[test_start] = err * layer.run_der(actv[off + i]);
                test_start += 1;
            }

            last_layer_nodes = layer.ndc;
            last_layer = layer;
        }
    }

    ///Takes input, corresponding activations, corresponding error to that dataset and corrects to network
    ///using gradient descent function 
    /// 
    ///`input` => given input
    /// 
    ///`actv` => current networks pass throught of that input
    /// 
    ///`dlt` => corresponding calculater error
    /// 
    ///`nth` => n-th element
    /// 
    ///`lr` => learn rate
    /// 
    ///`Note`: no checks are done so random panics could happen resulting in a non-recoverable state
    fn apply_change(&mut self, input: &[f64], actv: &Vec<f64>, dlt: &Vec<f64>, nth: usize, lr: f64) {

        let mut layer_ref = self.layers.iter_mut().rev();
        let mut off: usize = nth*self.nodes_total;
        let mut dlt_off = (nth-1)*self.nodes_total;

        for _ in 0..self.layers_total-1 {

            let layer = layer_ref.next().unwrap();
            off -= layer.ndc;

            for i in 0..layer.ndc {

                layer.bias[i] += dlt[dlt_off + i] * lr;
                for j in 0..layer.n {
                    layer.nodes[j].w[i] += actv[off - layer.n + j] * dlt[dlt_off + i] * lr;
                }
            }

            dlt_off += layer.ndc;
        }

        let last_layer = layer_ref.next().unwrap();

        for i in 0..last_layer.ndc  {

            last_layer.bias[i] += dlt[dlt_off + i] * lr;
            for j in 0..last_layer.n {
                last_layer.nodes[j].w[i] += input[j] * dlt[dlt_off + i] * lr;
            }
        }
    }

    ///Trains the network with the data supplied
    /// 
    ///`inp` => input
    /// 
    ///`correct` => corresponding labels
    /// 
    ///`n_epochs` => number of epochs
    /// 
    ///`lr` => learn rate
    ///
    ///`Note`: no checks are done so random panics could happen resulting in a non-recoverable state
    pub fn learn(&mut self, inp: &mut Vec<f64>, correct: &mut Vec<f64>, n_epochs: usize, lr: f64) {

        let elm_am: usize = correct.len() / self.shape_out;
        let mut rng = thread_rng();
        let mut cur_pred: Vec<f64> = self.helper_init_actv(false);
        let mut delta: Vec<f64> = self.helper_init_actv(false); 
        let zero = 0;

        for _ in 0..n_epochs {
            for i in 0..elm_am {

                //shuffle, calculate delta, apply change, repeat
                self.helper_shuffle_in(inp, correct, &mut rng);
                self.feedforward(
                    &inp[(i*self.shape_in)..((i+1)*self.shape_in)], 
                    &mut cur_pred,
                    zero,
                );

                self.calc_delta(
                    &cur_pred, 
                    &correct[(i*self.shape_out)..((i+1)*self.shape_out)], 
                    &mut delta,
                    1
                    );

                self.apply_change(
                    &inp[(i*self.shape_in)..((i+1)*self.shape_in)], 
                    &cur_pred, 
                    &delta, 
                    1,
                    lr,
                );

                // cur_pred.reverse();
            }
        }
        println!("EXIT!!!!");
    }

    ///Helper function used in `learn` method, initializes vector for activations and error
    /// 
    ///`rev` => should/should not reverse vector
    pub fn helper_init_actv(&self, rev: bool) -> Vec<f64> {

        let mut v = Vec::with_capacity(self.nodes_total);
        unsafe { v.set_len(self.nodes_total ); }

        if rev {
            v.reverse()
        }

        v
    }

    ///Multi-threaded version of the `learn` method, benefits in speed and uses batching
    ///for more effecient data distribution and better correction
    /// 
    ///`max_workers` => max threads
    /// 
    ///`train_data` => `input` field in the `learn` method
    /// 
    ///`correct` => `correct` field in the `learn` method
    /// 
    ///`epochs` => `n_epochs` field in the `learn` method
    /// 
    ///`lr` => `lr` field in the `learn` method 
    /// 
    ///`batch_size` => batch size

    pub fn multthrd_learn(&mut self, max_workers: usize, train_data: Arc<sync::RwLock<Vec<f64>>>, correct: Arc<sync::RwLock<Vec<f64>>>, epochs: usize, lr: f64, batch_size: usize) {

        assert!(max_workers > 1 && max_workers < usize::from(std::thread::available_parallelism().unwrap()), "max_workers was zero or the number exceeded number of physical procesors");
        //check if the number is a power of two
        assert!(batch_size >= 1 && (batch_size & (batch_size - 1) == 0));

        let traindat_lock = {

            if train_data.is_poisoned() {

                panic!("Poisoned RwLock");
            }

            train_data.read().unwrap()

        };

        let amount_input = train_data.read().unwrap().len() / self.shape_in;
        let amount_correct = correct.read().unwrap().len() / self.shape_out;

        assert!((amount_input == amount_correct), "Number of elements in input and labels doesn't match, {} {}", amount_input, amount_correct);

        let batch_iters = traindat_lock.len() / (batch_size*self.shape_in);
        let thread_iters = batch_iters / max_workers;
        let iters_left = amount_input % batch_iters;
        let pool = rust_thread_pool::pool::ThreadPool::new(max_workers);
        let mut thread_output: Vec<Arc<Mutex<ThreadDatSync<Vec<f64>>>>> = Vec::with_capacity(max_workers);
        let mut rng = thread_rng();

        std::mem::drop(traindat_lock);

        for i in 0..max_workers {

            thread_output.push(Arc::new(Mutex::new(ThreadDatSync::new(Vec::with_capacity(self.nodes_total*batch_size)))));

            let x = &thread_output[i];
            unsafe { 
                let mut lock = x.lock().unwrap();
                lock.dat.set_len(self.nodes_total*batch_size);
            }
        }

        for e in 0..epochs {

            let self_copy = Arc::new(self.clone());
            for elm in 0..thread_iters {
                for thitr in 0..max_workers {

                    let output = Arc::clone(&thread_output[elm]);
                    let (start, end) = (elm*max_workers*self.shape_in + self.shape_in*thitr, elm*max_workers*self.shape_in + self.shape_in*(thitr + 1));
                    let arc_train = Arc::clone(&train_data);
                    let self_arc = Arc::clone(&self_copy);

                    pool.execute(move || {

                        //"lock" thread output object so that we can keep them in sync and avoid bugs 
                        let out = &mut output.lock().unwrap();
                        println!("{:?}", &arc_train.read().unwrap()[start..end]);

                        self_arc.shape_in;

                        std::mem::drop(out);
                    });
                }

                //wait for all threads to finish
                let mut sync = max_workers;
                while sync > 0 {

                    sync = max_workers;
                    for x in &thread_output {
                        
                        match x.try_lock() {
                            Ok(_) => sync -= 1,
                            Err(_) => (),
                        }
                    }
                }

                //thread::sleep(Duration::from_millis(10));
                println!("========================");
            }

            println!("PUK");
            self.helper_shuffle_in_rw(train_data.write().unwrap(), correct.write().unwrap(), &mut rng);
            println!("MEGA");
        }
    }

    ///Helper function that shuffles the input while making sure the labels match the new order
    /// 
    ///`arg1` => generally the input
    /// 
    ///`arg2` => generally the labels
    /// 
    ///`rng` => `ThreadRng` object to use
    fn helper_shuffle_in(&self, arg1: &mut Vec<f64>, arg2: &mut Vec<f64>, rng: &mut ThreadRng) {

        let el_am: usize = arg1.len() / self.shape_in;
        for i in 0..el_am {

            let random_indx: usize = rng.gen_range(0..el_am-i) + i;

            for x in 0..self.shape_in {
                arg1.swap(random_indx*self.shape_in + x, i*self.shape_in + x);
            }

            for y in 0..self.shape_out {
                arg2.swap(random_indx*self.shape_out + y, i*self.shape_out + y);
            }

        }

    }

    fn helper_shuffle_in_rw(&self, mut arg1: RwLockWriteGuard<Vec<f64>>, mut arg2: RwLockWriteGuard<Vec<f64>>, rng: &mut ThreadRng) {

        println!("GARLIC");
        let el_am: usize = arg1.len() / self.shape_in;
        for i in 0..el_am {

            let random_indx: usize = rng.gen_range(0..el_am-i) + i;

            for x in 0..self.shape_in {
                arg1.swap(random_indx*self.shape_in + x, i*self.shape_in + x);
                //let swp = arg1[random_indx*self.shape_in + x];
                //arg1[random_indx*self.shape_in + x] = arg1[i*self.shape_in + x];
                //arg1[i*self.shape_in + x] = swp;

            }

            for y in 0..self.shape_out {
                arg2.swap(random_indx*self.shape_out + y, i*self.shape_out + y);
            }

        }

    }

    ///Clears given vector
    /// 
    ///`a` => vector to be cleared
    #[inline]
    pub fn clear_actv(a: &mut Vec<Vec<f64>>) {
        for x in a {
            x.clear();
        }
    }

}