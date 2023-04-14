#[derive(Clone)]
pub struct FunctionPair {

    pub actv: fn(f64) -> f64,
    pub der: fn(f64) -> f64,

}

pub enum FnOption {

    Sigmoid, 
    LnReg,

}

impl FunctionPair {

    pub fn new(e: FnOption) -> FunctionPair {

        match e {

            FnOption::Sigmoid => FunctionPair { actv: sig, der: der_sig },
            FnOption::LnReg => FunctionPair { actv: ln_reg, der: der_ln_reg },
        }

    }

}

fn sig(x: f64) -> f64 {
    return 1.0 / (1.0 + (-x).exp());
}

fn der_sig(x: f64) -> f64 {
    return x * (1.0 - x);
}

fn ln_reg(x: f64) -> f64 {

    if x > 0.0 {
        return x;
    }
    return 0.0;
}

fn der_ln_reg(x: f64) -> f64 {
    ln_reg(x)
}
