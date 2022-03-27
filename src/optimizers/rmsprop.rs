use std::iter::zip;

use crate::fmod;
use ndarray::{Array1, Array2};

use super::optimizers::*;

// numerical stability for float division
const NUM_EPS: fmod = 1e-12;

#[inline(always)]
fn rmsprop_memoizer(
    params: &Vec<fmod>,
    mem: &OptimizerMemory,
    weight_grad: &Vec<Array2<fmod>>,
    bias_grad: &Vec<Array1<fmod>>,
) -> OptimizerMemory {
    let decay = params[1];
    let invdecay = 1.0 - decay;

    if mem.len() != 0 {
        let avgweight: Vec<Array2<fmod>> = (&(mem[0]).0)
            .into_iter()
            .zip(weight_grad.into_iter())
            .map(|(w_mem, w_grad)| decay * w_mem + invdecay * w_grad.mapv(|v| v.powi(2)))
            .collect();
        let avgbias: Vec<Array1<fmod>> = (&(mem[0]).1)
            .into_iter()
            .zip(bias_grad.into_iter())
            .map(|(b_mem, b_grad)| decay * b_mem + invdecay * b_grad.mapv(|v| v.powi(2)))
            .collect();

        vec![(avgweight, avgbias)]
    } else {
        let avgweight: Vec<Array2<fmod>> = weight_grad
            .into_iter()
            .map(|w_grad| invdecay * w_grad.mapv(|v| v.powi(2)))
            .collect();
        let avgbias: Vec<Array1<fmod>> = bias_grad
            .into_iter()
            .map(|b_grad| invdecay * b_grad.mapv(|v| v.powi(2)))
            .collect();

        vec![(avgweight, avgbias)]
    }
}
#[inline(always)]
fn rmsprop_bias_update(
    params: &Vec<fmod>,
    mem: &OptimizerMemory,
    bias_grad: &Vec<Array1<fmod>>,
) -> BiasUpdate {
    let eps = params[0];

    let bias_avg_grads = &mem[0].1;

    zip(bias_avg_grads.into_iter(), bias_grad.into_iter())
        .map(|(b_avg, b_grad)| eps / (NUM_EPS + b_avg.mapv(fmod::sqrt)) * b_grad)
        .collect()
}
#[inline(always)]
fn rmsprop_weight_update(
    params: &Vec<fmod>,
    mem: &OptimizerMemory,
    weight_grad: &Vec<Array2<fmod>>,
) -> WeightUpdate {
    let eps = params[0];

    let weight_avg_grads = &mem[0].0;

    zip(weight_avg_grads.into_iter(), weight_grad.into_iter())
        .map(|(w_avg, w_grad)| eps / (NUM_EPS + w_avg.mapv(fmod::sqrt)) * w_grad)
        .collect()
}

impl Optimizer {
    pub fn rmsprop(eps: fmod, decay: fmod) -> Optimizer {
        Optimizer {
            parameters: vec![eps, decay],
            memoizer: rmsprop_memoizer,
            memory: vec![],
            bias_update_rule: rmsprop_bias_update,
            weight_update_rule: rmsprop_weight_update,
        }
    }
}
