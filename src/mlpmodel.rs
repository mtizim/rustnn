use crate::optimizers::optimizers::Optimizer;
use crate::*;
use std::iter::zip;

use itertools::izip;
use ndarray_rand::rand::prelude::StdRng;
use ndarray_rand::rand::SeedableRng;
use ndarray_rand::rand_distr::StandardNormal;
use ndarray_rand::RandomExt;
use rand::prelude::SliceRandom;
use rayon::prelude::*;

pub struct Gradients(
    pub Vec<Array2<fmod>>,
    pub Vec<Array1<fmod>>, //
);
pub struct MultilayerPerceptron {
    pub weights: Vec<Array2<fmod>>,
    pub biases: Vec<Array1<fmod>>,
    pub shape: Vec<u16>,
    pub activations: Vec<NeuronActivation>,
    pub optimizer: Optimizer,
    softmaxed: bool,
}

impl MultilayerPerceptron {
    pub fn new(
        shape: Vec<u16>,
        activations: Vec<NeuronActivation>,
        optimizer: Optimizer,
        softmaxed: bool,
    ) -> MultilayerPerceptron {
        let mut rng: StdRng = SeedableRng::from_seed([0u8; 32]);

        let mut weights: Vec<Array2<fmod>> = Vec::new();
        let mut biases: Vec<Array1<fmod>> = Vec::new();
        for layeridx in 1..shape.len() {
            let dimfrom = shape[layeridx - 1] as usize;
            let dimto = shape[layeridx] as usize;
            weights.push(Array::random_using(
                (dimfrom, dimto),
                StandardNormal,
                &mut rng,
            ));
            biases.push(Array::random_using(dimto, StandardNormal, &mut rng));
        }
        assert!(activations.len() == shape.len() - 1);

        MultilayerPerceptron {
            weights,
            biases,
            shape,
            activations,
            optimizer,
            softmaxed,
        }
    }

    pub fn predict_sample(&self, x: &Array1<fmod>) -> Array1<fmod> {
        let mut x_layer = x.to_owned();

        for (w, b, activation) in izip!(&self.weights, &self.biases, &self.activations) {
            x_layer = (w.t().dot(&x_layer) + b).mapv(activation.activation);
        }
        if self.softmaxed {
            x_layer = softmax(&x_layer);
        }
        x_layer.to_owned()
    }
    pub fn predict(&self, x: &Vec<Array1<fmod>>) -> Vec<Array1<fmod>> {
        x.into_iter().map(|s| self.predict_sample(s)).collect()
    }

    fn backprop_once(&self, x: &Array1<fmod>, y: &Array1<fmod>) -> Gradients {
        let mut outputs: Vec<Array1<fmod>> = vec![x.to_owned()];
        let mut grads: Vec<Array1<fmod>> = Vec::new();

        let weights = &self.weights;
        let activations = &self.activations;
        let biases = &self.biases;

        // forward pass
        {
            for layeridx in 0..self.shape.len() - 1 {
                let out = weights[layeridx].t().dot(&outputs[layeridx]) + &biases[layeridx];
                let currentlayer = out.mapv(activations[layeridx].activation);
                let grad = currentlayer.mapv(activations[layeridx].gradient).to_owned();

                outputs.push(currentlayer);
                grads.push(grad);
            }
        }

        let lastlayeridx = self.shape.len() - 1;

        let mut current_layer_errs = if self.softmaxed {
            let outputs = &outputs[lastlayeridx];
            let outputs_softmaxed = softmax(&outputs);
            let cel_errs = outputs_softmaxed - y;

            &grads[lastlayeridx - 1] * cel_errs
        } else {
            let errs = &outputs[lastlayeridx] - y;
            &grads[lastlayeridx - 1] * errs
        };

        let mut weight_deltas: Vec<Array2<fmod>> = Vec::new();
        let mut bias_deltas: Vec<Array1<fmod>> = Vec::new();

        for layeridx in (0..=lastlayeridx - 1).rev() {
            let errs2d = into_col(current_layer_errs.to_owned());
            let outs2d = into_col(outputs[layeridx].to_owned());
            let weight_deltas_ = outs2d.dot(&errs2d.t());
            weight_deltas.push(weight_deltas_);

            let bias_deltas_ = flatten(errs2d);
            bias_deltas.push(bias_deltas_);

            if layeridx == 0 {
                break;
            }
            current_layer_errs = weights[layeridx].dot(&current_layer_errs) * &grads[layeridx - 1];
        }
        weight_deltas.reverse();
        bias_deltas.reverse();
        Gradients(weight_deltas, bias_deltas)
    }

    fn train_batch(&mut self, batch: &Batch) {
        let batchsize = batch.x.len();
        assert!(batchsize == batch.y.len());

        let layercount = &self.shape.len() - 1;

        // async gradients calculation
        // it's just a map(to gradients) -> reduce(sum) in principle,
        // but the fold makes it a bit faster
        let mut gradients = batch
            .x
            .into_par_iter()
            .zip(batch.y.into_par_iter())
            .fold(
                || None,
                |o_cumgrad, (x, y)| {
                    let grad = self.backprop_once(x, y);
                    match o_cumgrad {
                        None => Some(grad),
                        Some(mut cumgrad) => {
                            for layer in 0..layercount {
                                cumgrad.0[layer] += &grad.0[layer];
                                cumgrad.1[layer] += &grad.1[layer];
                            }
                            Some(cumgrad)
                        }
                    }
                },
            )
            .reduce(
                || None,
                |o_cumgrad, o_subcumgrad| {
                    let subcumgrad = o_subcumgrad.unwrap();
                    match o_cumgrad {
                        None => Some(subcumgrad),
                        Some(mut cumgrad) => {
                            for layer in 0..layercount {
                                cumgrad.0[layer] += &subcumgrad.0[layer];
                                cumgrad.1[layer] += &subcumgrad.1[layer];
                            }
                            Some(cumgrad)
                        }
                    }
                },
            )
            .unwrap();

        for layeridx in 0..layercount {
            gradients.0[layeridx] /= batchsize as fmod;
            gradients.1[layeridx] /= batchsize as fmod;
        }
        // update model
        let (weight_update, bias_update) = self.optimizer.get_updates(&gradients);
        for (layeridx, (wd, bd)) in zip(weight_update, bias_update).enumerate() {
            self.weights[layeridx] -= &wd;
            self.biases[layeridx] -= &bd;
        }
    }

    pub fn train_epoch(&mut self, data: &DataRef, batchsize: usize) {
        assert!(data.x.len() == data.y.len());

        let mut rng: StdRng = SeedableRng::from_seed([1u8; 32]);

        let mut data: Vec<(Array1<fmod>, Array1<fmod>)> =
            zip(data.x.clone(), data.y.clone()).collect();
        data.shuffle(&mut rng);

        let (datax, datay): (Vec<Array1<fmod>>, Vec<Array1<fmod>>) = data.into_iter().unzip();

        if batchsize == 0 {
            self.train_batch(&Batch {
                x: &datax,
                y: &datay,
            });
        } else {
            for (xslice, yslice) in zip(datax.chunks(batchsize), datay.chunks(batchsize)) {
                self.train_batch(&Batch {
                    x: xslice,
                    y: yslice,
                });
            }
        }
    }
}

fn softmax(x: &Array1<fmod>) -> Array1<fmod> {
    let e = fmod::from(std::f64::consts::E);
    let max = x
        .into_iter()
        .max_by(|a, b| a.partial_cmp(b).unwrap())
        .expect("nonempty");

    let exps = (&x).mapv(|v| e.powf(v - max));
    &exps / (&exps).sum()
}
