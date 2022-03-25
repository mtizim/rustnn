use crate::*;
use std::iter::zip;

use itertools::izip;
use ndarray_rand::rand::prelude::StdRng;
use ndarray_rand::rand::SeedableRng;
use ndarray_rand::rand_distr::StandardNormal;
use ndarray_rand::RandomExt;
use rayon::prelude::*;

pub struct MLPGradientUpdate(
    pub Vec<Array2<fmod>>,
    pub Vec<Array1<fmod>>, //
);
pub struct MultilayerPerceptron {
    pub weights: Vec<Array2<fmod>>,
    pub biases: Vec<Array1<fmod>>,
    pub shape: Vec<u16>,
    pub activations: Vec<NeuronActivation>,
    pub eps: fmod,
}

impl MultilayerPerceptron {
    pub fn new(
        shape: Vec<u16>,
        activations: Vec<NeuronActivation>,
        eps: fmod,
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
            eps: eps, //TODO throw eps adjusting strategies into the constructor
        }
    }

    pub fn predict_sample(&self, x: &Array1<fmod>) -> Array1<fmod> {
        let mut x_layer = x.to_owned();

        for (w, b, activation) in izip!(&self.weights, &self.biases, &self.activations) {
            x_layer = (w.t().dot(&x_layer) + b).mapv(activation.activation);
        }
        x_layer.to_owned()
    }
    pub fn predict(&self, x: &Vec<Array1<fmod>>) -> Vec<Array1<fmod>> {
        x.into_iter().map(|s| self.predict_sample(s)).collect()
    }

    fn backprop_once(&self, x: &Array1<fmod>, y: &Array1<fmod>) -> MLPGradientUpdate {
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

        let errs = &outputs[self.shape.len() - 1] - y;
        let mut current_layer_errs: Array1<fmod> = &grads[self.shape.len() - 2] * errs;

        let mut weight_deltas: Vec<Array2<fmod>> = Vec::new();
        let mut bias_deltas: Vec<Array1<fmod>> = Vec::new();

        for layeridx in (0..=self.shape.len() - 2).rev() {
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
        MLPGradientUpdate(weight_deltas, bias_deltas)
    }

    fn train_batch(&mut self, batch: &Batch) {
        // todo shuffle batch
        let batchsize = batch.x.len();
        assert!(batchsize == batch.y.len());

        let layercount = &self.shape.len() - 1;

        // async update calculation
        // it's just a map(to updates) -> reduce(sum) in principle,
        // but the fold makes it a bit faster
        let update = batch
            .x
            .into_par_iter()
            .zip(batch.y.into_par_iter())
            .fold(
                || None,
                |o_cumupdate, (x, y)| {
                    let update = self.backprop_once(x, y);
                    match o_cumupdate {
                        None => Some(update),
                        Some(mut cumupdate) => {
                            for layer in 0..layercount {
                                cumupdate.0[layer] += &update.0[layer];
                                cumupdate.1[layer] += &update.1[layer];
                            }
                            Some(cumupdate)
                        }
                    }
                },
            )
            .reduce(
                || None,
                |o_cumupdate, o_subcumupdate| {
                    let subcumupdate = o_subcumupdate.unwrap();
                    match o_cumupdate {
                        None => Some(subcumupdate),
                        Some(mut cumupdate) => {
                            for layer in 0..layercount {
                                cumupdate.0[layer] += &subcumupdate.0[layer];
                                cumupdate.1[layer] += &subcumupdate.1[layer];
                            }
                            Some(cumupdate)
                        }
                    }
                },
            )
            .unwrap();

        // update model
        let eps = self.eps / (batchsize as fmod);
        for (layeridx, (wd, bd)) in zip(update.0.into_iter(), update.1.into_iter()).enumerate() {
            self.weights[layeridx] -= &(eps * wd);
            self.biases[layeridx] -= &(eps * bd);
        }
    }

    pub fn train_epoch(&mut self, data: &DataRef, batchsize: usize) {
        assert!(data.x.len() == data.y.len());
        if batchsize == 0 {
            self.train_batch(&Batch {
                x: data.x,
                y: data.y,
            });
        } else {
            for (xslice, yslice) in zip(data.x.chunks(batchsize), data.y.chunks(batchsize)) {
                self.train_batch(&Batch {
                    x: xslice,
                    y: yslice,
                });
            }
        }
    }
}
