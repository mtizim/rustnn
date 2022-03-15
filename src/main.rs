use std::iter::zip;

use itertools::izip;
use ndarray::{arr1, Array, Array1, Array2};
use ndarray_linalg::{flatten, into_col};
use num::Float;
use rayon::prelude::*;

use ndarray_rand::rand_distr::StandardNormal;
use ndarray_rand::RandomExt;

type Activation = fn(f64) -> f64;
type Gradient = fn(f64) -> f64;

pub struct MultilayerPerceptron {
    pub weights: Vec<Array2<f64>>,
    pub biases: Vec<Array1<f64>>,
    pub shape: Vec<u16>,
    pub activations: Vec<NeuronActivation>,
    pub eps: f64,
}

pub struct NeuronActivation {
    pub activation: Activation,
    pub gradient: Gradient,
}
//  TODO split into files:
#[inline(always)]
fn sigmoid<F: Float>(f: F) -> F {
    use std::f64::consts::E;
    let e = F::from(E).unwrap();
    F::one() / (F::one() + e.powf(-f))
}
#[inline(always)]
fn sigmoid_grad<F: Float>(f: F) -> F {
    f * (F::one() - f)
}
#[inline(always)]
fn linear<F: Float>(f: F) -> F {
    f
}
#[inline(always)]
fn linear_grad<F: Float>(_f: F) -> F {
    F::one()
}

impl NeuronActivation {
    pub fn sigmoid() -> NeuronActivation {
        NeuronActivation {
            activation: sigmoid,
            gradient: sigmoid_grad,
        }
    }
    pub fn linear() -> NeuronActivation {
        NeuronActivation {
            activation: linear,
            gradient: linear_grad,
        }
    }
}

struct MLPGradientUpdate(Vec<Array2<f64>>, Vec<Array1<f64>>);
struct Batch {
    x: Vec<Array1<f64>>,
    y: Vec<Array1<f64>>,
}

impl MultilayerPerceptron {
    pub fn new(shape: Vec<u16>, activations: Vec<NeuronActivation>) -> MultilayerPerceptron {
        let mut weights: Vec<Array2<f64>> = Vec::new();
        let mut biases: Vec<Array1<f64>> = Vec::new();
        for layeridx in 1..shape.len() {
            let dimfrom = shape[layeridx - 1] as usize;
            let dimto = shape[layeridx] as usize;
            weights.push(Array::random((dimfrom, dimto), StandardNormal));
            biases.push(Array::random(dimto, StandardNormal));
        }
        assert!(activations.len() == shape.len() - 1);

        MultilayerPerceptron {
            weights,
            biases,
            shape,
            activations,
            eps: 0.01,
        }
    }

    pub fn predict(&self, x: Array1<f64>) -> Array1<f64> {
        let mut x_layer: Array1<f64> = x;

        for (w, b, activation) in izip!(&self.weights, &self.biases, &self.activations) {
            x_layer = (w.t().dot(&x_layer) + b).mapv(activation.activation);
        }
        x_layer
    }

    fn backprop_once(&self, x: Array1<f64>, y: Array1<f64>) -> MLPGradientUpdate {
        let mut outputs: Vec<Array1<f64>> = vec![x];
        let mut grads: Vec<Array1<f64>> = Vec::new();

        let weights = &self.weights;
        let activations = &self.activations;
        let biases = &self.biases;

        {
            for layeridx in 0..self.shape.len() - 1 {
                let previouslayer = &outputs[layeridx];
                let out = weights[layeridx].t().dot(previouslayer) + &biases[layeridx];

                let currentlayer = out.mapv(activations[layeridx].activation);
                let grad = currentlayer.mapv(activations[layeridx].gradient);

                outputs.push(currentlayer);
                grads.push(grad);
            }
        }

        let errs = &outputs[self.shape.len() - 1] - y;
        let mut current_layer_errs: Array1<f64> = &grads[self.shape.len() - 2] * errs;

        let mut weight_deltas: Vec<Array2<f64>> = Vec::new();
        let mut bias_deltas: Vec<Array1<f64>> = Vec::new();

        for layeridx in (0..=self.shape.len() - 2).rev() {
            let errs2d = into_col(current_layer_errs.to_owned());
            let outs2d = into_col(outputs[layeridx].to_owned());
            let weight_deltas_ = outs2d.dot(&errs2d.t());
            println!("{layeridx}\nerrs {errs2d} \n outs\n {outs2d} \n dp\n ");
            dbg!(&weight_deltas);
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

    fn train_batch(&mut self, batch: Batch) {
        let batchsize = batch.x.len();
        assert!(batchsize == batch.y.len());

        let mut updates = Vec::new();
        batch
            .x
            .into_par_iter()
            .zip(batch.y.into_par_iter())
            .map(|(x, y)| self.backprop_once(x, y)) //todo use fold here
            .collect_into_vec(&mut updates);

        // update model
        let eps = self.eps / (batchsize as f64);
        for update in updates {
            for (layeridx, (wd, bd)) in zip(update.0.into_iter(), update.1.into_iter()).enumerate()
            {
                self.weights[layeridx] -= &(eps * wd);
                self.biases[layeridx] -= &(eps * bd);
            }
        }
    }
}

fn read_data<P: AsRef<std::path::Path>>(path: P) -> (Array1<f64>, Array1<f64>) {
    // TODO return a Data struct
    let mut reader = csv::Reader::from_path(path).expect("Path existence");

    let mut xs: Vec<f64> = Vec::new();
    let mut ys: Vec<f64> = Vec::new();

    for result in reader.records() {
        let record = result.expect("a CSV record");

        let cx = (&record[1]).parse().expect("Formatting");
        let cy = (&record[2]).parse().expect("Formatting");
        xs.push(cx);
        ys.push(cy);
    }
    (arr1(&xs), arr1(&ys))
}

fn mse(ypred: Array1<f64>, y: Array1<f64>) -> f64 {
    let count = ypred.len() as f64;
    (ypred - y).mapv(|e| e.powi(2)).sum() / count
}

fn main() {
    let (xs, ys) = read_data("square-simple-test.csv");
    let mut bx = Vec::new();
    let mut by = Vec::new();
    for (xs, ys) in zip(xs, ys) {
        bx.push(arr1(&[xs]));
        by.push(arr1(&[ys]));
        break;
    }
    {
        let shape = vec![1, 5, 1];
        let activations = vec![
            NeuronActivation::sigmoid(),
            NeuronActivation::linear(),
            //
        ];
        let mut mlp = MultilayerPerceptron::new(shape, activations);

        for _ in 0..1 {
            mlp.train_batch(Batch {
                x: bx.clone(),
                y: by.clone(),
            });
        }
        let first = mlp.predict(arr1(&[1.22778856509831]))[0];

        let w = mlp.weights[0].to_owned();
        let b = mlp.biases[0].to_owned();

        println!("Got {first}");
        println!("Expected 5.6718284527546");
    }
}
