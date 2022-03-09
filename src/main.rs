use itertools::izip;
use ndarray::{arr1, Array, Array1, Array2, ShapeError};
use ndarray_linalg::{into_col, into_row};
use num::Float;

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

impl MultilayerPerceptron {
    pub fn new(shape: Vec<u16>, activations: Vec<NeuronActivation>) -> MultilayerPerceptron {
        let mut weights: Vec<Array2<f64>> = Vec::new();
        let mut biases: Vec<Array1<f64>> = Vec::new();
        for layeridx in 1..shape.len() {
            let dimfrom = shape[layeridx - 1] as usize;
            let dimto = shape[layeridx] as usize;
            weights.push(Array::random((dimto, dimfrom), StandardNormal));
            biases.push(Array::random(dimto, StandardNormal));
        }
        assert!(
            weights.len() == activations.len()
                && activations.len() == biases.len()
                && biases.len() == shape.len() - 1
        );
        for layeridx in 0..weights.len() {
            assert!(
                weights[layeridx].shape()
                    == &[shape[layeridx + 1] as usize, shape[layeridx] as usize]
            )
        }
        for layeridx in 0..weights.len() {
            assert!(
                weights[layeridx].shape()
                    == &[shape[layeridx + 1] as usize, shape[layeridx] as usize]
            )
        }

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
            x_layer = (w.dot(&x_layer) + b).mapv(activation.activation);
        }
        x_layer
    }

    pub fn train(&mut self, x: Vec<Array1<f64>>, y: Vec<Array1<f64>>) -> Result<(), ShapeError> {
        let first = x[0].to_owned();
        let firsty = y[0].to_owned();
        // TODO batch this whole thing (add weight deltas together)

        // forward propagation
        let mut outputs: Vec<Array1<f64>> = vec![first.to_owned()];
        let mut grads: Vec<Array1<f64>> = Vec::new();

        let weights = &self.weights;
        let biases = &self.biases;
        let activations = &self.activations;
        let eps = &self.eps;

        {
            for layeridx in 0..self.shape.len() - 1 {
                let previouslayer = &outputs[layeridx];
                let out = weights[layeridx].dot(previouslayer) + biases[layeridx].to_owned();

                let currentlayer = out.mapv(activations[layeridx].activation);
                let grad = currentlayer.mapv(activations[layeridx].gradient);

                outputs.push(currentlayer);
                grads.push(grad);
            }
        }

        let errs = outputs[self.shape.len()-1].to_owned() - firsty;
        let mut current_layer_errs: Array1<f64> = errs * (grads[self.shape.len() - 2].to_owned());

        // backprop
        let (weight_deltas, bias_deltas) = {
            let mut weight_deltas: Vec<Array2<f64>> = Vec::new();
            let mut bias_deltas: Vec<Array1<f64>> = Vec::new();

            for layeridx in (0..=self.shape.len() - 2).rev() {
                let errs2d = into_col(current_layer_errs.to_owned());
                let outs2d = into_col(outputs[layeridx].to_owned());



                let weight_deltas_ = -eps * (errs2d).dot(&outs2d.t()).to_owned();

                weight_deltas.push(weight_deltas_);
                let bias_deltas_ = -eps * current_layer_errs.to_owned();
                bias_deltas.push(bias_deltas_);

                if layeridx == 0 {
                    break;
                }
                current_layer_errs =
                    current_layer_errs.dot(&weights[layeridx]) * (grads[layeridx - 1].to_owned());
            }
            weight_deltas.reverse();
            bias_deltas.reverse();
            (weight_deltas, bias_deltas)
        };

        // update model
        for layeridx in 0..self.shape.len() - 1 {
            self.weights[layeridx] =
                self.weights[layeridx].to_owned() + weight_deltas[layeridx].to_owned();
            self.biases[layeridx] =
                self.biases[layeridx].to_owned() + bias_deltas[layeridx].to_owned();
        }
        // println!("update");

        Ok(())
    }
}

fn read_data<P: AsRef<std::path::Path>>(path: P) -> (Array1<f64>, Array1<f64>) {
    let mut reader = csv::Reader::from_path(path).expect("Path existence");

    let mut xs: Vec<f64> = Vec::new();
    let mut ys: Vec<f64> = Vec::new();

    for result in reader.records() {
        let record = result.expect("a CSV record"); // this panics

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
    {
        let shape = vec![1, 2, 3, 1];
        let activations = vec![
            NeuronActivation::linear(),
            NeuronActivation::linear(),
            NeuronActivation::linear(),
        ];
        let mut mlp = MultilayerPerceptron::new(shape, activations);

        for _ in 0..1 {
            mlp.train(vec![arr1(&[1.])], vec![arr1(&[2.])])
                .expect("Pls dont fail");
            mlp.train(vec![arr1(&[20.])], vec![arr1(&[40.])])
                .expect("Pls dont fail");
            mlp.train(vec![arr1(&[0.1])], vec![arr1(&[0.2])])
                .expect("Pls dont fail");
        }
        let first = mlp.predict(arr1(&[3.]))[0];
    }
    {
        let shape = vec![2, 1];
        let activations = vec![
            // NeuronActivation::linear(),
            NeuronActivation::linear(),
        ];
        let mut mlp = MultilayerPerceptron::new(shape, activations);

        for _ in 0..5 {
            mlp.train(vec![arr1(&[100.,120.])], vec![arr1(&[220.])])
                .expect("Pls dont fail");
            mlp.train(vec![arr1(&[10.,50.])], vec![arr1(&[60.])])
                .expect("Pls dont fail");
            mlp.train(vec![arr1(&[-2.,4.])], vec![arr1(&[2.])])
                .expect("Pls dont fail");
            mlp.train(vec![arr1(&[-100.,-100.])], vec![arr1(&[-200.])])
                .expect("Pls dont fail");

        }
        let first = mlp.predict(arr1(&[50.,10.]));

        let w = mlp.weights[0].to_owned();
        let b = mlp.biases[0].to_owned();

        println!("Got {first}, w: {w},b: {b}");
    }
    // {
    //     let (mut xs, ys) = read_data("steps-large-test.csv");

    //     let shape = vec![1, 5, 1];

    //     let activations = vec![sigmoid, linear];

    //     let mut mlp = MultilayerPerceptron::new(shape, activations);
    //     xs.par_mapv_inplace(|x| mlp.predict(arr1(&[x]))[0]);

    //     let mse = mse(xs, ys);

    //     println!("MSE for random init: {mse}"); // MSE for steps-large: 7.029725723660864
    // }
}
