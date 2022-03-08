use itertools::izip;
use ndarray::{arr1, arr2, Array1, Array2};
use num::Float;

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
fn linear_grad<F: Float>(f: F) -> F {
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
    pub fn new(
        shape: Vec<u16>,
        weights: Vec<Array2<f64>>,
        biases: Vec<Array1<f64>>,
        activations: Vec<NeuronActivation>,
    ) -> MultilayerPerceptron {
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
            eps: 0.001,
        }
    }

    pub fn predict(&self, x: Array1<f64>) -> Array1<f64> {
        let mut x_layer: Array1<f64> = x;

        for (w, b, activation) in izip!(&self.weights, &self.biases, &self.activations) {
            x_layer = (w.dot(&x_layer) + b).mapv(activation.activation);
        }
        x_layer
    }

    pub fn train(&self, x: Vec<Array1<f64>>, y: Vec<Array1<f64>>) -> Result<(), > {
        let first = x[0].to_owned();
        // TODO batch this whole thing (add weight deltas together)

        // forward propagation
        let mut outputs: Vec<Array1<f64>> = vec![first];
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

        // TODO calculate errors
        let last_layer_errs = arr1(&[1.0, 2.0, 3.0]);

        // backprop
        {
            let mut current_layer_errs: Array1<f64> = last_layer_errs.to_owned();

            let mut weight_deltas: Vec<Array2<f64>>;
            let mut bias_deltas: Vec<Array1<f64>>;

            for layeridx in self.shape.len() - 1..0 {
                // errs =  weights @ curr layer errs * grads
                let  weight_deltas_ = -eps * (grads[layeridx].to_owned().into_dimensionality()?).dot(&current_layer_errs);
                weight_deltas.push(weight_deltas_);
                // -eps * current layer grad * prev layer errs
                let bias_deltas_ = -eps * current_layer_errs;
                bias_deltas.push(bias_deltas_);

                current_layer_errs = weights[layeridx].dot(&current_layer_errs) * (grads[layeridx].to_owned());

            }
        }

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
    let sigmoid = NeuronActivation::sigmoid();
    let linear = NeuronActivation::linear();
    {
        let (mut xs, ys) = read_data("steps-large-test.csv");

        let shape = vec![1, 5, 1];
        let weights = vec![
            arr2(&[[300.0], [300.0], [300.0], [0.0], [0.0]]),
            arr2(&[[80.0, 80.0, 80.0, 0.0, 0.0]]),
        ];
        let biases = vec![arr1(&[150.0, -150.0, -450.0, 0.0, 0.0]), arr1(&[-80.0])];
        let activations = vec![sigmoid, linear];

        let mlp = MultilayerPerceptron::new(shape, weights, biases, activations);
        xs.par_mapv_inplace(|x| mlp.predict(arr1(&[x]))[0]);

        let mse = mse(xs, ys);

        println!("MSE for steps-large: {mse}"); // MSE for steps-large: 7.029725723660864
    }
    let sigmoid = NeuronActivation::sigmoid();
    let linear = NeuronActivation::linear();
    {
        let (mut xs, ys) = read_data("square-simple-test.csv");

        let shape = vec![1, 5, 1];
        let weights = vec![
            arr2(&[[1.7], [0.0], [0.0], [0.0], [1.7]]),
            arr2(&[[-663.0, 0.0, 0.0, 0.0, 700.0]]),
        ];
        let biases = vec![arr1(&[3.0, 0.0, 0.0, 0.0, -3.1]), arr1(&[475.0])];
        let activations = vec![sigmoid, linear];

        let mlp = MultilayerPerceptron::new(shape, weights, biases, activations);
        xs.par_mapv_inplace(|x| mlp.predict(arr1(&[x]))[0]);

        let mse = mse(xs, ys);
        println!("MSE for square-simple: {mse}") // MSE for square-simple: 8.516778537293016
    }
}
