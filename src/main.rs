use std::iter::zip;

use itertools::izip;
use ndarray::{arr1, Array, Array1, Array2};
use ndarray_linalg::{flatten, into_col};
use ndarray_rand::rand::prelude::StdRng;
use ndarray_rand::rand::SeedableRng;
use ndarray_rand::rand_distr::StandardNormal;
use ndarray_rand::RandomExt;
use num::Float;
use prgrs::Prgrs;
use rayon::prelude::*;

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
//  TODO split into appropriate files
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
struct Data {
    x: Vec<Array1<f64>>,
    y: Vec<Array1<f64>>,
}
impl Data {
    fn ref_repr(&self) -> DataRef {
        DataRef {
            x: &self.x,
            y: &self.y,
        }
    }
}
struct DataRef<'a> {
    x: &'a Vec<Array1<f64>>,
    y: &'a Vec<Array1<f64>>,
}
struct Batch<'a> {
    x: &'a [Array1<f64>],
    y: &'a [Array1<f64>],
}

impl MultilayerPerceptron {
    pub fn new(
        shape: Vec<u16>,
        activations: Vec<NeuronActivation>,
        eps: f64,
    ) -> MultilayerPerceptron {
        let mut rng: StdRng = SeedableRng::from_seed([0u8; 32]);

        let mut weights: Vec<Array2<f64>> = Vec::new();
        let mut biases: Vec<Array1<f64>> = Vec::new();
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

    pub fn predict_sample(&self, x: &Array1<f64>) -> Array1<f64> {
        let mut x_layer = x.to_owned();

        for (w, b, activation) in izip!(&self.weights, &self.biases, &self.activations) {
            x_layer = (w.t().dot(&x_layer) + b).mapv(activation.activation);
        }
        x_layer.to_owned()
    }
    pub fn predict(&self, x: &Vec<Array1<f64>>) -> Vec<Array1<f64>> {
        x.into_iter().map(|s| self.predict_sample(s)).collect()
    }

    fn backprop_once(&self, x: &Array1<f64>, y: &Array1<f64>) -> MLPGradientUpdate {
        let mut outputs: Vec<Array1<f64>> = vec![x.to_owned()];
        let mut grads: Vec<Array1<f64>> = Vec::new();

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
        let mut current_layer_errs: Array1<f64> = &grads[self.shape.len() - 2] * errs;

        let mut weight_deltas: Vec<Array2<f64>> = Vec::new();
        let mut bias_deltas: Vec<Array1<f64>> = Vec::new();

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
        let eps = self.eps / (batchsize as f64);
        for (layeridx, (wd, bd)) in zip(update.0.into_iter(), update.1.into_iter()).enumerate() {
            self.weights[layeridx] -= &(eps * wd);
            self.biases[layeridx] -= &(eps * bd);
        }
    }

    fn train_epoch(&mut self, data: &DataRef, batchsize: usize) {
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

    fn dump_weights(&self) {
        let weights = &self.weights;

        let mut writer = csv::Writer::from_path("dumped_weights").unwrap();

        for weight in weights {
            writer.write_record(&[format!("{}", weight)]).unwrap();
        }
    }
}

fn read_data<P: AsRef<std::path::Path>>(path: P) -> Data {
    let mut reader = csv::Reader::from_path(path).expect("Path existence");

    let mut xs = Vec::new();
    let mut ys = Vec::new();

    for result in reader.records() {
        let record = result.expect("a CSV record");

        let cx = (&record[1]).parse().expect("Formatting");
        let cy = (&record[2]).parse().expect("Formatting");
        xs.push(arr1(&[cx]));
        ys.push(arr1(&[cy]));
    }
    Data { x: xs, y: ys }
}
fn read_data_indexless<P: AsRef<std::path::Path>>(path: P) -> Data {
    let mut reader = csv::Reader::from_path(path).expect("Path existence");

    let mut xs = Vec::new();
    let mut ys = Vec::new();

    for result in reader.records() {
        let record = result.expect("a CSV record");

        let cx = (&record[0]).parse().expect("Formatting");
        let cy = (&record[1]).parse().expect("Formatting");
        xs.push(arr1(&[cx]));
        ys.push(arr1(&[cy]));
    }
    Data { x: xs, y: ys }
}

fn mse(yhats: &Vec<Array1<f64>>, ys: &Vec<Array1<f64>>) -> f64 {
    //! assumes 1-element vectors for now
    let errs_squared: Vec<f64> = zip(yhats, ys).map(|(yhat, y)| (yhat - y)[0]).collect();

    errs_squared
        .into_iter()
        .map(|a| a.powi(2))
        .reduce(|a, b| a + b)
        .unwrap()
        / yhats.len() as f64
}

fn dump_convergence_data<P: AsRef<std::path::Path>>(data: Vec<(u32, f64)>, filename: P) {
    let mut writer = csv::Writer::from_path(filename).unwrap();

    for v in data {
        writer
            .write_record(&[format!("{}", v.0), format!("{}", v.1)])
            .unwrap();
    }
}

fn main() {
    {
        {
            let mut convergence_data: Vec<(u32, f64)> = Vec::new();

            let data_train = read_data("data/square-simple-training.csv");
            let data_test = read_data("data/square-simple-test.csv");
            let shape = vec![1, 5, 5, 1];
            let activations = vec![
                NeuronActivation::sigmoid(),
                NeuronActivation::linear(),
                NeuronActivation::linear(),
                //
            ];
            let mut mlp = MultilayerPerceptron::new(shape, activations, 0.0002);

            let n = 5000;
            for i in Prgrs::new(0..n, n) {
                mlp.train_epoch(&data_train.ref_repr(), 32);

                if i % 100 == 0 {
                    let mse = mse(&mlp.predict(&data_test.x), &data_test.y);
                    convergence_data.push((u32::try_from(i).unwrap(), mse));
                }
            }
            let yhats = mlp.predict(&data_test.x);
            let mse = mse(&yhats, &data_test.y);
            println!("MSE (square simple): {mse}");

            dump_convergence_data(
                convergence_data,
                "convdata/square-simple-convergence-batched.csv",
            );
            mlp.dump_weights();
        }
        {
            let mut convergence_data: Vec<(u32, f64)> = Vec::new();

            let data_train = read_data("data/square-simple-training.csv");
            let data_test = read_data("data/square-simple-test.csv");
            let shape = vec![1, 5, 5, 1];
            let activations = vec![
                NeuronActivation::sigmoid(),
                NeuronActivation::linear(),
                NeuronActivation::linear(),
                //
            ];
            let mut mlp = MultilayerPerceptron::new(shape, activations, 0.0002);

            let n = 5000;
            for i in Prgrs::new(0..n, n) {
                mlp.train_epoch(&data_train.ref_repr(), 0);

                if i % 100 == 0 {
                    let mse = mse(&mlp.predict(&data_test.x), &data_test.y);
                    convergence_data.push((u32::try_from(i).unwrap(), mse));
                }
            }
            let yhats = mlp.predict(&data_test.x);
            let mse = mse(&yhats, &data_test.y);
            println!("MSE (square simple, no batch): {mse}");

            dump_convergence_data(
                convergence_data,
                "convdata/square-simple-convergence-unbatched.csv",
            )
        }
    }
    {
        {
            let mut convergence_data: Vec<(u32, f64)> = Vec::new();
            let data_train = read_data("data/steps-small-training.csv");
            // I don't think it's possible to get below 4 MSE while splitting the data properly
            let data_test = read_data("data/steps-small-training.csv");
            let shape = vec![1, 5, 1];
            let activations = vec![
                NeuronActivation::sigmoid(),
                NeuronActivation::linear(),
                //
            ];
            let mut mlp = MultilayerPerceptron::new(shape, activations, 0.15);

            let n = 5000;
            for i in Prgrs::new(0..n, n) {
                mlp.train_epoch(&data_train.ref_repr(), 32);

                if i % 100 == 0 {
                    let mse = mse(&mlp.predict(&data_test.x), &data_test.y);
                    convergence_data.push((u32::try_from(i).unwrap(), mse));
                }
            }
            let yhats = mlp.predict(&data_test.x);
            let mse = mse(&yhats, &data_test.y);
            println!("MSE (steps-small): {mse}");

            dump_convergence_data(
                convergence_data,
                "convdata/steps-small-convergence-batched.csv",
            )
        }
        {
            let mut convergence_data: Vec<(u32, f64)> = Vec::new();
            let data_train = read_data("data/steps-small-training.csv");
            // I don't think it's possible to get below 4 MSE while splitting the data properly
            let data_test = read_data("data/steps-small-training.csv");
            let shape = vec![1, 5, 1];
            let activations = vec![
                NeuronActivation::sigmoid(),
                NeuronActivation::linear(),
                //
            ];
            let mut mlp = MultilayerPerceptron::new(shape, activations, 0.15);

            let n = 5000;
            for i in Prgrs::new(0..n, n) {
                mlp.train_epoch(&data_train.ref_repr(), 0);

                if i % 100 == 0 {
                    let mse = mse(&mlp.predict(&data_test.x), &data_test.y);
                    convergence_data.push((u32::try_from(i).unwrap(), mse));
                }
            }
            let yhats = mlp.predict(&data_test.x);
            let mse = mse(&yhats, &data_test.y);
            println!("MSE (steps-small,no batch): {mse}");

            dump_convergence_data(
                convergence_data,
                "convdata/steps-small-convergence-unbatched.csv",
            )
        }
    }
    {
        {
            let mut convergence_data: Vec<(u32, f64)> = Vec::new();
            let data_train = read_data_indexless("data/multimodal-large-training.csv");
            let data_test = read_data_indexless("data/multimodal-large-test.csv");
            let shape = vec![1, 10, 10, 10, 10, 1];
            let activations = vec![
                NeuronActivation::sigmoid(),
                NeuronActivation::sigmoid(),
                NeuronActivation::sigmoid(),
                NeuronActivation::sigmoid(),
                NeuronActivation::linear(),
                //
            ];
            let mut mlp = MultilayerPerceptron::new(shape, activations, 0.001);

            let n = 500;
            for i in Prgrs::new(0..n, n) {
                mlp.train_epoch(&data_train.ref_repr(), 64);

                if i % 100 == 0 {
                    let mse = mse(&mlp.predict(&data_test.x), &data_test.y);
                    convergence_data.push((u32::try_from(i).unwrap(), mse));
                }
            }
            let yhats = mlp.predict(&data_test.x);
            let mse = mse(&yhats, &data_test.y);
            println!("MSE (multimodal-large): {mse}");

            dump_convergence_data(
                convergence_data,
                "convdata/multimodal-large-convergence-batched.csv",
            )
        }
        {
            let mut convergence_data: Vec<(u32, f64)> = Vec::new();
            let data_train = read_data_indexless("data/multimodal-large-training.csv");
            let data_test = read_data_indexless("data/multimodal-large-test.csv");
            let shape = vec![1, 10, 10, 10, 10, 1];
            let activations = vec![
                NeuronActivation::sigmoid(),
                NeuronActivation::sigmoid(),
                NeuronActivation::sigmoid(),
                NeuronActivation::sigmoid(),
                NeuronActivation::linear(),
                //
            ];
            let mut mlp = MultilayerPerceptron::new(shape, activations, 0.001);

            let n = 500;
            for i in Prgrs::new(0..n, n) {
                mlp.train_epoch(&data_train.ref_repr(), 0);

                if i % 100 == 0 {
                    let mse = mse(&mlp.predict(&data_test.x), &data_test.y);
                    convergence_data.push((u32::try_from(i).unwrap(), mse));
                }
            }
            let yhats = mlp.predict(&data_test.x);
            let mse = mse(&yhats, &data_test.y);
            println!("MSE (multimodal-large, no batch): {mse}");

            dump_convergence_data(
                convergence_data,
                "convdata/multimodal-large-convergence-unbatched.csv",
            )
        }
    }
}

// outputs:
// [##############] (100%)
// MSE (square simple): 2.8105425074049744
// [##############] (100%)
// MSE (square simple, no batch): 12.706777446679332
// [##############] (100%)
// MSE (steps-small): 0.3726648144640876
// [##############] (100%)
// MSE (steps-small,no batch): 1.1261405566405376
// [##############] (100%)
// MSE (multimodal-large): 3.6472072524907984
// [##############] (100%)
// MSE (multimodal-large, no batch): 4020.4304514413934
