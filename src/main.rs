#![allow(dead_code)]
use crate::activations::*;
use crate::datastructs::*;
use crate::helpers::*;
use crate::mlpmodel::MultilayerPerceptron;
use crate::optimizers::optimizers::Optimizer;
use ndarray::{Array, Array1, Array2};
use ndarray_linalg::{flatten, into_col};

use prgrs::Prgrs;

mod activations;
mod datastructs;
mod helpers;
mod mlpmodel;
mod optimizers;

#[allow(non_camel_case_types)]
type fmod = f64;

fn main() {
    // easy
    {
        let data_train = helpers::read_data_truefalse("data/easy-training.csv", 0);
        let data_test = helpers::read_data_truefalse("data/easy-test.csv", 0);
        let shape = vec![2, 5, 5, 5, 2];
        let activations = vec![
            NeuronActivation::relu(),
            NeuronActivation::relu(),
            NeuronActivation::relu(),
            NeuronActivation::sigmoid(),
        ];

        let mut mlp =
            MultilayerPerceptron::new(shape, activations, Optimizer::rmsprop(0.001, 0.99), true);

        let n = 5000;
        for _ in Prgrs::new(0..n, n) {
            mlp.train_epoch(&data_train.ref_repr(), 32);
        }
        let yhats = mlp.predict(&data_test.x);
        let f1 = f1(&yhats, &data_test.y, 2);
        println!("f1 - easy: {:?}", f1);
        save_preds(yhats, String::from("easy"));
    }

    // rings3
    {
        let data_train = helpers::read_data_r("data/rings3-regular-training.csv", 0, 3);
        let data_test = helpers::read_data_r("data/rings3-regular-test.csv", 0, 3);
        let shape = vec![
            3, //
            16, 16, //
            16, 16, //
            16, 16, //
            3,  //
        ];
        let activations = vec![
            NeuronActivation::sigmoid(),
            NeuronActivation::relu(),
            NeuronActivation::sigmoid(),
            NeuronActivation::relu(),
            NeuronActivation::sigmoid(),
            NeuronActivation::relu(),
            NeuronActivation::sigmoid(),
        ];

        let mut mlp =
            MultilayerPerceptron::new(shape, activations, Optimizer::rmsprop(0.001, 0.99), true);

        let n = 5000;
        for _ in Prgrs::new(0..n, n) {
            mlp.train_epoch(&data_train.ref_repr(), 64);
        }
        let yhats = mlp.predict(&data_test.x);
        let f1 = f1(&yhats, &data_test.y, 3);
        println!("f1 - rings3: {:?}", f1);
        save_preds(yhats, String::from("rings3"));
    }

    // xor3

    {
        let data_train = helpers::read_data_sins("data/xor3-training.csv", 0, 2);
        let data_test = helpers::read_data_sins("data/xor3-test.csv", 0, 2);
        let shape = vec![
            2, //
            16, 16, //
            16, 16, //
            16, 16, //
            2,  //
        ];
        let activations = vec![
            NeuronActivation::sigmoid(),
            NeuronActivation::relu(),
            NeuronActivation::sigmoid(),
            NeuronActivation::relu(),
            NeuronActivation::sigmoid(),
            NeuronActivation::relu(),
            NeuronActivation::sigmoid(),
        ];

        let mut mlp =
            MultilayerPerceptron::new(shape, activations, Optimizer::rmsprop(0.0001, 0.99), true);

        let n = 5000;
        for _ in Prgrs::new(0..n, n) {
            mlp.train_epoch(&data_train.ref_repr(), 64);
        }
        let yhats = mlp.predict(&data_test.x);
        let f1 = f1(&yhats, &data_test.y, 2);
        println!("f1 - xor3: {:?}", f1);
        save_preds(yhats, String::from("xor3"));
    }
}

fn save_preds(preds: Vec<Array1<fmod>>, name: String) {
    use std::fs;
    let predictions: Vec<usize> = preds
        .into_iter()
        .map(|yh| {
            yh.iter()
                .enumerate()
                .fold((0, 0.0), |(idx_max, val_max), (idx, val)| {
                    if &val_max > val {
                        (idx_max, val_max)
                    } else {
                        (idx, *val)
                    }
                })
                .0
        })
        .collect();
    let mut out = String::new();
    for c in predictions {
        let data = c.to_string();
        out = out + "\n" + &data;
    }
    out = String::from("'c'\n") + &out;
    fs::write(name, out).expect("ok");
}

// OUTPUTS:
// [###########] (100%)
// [0.7310585786300049, 0.2689414213699951]
// f1 - easy: [0.994012, 0.993988]
// [###########] (100%)
// [0.21194155761663241, 0.21194155761877007, 0.5761168847645975]
// f1 - rings3: [0.95035464, 0.94304633, 0.94160587]
// [###########] (100%)
// [0.7309826937861038, 0.26901730621389613]
// f1 - xor3: [0.9685315, 0.9579439]

// WRITES FILES:
// xor3
// rings3
// easy
