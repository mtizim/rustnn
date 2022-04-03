use crate::datastructs::Data;
use crate::fmod;
use core::iter::zip;
use ndarray::arr1;
use ndarray::Array;
use ndarray::Array1;

pub fn read_data_r<P: AsRef<std::path::Path>>(path: P, startidx: usize, n_classes: usize) -> Data {
    let mut reader = csv::Reader::from_path(path).expect("Path existence");

    let mut xs = Vec::new();
    let mut ys = Vec::new();

    for result in reader.records() {
        let record = result.expect("a CSV record");

        let cx: fmod = (&record[startidx]).parse().expect("Formatting");
        let cy: fmod = (&record[startidx + 1]).parse().expect("Formatting");
        let mut class: Array1<fmod> = Array::zeros(n_classes);
        let classidx: usize = (&record[startidx + 2]).parse().expect("Formatting");
        class[classidx] = 1.0;

        // aug
        let r = (cx.powi(2) + cy.powi(2)).sqrt();

        xs.push(arr1(&[cx, cy, r]));
        ys.push(class);
    }
    Data { x: xs, y: ys }
}
pub fn read_data_sins<P: AsRef<std::path::Path>>(
    path: P,
    startidx: usize,
    n_classes: usize,
) -> Data {
    let mut reader = csv::Reader::from_path(path).expect("Path existence");

    let mut xs = Vec::new();
    let mut ys = Vec::new();

    for result in reader.records() {
        let record = result.expect("a CSV record");

        let cx: fmod = (&record[startidx]).parse().expect("Formatting");
        let cy: fmod = (&record[startidx + 1]).parse().expect("Formatting");
        let mut class: Array1<fmod> = Array::zeros(n_classes);
        let classidx: usize = (&record[startidx + 2]).parse().expect("Formatting");
        class[classidx] = 1.0;

        // aug
        let sx = (cx / 50.0).cos();
        let sy = (cy / 50.0).cos();

        xs.push(arr1(&[sx, sy]));
        ys.push(class);
    }
    Data { x: xs, y: ys }
}
pub fn read_data_truefalse<P: AsRef<std::path::Path>>(path: P, startidx: usize) -> Data {
    let mut reader = csv::Reader::from_path(path).expect("Path existence");

    let mut xs = Vec::new();
    let mut ys = Vec::new();

    for result in reader.records() {
        let record = result.expect("a CSV record");

        let cx = (&record[startidx]).parse().expect("Formatting");
        let cy = (&record[startidx + 1]).parse().expect("Formatting");
        let class = if (&record[startidx + 2]) == "TRUE" {
            arr1(&[1.0, 0.0])
        } else {
            arr1(&[0.0, 1.0])
        };
        xs.push(arr1(&[cx, cy]));
        ys.push(class);
    }
    Data { x: xs, y: ys }
}

pub fn mse(yhats: &Vec<Array1<fmod>>, ys: &Vec<Array1<fmod>>) -> fmod {
    //! assumes 1-element vectors for now

    zip(yhats, ys)
        .map(|(yhat, y)| (yhat - y)[0])
        .map(|a| a.powi(2))
        .reduce(|a, b| a + b)
        .unwrap()
        / yhats.len() as fmod
}
pub fn f1(yhats: &Vec<Array1<fmod>>, ys: &Vec<Array1<fmod>>, classes: u32) -> Vec<f32> {
    let predictions: Vec<Array1<fmod>> = yhats
        .into_iter()
        .map(|yh| {
            let mut yhclassed: Array1<fmod> = Array::zeros(yh.raw_dim());
            let idx = yh
                .iter()
                .enumerate()
                .fold((0, 0.0), |(idx_max, val_max), (idx, val)| {
                    if &val_max > val {
                        (idx_max, val_max)
                    } else {
                        (idx, *val)
                    }
                })
                .0;
            yhclassed[idx] = 1.0;
            yhclassed
        })
        .collect();
    let x = &yhats[0];
    println!("{x}");
    let f1s: Vec<f32> = (0..classes)
        .map(|i| {
            let c = i as usize;
            let tp: f32 = zip(ys, &predictions)
                .map(|(y, p)| if y[c] == 1.0 && p[c] == 1.0 { 1.0 } else { 0.0 })
                .sum();
            let fp: f32 = zip(ys, &predictions)
                .map(|(y, p)| if y[c] == 0.0 && p[c] == 1.0 { 1.0 } else { 0.0 })
                .sum();
            let fn_: f32 = zip(ys, &predictions)
                .map(|(y, p)| if y[c] == 1.0 && p[c] == 0.0 { 1.0 } else { 0.0 })
                .sum();

            tp / (tp + (fp + fn_) / 2.0)
        })
        .collect();
    f1s
}
