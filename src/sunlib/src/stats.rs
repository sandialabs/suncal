// Statistics

pub fn mean(data: &Vec<f64>) -> f64 {
    data.iter().sum::<f64>() / data.len() as f64
}

// pub fn median(data: &Vec<f64>) -> f64 {
//     let mut d = data.clone();
//     d.sort_by(|a, b| a.partial_cmp(b).unwrap());
//     let mid = d.len() / 2;
//     if d.len() % 2 == 0 {
//         (d[mid - 1] + d[mid]) / 2.0
//     } else {
//         d[mid]
//     }
// }

pub fn variance(data: &Vec<f64>) -> f64 {
    let sum: f64 = data.iter().sum();
    let n = data.len() as f64;
    let xbar: f64 = sum / n;
    let sumsq: f64 = data.iter().map(|x| ((x - xbar).powi(2))).sum();
    sumsq / (n - 1.0)
}

pub fn std_dev(data: &Vec<f64>) -> f64 {
    variance(data).sqrt()
}


