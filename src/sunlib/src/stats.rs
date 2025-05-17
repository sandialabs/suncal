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

pub fn grand_mean(data: &Vec<Vec<f64>>) -> f64 {
    let means: Vec<f64> = data.iter().map(|v| mean(v)).collect();
    mean(&means)
}

pub fn reproducibility(data: &Vec<Vec<f64>>) -> f64 {
    // Calculate reproducibility uncertainty
    let means: Vec<f64> = data.iter().map(|v| mean(v)).collect();
    std_dev(&means)
}

pub fn reproducibility_degf(data: &Vec<Vec<f64>>) -> f64 {
    data.len() as f64 - 1.0
}

pub fn quantiles(data: &Vec<f64>, quants: Vec<f64>) -> Vec<f64> {
    // Calculate quantiles of data
    // Data should already be sorted
    let mut samples = data.clone();
    samples.sort_by(|a, b| a.partial_cmp(b).unwrap());

    let n = samples.len();
    let mut out: Vec<f64> = Vec::new();
    for q in quants.iter() {
        out.push( samples[(q * (n + 1) as f64) as usize] );
    }
    out
}