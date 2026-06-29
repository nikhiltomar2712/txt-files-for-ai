// Simple MLP classifier demo in Rust (no external crates).
// This is NOT an LLM, but it demonstrates "training a small model" pipeline.
// For a tiny binary dataset, it trains a 2-layer MLP using SGD.

use std::f32;

fn sigmoid(x: f32) -> f32 { 1.0 / (1.0 + (-x).exp()) }

fn main() {
    // Toy dataset: points in 2D. Label 1 if x+y > 0 else 0.
    let mut xs: Vec<[f32; 2]> = Vec::new();
    let mut ys: Vec<f32> = Vec::new();

    for i in -10..=10 {
        for j in -10..=10 {
            let x1 = i as f32 / 10.0;
            let x2 = j as f32 / 10.0;
            xs.push([x1, x2]);
            ys.push(if x1 + x2 > 0.0 { 1.0 } else { 0.0 });
        }
    }

    let hidden = 8usize;
    let lr = 0.05f32;
    let epochs = 60usize;

    // weights: input(2)->hidden(hidden), hidden->out(1)
    let mut w1 = vec![[0.0f32; 2]; hidden];
    let mut b1 = vec![0.0f32; hidden];
    let mut w2 = vec![0.0f32; hidden];
    let mut b2 = 0.0f32;

    // init
    let mut seed: u64 = 1;
    for h in 0..hidden {
        seed ^= seed << 13;
        seed ^= seed >> 7;
        seed ^= seed << 17;
        let r = (seed as f32 / u64::MAX as f32) - 0.5;
        w2[h] = r * 0.5;
        for k in 0..2 {
            w1[h][k] = ((seed as f32 / u64::MAX as f32) - 0.5) * 0.5;
        }
    }

    for epoch in 1..=epochs {
        let mut total_loss = 0.0f32;

        for (x, &y_true) in xs.iter().zip(ys.iter()) {
            // forward
            let mut z1 = vec![0.0f32; hidden];
            let mut a1 = vec![0.0f32; hidden];
            for h in 0..hidden {
                z1[h] = w1[h][0] * x[0] + w1[h][1] * x[1] + b1[h];
                a1[h] = z1[h].tanh();
            }

            let z2: f32 = w2.iter().enumerate().map(|(h, &wh)| wh * a1[h]).sum::<f32>() + b2;
            let y_pred = sigmoid(z2);

            // loss: binary cross entropy
            let eps = 1e-7f32;
            let loss = -(y_true * (y_pred + eps).ln() + (1.0 - y_true) * (1.0 - y_pred + eps).ln());
            total_loss += loss;

            // backward (gradients)
            // dL/dz2 = y_pred - y_true
            let dz2 = y_pred - y_true;

            for h in 0..hidden {
                let dw2 = dz2 * a1[h];
                w2[h] -= lr * dw2;
            }
            b2 -= lr * dz2;

            // backprop into hidden: dz1 = dz2*w2[h]*(1 - tanh^2)
            for h in 0..hidden {
                let da1 = dz2 * w2[h];
                let dz1 = da1 * (1.0 - a1[h] * a1[h]);

                for k in 0..2 {
                    let dw1 = dz1 * x[k];
                    w1[h][k] -= lr * dw1;
                }
                b1[h] -= lr * dz1;
            }
        }

        if epoch == 1 || epoch % 10 == 0 {
            println!("epoch={epoch} avg_loss={:.6}", total_loss / xs.len() as f32);
        }
    }

    // quick sanity test
    let test_points = [[0.2, 0.1], [-0.3, 0.1], [0.5, -0.2], [-0.6, -0.6]];
    for x in test_points {
        let mut a1 = vec![0.0f32; hidden];
        for h in 0..hidden {
            let z1 = w1[h][0] * x[0] + w1[h][1] * x[1] + b1[h];
            a1[h] = z1.tanh();
        }
        let z2: f32 = w2.iter().enumerate().map(|(h, &wh)| wh * a1[h]).sum::<f32>() + b2;
        let y_pred = sigmoid(z2);
        println!("x={:?} pred={:.3} label={}", x, y_pred, if x[0] + x[1] > 0.0 { 1 } else { 0 });
    }
}

