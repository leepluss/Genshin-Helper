//! 4-DOF similarity RANSAC (`estimate_affine_partial_2d_ransac`)
//! with deterministic SplitMix64 PRNG and closed-form LSQ refit over inliers.

/// Estimates 2x3 partial affine (4-DOF similarity: scale, rotation, translation)
/// `[[a, -b, tx], [b, a, ty]]` mapping `src_pts` -> `dst_pts`.
/// Returns `Some(([a, -b, tx, b, a, ty], inlier_mask))` on success.
pub fn estimate_affine_partial_2d_ransac(
    src_pts: &[[f32; 2]],
    dst_pts: &[[f32; 2]],
    reproj_thresh: f64,
    max_iters: usize,
    confidence: f64,
) -> Option<([f64; 6], Vec<u8>)> {
    let n = src_pts.len();
    if n < 2 || dst_pts.len() != n || reproj_thresh <= 0.0 || max_iters == 0 {
        return None;
    }

    let thresh_sq = reproj_thresh * reproj_thresh;
    let conf = confidence.clamp(1e-6, 1.0 - 1e-9);
    let log_one_minus_conf = (1.0 - conf).ln();

    // If n == 2, deterministic exact fit
    if n == 2 {
        let model = fit_similarity_2pt(src_pts[0], src_pts[1], dst_pts[0], dst_pts[1])?;
        let inl = count_inliers(src_pts, dst_pts, &model, thresh_sq);
        let mask: Vec<u8> = (0..2)
            .map(|i| if inlier_err_sq(src_pts[i], dst_pts[i], &model) <= thresh_sq { 1 } else { 0 })
            .collect();
        if inl >= 2 {
            return Some((model, mask));
        }
        return None;
    }

    let mut rng_state = 0x517c_c1b7_2722_0a95u64 ^ (n as u64).wrapping_mul(0x9e37_79b9_7f4a_7c15);
    let mut best_model = [1.0, 0.0, 0.0, 0.0, 1.0, 0.0];
    let mut best_count = 0usize;
    let mut best_err_sum = f64::INFINITY;
    let mut niters = max_iters;

    for iter in 0..max_iters {
        if iter >= niters {
            break;
        }
        let i0 = (splitmix64(&mut rng_state) as usize) % n;
        let mut i1 = (splitmix64(&mut rng_state) as usize) % (n - 1);
        if i1 >= i0 {
            i1 += 1;
        }

        let Some(model) = fit_similarity_2pt(src_pts[i0], src_pts[i1], dst_pts[i0], dst_pts[i1])
        else {
            continue;
        };

        let (cnt, err_sum) = score_model(src_pts, dst_pts, &model, thresh_sq);
        if cnt > best_count || (cnt == best_count && err_sum < best_err_sum) {
            best_count = cnt;
            best_err_sum = err_sum;
            best_model = model;

            let ep = (n - cnt) as f64 / (n as f64);
            let p_good = (1.0 - ep).powi(2);
            if p_good >= 1.0 - 1e-12 {
                niters = iter + 1;
            } else if p_good > 1e-12 {
                let est = (log_one_minus_conf / (1.0 - p_good).ln()).ceil();
                if est.is_finite() && est >= 1.0 {
                    niters = niters.min(est as usize);
                }
            }
        }
    }

    if best_count < 2 {
        return None;
    }

    // Closed-form LSQ refit on inliers (up to 2 polish passes)
    let mut cur_model = best_model;
    let mut mask = vec![0u8; n];
    for _ in 0..2 {
        let mut inlier_indices = Vec::with_capacity(n);
        for i in 0..n {
            if inlier_err_sq(src_pts[i], dst_pts[i], &cur_model) <= thresh_sq {
                mask[i] = 1;
                inlier_indices.push(i);
            } else {
                mask[i] = 0;
            }
        }
        if inlier_indices.len() < 2 {
            break;
        }
        if let Some(refit) = fit_similarity_lsq(src_pts, dst_pts, &inlier_indices) {
            cur_model = refit;
        }
    }

    let mut final_cnt = 0usize;
    for i in 0..n {
        if inlier_err_sq(src_pts[i], dst_pts[i], &cur_model) <= thresh_sq {
            mask[i] = 1;
            final_cnt += 1;
        } else {
            mask[i] = 0;
        }
    }
    if final_cnt < 2 {
        return None;
    }

    Some((cur_model, mask))
}

#[inline]
fn splitmix64(state: &mut u64) -> u64 {
    *state = state.wrapping_add(0x9e37_79b9_7f4a_7c15);
    let mut z = *state;
    z = (z ^ (z >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
    z ^ (z >> 31)
}

#[inline]
fn fit_similarity_2pt(p0: [f32; 2], p1: [f32; 2], q0: [f32; 2], q1: [f32; 2]) -> Option<[f64; 6]> {
    let dx = p1[0] as f64 - p0[0] as f64;
    let dy = p1[1] as f64 - p0[1] as f64;
    let denom = dx * dx + dy * dy;
    if denom < 1e-8 {
        return None;
    }
    let du = q1[0] as f64 - q0[0] as f64;
    let dv = q1[1] as f64 - q0[1] as f64;
    let a = (dx * du + dy * dv) / denom;
    let b = (dx * dv - dy * du) / denom;
    let tx = q0[0] as f64 - (a * p0[0] as f64 - b * p0[1] as f64);
    let ty = q0[1] as f64 - (b * p0[0] as f64 + a * p0[1] as f64);
    Some([a, -b, tx, b, a, ty])
}

#[inline]
fn inlier_err_sq(p: [f32; 2], q: [f32; 2], m: &[f64; 6]) -> f64 {
    let x = p[0] as f64;
    let y = p[1] as f64;
    let ex = m[0] * x + m[1] * y + m[2] - q[0] as f64;
    let ey = m[3] * x + m[4] * y + m[5] - q[1] as f64;
    ex * ex + ey * ey
}

#[inline]
fn count_inliers(src: &[[f32; 2]], dst: &[[f32; 2]], m: &[f64; 6], thresh_sq: f64) -> usize {
    src.iter()
        .zip(dst.iter())
        .filter(|&(p, q)| inlier_err_sq(*p, *q, m) <= thresh_sq)
        .count()
}

#[inline]
fn score_model(
    src: &[[f32; 2]],
    dst: &[[f32; 2]],
    m: &[f64; 6],
    thresh_sq: f64,
) -> (usize, f64) {
    let mut cnt = 0usize;
    let mut err_sum = 0.0f64;
    for (p, q) in src.iter().zip(dst.iter()) {
        let e2 = inlier_err_sq(*p, *q, m);
        if e2 <= thresh_sq {
            cnt += 1;
            err_sum += e2;
        } else {
            err_sum += thresh_sq;
        }
    }
    (cnt, err_sum)
}

fn fit_similarity_lsq(
    src: &[[f32; 2]],
    dst: &[[f32; 2]],
    indices: &[usize],
) -> Option<[f64; 6]> {
    let k = indices.len() as f64;
    if indices.len() < 2 {
        return None;
    }
    let mut mx = 0.0f64;
    let mut my = 0.0f64;
    let mut mu = 0.0f64;
    let mut mv = 0.0f64;
    for &i in indices {
        mx += src[i][0] as f64;
        my += src[i][1] as f64;
        mu += dst[i][0] as f64;
        mv += dst[i][1] as f64;
    }
    mx /= k;
    my /= k;
    mu /= k;
    mv /= k;

    let mut denom = 0.0f64;
    let mut num_a = 0.0f64;
    let mut num_b = 0.0f64;
    for &i in indices {
        let dx = src[i][0] as f64 - mx;
        let dy = src[i][1] as f64 - my;
        let du = dst[i][0] as f64 - mu;
        let dv = dst[i][1] as f64 - mv;
        denom += dx * dx + dy * dy;
        num_a += dx * du + dy * dv;
        num_b += dx * dv - dy * du;
    }
    if denom < 1e-10 {
        return None;
    }
    let a = num_a / denom;
    let b = num_b / denom;
    let tx = mu - (a * mx - b * my);
    let ty = mv - (b * mx + a * my);
    Some([a, -b, tx, b, a, ty])
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn ransac_recovers_exact_similarity_with_50pct_outliers() {
        let theta = 37.5f64.to_radians();
        let scale = 1.18f64;
        let a = scale * theta.cos();
        let b = scale * theta.sin();
        let tx = 42.3;
        let ty = -19.7;

        let mut src = Vec::new();
        let mut dst = Vec::new();
        for i in 0..30 {
            let x = (i * 11 % 100) as f32 + 10.0;
            let y = (i * 17 % 90) as f32 + 5.0;
            src.push([x, y]);
            if i < 15 {
                let u = (a * x as f64 - b * y as f64 + tx) as f32;
                let v = (b * x as f64 + a * y as f64 + ty) as f32;
                dst.push([u, v]);
            } else {
                // Gross outliers
                dst.push([(i * 53 % 200) as f32, (i * 71 % 200) as f32]);
            }
        }

        let (m, mask) =
            estimate_affine_partial_2d_ransac(&src, &dst, 2.0, 2000, 0.99).unwrap();
        let inl: usize = mask.iter().map(|&v| v as usize).sum();
        assert_eq!(inl, 15);
        assert!((m[0] - a).abs() < 1e-4);
        assert!((m[3] - b).abs() < 1e-4);
        assert!((m[2] - tx).abs() < 1e-3);
        assert!((m[5] - ty).abs() < 1e-3);
    }
}
