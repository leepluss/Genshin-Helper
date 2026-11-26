//! Normalized cross-correlation (`match_template`) for unmasked 1-ch/3-ch images
//! and Padmanabhan (2012) 6-FFT masked `TM_CCOEFF_NORMED` + fast spatial branch.

use crate::ops::{CvError, CvResult, TemplateMatchMode};
use ndarray::{Array2, ArrayView2, ArrayView3};
use rayon::prelude::*;
use rustfft::{Fft, FftNum, FftPlanner, num_complex::Complex};
use std::cell::RefCell;
use std::collections::HashMap;
use std::sync::{Arc, Mutex, OnceLock};

const MAX_SCRATCH_CAP: usize = 512 * 1024;

#[derive(Default)]
struct NccScratch {
    a: Vec<Complex<f64>>,
    at: Vec<Complex<f64>>,
    b: Vec<Complex<f64>>,
    bt: Vec<Complex<f64>>,
    int_i: Vec<f64>,
    int_i2: Vec<f64>,
}

impl NccScratch {
    fn trim_if_huge(&mut self) {
        if self.a.capacity() > MAX_SCRATCH_CAP {
            self.a = Vec::new();
        }
        if self.at.capacity() > MAX_SCRATCH_CAP {
            self.at = Vec::new();
        }
        if self.b.capacity() > MAX_SCRATCH_CAP {
            self.b = Vec::new();
        }
        if self.bt.capacity() > MAX_SCRATCH_CAP {
            self.bt = Vec::new();
        }
        if self.int_i.capacity() > MAX_SCRATCH_CAP {
            self.int_i = Vec::new();
        }
        if self.int_i2.capacity() > MAX_SCRATCH_CAP {
            self.int_i2 = Vec::new();
        }
    }
}

thread_local! {
    static NCC_SCRATCH: RefCell<NccScratch> = RefCell::new(NccScratch::default());
}

pub type MinMaxLocResult = (f32, f32, (usize, usize), (usize, usize));

/// Returns `(min_val, max_val, (min_x, min_y), (max_x, max_y))` matching `cv2.minMaxLoc`.
pub fn min_max_loc_f32(map: ArrayView2<'_, f32>) -> CvResult<MinMaxLocResult> {
    let (h, w) = map.dim();
    if h == 0 || w == 0 {
        return Err(CvError::EmptyInput("min_max_loc_f32 empty".into()));
    }
    let mut min_val = f32::INFINITY;
    let mut max_val = f32::NEG_INFINITY;
    let mut min_loc = (0usize, 0usize);
    let mut max_loc = (0usize, 0usize);
    if let Some(s) = map.as_slice() {
        let mut min_idx = 0usize;
        let mut max_idx = 0usize;
        for (i, &v) in s.iter().enumerate() {
            if v.is_nan() {
                continue;
            }
            if v < min_val {
                min_val = v;
                min_idx = i;
            }
            if v > max_val {
                max_val = v;
                max_idx = i;
            }
        }
        if min_val.is_finite() {
            min_loc = (min_idx % w, min_idx / w);
        }
        if max_val.is_finite() {
            max_loc = (max_idx % w, max_idx / w);
        }
    } else {
        for y in 0..h {
            for x in 0..w {
                let v = map[[y, x]];
                if v.is_nan() {
                    continue;
                }
                if v < min_val {
                    min_val = v;
                    min_loc = (x, y);
                }
                if v > max_val {
                    max_val = v;
                    max_loc = (x, y);
                }
            }
        }
    }
    if !min_val.is_finite() {
        min_val = 0.0;
    }
    if !max_val.is_finite() {
        max_val = 0.0;
    }
    Ok((min_val, max_val, min_loc, max_loc))
}

#[inline]
pub(crate) fn next_smooth_fft_len(n: usize) -> usize {
    if n <= 1 {
        return 1;
    }
    let mut cur = n;
    loop {
        let mut x = cur;
        for p in [2usize, 3, 5] {
            while x.is_multiple_of(p) {
                x /= p;
            }
        }
        if x == 1 {
            return cur;
        }
        cur += 1;
    }
}

type FftPlanPair<T> = (Arc<dyn Fft<T>>, Arc<dyn Fft<T>>);
type FftPool<T> = OnceLock<Mutex<HashMap<usize, FftPlanPair<T>>>>;

pub(crate) fn plan_fft_pair_f64(len: usize) -> FftPlanPair<f64> {
    static POOL: FftPool<f64> = OnceLock::new();
    let map = POOL.get_or_init(|| Mutex::new(HashMap::new()));
    if let Some(pair) = map.lock().unwrap_or_else(|e| e.into_inner()).get(&len) {
        return pair.clone();
    }
    let mut planner = FftPlanner::<f64>::new();
    let pair = (planner.plan_fft_forward(len), planner.plan_fft_inverse(len));
    map.lock()
        .unwrap_or_else(|e| e.into_inner())
        .entry(len)
        .or_insert_with(|| pair.clone())
        .clone()
}

fn transpose_blocked<T: Copy>(src: &[T], dst: &mut [T], rows: usize, cols: usize) {
    debug_assert_eq!(src.len(), rows * cols);
    debug_assert_eq!(dst.len(), rows * cols);
    const BLK: usize = 8;
    let mut r0 = 0;
    while r0 < rows {
        let r1 = (r0 + BLK).min(rows);
        let mut c0 = 0;
        while c0 < cols {
            let c1 = (c0 + BLK).min(cols);
            for r in r0..r1 {
                let srow = &src[r * cols..];
                for c in c0..c1 {
                    dst[c * rows + r] = srow[c];
                }
            }
            c0 += BLK;
        }
        r0 += BLK;
    }
}

fn process_rows_fft<T: FftNum + Send>(
    buf: &mut [Complex<T>],
    row_len: usize,
    fft: &Arc<dyn Fft<T>>,
) {
    if buf.is_empty() || row_len == 0 {
        return;
    }
    let scratch_len = fft.get_inplace_scratch_len();
    if buf.len() >= 32_768 && buf.len() / row_len >= 8 {
        buf.par_chunks_mut(row_len).for_each_init(
            || vec![Complex::new(T::zero(), T::zero()); scratch_len],
            |scratch, row| {
                fft.process_with_scratch(row, scratch);
            },
        );
    } else {
        let mut scratch = vec![Complex::new(T::zero(), T::zero()); scratch_len];
        for row in buf.chunks_exact_mut(row_len) {
            fft.process_with_scratch(row, &mut scratch);
        }
    }
}

pub(crate) fn fft2d_c64(
    buf: &mut [Complex<f64>],
    rows: usize,
    cols: usize,
    row_fft: &Arc<dyn Fft<f64>>,
    col_fft: &Arc<dyn Fft<f64>>,
) {
    process_rows_fft(buf, cols, row_fft);
    let mut trans = vec![Complex::new(0.0, 0.0); rows * cols];
    transpose_blocked(buf, &mut trans, rows, cols);
    process_rows_fft(&mut trans, rows, col_fft);
    transpose_blocked(&trans, buf, cols, rows);
}

/// Computes `I ★ K` (valid cross-correlation of size `(ih - kh + 1, iw - kw + 1)`) via 2D `f64` FFT
/// in transposed layout (halving transpose count, reusing thread-local scratch, and skipping empty rows).
fn xcorr_valid_fft_f64(
    img: &[f64],
    ih: usize,
    iw: usize,
    kern: &[f64],
    kh: usize,
    kw: usize,
) -> Vec<f64> {
    let fh = next_smooth_fft_len(ih);
    let fw = next_smooth_fft_len(iw);
    let oh = ih - kh + 1;
    let ow = iw - kw + 1;
    let (row_fwd, row_inv) = plan_fft_pair_f64(fw);
    let (col_fwd, col_inv) = plan_fft_pair_f64(fh);

    NCC_SCRATCH.with(|cell| {
        let sc = &mut *cell.borrow_mut();
        let n_fft = fh * fw;
        sc.a.resize(n_fft, Complex::new(0.0, 0.0));
        sc.b.resize(n_fft, Complex::new(0.0, 0.0));
        sc.at.resize(n_fft, Complex::new(0.0, 0.0));
        sc.bt.resize(n_fft, Complex::new(0.0, 0.0));

        let NccScratch { a, at, b, bt, .. } = sc;
        for y in 0..ih {
            let srow = &img[y * iw..(y + 1) * iw];
            let arow = &mut a[y * fw..(y + 1) * fw];
            for (dst, &v) in arow[..iw].iter_mut().zip(srow.iter()) {
                *dst = Complex::new(v, 0.0);
            }
            arow[iw..].fill(Complex::new(0.0, 0.0));
        }
        a[ih * fw..n_fft].fill(Complex::new(0.0, 0.0));

        for y in 0..kh {
            let krow = &kern[y * kw..(y + 1) * kw];
            let brow = &mut b[y * fw..(y + 1) * fw];
            for (dst, &v) in brow[..kw].iter_mut().zip(krow.iter()) {
                *dst = Complex::new(v, 0.0);
            }
            brow[kw..].fill(Complex::new(0.0, 0.0));
        }
        b[kh * fw..n_fft].fill(Complex::new(0.0, 0.0));

        process_rows_fft(&mut a[..ih * fw], fw, &row_fwd);
        process_rows_fft(&mut b[..kh * fw], fw, &row_fwd);

        transpose_blocked(&a[..n_fft], &mut at[..n_fft], fh, fw);
        transpose_blocked(&b[..n_fft], &mut bt[..n_fft], fh, fw);

        process_rows_fft(&mut at[..n_fft], fh, &col_fwd);
        process_rows_fft(&mut bt[..n_fft], fh, &col_fwd);
        for (va, &vb) in at[..n_fft].iter_mut().zip(bt[..n_fft].iter()) {
            *va *= vb.conj();
        }
        process_rows_fft(&mut at[..n_fft], fh, &col_inv);
        transpose_blocked(&at[..n_fft], &mut a[..n_fft], fw, fh);
        process_rows_fft(&mut a[..oh * fw], fw, &row_inv);

        let norm = 1.0 / (n_fft as f64);
        let mut out = Vec::with_capacity(oh * ow);
        for y in 0..oh {
            for &v in &a[y * fw..y * fw + ow] {
                out.push(v.re * norm);
            }
        }
        sc.trim_if_huge();
        out
    })
}

/// Computes `(img1 ★ kern1, img1 ★ kern2, img2 ★ kern2)` simultaneously via complex packing
/// in transposed 2D `f64` FFT layout.
fn xcorr_valid_pair_img_fft_f64(
    img1: &[f64],
    img2: &[f64],
    img_shape: (usize, usize),
    kern1: &[f64],
    kern2: &[f64],
    kern_shape: (usize, usize),
) -> (Vec<f64>, Vec<f64>, Vec<f64>) {
    let (ih, iw) = img_shape;
    let (kh, kw) = kern_shape;
    let fh = next_smooth_fft_len(ih);
    let fw = next_smooth_fft_len(iw);
    let (row_fwd, row_inv) = plan_fft_pair_f64(fw);
    let (col_fwd, col_inv) = plan_fft_pair_f64(fh);

    let mut f_i_pair = vec![Complex::new(0.0, 0.0); fh * fw];
    let mut f_m = vec![Complex::new(0.0, 0.0); fh * fw];
    let mut f_t = vec![Complex::new(0.0, 0.0); fh * fw];

    for y in 0..ih {
        let r1 = &img1[y * iw..(y + 1) * iw];
        let r2 = &img2[y * iw..(y + 1) * iw];
        let dst = &mut f_i_pair[y * fw..y * fw + iw];
        for ((d, &v1), &v2) in dst.iter_mut().zip(r1.iter()).zip(r2.iter()) {
            *d = Complex::new(v1, v2);
        }
    }
    for y in 0..kh {
        let k1 = &kern1[y * kw..(y + 1) * kw];
        let k2 = &kern2[y * kw..(y + 1) * kw];
        let dst_t = &mut f_t[y * fw..y * fw + kw];
        let dst_m = &mut f_m[y * fw..y * fw + kw];
        for ((dt, dm), (&vk1, &vk2)) in dst_t
            .iter_mut()
            .zip(dst_m.iter_mut())
            .zip(k1.iter().zip(k2.iter()))
        {
            dt.re = vk1;
            dm.re = vk2;
        }
    }

    process_rows_fft(&mut f_i_pair, fw, &row_fwd);
    process_rows_fft(&mut f_m[..kh * fw], fw, &row_fwd);
    process_rows_fft(&mut f_t[..kh * fw], fw, &row_fwd);

    let mut ft_i_pair = vec![Complex::new(0.0, 0.0); fw * fh];
    let mut ft_m = vec![Complex::new(0.0, 0.0); fw * fh];
    let mut prod_t = vec![Complex::new(0.0, 0.0); fw * fh];
    transpose_blocked(&f_i_pair, &mut ft_i_pair, fh, fw);
    transpose_blocked(&f_m, &mut ft_m, fh, fw);
    // Reuse `f_m` temporarily as `ft_t` to avoid an extra allocation.
    transpose_blocked(&f_t, &mut f_m, fh, fw);

    process_rows_fft(&mut ft_i_pair, fh, &col_fwd);
    process_rows_fft(&mut ft_m, fh, &col_fwd);
    process_rows_fft(&mut f_m, fh, &col_fwd);

    // Extract F(img1) from F(img1 + i*img2) in transposed `(c, r)` layout via 2D conjugate symmetry:
    // F_re[c, r] = 0.5 * (Z[c, r] + conj(Z[(fw - c) % fw, (fh - r) % fh])).
    for c in 0..fw {
        let c_neg = if c == 0 { 0 } else { fw - c };
        let z_row = &ft_i_pair[c * fh..(c + 1) * fh];
        let zneg_row = &ft_i_pair[c_neg * fh..(c_neg + 1) * fh];
        let t_row = &f_m[c * fh..(c + 1) * fh];
        let out_row = &mut prod_t[c * fh..(c + 1) * fh];
        for r in 0..fh {
            let r_neg = if r == 0 { 0 } else { fh - r };
            let f_img1 = (z_row[r] + zneg_row[r_neg].conj()) * 0.5;
            out_row[r] = f_img1 * t_row[r].conj();
        }
    }

    // S_I + i * S_I2 = IFFT(ft_i_pair * conj(ft_m))
    for (vi, vm) in ft_i_pair.iter_mut().zip(ft_m.iter()) {
        *vi *= vm.conj();
    }

    process_rows_fft(&mut prod_t, fh, &col_inv);
    process_rows_fft(&mut ft_i_pair, fh, &col_inv);

    transpose_blocked(&prod_t, &mut f_t, fw, fh);
    transpose_blocked(&ft_i_pair, &mut f_i_pair, fw, fh);

    process_rows_fft(&mut f_t, fw, &row_inv);
    process_rows_fft(&mut f_i_pair, fw, &row_inv);

    let norm = 1.0 / ((fh * fw) as f64);
    let oh = ih - kh + 1;
    let ow = iw - kw + 1;
    let mut c_it = Vec::with_capacity(oh * ow);
    let mut s_i = Vec::with_capacity(oh * ow);
    let mut s_i2 = Vec::with_capacity(oh * ow);
    for y in 0..oh {
        let pt_row = &f_t[y * fw..y * fw + ow];
        let pm_row = &f_i_pair[y * fw..y * fw + ow];
        for (&vt, &vm) in pt_row.iter().zip(pm_row.iter()) {
            c_it.push(vt.re * norm);
            s_i.push(vm.re * norm);
            s_i2.push(vm.im * norm);
        }
    }
    (c_it, s_i, s_i2)
}

/// Unmasked 1-channel `match_template` on `f32` images (`CcoeffNormed`, `CcorrNormed`, `SqdiffNormed`).
pub fn match_template_f32(
    img: ArrayView2<'_, f32>,
    tpl: ArrayView2<'_, f32>,
    mode: TemplateMatchMode,
) -> CvResult<Array2<f32>> {
    let (ih, iw) = img.dim();
    let (kh, kw) = tpl.dim();
    if ih == 0 || iw == 0 || kh == 0 || kw == 0 {
        return Err(CvError::EmptyInput("match_template_f32 empty".into()));
    }
    if kh > ih || kw > iw {
        return Err(CvError::ShapeMismatch("template larger than image".into()));
    }
    let oh = ih - kh + 1;
    let ow = iw - kw + 1;
    let out_cells = oh * ow;
    let n_tpl = (kh * kw) as f64;

    // Mean-shift image for numerical stability in CcoeffNormed (invariant to additive shift!)
    let mut sum_i_all = 0.0f64;
    for &v in img.iter() {
        sum_i_all += v as f64;
    }
    let mean_shift = if matches!(mode, TemplateMatchMode::CcoeffNormed) {
        sum_i_all / ((ih * iw) as f64)
    } else {
        0.0
    };

    let mut t_mean = 0.0f64;
    if matches!(mode, TemplateMatchMode::CcoeffNormed) {
        for &v in tpl.iter() {
            t_mean += v as f64;
        }
        t_mean /= n_tpl;
    }
    let mut t_vec = Vec::with_capacity(kh * kw);
    let mut t_sq_sum = 0.0f64;
    for y in 0..kh {
        for x in 0..kw {
            let tv = tpl[[y, x]] as f64 - t_mean;
            t_sq_sum += tv * tv;
            t_vec.push(tv);
        }
    }

    if matches!(mode, TemplateMatchMode::CcoeffNormed | TemplateMatchMode::CcorrNormed)
        && t_sq_sum <= 1e-12
    {
        return Ok(Array2::<f32>::zeros((oh, ow)));
    }

    // Fast spatial branch for small templates / output windows
    if out_cells <= 256 || (out_cells * kh * kw) <= 250_000 {
        let mut out = Array2::<f32>::zeros((oh, ow));
        for oy in 0..oh {
            for ox in 0..ow {
                let mut sum_i = 0.0f64;
                let mut sum_i2 = 0.0f64;
                let mut dot = 0.0f64;
                let mut sqdiff = 0.0f64;
                for ky in 0..kh {
                    let trow = &t_vec[ky * kw..(ky + 1) * kw];
                    for kx in 0..kw {
                        let iv = img[[oy + ky, ox + kx]] as f64 - mean_shift;
                        let tv = trow[kx];
                        sum_i += iv;
                        sum_i2 += iv * iv;
                        dot += iv * tv;
                        if matches!(mode, TemplateMatchMode::SqdiffNormed) {
                            let d = iv - tv;
                            sqdiff += d * d;
                        }
                    }
                }
                let val = match mode {
                    TemplateMatchMode::CcoeffNormed => {
                        let var_i = (sum_i2 - (sum_i * sum_i) / n_tpl).max(0.0);
                        let denom = (var_i * t_sq_sum).sqrt();
                        if denom <= 1e-12 || var_i <= 1e-12 * n_tpl {
                            0.0
                        } else {
                            (dot / denom).clamp(-1.0, 1.0)
                        }
                    }
                    TemplateMatchMode::CcorrNormed => {
                        let denom = (sum_i2 * t_sq_sum).sqrt();
                        if denom <= 1e-12 {
                            0.0
                        } else {
                            (dot / denom).clamp(-1.0, 1.0)
                        }
                    }
                    TemplateMatchMode::SqdiffNormed => {
                        let denom = (sum_i2 * t_sq_sum).sqrt();
                        if denom <= 1e-12 {
                            0.0
                        } else {
                            (sqdiff / denom).clamp(0.0, 1.0)
                        }
                    }
                };
                out[[oy, ox]] = if val.is_finite() { val as f32 } else { 0.0 };
            }
        }
        return Ok(out);
    }

    // Integral images for sum(I) and sum(I^2) + 2D FFT cross-correlation
    let stride_int = iw + 1;
    let mut int_i = vec![0.0f64; (ih + 1) * stride_int];
    let mut int_i2 = vec![0.0f64; (ih + 1) * stride_int];
    let mut img_shifted = Vec::with_capacity(ih * iw);
    let img_contig;
    let img_s = if let Some(s) = img.as_slice() {
        s
    } else {
        img_contig = img.to_owned();
        img_contig.as_slice().unwrap()
    };
    for y in 0..ih {
        let mut row_s1 = 0.0f64;
        let mut row_s2 = 0.0f64;
        let srow = &img_s[y * iw..(y + 1) * iw];
        for (x, &raw) in srow.iter().enumerate() {
            let v = raw as f64 - mean_shift;
            img_shifted.push(v);
            row_s1 += v;
            row_s2 += v * v;
            let idx = (y + 1) * stride_int + (x + 1);
            int_i[idx] = int_i[y * stride_int + (x + 1)] + row_s1;
            int_i2[idx] = int_i2[y * stride_int + (x + 1)] + row_s2;
        }
    }

    let corr = xcorr_valid_fft_f64(&img_shifted, ih, iw, &t_vec, kh, kw);
    let mut out = Array2::<f32>::zeros((oh, ow));
    let out_s = out.as_slice_mut().expect("contiguous out");
    for oy in 0..oh {
        let y0 = oy * stride_int;
        let y1 = (oy + kh) * stride_int;
        let out_row = &mut out_s[oy * ow..(oy + 1) * ow];
        let corr_row = &corr[oy * ow..(oy + 1) * ow];
        for ox in 0..ow {
            let x0 = ox;
            let x1 = ox + kw;
            let s1 = int_i[y1 + x1] - int_i[y0 + x1] - int_i[y1 + x0] + int_i[y0 + x0];
            let s2 = int_i2[y1 + x1] - int_i2[y0 + x1] - int_i2[y1 + x0] + int_i2[y0 + x0];
            let dot = corr_row[ox];
            let val = match mode {
                TemplateMatchMode::CcoeffNormed => {
                    let var_i = (s2 - (s1 * s1) / n_tpl).max(0.0);
                    let denom = (var_i * t_sq_sum).sqrt();
                    if denom <= 1e-12 || var_i <= 1e-12 * n_tpl {
                        0.0
                    } else {
                        (dot / denom).clamp(-1.0, 1.0)
                    }
                }
                TemplateMatchMode::CcorrNormed => {
                    let denom = (s2.max(0.0) * t_sq_sum).sqrt();
                    if denom <= 1e-12 {
                        0.0
                    } else {
                        (dot / denom).clamp(-1.0, 1.0)
                    }
                }
                TemplateMatchMode::SqdiffNormed => {
                    let denom = (s2.max(0.0) * t_sq_sum).sqrt();
                    if denom <= 1e-12 {
                        0.0
                    } else {
                        ((s2 - 2.0 * dot + t_sq_sum).max(0.0) / denom).clamp(0.0, 1.0)
                    }
                }
            };
            out_row[ox] = if val.is_finite() { val as f32 } else { 0.0 };
        }
    }
    Ok(out)
}

/// Cached 2D forward FFT of a zero-mean `u8` template for fixed `(ih, iw)` image dimensions.
#[derive(Debug)]
pub(crate) struct PreparedTemplateFft {
    pub ih: usize,
    pub iw: usize,
    pub fh: usize,
    pub fw: usize,
    pub kh: usize,
    pub kw: usize,
    pub t_sq_sum: f64,
    pub bt_conj: Vec<Complex<f64>>,
}

/// Precomputes the zero-mean template FFT (conjugated in transposed `(fw, fh)` layout)
/// for repeated `CcoeffNormed` matching against `u8` images of size `(ih, iw)`.
pub(crate) fn prepare_ccoeff_template_fft_u8(
    tpl: ArrayView2<'_, u8>,
    ih: usize,
    iw: usize,
) -> CvResult<Arc<PreparedTemplateFft>> {
    let (kh, kw) = tpl.dim();
    if ih == 0 || iw == 0 || kh == 0 || kw == 0 {
        return Err(CvError::EmptyInput("prepare_ccoeff_template_fft_u8 empty".into()));
    }
    if kh > ih || kw > iw {
        return Err(CvError::ShapeMismatch("template larger than image".into()));
    }
    let tpl_contig;
    let tpl_s = if let Some(s) = tpl.as_slice() {
        s
    } else {
        tpl_contig = tpl.to_owned();
        tpl_contig.as_slice().unwrap()
    };
    let n_tpl = (kh * kw) as f64;
    let t_mean = tpl_s.iter().map(|&v| v as u64).sum::<u64>() as f64 / n_tpl;

    let fh = next_smooth_fft_len(ih);
    let fw = next_smooth_fft_len(iw);
    let (row_fwd, _) = plan_fft_pair_f64(fw);
    let (col_fwd, _) = plan_fft_pair_f64(fh);

    let mut b = vec![Complex::new(0.0f64, 0.0f64); fh * fw];
    let mut t_sq_sum = 0.0f64;
    for y in 0..kh {
        let trow = &tpl_s[y * kw..(y + 1) * kw];
        let brow = &mut b[y * fw..y * fw + kw];
        for (dst, &raw) in brow.iter_mut().zip(trow.iter()) {
            let tv = raw as f64 - t_mean;
            t_sq_sum += tv * tv;
            dst.re = tv;
        }
    }

    let mut bt_conj = Vec::new();
    if t_sq_sum > 1e-12 {
        process_rows_fft(&mut b[..kh * fw], fw, &row_fwd);
        bt_conj = vec![Complex::new(0.0f64, 0.0f64); fw * fh];
        transpose_blocked(&b, &mut bt_conj, fh, fw);
        process_rows_fft(&mut bt_conj, fh, &col_fwd);
        for v in &mut bt_conj {
            *v = v.conj();
        }
    }

    Ok(Arc::new(PreparedTemplateFft {
        ih,
        iw,
        fh,
        fw,
        kh,
        kw,
        t_sq_sum,
        bt_conj,
    }))
}

#[inline]
fn eval_prepared_u8_ccoeff<R>(
    img: ArrayView2<'_, u8>,
    prep: &PreparedTemplateFft,
    mut on_cell: impl FnMut(usize, usize, f32),
    finish: impl FnOnce() -> R,
) -> CvResult<R> {
    let (ih, iw) = img.dim();
    if ih != prep.ih || iw != prep.iw {
        return Err(CvError::ShapeMismatch(
            "image dimensions differ from PreparedTemplateFft".into(),
        ));
    }
    let oh = ih - prep.kh + 1;
    let ow = iw - prep.kw + 1;
    if prep.t_sq_sum <= 1e-12 || prep.bt_conj.is_empty() {
        for oy in 0..oh {
            for ox in 0..ow {
                on_cell(oy, ox, 0.0);
            }
        }
        return Ok(finish());
    }

    let fh = prep.fh;
    let fw = prep.fw;
    let kh = prep.kh;
    let kw = prep.kw;
    let n_tpl = (kh * kw) as f64;
    let n_fft = fh * fw;

    let img_contig;
    let img_s = if let Some(s) = img.as_slice() {
        s
    } else {
        img_contig = img.to_owned();
        img_contig.as_slice().unwrap()
    };
    let sum_u64: u64 = img_s.iter().map(|&v| v as u64).sum();
    let mean_shift = (sum_u64 as f64) / ((ih * iw) as f64);

    let stride_int = iw + 1;
    let int_len = (ih + 1) * stride_int;
    let (row_fwd, row_inv) = plan_fft_pair_f64(fw);
    let (col_fwd, col_inv) = plan_fft_pair_f64(fh);

    NCC_SCRATCH.with(|cell| {
        let mut sc = cell.borrow_mut();
        sc.int_i.resize(int_len, 0.0);
        sc.int_i2.resize(int_len, 0.0);
        sc.a.resize(n_fft, Complex::new(0.0, 0.0));
        sc.at.resize(n_fft, Complex::new(0.0, 0.0));

        sc.int_i[..stride_int].fill(0.0);
        sc.int_i2[..stride_int].fill(0.0);

        let NccScratch {
            a,
            at,
            int_i,
            int_i2,
            ..
        } = &mut *sc;

        for y in 0..ih {
            let mut row_s1 = 0.0f64;
            let mut row_s2 = 0.0f64;
            let srow = &img_s[y * iw..(y + 1) * iw];
            let arow = &mut a[y * fw..(y + 1) * fw];
            let prev_int_row = y * stride_int;
            let cur_int_row = (y + 1) * stride_int;
            int_i[cur_int_row] = 0.0;
            int_i2[cur_int_row] = 0.0;
            for (x, (&raw, dst_a)) in srow.iter().zip(arow[..iw].iter_mut()).enumerate() {
                let v = raw as f64 - mean_shift;
                *dst_a = Complex::new(v, 0.0);
                row_s1 += v;
                row_s2 += v * v;
                let xi = x + 1;
                int_i[cur_int_row + xi] = int_i[prev_int_row + xi] + row_s1;
                int_i2[cur_int_row + xi] = int_i2[prev_int_row + xi] + row_s2;
            }
            arow[iw..].fill(Complex::new(0.0, 0.0));
        }
        a[ih * fw..n_fft].fill(Complex::new(0.0, 0.0));

        process_rows_fft(&mut a[..ih * fw], fw, &row_fwd);
        transpose_blocked(&a[..n_fft], &mut at[..n_fft], fh, fw);
        process_rows_fft(&mut at[..n_fft], fh, &col_fwd);
        for (va, &vb) in at[..n_fft].iter_mut().zip(prep.bt_conj.iter()) {
            *va *= vb;
        }
        process_rows_fft(&mut at[..n_fft], fh, &col_inv);
        transpose_blocked(&at[..n_fft], &mut a[..n_fft], fw, fh);
        process_rows_fft(&mut a[..oh * fw], fw, &row_inv);

        let norm = 1.0 / (n_fft as f64);
        for oy in 0..oh {
            let y0 = oy * stride_int;
            let y1 = (oy + kh) * stride_int;
            let a_row = &a[oy * fw..oy * fw + ow];
            for (ox, ca) in a_row.iter().enumerate() {
                let x0 = ox;
                let x1 = ox + kw;
                let s1 = int_i[y1 + x1] - int_i[y0 + x1] - int_i[y1 + x0] + int_i[y0 + x0];
                let s2 = int_i2[y1 + x1] - int_i2[y0 + x1] - int_i2[y1 + x0] + int_i2[y0 + x0];
                let var_i = (s2 - (s1 * s1) / n_tpl).max(0.0);
                let denom = (var_i * prep.t_sq_sum).sqrt();
                let val = if denom <= 1e-12 || var_i <= 1e-12 * n_tpl {
                    0.0
                } else {
                    ((ca.re * norm) / denom).clamp(-1.0, 1.0)
                };
                on_cell(oy, ox, if val.is_finite() { val as f32 } else { 0.0 });
            }
        }
        sc.trim_if_huge();
    });
    Ok(finish())
}

/// Computes `TM_CCOEFF_NORMED` for a `u8` grayscale image using a cached `PreparedTemplateFft`,
/// avoiding `img.mapv(f32)` and kernel FFT recomputation.
#[cfg(test)]
pub(crate) fn match_template_u8_ccoeff_prepared(
    img: ArrayView2<'_, u8>,
    prep: &PreparedTemplateFft,
) -> CvResult<Array2<f32>> {
    let (ih, iw) = img.dim();
    let oh = ih.saturating_sub(prep.kh) + 1;
    let ow = iw.saturating_sub(prep.kw) + 1;
    let mut out = Array2::<f32>::zeros((oh, ow));
    let out_s = out.as_slice_mut().expect("contiguous out");
    eval_prepared_u8_ccoeff(
        img,
        prep,
        |oy, ox, v| {
            out_s[oy * ow + ox] = v;
        },
        || {},
    )?;
    Ok(out)
}

/// Computes `(max_val, (max_x, max_y))` of `TM_CCOEFF_NORMED` directly from a cached `PreparedTemplateFft`
/// without allocating the full output correlation image.
pub(crate) fn match_template_u8_ccoeff_prepared_argmax(
    img: ArrayView2<'_, u8>,
    prep: &PreparedTemplateFft,
) -> CvResult<(f32, (usize, usize))> {
    let mut max_val = f32::NEG_INFINITY;
    let mut max_loc = (0usize, 0usize);
    eval_prepared_u8_ccoeff(
        img,
        prep,
        |oy, ox, v| {
            if !v.is_nan() && v > max_val {
                max_val = v;
                max_loc = (ox, oy);
            }
        },
        || {},
    )?;
    if !max_val.is_finite() {
        max_val = 0.0;
    }
    Ok((max_val, max_loc))
}

/// Unmasked 1-channel `match_template` on `u8` grayscale images.
pub fn match_template_u8(
    img: ArrayView2<'_, u8>,
    tpl: ArrayView2<'_, u8>,
    mode: TemplateMatchMode,
) -> CvResult<Array2<f32>> {
    let img_f = img.mapv(|v| v as f32);
    let tpl_f = tpl.mapv(|v| v as f32);
    match_template_f32(img_f.view(), tpl_f.view(), mode)
}

/// Unmasked 3-channel `(H, W, 3)` `match_template` on `u8` BGR/RGB images matching `cv2.matchTemplate`.
pub fn match_template_u8_hwc(
    img: ArrayView3<'_, u8>,
    tpl: ArrayView3<'_, u8>,
    mode: TemplateMatchMode,
) -> CvResult<Array2<f32>> {
    let (ih, iw, ic) = img.dim();
    let (kh, kw, kc) = tpl.dim();
    if ih == 0 || iw == 0 || kh == 0 || kw == 0 {
        return Err(CvError::EmptyInput("match_template_u8_hwc empty".into()));
    }
    if ic != kc || ic == 0 {
        return Err(CvError::ShapeMismatch("channel mismatch".into()));
    }
    if kh > ih || kw > iw {
        return Err(CvError::ShapeMismatch("template larger than image".into()));
    }

    let oh = ih - kh + 1;
    let ow = iw - kw + 1;
    let out_cells = oh * ow;
    let n_tpl = (kh * kw) as f64;

    let mut t_means = vec![0.0f64; ic];
    if matches!(mode, TemplateMatchMode::CcoeffNormed) {
        for y in 0..kh {
            for x in 0..kw {
                for c in 0..ic {
                    t_means[c] += tpl[[y, x, c]] as f64;
                }
            }
        }
        for m in &mut t_means {
            *m /= n_tpl;
        }
    }
    let mut t_sq_total = 0.0f64;
    let mut t_planes = vec![vec![0.0f64; kh * kw]; ic];
    for c in 0..ic {
        for y in 0..kh {
            for x in 0..kw {
                let tv = tpl[[y, x, c]] as f64 - t_means[c];
                t_planes[c][y * kw + x] = tv;
                t_sq_total += tv * tv;
            }
        }
    }

    if matches!(mode, TemplateMatchMode::CcoeffNormed | TemplateMatchMode::CcorrNormed)
        && t_sq_total <= 1e-12
    {
        return Ok(Array2::<f32>::zeros((oh, ow)));
    }

    let mut num_sum = vec![0.0f64; out_cells];
    let mut var_i_sum = vec![0.0f64; out_cells];
    let mut sq_i_sum = vec![0.0f64; out_cells];

    for c in 0..ic {
        if out_cells <= 256 || (out_cells * kh * kw) <= 250_000 {
            let t_plane = &t_planes[c];
            for oy in 0..oh {
                for ox in 0..ow {
                    let mut s1 = 0.0f64;
                    let mut s2 = 0.0f64;
                    let mut dot = 0.0f64;
                    for ky in 0..kh {
                        let trow = &t_plane[ky * kw..(ky + 1) * kw];
                        for kx in 0..kw {
                            let iv = img[[oy + ky, ox + kx, c]] as f64;
                            s1 += iv;
                            s2 += iv * iv;
                            dot += iv * trow[kx];
                        }
                    }
                    let idx = oy * ow + ox;
                    num_sum[idx] += dot;
                    sq_i_sum[idx] += s2;
                    var_i_sum[idx] += (s2 - (s1 * s1) / n_tpl).max(0.0);
                }
            }
        } else {
            let stride_int = iw + 1;
            let mut int_i = vec![0.0f64; (ih + 1) * stride_int];
            let mut int_i2 = vec![0.0f64; (ih + 1) * stride_int];
            let mut plane = Vec::with_capacity(ih * iw);
            for y in 0..ih {
                let mut rs1 = 0.0f64;
                let mut rs2 = 0.0f64;
                for x in 0..iw {
                    let v = img[[y, x, c]] as f64;
                    plane.push(v);
                    rs1 += v;
                    rs2 += v * v;
                    let idx = (y + 1) * stride_int + (x + 1);
                    int_i[idx] = int_i[y * stride_int + (x + 1)] + rs1;
                    int_i2[idx] = int_i2[y * stride_int + (x + 1)] + rs2;
                }
            }
            let corr = xcorr_valid_fft_f64(&plane, ih, iw, &t_planes[c], kh, kw);
            for oy in 0..oh {
                let y0 = oy;
                let y1 = oy + kh;
                for ox in 0..ow {
                    let x0 = ox;
                    let x1 = ox + kw;
                    let s1 = int_i[y1 * stride_int + x1]
                        - int_i[y0 * stride_int + x1]
                        - int_i[y1 * stride_int + x0]
                        + int_i[y0 * stride_int + x0];
                    let s2 = int_i2[y1 * stride_int + x1]
                        - int_i2[y0 * stride_int + x1]
                        - int_i2[y1 * stride_int + x0]
                        + int_i2[y0 * stride_int + x0];
                    let idx = oy * ow + ox;
                    num_sum[idx] += corr[idx];
                    sq_i_sum[idx] += s2.max(0.0);
                    var_i_sum[idx] += (s2 - (s1 * s1) / n_tpl).max(0.0);
                }
            }
        }
    }

    let mut out = Array2::<f32>::zeros((oh, ow));
    for oy in 0..oh {
        for ox in 0..ow {
            let idx = oy * ow + ox;
            let val = match mode {
                TemplateMatchMode::CcoeffNormed => {
                    let denom = (var_i_sum[idx] * t_sq_total).sqrt();
                    if denom <= 1e-12 || var_i_sum[idx] <= 1e-12 * n_tpl * (ic as f64) {
                        0.0
                    } else {
                        (num_sum[idx] / denom).clamp(-1.0, 1.0)
                    }
                }
                TemplateMatchMode::CcorrNormed => {
                    let denom = (sq_i_sum[idx] * t_sq_total).sqrt();
                    if denom <= 1e-12 {
                        0.0
                    } else {
                        (num_sum[idx] / denom).clamp(-1.0, 1.0)
                    }
                }
                TemplateMatchMode::SqdiffNormed => {
                    let denom = (sq_i_sum[idx] * t_sq_total).sqrt();
                    if denom <= 1e-12 {
                        0.0
                    } else {
                        ((sq_i_sum[idx] - 2.0 * num_sum[idx] + t_sq_total).max(0.0) / denom)
                            .clamp(0.0, 1.0)
                    }
                }
            };
            out[[oy, ox]] = if val.is_finite() { val as f32 } else { 0.0 };
        }
    }
    Ok(out)
}

/// Masked `match_template` on `f32` images (`mask > 0` active pixels).
/// Uses fast spatial sparse-mask inner loop for small outputs (`<= 256` cells) or small templates,
/// and Padmanabhan (2012) FFT masked NCC (`I, I^2, M, MT`) in `f64` for large patches.
pub fn match_template_masked_f32(
    img: ArrayView2<'_, f32>,
    tpl: ArrayView2<'_, f32>,
    mask: ArrayView2<'_, u8>,
    mode: TemplateMatchMode,
) -> CvResult<Array2<f32>> {
    let (ih, iw) = img.dim();
    let (kh, kw) = tpl.dim();
    if ih == 0 || iw == 0 || kh == 0 || kw == 0 {
        return Err(CvError::EmptyInput("match_template_masked_f32 empty".into()));
    }
    if tpl.dim() != mask.dim() {
        return Err(CvError::ShapeMismatch("tpl and mask dimensions differ".into()));
    }
    if kh > ih || kw > iw {
        return Err(CvError::ShapeMismatch("template larger than image".into()));
    }

    let oh = ih - kh + 1;
    let ow = iw - kw + 1;
    let out_cells = oh * ow;

    let mut active_pts: Vec<(usize, usize, f64)> = Vec::new();
    let mut sum_mt = 0.0f64;
    for ky in 0..kh {
        for kx in 0..kw {
            if mask[[ky, kx]] > 0 {
                let tv = tpl[[ky, kx]] as f64;
                active_pts.push((ky, kx, tv));
                sum_mt += tv;
            }
        }
    }
    let n_m = active_pts.len() as f64;
    if active_pts.is_empty() {
        return Ok(Array2::<f32>::zeros((oh, ow)));
    }
    let t_mean = if matches!(mode, TemplateMatchMode::CcoeffNormed) {
        sum_mt / n_m
    } else {
        0.0
    };

    let mut t_zm = vec![0.0f64; kh * kw];
    let mut m_bin = vec![0.0f64; kh * kw];
    let mut t_var = 0.0f64;
    for &(ky, kx, tv) in &active_pts {
        let d = tv - t_mean;
        t_zm[ky * kw + kx] = d;
        m_bin[ky * kw + kx] = 1.0;
        t_var += d * d;
    }

    if matches!(mode, TemplateMatchMode::CcoeffNormed | TemplateMatchMode::CcorrNormed)
        && t_var <= 1e-12
    {
        return Ok(Array2::<f32>::zeros((oh, ow)));
    }

    // Spatial SIMD-friendly sparse mask path for small ROIs / Paimon refine (<= 256 cells)
    if out_cells <= 256 || (out_cells * active_pts.len()) <= 180_000 {
        let active_zm: Vec<(usize, usize, f64)> = active_pts
            .iter()
            .map(|&(ky, kx, tv)| (ky, kx, tv - t_mean))
            .collect();
        let mut out = Array2::<f32>::zeros((oh, ow));
        for oy in 0..oh {
            for ox in 0..ow {
                let mut s_i = 0.0f64;
                let mut s_i2 = 0.0f64;
                let mut dot = 0.0f64;
                let mut sqdiff = 0.0f64;
                for &(ky, kx, tv) in &active_zm {
                    let iv = img[[oy + ky, ox + kx]] as f64;
                    s_i += iv;
                    s_i2 += iv * iv;
                    dot += iv * tv;
                    if matches!(mode, TemplateMatchMode::SqdiffNormed) {
                        let d = iv - tv;
                        sqdiff += d * d;
                    }
                }
                let val = match mode {
                    TemplateMatchMode::CcoeffNormed => {
                        let var_i = (s_i2 - (s_i * s_i) / n_m).max(0.0);
                        let denom = (var_i * t_var).sqrt();
                        if denom <= 1e-12 || var_i <= 1e-12 * n_m {
                            0.0
                        } else {
                            (dot / denom).clamp(-1.0, 1.0)
                        }
                    }
                    TemplateMatchMode::CcorrNormed => {
                        let denom = (s_i2.max(0.0) * t_var).sqrt();
                        if denom <= 1e-12 {
                            0.0
                        } else {
                            (dot / denom).clamp(-1.0, 1.0)
                        }
                    }
                    TemplateMatchMode::SqdiffNormed => {
                        let denom = (s_i2.max(0.0) * t_var).sqrt();
                        if denom <= 1e-12 {
                            0.0
                        } else {
                            (sqdiff / denom).max(0.0)
                        }
                    }
                };
                out[[oy, ox]] = if val.is_finite() { val as f32 } else { 0.0 };
            }
        }
        return Ok(out);
    }

    // Padmanabhan (2012) masked NCC via 2D FFT
    let mut sum_i_all = 0.0f64;
    for &v in img.iter() {
        sum_i_all += v as f64;
    }
    let shift = if matches!(mode, TemplateMatchMode::CcoeffNormed) {
        sum_i_all / ((ih * iw) as f64)
    } else {
        0.0
    };

    let mut i_arr = Vec::with_capacity(ih * iw);
    let mut i2_arr = Vec::with_capacity(ih * iw);
    let img_contig;
    let img_s = if let Some(s) = img.as_slice() {
        s
    } else {
        img_contig = img.to_owned();
        img_contig.as_slice().unwrap()
    };
    for &raw in img_s {
        let v = raw as f64 - shift;
        i_arr.push(v);
        i2_arr.push(v * v);
    }

    let (c_it, s_i, s_i2) =
        xcorr_valid_pair_img_fft_f64(&i_arr, &i2_arr, (ih, iw), &t_zm, &m_bin, (kh, kw));

    let mut out = Array2::<f32>::zeros((oh, ow));
    let out_s = out.as_slice_mut().expect("contiguous out");
    for (idx, dst) in out_s.iter_mut().enumerate() {
        let val = match mode {
            TemplateMatchMode::CcoeffNormed => {
                let var_i = (s_i2[idx] - (s_i[idx] * s_i[idx]) / n_m).max(0.0);
                let denom = (var_i * t_var).sqrt();
                if denom <= 1e-12 || var_i <= 1e-12 * n_m {
                    0.0
                } else {
                    (c_it[idx] / denom).clamp(-1.0, 1.0)
                }
            }
            TemplateMatchMode::CcorrNormed => {
                let denom = (s_i2[idx].max(0.0) * t_var).sqrt();
                if denom <= 1e-12 {
                    0.0
                } else {
                    (c_it[idx] / denom).clamp(-1.0, 1.0)
                }
            }
            TemplateMatchMode::SqdiffNormed => {
                let denom = (s_i2[idx].max(0.0) * t_var).sqrt();
                if denom <= 1e-12 {
                    0.0
                } else {
                    ((s_i2[idx] - 2.0 * c_it[idx] + t_var).max(0.0) / denom).max(0.0)
                }
            }
        };
        *dst = if val.is_finite() { val as f32 } else { 0.0 };
    }
    Ok(out)
}

/// Masked `match_template` on `u8` grayscale images (`mask > 0` active pixels).
pub fn match_template_masked_u8(
    img: ArrayView2<'_, u8>,
    tpl: ArrayView2<'_, u8>,
    mask: ArrayView2<'_, u8>,
    mode: TemplateMatchMode,
) -> CvResult<Array2<f32>> {
    let img_f = img.mapv(|v| v as f32);
    let tpl_f = tpl.mapv(|v| v as f32);
    match_template_masked_f32(img_f.view(), tpl_f.view(), mask, mode)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn unmasked_and_masked_ncc_find_exact_peak() {
        let mut img = Array2::<f32>::zeros((64, 64));
        for y in 0..64 {
            for x in 0..64 {
                img[[y, x]] = (((y * 7 + x * 13) % 97) + ((y * x) % 53)) as f32;
            }
        }
        let tpl = img.slice(ndarray::s![18..38, 22..42]).to_owned();
        let res = match_template_f32(img.view(), tpl.view(), TemplateMatchMode::CcoeffNormed).unwrap();
        let (_, max_v, _, max_loc) = min_max_loc_f32(res.view()).unwrap();
        assert!((max_v - 1.0).abs() < 1e-4, "max_v={max_v}");
        assert_eq!(max_loc, (22, 18));

        let mut mask = Array2::<u8>::zeros((20, 20));
        for y in 2..18 {
            for x in 2..18 {
                if (y as i32 - 10).pow(2) + (x as i32 - 10).pow(2) <= 49 {
                    mask[[y, x]] = 255;
                }
            }
        }
        let res_m =
            match_template_masked_f32(img.view(), tpl.view(), mask.view(), TemplateMatchMode::CcoeffNormed)
                .unwrap();
        let (_, mv_m, _, ml_m) = min_max_loc_f32(res_m.view()).unwrap();
        assert!((mv_m - 1.0).abs() < 1e-4, "mv_m={mv_m}");
        assert_eq!(ml_m, (22, 18));
    }

    #[test]
    fn zero_variance_returns_zero_never_nan() {
        let flat = Array2::<u8>::from_elem((40, 40), 128);
        let tpl = Array2::<u8>::from_elem((16, 16), 128);
        let mask = Array2::<u8>::from_elem((16, 16), 255);
        let res =
            match_template_masked_u8(flat.view(), tpl.view(), mask.view(), TemplateMatchMode::CcoeffNormed)
                .unwrap();
        assert!(res.iter().all(|&v| v == 0.0 && v.is_finite()));

        // Unmasked flat patch alongside high-contrast region (tests mean_shift cancellation guard)
        let mut img_f = Array2::<f32>::zeros((80, 80));
        for y in 0..40 {
            for x in 0..80 {
                img_f[[y, x]] = 5000.0 + ((y * 13 + x * 7) % 100) as f32;
            }
        }
        let tpl_f = img_f.slice(ndarray::s![0..20, 0..20]).to_owned();
        let res_u =
            match_template_f32(img_f.view(), tpl_f.view(), TemplateMatchMode::CcoeffNormed).unwrap();
        // Bottom rows (y >= 40) are pure 0.0 constant windows and must return exact 0.0
        for y in 40..res_u.dim().0 {
            for x in 0..res_u.dim().1 {
                assert_eq!(res_u[[y, x]], 0.0, "flat window ({y},{x}) returned {}", res_u[[y, x]]);
            }
        }
    }

    #[test]
    fn unmasked_sqdiff_normed_clamped_to_unit_interval() {
        let mut img = Array2::<f32>::from_elem((80, 80), 1.0);
        img[[10, 10]] = 2.0;
        let tpl = Array2::<f32>::from_elem((20, 20), 250.0);
        let res = match_template_f32(img.view(), tpl.view(), TemplateMatchMode::SqdiffNormed).unwrap();
        assert!(res.iter().all(|&v| (0.0..=1.0).contains(&v)), "SqdiffNormed f32 exceeded 1.0");

        // Spatial branch (small output)
        let small_img = Array2::<f32>::from_elem((10, 10), 2.0);
        let small_tpl = Array2::<f32>::from_elem((5, 5), 200.0);
        let res_sp =
            match_template_f32(small_img.view(), small_tpl.view(), TemplateMatchMode::SqdiffNormed)
                .unwrap();
        assert!(res_sp.iter().all(|&v| (0.0..=1.0).contains(&v)));

        // HWC branch
        let img3 = ndarray::Array3::<u8>::from_elem((40, 40, 3), 2);
        let tpl3 = ndarray::Array3::<u8>::from_elem((16, 16, 3), 200);
        let res_hwc =
            match_template_u8_hwc(img3.view(), tpl3.view(), TemplateMatchMode::SqdiffNormed)
                .unwrap();
        assert!(res_hwc.iter().all(|&v| (0.0..=1.0).contains(&v)));
    }

    #[test]
    fn rectangular_and_large_f32_ncc_matches_spatial_reference() {
        for &(ih, iw, kh, kw) in &[(97usize, 63usize, 21usize, 17usize), (663, 128, 31, 31)] {
            let mut img = Array2::<f32>::zeros((ih, iw));
            for y in 0..ih {
                for x in 0..iw {
                    img[[y, x]] = (((y * y * 7 + x * x * 13 + y * x * 5) % 251) as f32) * 0.5;
                }
            }
            let py0 = ih / 3;
            let px0 = iw / 3;
            let tpl = img.slice(ndarray::s![py0..py0 + kh, px0..px0 + kw]).to_owned();
            let res = match_template_f32(img.view(), tpl.view(), TemplateMatchMode::CcoeffNormed)
                .unwrap();
            let (_, max_v, _, max_loc) = min_max_loc_f32(res.view()).unwrap();
            assert_eq!(max_loc, (px0, py0), "shape ({ih},{iw})x({kh},{kw}) peak {max_loc:?}, max_v={max_v}");
            assert!((max_v - 1.0).abs() < 1e-4, "shape ({ih},{iw}) max_v={max_v}");

            // Check a deterministic grid of output cells against naive spatial formula
            let t_mean = tpl.iter().map(|&v| v as f64).sum::<f64>() / (kh * kw) as f64;
            let t_sq: f64 = tpl.iter().map(|&v| (v as f64 - t_mean).powi(2)).sum();
            let mut max_d = 0.0f64;
            for oy in (0..(ih - kh + 1)).step_by(7) {
                for ox in (0..(iw - kw + 1)).step_by(5) {
                    let mut s1 = 0.0f64;
                    let mut s2 = 0.0f64;
                    let mut dot = 0.0f64;
                    for ky in 0..kh {
                        for kx in 0..kw {
                            let iv = img[[oy + ky, ox + kx]] as f64;
                            let tv = tpl[[ky, kx]] as f64 - t_mean;
                            s1 += iv;
                            s2 += iv * iv;
                            dot += iv * tv;
                        }
                    }
                    let var = (s2 - s1 * s1 / (kh * kw) as f64).max(0.0);
                    let ref_v = dot / (var * t_sq).sqrt();
                    max_d = max_d.max((res[[oy, ox]] as f64 - ref_v).abs());
                }
            }
            assert!(max_d < 1e-4, "shape ({ih},{iw}) max_d vs spatial={max_d}");
        }
    }

    #[test]
    fn prepared_u8_ccoeff_matches_standard_match_template_u8() {
        let ih = 220usize;
        let iw = 96usize;
        let mut img = Array2::<u8>::from_elem((ih, iw), 35);
        for y in 0..ih {
            for x in 0..iw {
                if y < 140 {
                    img[[y, x]] = (((y * 19 + x * 37 + y * x * 3) % 210) + 20) as u8;
                }
            }
        }
        let tpl = img.slice(ndarray::s![45..76, 30..61]).to_owned();
        let ref_map = match_template_u8(img.view(), tpl.view(), TemplateMatchMode::CcoeffNormed)
            .unwrap();
        let prep = prepare_ccoeff_template_fft_u8(tpl.view(), ih, iw).unwrap();
        let fast_map = match_template_u8_ccoeff_prepared(img.view(), &prep).unwrap();
        assert_eq!(ref_map.dim(), fast_map.dim());
        let mut max_d = 0.0f32;
        for (&a, &b) in ref_map.iter().zip(fast_map.iter()) {
            max_d = max_d.max((a - b).abs());
        }
        assert!(max_d < 1e-5, "prepared max_d={max_d}");

        let (_, max_v_ref, _, max_loc_ref) = min_max_loc_f32(ref_map.view()).unwrap();
        let (max_v_fast, max_loc_fast) =
            match_template_u8_ccoeff_prepared_argmax(img.view(), &prep).unwrap();
        assert_eq!(max_loc_fast, max_loc_ref);
        assert!((max_v_fast - max_v_ref).abs() < 1e-5);
    }
}
