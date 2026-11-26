//! Edge detection (`canny_u8`), 8-connected components with stats, external contour extraction,
//! polygon moments (Green's theorem), convex hull, Douglas-Peucker, minimum enclosing triangle, and line fit.

use crate::ops::sobel::sobel_3x3_u8_f32;
use crate::ops::{ContourMethod, CvError, CvResult, Point2f, Point2i, Rect};
use ndarray::{Array2, ArrayView2};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct ComponentStats {
    pub x: i32,
    pub y: i32,
    pub width: i32,
    pub height: i32,
    pub area: i32,
}

#[derive(Debug, Clone)]
pub struct ConnectedComponents {
    pub num_labels: usize,
    pub labels: Array2<i32>,
    pub stats: Vec<ComponentStats>,
    pub centroids: Vec<[f64; 2]>,
}

#[derive(Debug, Clone, Copy, PartialEq, Default)]
pub struct Moments {
    pub m00: f64,
    pub m10: f64,
    pub m01: f64,
}

/// Canny edge detector matching `cv2.Canny(src, low_thresh, high_thresh, apertureSize=3, L2gradient=False)`.
pub fn canny_u8(src: ArrayView2<'_, u8>, low_thresh: f64, high_thresh: f64) -> CvResult<Array2<u8>> {
    let (h, w) = src.dim();
    if h == 0 || w == 0 {
        return Err(CvError::EmptyInput("canny_u8 empty".into()));
    }
    let (lo, hi) = if low_thresh <= high_thresh {
        (low_thresh as i32, high_thresh as i32)
    } else {
        (high_thresh as i32, low_thresh as i32)
    };
    let (gx_f, gy_f) = sobel_3x3_u8_f32(src)?;
    let n = h * w;
    let gx_s = gx_f.as_slice().expect("contiguous sobel gx");
    let gy_s = gy_f.as_slice().expect("contiguous sobel gy");
    let mut mag = vec![0i32; n];
    let mut gx = vec![0i32; n];
    let mut gy = vec![0i32; n];
    for idx in 0..n {
        let ix = gx_s[idx].round() as i32;
        let iy = gy_s[idx].round() as i32;
        gx[idx] = ix;
        gy[idx] = iy;
        mag[idx] = ix.abs() + iy.abs();
    }

    // 4-sector NMS (0=horiz, 1=diag45, 2=vert, 3=diag135) matching OpenCV canny.cpp
    // map: 0 = non-edge, 1 = weak candidate, 2 = strong edge
    let mut state = vec![0u8; n];
    let mut stack = Vec::<usize>::with_capacity(n / 8 + 16);
    const TAN225: i64 = 13_573; // round(tan(22.5) * 32768)
    const TAN675: i64 = 79_109; // round(tan(67.5) * 32768)

    if h >= 3 && w >= 3 {
        for y in 1..(h - 1) {
            let row_off = y * w;
            for x in 1..(w - 1) {
                let idx = row_off + x;
                let m = mag[idx];
                if m <= lo {
                    continue;
                }
                let gxv = gx[idx];
                let gyv = gy[idx];
                let ax = gxv.abs() as i64;
                let ay = (gyv.abs() as i64) << 15;
                let (n1, n2) = if ay < ax * TAN225 {
                    (mag[idx - 1], mag[idx + 1])
                } else if ay > ax * TAN675 {
                    (mag[idx - w], mag[idx + w])
                } else {
                    let s = if (gxv ^ gyv) < 0 { -1isize } else { 1isize };
                    let idx_i = idx as isize;
                    let wi = w as isize;
                    (
                        mag[(idx_i - wi - s) as usize],
                        mag[(idx_i + wi + s) as usize],
                    )
                };
                if m > n1 && m >= n2 {
                    if m > hi {
                        state[idx] = 2;
                        stack.push(idx);
                    } else {
                        state[idx] = 1;
                    }
                }
            }
        }
    }

    // Border rows/cols (plus tiny images < 3x3)
    for y in 0..h {
        let on_y_border = y == 0 || y + 1 == h;
        let x_iter_full = on_y_border || h < 3 || w < 3;
        let mut handle_border = |x: usize| {
            let idx = y * w + x;
            let m = mag[idx];
            if m <= lo {
                return;
            }
            let ax = gx[idx].abs() as i64;
            let ay = (gy[idx].abs() as i64) << 15;
            let (n1, n2) = if ay < ax * TAN225 {
                let m1 = if x > 0 { mag[idx - 1] } else { 0 };
                let m2 = if x + 1 < w { mag[idx + 1] } else { 0 };
                (m1, m2)
            } else if ay > ax * TAN675 {
                let m1 = if y > 0 { mag[idx - w] } else { 0 };
                let m2 = if y + 1 < h { mag[idx + w] } else { 0 };
                (m1, m2)
            } else {
                let s = if (gx[idx] ^ gy[idx]) < 0 { -1isize } else { 1isize };
                let nx1 = x as isize - s;
                let nx2 = x as isize + s;
                let m1 = if y > 0 && nx1 >= 0 && nx1 < w as isize {
                    mag[(y - 1) * w + nx1 as usize]
                } else {
                    0
                };
                let m2 = if y + 1 < h && nx2 >= 0 && nx2 < w as isize {
                    mag[(y + 1) * w + nx2 as usize]
                } else {
                    0
                };
                (m1, m2)
            };
            if m > n1 && m >= n2 {
                if m > hi {
                    state[idx] = 2;
                    stack.push(idx);
                } else {
                    state[idx] = 1;
                }
            }
        };
        if x_iter_full {
            for x in 0..w {
                handle_border(x);
            }
        } else {
            handle_border(0);
            if w > 1 {
                handle_border(w - 1);
            }
        }
    }

    // Hysteresis connectivity (flat stack reachability is invariant to pop order)
    while let Some(cidx) = stack.pop() {
        let cx = cidx % w;
        let cy = cidx / w;
        if cx > 0 && cx + 1 < w && cy > 0 && cy + 1 < h {
            for nidx in [
                cidx - w - 1,
                cidx - w,
                cidx - w + 1,
                cidx - 1,
                cidx + 1,
                cidx + w - 1,
                cidx + w,
                cidx + w + 1,
            ] {
                if state[nidx] == 1 {
                    state[nidx] = 2;
                    stack.push(nidx);
                }
            }
        } else {
            for dy in -1isize..=1 {
                let ny = cy as isize + dy;
                if ny < 0 || ny >= h as isize {
                    continue;
                }
                for dx in -1isize..=1 {
                    if dx == 0 && dy == 0 {
                        continue;
                    }
                    let nx = cx as isize + dx;
                    if nx < 0 || nx >= w as isize {
                        continue;
                    }
                    let nidx = ny as usize * w + nx as usize;
                    if state[nidx] == 1 {
                        state[nidx] = 2;
                        stack.push(nidx);
                    }
                }
            }
        }
    }

    let mut out = Array2::<u8>::zeros((h, w));
    let out_s = out.as_slice_mut().unwrap();
    for (dst, &st) in out_s.iter_mut().zip(state.iter()) {
        *dst = if st == 2 { 255 } else { 0 };
    }
    Ok(out)
}

/// 8-connected components with stats matching `cv2.connectedComponentsWithStats(mask, connectivity=8)`.
pub fn connected_components_with_stats_8(mask: ArrayView2<'_, u8>) -> CvResult<ConnectedComponents> {
    let (h, w) = mask.dim();
    if h == 0 || w == 0 {
        return Err(CvError::EmptyInput("connected_components_with_stats_8 empty".into()));
    }
    let mask_contig;
    let m_s = if let Some(s) = mask.as_slice() {
        s
    } else {
        mask_contig = mask.to_owned();
        mask_contig.as_slice().unwrap()
    };

    let mut labels = Array2::<i32>::zeros((h, w));
    let lbl_s = labels.as_slice_mut().unwrap();
    let mut stats = vec![ComponentStats {
        x: 0,
        y: 0,
        width: w as i32,
        height: h as i32,
        area: 0,
    }];
    let mut centroids = vec![[0.0f64; 2]];
    let mut bg_sum_x = 0.0f64;
    let mut bg_sum_y = 0.0f64;
    let mut bg_area = 0i32;

    let mut stack = Vec::<usize>::with_capacity(256);
    let mut cur_label = 0i32;

    for y in 0..h {
        let row_off = y * w;
        for x in 0..w {
            let start_idx = row_off + x;
            if m_s[start_idx] == 0 {
                bg_area += 1;
                bg_sum_x += x as f64;
                bg_sum_y += y as f64;
                continue;
            }
            if lbl_s[start_idx] != 0 {
                continue;
            }
            cur_label += 1;
            lbl_s[start_idx] = cur_label;
            stack.push(start_idx);

            let mut min_x = x as i32;
            let mut max_x = x as i32;
            let mut min_y = y as i32;
            let mut max_y = y as i32;
            let mut area = 0i32;
            let mut sum_x = 0.0f64;
            let mut sum_y = 0.0f64;

            while let Some(pidx) = stack.pop() {
                let px = pidx % w;
                let py = pidx / w;
                let px_i = px as i32;
                let py_i = py as i32;
                area += 1;
                sum_x += px as f64;
                sum_y += py as f64;
                if px_i < min_x {
                    min_x = px_i;
                }
                if px_i > max_x {
                    max_x = px_i;
                }
                if py_i < min_y {
                    min_y = py_i;
                }
                if py_i > max_y {
                    max_y = py_i;
                }

                if px > 0 && px + 1 < w && py > 0 && py + 1 < h {
                    for nidx in [
                        pidx - w - 1,
                        pidx - w,
                        pidx - w + 1,
                        pidx - 1,
                        pidx + 1,
                        pidx + w - 1,
                        pidx + w,
                        pidx + w + 1,
                    ] {
                        if m_s[nidx] != 0 && lbl_s[nidx] == 0 {
                            lbl_s[nidx] = cur_label;
                            stack.push(nidx);
                        }
                    }
                } else {
                    for dy in -1isize..=1 {
                        let ny = py as isize + dy;
                        if ny < 0 || ny >= h as isize {
                            continue;
                        }
                        for dx in -1isize..=1 {
                            if dx == 0 && dy == 0 {
                                continue;
                            }
                            let nx = px as isize + dx;
                            if nx < 0 || nx >= w as isize {
                                continue;
                            }
                            let nidx = ny as usize * w + nx as usize;
                            if m_s[nidx] != 0 && lbl_s[nidx] == 0 {
                                lbl_s[nidx] = cur_label;
                                stack.push(nidx);
                            }
                        }
                    }
                }
            }

            stats.push(ComponentStats {
                x: min_x,
                y: min_y,
                width: max_x - min_x + 1,
                height: max_y - min_y + 1,
                area,
            });
            centroids.push([sum_x / area as f64, sum_y / area as f64]);
        }
    }

    stats[0].area = bg_area;
    if bg_area > 0 {
        centroids[0] = [bg_sum_x / bg_area as f64, bg_sum_y / bg_area as f64];
    }

    Ok(ConnectedComponents {
        num_labels: (cur_label + 1) as usize,
        labels,
        stats,
        centroids,
    })
}

/// Traces outer external contours (`RETR_EXTERNAL`) matching `cv2.findContours`.
pub fn find_contours_external(
    mask: ArrayView2<'_, u8>,
    method: ContourMethod,
) -> CvResult<Vec<Vec<Point2i>>> {
    let (h, w) = mask.dim();
    if h == 0 || w == 0 {
        return Err(CvError::EmptyInput("find_contours_external empty".into()));
    }
    let cc = connected_components_with_stats_8(mask)?;
    let lbl_s = cc.labels.as_slice().expect("contiguous cc labels");

    // Find first (top-left) pixel for each label
    let mut first_px = vec![None::<(usize, usize)>; cc.num_labels];
    let mut remaining = cc.num_labels.saturating_sub(1);
    for (idx, &lbl_i) in lbl_s.iter().enumerate() {
        if lbl_i > 0 {
            let lbl = lbl_i as usize;
            if first_px[lbl].is_none() {
                first_px[lbl] = Some((idx % w, idx / w));
                remaining -= 1;
                if remaining == 0 {
                    break;
                }
            }
        }
    }

    // Clockwise 8-neighbor table starting from West (-1, 0)
    const CW8: [(isize, isize); 8] = [
        (-1, 0),
        (-1, -1),
        (0, -1),
        (1, -1),
        (1, 0),
        (1, 1),
        (0, 1),
        (-1, 1),
    ];

    let mut tagged_contours: Vec<(usize, Vec<Point2i>)> = Vec::with_capacity(cc.num_labels);
    for (lbl, pos_opt) in first_px.iter().enumerate().skip(1) {
        let Some((sx, sy)) = *pos_opt else {
            continue;
        };

        // Moore-Neighbor contour tracing around component `lbl`
        let target = lbl as i32;
        let is_fg = |x: isize, y: isize| -> bool {
            x >= 0
                && x < w as isize
                && y >= 0
                && y < h as isize
                && lbl_s[y as usize * w + x as usize] == target
        };

        let mut pts = Vec::new();
        pts.push(Point2i {
            x: sx as i32,
            y: sy as i32,
        });

        let mut cx = sx as isize;
        let mut cy = sy as isize;
        let mut start_dir = 0usize;
        let mut first_next: Option<(isize, isize)> = None;
        let max_steps = (h * w * 4).max(64);
        for _ in 0..max_steps {
            let mut found = None;
            for step in 0..8 {
                let d = (start_dir + step) % 8;
                let nx = cx + CW8[d].0;
                let ny = cy + CW8[d].1;
                if is_fg(nx, ny) {
                    found = Some((nx, ny, d));
                    break;
                }
            }
            let Some((nx, ny, hit_dir)) = found else {
                break;
            };
            if let Some((fnx, fny)) = first_next {
                if cx == sx as isize && cy == sy as isize && nx == fnx && ny == fny {
                    break;
                }
            } else {
                first_next = Some((nx, ny));
            }
            if !(nx == sx as isize && ny == sy as isize) || pts.len() == 1 {
                if nx == sx as isize && ny == sy as isize && first_next.is_some() {
                    // Wait until next step verifies Jacob's stopping criterion
                } else {
                    pts.push(Point2i {
                        x: nx as i32,
                        y: ny as i32,
                    });
                }
            }
            cx = nx;
            cy = ny;
            start_dir = (hit_dir + 5) % 8;
        }
        if pts.len() > 1 && pts.last() == pts.first() {
            pts.pop();
        }

        let final_pts = match method {
            ContourMethod::None => pts,
            ContourMethod::Simple => compress_chain_simple(&pts),
        };
        if !final_pts.is_empty() {
            tagged_contours.push((lbl, final_pts));
        }
    }

    // Filter out contours strictly enclosed inside another component's hole using border-reachable background.
    // Fast skip: a component `lbl` starting at `(p.x, p.y)` can only be enclosed in a hole if some other
    // component's bounding box strictly contains `(p.x, p.y)`.
    if tagged_contours.len() > 1 {
        let maybe_enclosed = tagged_contours.iter().any(|(lbl, c)| {
            let p = c[0];
            cc.stats.iter().enumerate().skip(1).any(|(other_lbl, st)| {
                other_lbl != *lbl
                    && st.width >= 3
                    && st.height >= 3
                    && p.x > st.x
                    && p.x < st.x + st.width - 1
                    && p.y > st.y
                    && p.y < st.y + st.height - 1
            })
        });

        if maybe_enclosed {
            let mask_contig;
            let m_s = if let Some(s) = mask.as_slice() {
                s
            } else {
                mask_contig = mask.to_owned();
                mask_contig.as_slice().unwrap()
            };
            let mut ext_bg = vec![0u8; h * w];
            let mut stack = Vec::<usize>::with_capacity((h * w) / 4 + 16);
            for x in 0..w {
                for y in [0usize, h - 1] {
                    let idx = y * w + x;
                    if m_s[idx] == 0 && ext_bg[idx] == 0 {
                        ext_bg[idx] = 1;
                        stack.push(idx);
                    }
                }
            }
            for y in 0..h {
                for x in [0usize, w - 1] {
                    let idx = y * w + x;
                    if m_s[idx] == 0 && ext_bg[idx] == 0 {
                        ext_bg[idx] = 1;
                        stack.push(idx);
                    }
                }
            }
            while let Some(bidx) = stack.pop() {
                let bx = bidx % w;
                let by = bidx / w;
                if bx > 0 {
                    let n = bidx - 1;
                    if m_s[n] == 0 && ext_bg[n] == 0 {
                        ext_bg[n] = 1;
                        stack.push(n);
                    }
                }
                if bx + 1 < w {
                    let n = bidx + 1;
                    if m_s[n] == 0 && ext_bg[n] == 0 {
                        ext_bg[n] = 1;
                        stack.push(n);
                    }
                }
                if by > 0 {
                    let n = bidx - w;
                    if m_s[n] == 0 && ext_bg[n] == 0 {
                        ext_bg[n] = 1;
                        stack.push(n);
                    }
                }
                if by + 1 < h {
                    let n = bidx + w;
                    if m_s[n] == 0 && ext_bg[n] == 0 {
                        ext_bg[n] = 1;
                        stack.push(n);
                    }
                }
            }
            // Keep only contours whose starting pixel touches exterior background (or image boundary)
            tagged_contours.retain(|(_, c)| {
                let p = c[0];
                let x = p.x as usize;
                let y = p.y as usize;
                x == 0
                    || y == 0
                    || x + 1 == w
                    || y + 1 == h
                    || ext_bg[y * w + (x - 1)] != 0
                    || ext_bg[(y - 1) * w + x] != 0
                    || ext_bg[y * w + (x + 1)] != 0
                    || ext_bg[(y + 1) * w + x] != 0
            });
        }
    }

    Ok(tagged_contours.into_iter().map(|(_, c)| c).collect())
}

fn compress_chain_simple(pts: &[Point2i]) -> Vec<Point2i> {
    if pts.len() <= 2 {
        return pts.to_vec();
    }
    let n = pts.len();
    let mut out = Vec::with_capacity(n);
    for i in 0..n {
        let prev = pts[(i + n - 1) % n];
        let curr = pts[i];
        let next = pts[(i + 1) % n];
        let dx1 = (curr.x - prev.x).signum();
        let dy1 = (curr.y - prev.y).signum();
        let dx2 = (next.x - curr.x).signum();
        let dy2 = (next.y - curr.y).signum();
        if dx1 != dx2 || dy1 != dy2 {
            out.push(curr);
        }
    }
    if out.is_empty() {
        pts.to_vec()
    } else {
        out
    }
}

/// Polygon area via Green's theorem matching `cv2.contourArea`.
pub fn contour_area(contour: &[Point2i]) -> f64 {
    if contour.len() < 3 {
        return 0.0;
    }
    let n = contour.len();
    let mut sum = 0.0f64;
    for i in 0..n {
        let p0 = contour[i];
        let p1 = contour[(i + 1) % n];
        sum += (p0.x as f64) * (p1.y as f64) - (p1.x as f64) * (p0.y as f64);
    }
    (0.5 * sum).abs()
}

/// Polygon spatial moments (`m00, m10, m01`) via Green's theorem matching `cv2.moments`.
pub fn contour_moments(contour: &[Point2i]) -> Moments {
    if contour.is_empty() {
        return Moments::default();
    }
    if contour.len() < 3 {
        return Moments::default();
    }
    let n = contour.len();
    let mut a2 = 0.0f64;
    let mut m10_6 = 0.0f64;
    let mut m01_6 = 0.0f64;
    for i in 0..n {
        let x0 = contour[i].x as f64;
        let y0 = contour[i].y as f64;
        let x1 = contour[(i + 1) % n].x as f64;
        let y1 = contour[(i + 1) % n].y as f64;
        let d = x0 * y1 - x1 * y0;
        a2 += d;
        m10_6 += (x0 + x1) * d;
        m01_6 += (y0 + y1) * d;
    }
    let sign = if a2 < 0.0 { -1.0 } else { 1.0 };
    Moments {
        m00: sign * a2 * 0.5,
        m10: sign * m10_6 / 6.0,
        m01: sign * m01_6 / 6.0,
    }
}

/// Axis-aligned bounding box matching `cv2.boundingRect` (`width = max_x - min_x + 1`).
pub fn bounding_rect(contour: &[Point2i]) -> Rect {
    if contour.is_empty() {
        return Rect::default();
    }
    let mut min_x = i32::MAX;
    let mut max_x = i32::MIN;
    let mut min_y = i32::MAX;
    let mut max_y = i32::MIN;
    for &p in contour {
        min_x = min_x.min(p.x);
        max_x = max_x.max(p.x);
        min_y = min_y.min(p.y);
        max_y = max_y.max(p.y);
    }
    Rect {
        x: min_x,
        y: min_y,
        width: max_x - min_x + 1,
        height: max_y - min_y + 1,
    }
}

/// Euclidean perimeter matching `cv2.arcLength`.
pub fn arc_length(contour: &[Point2i], closed: bool) -> f64 {
    if contour.len() < 2 {
        return 0.0;
    }
    let mut len = 0.0f64;
    for i in 1..contour.len() {
        let dx = (contour[i].x - contour[i - 1].x) as f64;
        let dy = (contour[i].y - contour[i - 1].y) as f64;
        len += (dx * dx + dy * dy).sqrt();
    }
    if closed {
        let last = contour[contour.len() - 1];
        let first = contour[0];
        let dx = (first.x - last.x) as f64;
        let dy = (first.y - last.y) as f64;
        len += (dx * dx + dy * dy).sqrt();
    }
    len
}

/// Convex hull via Andrew's monotone chain matching `cv2.convexHull`.
pub fn convex_hull(pts: &[Point2i]) -> Vec<Point2i> {
    if pts.len() <= 2 {
        return pts.to_vec();
    }
    let mut sorted = pts.to_vec();
    sorted.sort_by_key(|p| (p.x, p.y));
    sorted.dedup();
    if sorted.len() <= 2 {
        return sorted;
    }
    let cross = |o: Point2i, a: Point2i, b: Point2i| -> i64 {
        (a.x as i64 - o.x as i64) * (b.y as i64 - o.y as i64)
            - (a.y as i64 - o.y as i64) * (b.x as i64 - o.x as i64)
    };
    let mut lower = Vec::with_capacity(sorted.len());
    for &p in &sorted {
        while lower.len() >= 2 && cross(lower[lower.len() - 2], lower[lower.len() - 1], p) <= 0 {
            lower.pop();
        }
        lower.push(p);
    }
    let mut upper = Vec::with_capacity(sorted.len());
    for &p in sorted.iter().rev() {
        while upper.len() >= 2 && cross(upper[upper.len() - 2], upper[upper.len() - 1], p) <= 0 {
            upper.pop();
        }
        upper.push(p);
    }
    lower.pop();
    upper.pop();
    lower.extend(upper);
    lower
}

#[inline]
fn point_seg_dist_sq(p: Point2i, a: Point2i, b: Point2i) -> f64 {
    let dx = (b.x - a.x) as f64;
    let dy = (b.y - a.y) as f64;
    let len_sq = dx * dx + dy * dy;
    let px = (p.x - a.x) as f64;
    let py = (p.y - a.y) as f64;
    if len_sq <= 1e-12 {
        return px * px + py * py;
    }
    let cross = (px * dy - py * dx).abs();
    (cross * cross) / len_sq
}

fn rdp_recurse(pts: &[Point2i], start: usize, end: usize, eps_sq: f64, keep: &mut [bool]) {
    if end <= start + 1 {
        return;
    }
    let a = pts[start];
    let b = pts[end];
    let mut max_d = -1.0f64;
    let mut max_idx = start;
    for (idx, &p) in pts.iter().enumerate().take(end).skip(start + 1) {
        let d = point_seg_dist_sq(p, a, b);
        if d > max_d {
            max_d = d;
            max_idx = idx;
        }
    }
    if max_d > eps_sq {
        keep[max_idx] = true;
        rdp_recurse(pts, start, max_idx, eps_sq, keep);
        rdp_recurse(pts, max_idx, end, eps_sq, keep);
    }
}

/// Ramer-Douglas-Peucker polygon approximation matching `cv2.approxPolyDP`.
pub fn approx_poly_dp(contour: &[Point2i], epsilon: f64, closed: bool) -> Vec<Point2i> {
    let n = contour.len();
    if n <= 2 || epsilon <= 0.0 {
        return contour.to_vec();
    }
    let eps_sq = epsilon * epsilon;
    if !closed {
        let mut keep = vec![false; n];
        keep[0] = true;
        keep[n - 1] = true;
        rdp_recurse(contour, 0, n - 1, eps_sq, &mut keep);
        return contour
            .iter()
            .zip(keep.iter())
            .filter_map(|(&p, &k)| if k { Some(p) } else { None })
            .collect();
    }
    // Closed polygon: start from farthest point pair matching OpenCV approxPolyDP
    let mut s_idx = 0usize;
    let mut max_dist = -1.0f64;
    let p0 = contour[0];
    for (i, &p) in contour.iter().enumerate().skip(1) {
        let d = ((p.x - p0.x) as f64).powi(2) + ((p.y - p0.y) as f64).powi(2);
        if d > max_dist {
            max_dist = d;
            s_idx = i;
        }
    }
    let p_s = contour[s_idx];
    let mut e_idx = s_idx;
    let mut max_dist2 = -1.0f64;
    for (i, &p) in contour.iter().enumerate() {
        let d = ((p.x - p_s.x) as f64).powi(2) + ((p.y - p_s.y) as f64).powi(2);
        if d > max_dist2 {
            max_dist2 = d;
            e_idx = i;
        }
    }

    let mut ring = Vec::with_capacity(n + 1);
    for k in 0..=n {
        ring.push(contour[(s_idx + k) % n]);
    }
    let split = (e_idx + n - s_idx) % n;
    let mut keep = vec![false; n + 1];
    keep[0] = true;
    keep[n] = true;
    if split > 0 && split < n {
        keep[split] = true;
        rdp_recurse(&ring, 0, split, eps_sq, &mut keep);
        rdp_recurse(&ring, split, n, eps_sq, &mut keep);
    } else {
        rdp_recurse(&ring, 0, n, eps_sq, &mut keep);
    }

    let mut res = Vec::new();
    for i in 0..n {
        if keep[i] {
            res.push(ring[i]);
        }
    }
    res
}

#[inline]
fn line_intersection_f64(
    a1: [f64; 2],
    d1: [f64; 2],
    a2: [f64; 2],
    d2: [f64; 2],
) -> Option<[f64; 2]> {
    let det = d1[0] * d2[1] - d1[1] * d2[0];
    if det.abs() <= 1e-10 {
        return None;
    }
    let rx = a2[0] - a1[0];
    let ry = a2[1] - a1[1];
    let t = (rx * d2[1] - ry * d2[0]) / det;
    Some([a1[0] + t * d1[0], a1[1] + t * d1[1]])
}

#[inline]
fn tri_area_f64(a: [f64; 2], b: [f64; 2], c: [f64; 2]) -> f64 {
    0.5 * ((b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0])).abs()
}

/// Minimum-area enclosing triangle matching `cv2.minEnclosingTriangle` (O'Rourke 1986 flush-edge theorem).
pub fn min_enclosing_triangle(pts: &[Point2f]) -> Option<(f32, [Point2f; 3])> {
    if pts.len() < 3 {
        return None;
    }
    // Build convex hull in f64 (via scaled integer grid or direct Andrew's monotone chain)
    let mut sorted: Vec<[f64; 2]> = pts.iter().map(|p| [p.x as f64, p.y as f64]).collect();
    sorted.sort_by(|a, b| a[0].total_cmp(&b[0]).then_with(|| a[1].total_cmp(&b[1])));
    sorted.dedup_by(|a, b| (a[0] - b[0]).abs() < 1e-9 && (a[1] - b[1]).abs() < 1e-9);
    if sorted.len() < 3 {
        return None;
    }
    let cross = |o: [f64; 2], a: [f64; 2], b: [f64; 2]| -> f64 {
        (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])
    };
    let mut lower = Vec::new();
    for &p in &sorted {
        while lower.len() >= 2 && cross(lower[lower.len() - 2], lower[lower.len() - 1], p) <= 1e-12
        {
            lower.pop();
        }
        lower.push(p);
    }
    let mut upper = Vec::new();
    for &p in sorted.iter().rev() {
        while upper.len() >= 2 && cross(upper[upper.len() - 2], upper[upper.len() - 1], p) <= 1e-12
        {
            upper.pop();
        }
        upper.push(p);
    }
    lower.pop();
    upper.pop();
    let mut hull = lower;
    hull.extend(upper);
    let m = hull.len();
    if m < 3 {
        return None;
    }
    if m == 3 {
        let a = tri_area_f64(hull[0], hull[1], hull[2]) as f32;
        return Some((
            a,
            [
                Point2f {
                    x: hull[0][0] as f32,
                    y: hull[0][1] as f32,
                },
                Point2f {
                    x: hull[1][0] as f32,
                    y: hull[1][1] as f32,
                },
                Point2f {
                    x: hull[2][0] as f32,
                    y: hull[2][1] as f32,
                },
            ],
        ));
    }

    // Helper: check if all hull points lie on the non-negative (inside) side of directed edge (p -> p+dir)
    let encloses_hull = |v0: [f64; 2], v1: [f64; 2], v2: [f64; 2]| -> bool {
        let s_area = (v1[0] - v0[0]) * (v2[1] - v0[1]) - (v1[1] - v0[1]) * (v2[0] - v0[0]);
        if s_area.abs() <= 1e-9 {
            return false;
        }
        let (a, b, c) = if s_area > 0.0 {
            (v0, v1, v2)
        } else {
            (v0, v2, v1)
        };
        for &hp in &hull {
            if cross(a, b, hp) < -1e-4 || cross(b, c, hp) < -1e-4 || cross(c, a, hp) < -1e-4 {
                return false;
            }
        }
        true
    };

    let mut best_area = f64::INFINITY;
    let mut best_tri = [[0.0f64; 2]; 3];

    // Candidate 1: 3 flush hull edges
    for i in 0..m {
        let ai = hull[i];
        let di = [
            hull[(i + 1) % m][0] - ai[0],
            hull[(i + 1) % m][1] - ai[1],
        ];
        for j in (i + 1)..m {
            let aj = hull[j];
            let dj = [
                hull[(j + 1) % m][0] - aj[0],
                hull[(j + 1) % m][1] - aj[1],
            ];
            let Some(v_ij) = line_intersection_f64(ai, di, aj, dj) else {
                continue;
            };
            for k in (j + 1)..m {
                let ak = hull[k];
                let dk = [
                    hull[(k + 1) % m][0] - ak[0],
                    hull[(k + 1) % m][1] - ak[1],
                ];
                let Some(v_jk) = line_intersection_f64(aj, dj, ak, dk) else {
                    continue;
                };
                let Some(v_ki) = line_intersection_f64(ak, dk, ai, di) else {
                    continue;
                };
                let area = tri_area_f64(v_ij, v_jk, v_ki);
                if area < best_area && encloses_hull(v_ij, v_jk, v_ki) {
                    best_area = area;
                    best_tri = [v_ij, v_jk, v_ki];
                }
            }
        }
    }

    // Candidate 2: 2 flush hull edges + 3rd edge with midpoint at hull vertex p
    for i in 0..m {
        let ai = hull[i];
        let di = [
            hull[(i + 1) % m][0] - ai[0],
            hull[(i + 1) % m][1] - ai[1],
        ];
        for j in (i + 1)..m {
            let aj = hull[j];
            let dj = [
                hull[(j + 1) % m][0] - aj[0],
                hull[(j + 1) % m][1] - aj[1],
            ];
            let Some(apex) = line_intersection_f64(ai, di, aj, dj) else {
                continue;
            };
            // Orient rays from apex along the wedge containing the hull
            let test_p = hull[(i + 2) % m];
            let s1 = cross(ai, hull[(i + 1) % m], test_p);
            let _ = s1;
            let u_vec = {
                let len = di[0].hypot(di[1]);
                let mut u = [di[0] / len, di[1] / len];
                // Check which direction along di points into the half-plane of aj+dj containing hull
                let dot_side = (ai[0] - apex[0]) * u[0] + (ai[1] - apex[1]) * u[1];
                if dot_side < 0.0 {
                    u = [-u[0], -u[1]];
                }
                u
            };
            let v_vec = {
                let len = dj[0].hypot(dj[1]);
                let mut v = [dj[0] / len, dj[1] / len];
                let dot_side = (aj[0] - apex[0]) * v[0] + (aj[1] - apex[1]) * v[1];
                if dot_side < 0.0 {
                    v = [-v[0], -v[1]];
                }
                v
            };
            let det = u_vec[0] * v_vec[1] - u_vec[1] * v_vec[0];
            if det.abs() <= 1e-10 {
                continue;
            }
            for &hp in &hull {
                // Solve hp - apex = alpha * u_vec + beta * v_vec
                let rx = hp[0] - apex[0];
                let ry = hp[1] - apex[1];
                let alpha = (rx * v_vec[1] - ry * v_vec[0]) / det;
                let beta = (u_vec[0] * ry - u_vec[1] * rx) / det;
                if alpha > 1e-6 && beta > 1e-6 {
                    let b_pt = [apex[0] + 2.0 * alpha * u_vec[0], apex[1] + 2.0 * alpha * u_vec[1]];
                    let c_pt = [apex[0] + 2.0 * beta * v_vec[0], apex[1] + 2.0 * beta * v_vec[1]];
                    let area = tri_area_f64(apex, b_pt, c_pt);
                    if area < best_area && encloses_hull(apex, b_pt, c_pt) {
                        best_area = area;
                        best_tri = [apex, b_pt, c_pt];
                    }
                }
            }
        }
    }

    if !best_area.is_finite() {
        return None;
    }
    Some((
        best_area as f32,
        [
            Point2f {
                x: best_tri[0][0] as f32,
                y: best_tri[0][1] as f32,
            },
            Point2f {
                x: best_tri[1][0] as f32,
                y: best_tri[1][1] as f32,
            },
            Point2f {
                x: best_tri[2][0] as f32,
                y: best_tri[2][1] as f32,
            },
        ],
    ))
}

/// Fits a 2D line `(vx, vy, x0, y0)` via orthogonal total least squares (PCA) matching `cv2.fitLine(DIST_L2)`.
pub fn fit_line_2d(pts: &[Point2f]) -> Option<[f32; 4]> {
    if pts.len() < 2 {
        return None;
    }
    let n = pts.len() as f64;
    let mx: f64 = pts.iter().map(|p| p.x as f64).sum::<f64>() / n;
    let my: f64 = pts.iter().map(|p| p.y as f64).sum::<f64>() / n;
    let mut sxx = 0.0f64;
    let mut syy = 0.0f64;
    let mut sxy = 0.0f64;
    for p in pts {
        let dx = p.x as f64 - mx;
        let dy = p.y as f64 - my;
        sxx += dx * dx;
        syy += dy * dy;
        sxy += dx * dy;
    }
    let theta = 0.5 * (2.0 * sxy).atan2(sxx - syy);
    let vx = theta.cos() as f32;
    let vy = theta.sin() as f32;
    Some([vx, vy, mx as f32, my as f32])
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn contours_moments_bounding_rect_and_triangle() {
        let mut mask = Array2::<u8>::zeros((20, 20));
        for y in 4..10 {
            for x in 5..12 {
                mask[[y, x]] = 255;
            }
        }
        let cc = connected_components_with_stats_8(mask.view()).unwrap();
        assert_eq!(cc.num_labels, 2);
        assert_eq!(cc.stats[1].area, 42);

        let cnts = find_contours_external(mask.view(), ContourMethod::Simple).unwrap();
        assert_eq!(cnts.len(), 1);
        assert_eq!(
            bounding_rect(&cnts[0]),
            Rect {
                x: 5,
                y: 4,
                width: 7,
                height: 6
            }
        );

        let tri_pts = vec![
            Point2i { x: 10, y: 10 },
            Point2i { x: 20, y: 10 },
            Point2i { x: 15, y: 22 },
        ];
        assert!((contour_area(&tri_pts) - 60.0).abs() < 1e-6);
        let m = contour_moments(&tri_pts);
        assert!((m.m00 - 60.0).abs() < 1e-6);
        assert!((m.m10 - 900.0).abs() < 1e-6);
        assert!((m.m01 - 840.0).abs() < 1e-6);
        assert!((arc_length(&tri_pts, true) - 36.0).abs() < 1e-6);

        let tri_f: Vec<Point2f> = tri_pts
            .iter()
            .map(|p| Point2f {
                x: p.x as f32,
                y: p.y as f32,
            })
            .collect();
        let (area, _) = min_enclosing_triangle(&tri_f).unwrap();
        assert!((area - 60.0).abs() < 1e-3);
    }

    #[test]
    fn canny_detects_step_edge() {
        let mut img = Array2::<u8>::zeros((16, 16));
        for y in 0..16 {
            for x in 8..16 {
                img[[y, x]] = 200;
            }
        }
        let edges = canny_u8(img.view(), 20.0, 60.0).unwrap();
        assert!(edges[[8, 7]] == 255 || edges[[8, 8]] == 255);
        assert_eq!(edges[[8, 2]], 0);
    }

    #[test]
    fn find_contours_external_rejects_concentric_hole_and_keeps_bay() {
        let mut mask = Array2::<u8>::zeros((32, 32));
        // Outer square ring [4..28, 4..28] with interior hole [8..24, 8..24]
        for y in 4..28 {
            for x in 4..28 {
                if !(8..24).contains(&y) || !(8..24).contains(&x) {
                    mask[[y, x]] = 255;
                }
            }
        }
        // Inner square island [12..20, 12..20] strictly enclosed in hole
        for y in 12..20 {
            for x in 12..20 {
                mask[[y, x]] = 255;
            }
        }
        let cnts = find_contours_external(mask.view(), ContourMethod::Simple).unwrap();
        assert_eq!(cnts.len(), 1, "inner hole island must be filtered out");
        assert_eq!(
            bounding_rect(&cnts[0]),
            Rect {
                x: 4,
                y: 4,
                width: 24,
                height: 24
            }
        );

        // Open top wall at [4..8, 12..20] -> C-shaped bay (non-enclosed), inner island must be kept
        for y in 4..8 {
            for x in 12..20 {
                mask[[y, x]] = 0;
            }
        }
        let cnts_open = find_contours_external(mask.view(), ContourMethod::Simple).unwrap();
        assert_eq!(cnts_open.len(), 2, "bay island must be preserved");
    }
}
