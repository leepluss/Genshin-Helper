//! Pure-Rust `.npy`/`.npz` reader & writer supporting the 4 C-contiguous LE dtypes
//! used by Genshin-Helper datasets (`u8`, `i32`, `i64`, `f32`).
//! Replaces NumPy `np.load` / `np.savez_compressed` via pure-Rust `zip` deflate-only.

use crate::atomic_io;
use crate::{CoreError, CoreResult};
use std::collections::BTreeMap;
use std::fs::File;
use std::io::{BufReader, Cursor, Read, Seek, Write};
use std::path::Path;
use zip::write::SimpleFileOptions;
use zip::{CompressionMethod, ZipArchive, ZipWriter};

const NPY_MAGIC: &[u8; 6] = b"\x93NUMPY";
const MAX_NPY_HEADER_BYTES: usize = 1_048_576;
const NPY_READ_CHUNK_BYTES: usize = 64 * 1024;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum NpyDType {
    U8,
    I32,
    I64,
    F32,
}

impl NpyDType {
    pub const fn descr(self) -> &'static str {
        match self {
            Self::U8 => "|u1",
            Self::I32 => "<i4",
            Self::I64 => "<i8",
            Self::F32 => "<f4",
        }
    }

    pub const fn item_size(self) -> usize {
        match self {
            Self::U8 => 1,
            Self::I32 | Self::F32 => 4,
            Self::I64 => 8,
        }
    }

    pub fn from_descr(descr: &str) -> CoreResult<Self> {
        match descr.trim() {
            "|u1" | "u1" => Ok(Self::U8),
            "<i4" => Ok(Self::I32),
            "<i8" => Ok(Self::I64),
            "<f4" => Ok(Self::F32),
            other => Err(CoreError::InvalidData(format!(
                "unsupported NPY dtype '{other}'"
            ))),
        }
    }
}

#[derive(Debug, Clone, PartialEq)]
pub enum NpyData {
    U8(Vec<u8>),
    I32(Vec<i32>),
    I64(Vec<i64>),
    F32(Vec<f32>),
}

impl NpyData {
    pub fn dtype(&self) -> NpyDType {
        match self {
            Self::U8(_) => NpyDType::U8,
            Self::I32(_) => NpyDType::I32,
            Self::I64(_) => NpyDType::I64,
            Self::F32(_) => NpyDType::F32,
        }
    }

    pub fn len(&self) -> usize {
        match self {
            Self::U8(v) => v.len(),
            Self::I32(v) => v.len(),
            Self::I64(v) => v.len(),
            Self::F32(v) => v.len(),
        }
    }

    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct NpyArray {
    pub shape: Vec<usize>,
    pub data: NpyData,
}

impl NpyArray {
    pub fn new(shape: Vec<usize>, data: NpyData) -> CoreResult<Self> {
        let elem_count = checked_shape_product(&shape)?;
        if elem_count != data.len() {
            return Err(CoreError::InvalidArgument(format!(
                "shape {shape:?} expects {elem_count} elements, got {}",
                data.len()
            )));
        }
        Ok(Self { shape, data })
    }

    pub fn from_u8(shape: Vec<usize>, data: Vec<u8>) -> CoreResult<Self> {
        Self::new(shape, NpyData::U8(data))
    }

    pub fn from_i32(shape: Vec<usize>, data: Vec<i32>) -> CoreResult<Self> {
        Self::new(shape, NpyData::I32(data))
    }

    pub fn from_i64(shape: Vec<usize>, data: Vec<i64>) -> CoreResult<Self> {
        Self::new(shape, NpyData::I64(data))
    }

    pub fn from_f32(shape: Vec<usize>, data: Vec<f32>) -> CoreResult<Self> {
        Self::new(shape, NpyData::F32(data))
    }

    pub fn scalar_i32(v: i32) -> Self {
        Self {
            shape: Vec::new(),
            data: NpyData::I32(vec![v]),
        }
    }

    pub fn scalar_f32(v: f32) -> Self {
        Self {
            shape: Vec::new(),
            data: NpyData::F32(vec![v]),
        }
    }

    pub fn dtype(&self) -> NpyDType {
        self.data.dtype()
    }

    pub fn shape(&self) -> &[usize] {
        &self.shape
    }

    pub fn elem_count(&self) -> usize {
        self.data.len()
    }

    pub fn as_u8(&self) -> CoreResult<&[u8]> {
        match &self.data {
            NpyData::U8(v) => Ok(v),
            other => Err(CoreError::InvalidData(format!(
                "expected U8 array, found {:?}",
                other.dtype()
            ))),
        }
    }

    pub fn as_i32(&self) -> CoreResult<&[i32]> {
        match &self.data {
            NpyData::I32(v) => Ok(v),
            other => Err(CoreError::InvalidData(format!(
                "expected I32 array, found {:?}",
                other.dtype()
            ))),
        }
    }

    pub fn as_i64(&self) -> CoreResult<&[i64]> {
        match &self.data {
            NpyData::I64(v) => Ok(v),
            other => Err(CoreError::InvalidData(format!(
                "expected I64 array, found {:?}",
                other.dtype()
            ))),
        }
    }

    pub fn as_f32(&self) -> CoreResult<&[f32]> {
        match &self.data {
            NpyData::F32(v) => Ok(v),
            other => Err(CoreError::InvalidData(format!(
                "expected F32 array, found {:?}",
                other.dtype()
            ))),
        }
    }

    pub fn to_scalar_i32(&self) -> CoreResult<i32> {
        let slice = self.as_i32()?;
        if slice.len() != 1 {
            return Err(CoreError::InvalidData(format!(
                "expected 1-element I32 scalar, got shape {:?}",
                self.shape
            )));
        }
        Ok(slice[0])
    }

    pub fn to_scalar_f32(&self) -> CoreResult<f32> {
        let slice = self.as_f32()?;
        if slice.len() != 1 {
            return Err(CoreError::InvalidData(format!(
                "expected 1-element F32 scalar, got shape {:?}",
                self.shape
            )));
        }
        Ok(slice[0])
    }

    pub fn from_npy_reader<R: Read>(mut reader: R) -> CoreResult<Self> {
        let mut magic = [0u8; 6];
        reader.read_exact(&mut magic)?;
        if &magic != NPY_MAGIC {
            return Err(CoreError::InvalidData("invalid NPY magic header".to_owned()));
        }

        let mut version = [0u8; 2];
        reader.read_exact(&mut version)?;
        let header_len: usize = match version {
            [1, 0] => {
                let mut len_bytes = [0u8; 2];
                reader.read_exact(&mut len_bytes)?;
                u16::from_le_bytes(len_bytes) as usize
            }
            [2, 0] | [3, 0] => {
                let mut len_bytes = [0u8; 4];
                reader.read_exact(&mut len_bytes)?;
                u32::from_le_bytes(len_bytes) as usize
            }
            [major, minor] => {
                return Err(CoreError::InvalidData(format!(
                    "unsupported NPY version {major}.{minor}"
                )));
            }
        };

        if header_len == 0 || header_len > MAX_NPY_HEADER_BYTES {
            return Err(CoreError::InvalidData(format!(
                "invalid NPY header length: {header_len}"
            )));
        }

        let mut header_bytes = vec![0u8; header_len];
        reader.read_exact(&mut header_bytes)?;
        let header_str = if version[0] == 3 {
            std::str::from_utf8(&header_bytes)
                .map_err(|e| CoreError::InvalidData(format!("invalid UTF-8 NPY header: {e}")))?
                .to_owned()
        } else {
            header_bytes.iter().map(|&b| b as char).collect()
        };

        let (dtype, fortran_order, shape) = parse_npy_header(&header_str)?;
        if fortran_order && shape.len() > 1 {
            return Err(CoreError::InvalidData(
                "Fortran-ordered multidimensional NPY arrays are not supported".to_owned(),
            ));
        }

        let elem_count = checked_shape_product(&shape)?;
        let expected_bytes = elem_count
            .checked_mul(dtype.item_size())
            .ok_or_else(|| CoreError::InvalidData("NPY payload byte size overflow".to_owned()))?;

        let raw = read_exact_bounded(reader, expected_bytes)?;
        let data = match dtype {
            NpyDType::U8 => NpyData::U8(raw),
            NpyDType::I32 => {
                let mut vals = Vec::new();
                vals.try_reserve_exact(elem_count).map_err(|e| {
                    CoreError::InvalidData(format!("NPY I32 buffer allocation failed: {e}"))
                })?;
                vals.extend(raw.as_chunks::<4>().0.iter().map(|c| i32::from_le_bytes(*c)));
                NpyData::I32(vals)
            }
            NpyDType::I64 => {
                let mut vals = Vec::new();
                vals.try_reserve_exact(elem_count).map_err(|e| {
                    CoreError::InvalidData(format!("NPY I64 buffer allocation failed: {e}"))
                })?;
                vals.extend(raw.as_chunks::<8>().0.iter().map(|c| i64::from_le_bytes(*c)));
                NpyData::I64(vals)
            }
            NpyDType::F32 => {
                let mut vals = Vec::new();
                vals.try_reserve_exact(elem_count).map_err(|e| {
                    CoreError::InvalidData(format!("NPY F32 buffer allocation failed: {e}"))
                })?;
                vals.extend(raw.as_chunks::<4>().0.iter().map(|c| f32::from_le_bytes(*c)));
                NpyData::F32(vals)
            }
        };

        Ok(Self { shape, data })
    }

    pub fn from_npy_bytes(bytes: &[u8]) -> CoreResult<Self> {
        Self::from_npy_reader(Cursor::new(bytes))
    }

    pub fn write_npy<W: Write>(&self, mut writer: W) -> CoreResult<()> {
        let shape_str = format_shape_tuple(&self.shape);
        let dict = format!(
            "{{'descr': '{}', 'fortran_order': False, 'shape': {}, }}",
            self.dtype().descr(),
            shape_str
        );

        let dict_bytes = dict.as_bytes();
        // Check if v1.0 fits in u16 header_len after 64-byte alignment
        let v1_unaligned = 10 + dict_bytes.len() + 1;
        let v1_rem = v1_unaligned % 64;
        let v1_pad = if v1_rem == 0 { 0 } else { 64 - v1_rem };
        let v1_header_len = dict_bytes.len() + 1 + v1_pad;

        writer.write_all(NPY_MAGIC)?;
        if v1_header_len <= u16::MAX as usize {
            writer.write_all(&[1, 0])?;
            writer.write_all(&(v1_header_len as u16).to_le_bytes())?;
            writer.write_all(dict_bytes)?;
            for _ in 0..v1_pad {
                writer.write_all(b" ")?;
            }
            writer.write_all(b"\n")?;
        } else {
            let v2_unaligned = 12 + dict_bytes.len() + 1;
            let v2_rem = v2_unaligned % 64;
            let v2_pad = if v2_rem == 0 { 0 } else { 64 - v2_rem };
            let v2_header_len = dict_bytes.len() + 1 + v2_pad;
            writer.write_all(&[2, 0])?;
            writer.write_all(&(v2_header_len as u32).to_le_bytes())?;
            writer.write_all(dict_bytes)?;
            for _ in 0..v2_pad {
                writer.write_all(b" ")?;
            }
            writer.write_all(b"\n")?;
        }

        match &self.data {
            NpyData::U8(v) => writer.write_all(v)?,
            NpyData::I32(v) => {
                for &x in v {
                    writer.write_all(&x.to_le_bytes())?;
                }
            }
            NpyData::I64(v) => {
                for &x in v {
                    writer.write_all(&x.to_le_bytes())?;
                }
            }
            NpyData::F32(v) => {
                for &x in v {
                    writer.write_all(&x.to_le_bytes())?;
                }
            }
        }
        Ok(())
    }

    pub fn to_npy_bytes(&self) -> CoreResult<Vec<u8>> {
        let mut buf = Vec::new();
        self.write_npy(&mut buf)?;
        Ok(buf)
    }
}

#[derive(Debug, Clone, Default, PartialEq)]
pub struct NpzArchive {
    /// Stored arrays keyed without the trailing `.npy` suffix.
    pub arrays: BTreeMap<String, NpyArray>,
}

impl NpzArchive {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn insert(&mut self, name: impl Into<String>, array: NpyArray) {
        let raw = name.into();
        let clean = raw.strip_suffix(".npy").unwrap_or(&raw).to_owned();
        self.arrays.insert(clean, array);
    }

    pub fn get(&self, name: &str) -> Option<&NpyArray> {
        let clean = name.strip_suffix(".npy").unwrap_or(name);
        self.arrays.get(clean)
    }

    pub fn keys(&self) -> impl Iterator<Item = &str> {
        self.arrays.keys().map(String::as_str)
    }

    pub fn len(&self) -> usize {
        self.arrays.len()
    }

    pub fn is_empty(&self) -> bool {
        self.arrays.is_empty()
    }

    pub fn read_from<R: Read + Seek>(reader: R) -> CoreResult<Self> {
        let mut archive = ZipArchive::new(reader)?;
        let mut arrays = BTreeMap::new();
        for idx in 0..archive.len() {
            let mut entry = archive.by_index(idx)?;
            let name = entry.name()?.into_owned();
            if entry.is_dir() {
                continue;
            }
            let key = name.strip_suffix(".npy").ok_or_else(|| {
                CoreError::InvalidData(format!("NPZ entry '{name}' does not end with .npy"))
            })?;
            let key_owned = key.to_owned();
            let arr = NpyArray::from_npy_reader(&mut entry)?;
            arrays.insert(key_owned, arr);
        }
        Ok(Self { arrays })
    }

    pub fn read_file(path: &Path) -> CoreResult<Self> {
        let file = File::open(path)?;
        Self::read_from(BufReader::new(file))
    }

    pub fn read_single_from_file(path: &Path, key: &str) -> CoreResult<NpyArray> {
        let file = File::open(path)?;
        let mut archive = ZipArchive::new(BufReader::new(file))?;
        let clean = key.strip_suffix(".npy").unwrap_or(key);
        let entry_name = format!("{clean}.npy");
        let mut entry = match archive.by_name(&entry_name) {
            Ok(f) => f,
            Err(zip::result::ZipError::FileNotFound) => {
                return Err(CoreError::NotFound(format!(
                    "NPZ entry '{entry_name}' not found in {}",
                    path.display()
                )));
            }
            Err(e) => return Err(CoreError::Zip(e)),
        };
        NpyArray::from_npy_reader(&mut entry)
    }

    pub fn list_keys_from_file(path: &Path) -> CoreResult<Vec<String>> {
        let file = File::open(path)?;
        let archive = ZipArchive::new(BufReader::new(file))?;
        let mut keys = Vec::with_capacity(archive.len());
        for name_res in archive.file_names() {
            let name = name_res?;
            if let Some(stem) = name.strip_suffix(".npy") {
                keys.push(stem.to_owned());
            }
        }
        keys.sort();
        Ok(keys)
    }

    pub fn write_to<W: Write + Seek>(&self, writer: W) -> CoreResult<()> {
        let mut zw = ZipWriter::new(writer);
        let opts = SimpleFileOptions::default().compression_method(CompressionMethod::Deflated);
        for (name, arr) in &self.arrays {
            let entry_name = if name.ends_with(".npy") {
                name.clone()
            } else {
                format!("{name}.npy")
            };
            zw.start_file(entry_name, opts)?;
            arr.write_npy(&mut zw)?;
        }
        zw.finish()?;
        Ok(())
    }

    pub fn to_bytes(&self) -> CoreResult<Vec<u8>> {
        let mut cursor = Cursor::new(Vec::new());
        self.write_to(&mut cursor)?;
        Ok(cursor.into_inner())
    }

    pub fn write_file_atomic(&self, path: &Path) -> CoreResult<()> {
        let bytes = self.to_bytes()?;
        atomic_io::write_bytes_atomic(path, &bytes)
    }

    pub fn validate_npz_file(path: &Path) -> bool {
        let Ok(file) = File::open(path) else {
            return false;
        };
        let Ok(mut archive) = ZipArchive::new(BufReader::new(file)) else {
            return false;
        };
        if archive.is_empty() {
            return false;
        }
        let mut entry_count = 0usize;
        for idx in 0..archive.len() {
            let Ok(mut entry) = archive.by_index(idx) else {
                return false;
            };
            if entry.is_dir() {
                continue;
            }
            let Ok(name) = entry.name() else {
                return false;
            };
            if !name.ends_with(".npy") {
                return false;
            }
            let mut magic = [0u8; 6];
            if entry.read_exact(&mut magic).is_err() || &magic != NPY_MAGIC {
                return false;
            }
            entry_count += 1;
        }
        entry_count > 0
    }
}

fn read_exact_bounded<R: Read>(reader: R, expected_bytes: usize) -> CoreResult<Vec<u8>> {
    if expected_bytes == 0 {
        return Ok(Vec::new());
    }

    let mut limited = reader.take(expected_bytes as u64);
    let mut out = Vec::new();
    let initial_cap = expected_bytes.min(NPY_READ_CHUNK_BYTES);
    out.try_reserve(initial_cap).map_err(|e| {
        CoreError::InvalidData(format!(
            "NPY payload allocation failed ({expected_bytes} bytes): {e}"
        ))
    })?;

    let mut chunk = [0u8; NPY_READ_CHUNK_BYTES];
    while out.len() < expected_bytes {
        let to_read = (expected_bytes - out.len()).min(NPY_READ_CHUNK_BYTES);
        let n = match limited.read(&mut chunk[..to_read]) {
            Ok(n) => n,
            Err(e) if e.kind() == std::io::ErrorKind::Interrupted => continue,
            Err(e) => return Err(CoreError::Io(e)),
        };
        if n == 0 {
            return Err(CoreError::Io(std::io::Error::new(
                std::io::ErrorKind::UnexpectedEof,
                format!(
                    "truncated NPY payload: expected {expected_bytes} bytes, got {}",
                    out.len()
                ),
            )));
        }
        out.try_reserve(n).map_err(|e| {
            CoreError::InvalidData(format!(
                "NPY payload allocation failed while reading ({expected_bytes} bytes): {e}"
            ))
        })?;
        out.extend_from_slice(&chunk[..n]);
    }
    Ok(out)
}

fn checked_shape_product(shape: &[usize]) -> CoreResult<usize> {
    shape.iter().try_fold(1usize, |acc, &dim| {
        acc.checked_mul(dim)
            .ok_or_else(|| CoreError::InvalidArgument(format!("shape {shape:?} overflows usize")))
    })
}

fn format_shape_tuple(shape: &[usize]) -> String {
    match shape {
        [] => "()".to_owned(),
        [single] => format!("({single},)"),
        many => {
            let joined: Vec<String> = many.iter().map(usize::to_string).collect();
            format!("({})", joined.join(", "))
        }
    }
}

fn parse_npy_header(header: &str) -> CoreResult<(NpyDType, bool, Vec<usize>)> {
    let trimmed = header.trim();
    let inner = trimmed
        .strip_prefix('{')
        .and_then(|s| s.strip_suffix('}'))
        .ok_or_else(|| CoreError::InvalidData("NPY header is not a dict literal".to_owned()))?;

    let descr_raw = extract_dict_string(inner, "descr")
        .ok_or_else(|| CoreError::InvalidData("NPY header missing 'descr'".to_owned()))?;
    let dtype = NpyDType::from_descr(&descr_raw)?;

    let fortran_raw = extract_dict_ident(inner, "fortran_order")
        .ok_or_else(|| CoreError::InvalidData("NPY header missing 'fortran_order'".to_owned()))?;
    let fortran_order = match fortran_raw.as_str() {
        "False" => false,
        "True" => true,
        other => {
            return Err(CoreError::InvalidData(format!(
                "invalid NPY fortran_order '{other}'"
            )));
        }
    };

    let shape = extract_dict_shape(inner)?;
    Ok((dtype, fortran_order, shape))
}

fn find_key_colon<'a>(body: &'a str, key: &str) -> Option<&'a str> {
    let bytes = body.as_bytes();
    let mut i = 0usize;
    while i < bytes.len() {
        let q = bytes[i];
        if q == b'\'' || q == b'"' {
            let start = i + 1;
            let mut j = start;
            while j < bytes.len() && bytes[j] != q {
                if bytes[j] == b'\\' {
                    j += 2;
                } else {
                    j += 1;
                }
            }
            if j < bytes.len() && &body[start..j] == key {
                let rest = body[j + 1..].trim_start();
                if let Some(after_colon) = rest.strip_prefix(':') {
                    return Some(after_colon.trim_start());
                }
            }
            i = j + 1;
        } else {
            i += 1;
        }
    }
    None
}

fn extract_dict_string(body: &str, key: &str) -> Option<String> {
    let after = find_key_colon(body, key)?;
    let quote = after.chars().next()?;
    if quote != '\'' && quote != '"' {
        return None;
    }
    let rest = &after[1..];
    let end = rest.find(quote)?;
    Some(rest[..end].to_owned())
}

fn extract_dict_ident(body: &str, key: &str) -> Option<String> {
    let after = find_key_colon(body, key)?;
    let token: String = after
        .chars()
        .take_while(|c| c.is_ascii_alphanumeric() || *c == '_')
        .collect();
    if token.is_empty() {
        None
    } else {
        Some(token)
    }
}

fn extract_dict_shape(body: &str) -> CoreResult<Vec<usize>> {
    let after = find_key_colon(body, "shape")
        .ok_or_else(|| CoreError::InvalidData("NPY header missing 'shape'".to_owned()))?;
    let after_open = after
        .strip_prefix('(')
        .ok_or_else(|| CoreError::InvalidData("NPY 'shape' is not a tuple".to_owned()))?;
    let close_idx = after_open
        .find(')')
        .ok_or_else(|| CoreError::InvalidData("unclosed NPY 'shape' tuple".to_owned()))?;
    let inside = after_open[..close_idx].trim();
    if inside.is_empty() {
        return Ok(Vec::new());
    }
    let mut dims = Vec::new();
    for part in inside.split(',') {
        let token = part.trim();
        if token.is_empty() {
            continue;
        }
        let dim = token.parse::<usize>().map_err(|e| {
            CoreError::InvalidData(format!("invalid dimension '{token}' in NPY shape: {e}"))
        })?;
        dims.push(dim);
    }
    Ok(dims)
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::path::PathBuf;
    use std::sync::atomic::{AtomicUsize, Ordering};

    static TEST_SEQ: AtomicUsize = AtomicUsize::new(0);

    fn make_temp_dir(label: &str) -> PathBuf {
        let seq = TEST_SEQ.fetch_add(1, Ordering::Relaxed);
        let dir = std::env::temp_dir().join(format!(
            "gh_core_npz_{label}_{}_{seq}",
            std::process::id()
        ));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).expect("create temp dir");
        dir
    }

    #[test]
    fn roundtrip_all_dtypes_and_shapes_including_0d_scalars() {
        let mut archive = NpzArchive::new();
        archive.insert("scalar_ver", NpyArray::scalar_i32(3));
        archive.insert("scalar_scale", NpyArray::scalar_f32(1.5));
        archive.insert(
            "vec_i64",
            NpyArray::from_i64(vec![5], vec![10, -20, 30, 40, 50]).expect("vec_i64"),
        );
        archive.insert(
            "mat_u8",
            NpyArray::from_u8(vec![3, 4], (0u8..12).collect()).expect("mat_u8"),
        );
        archive.insert(
            "tensor_f32",
            NpyArray::from_f32(vec![2, 3, 4], (0..24).map(|x| x as f32 * 0.25).collect())
                .expect("tensor_f32"),
        );

        let bytes = archive.to_bytes().expect("serialize npz");
        let parsed = NpzArchive::read_from(Cursor::new(&bytes)).expect("read npz");

        assert_eq!(parsed.len(), 5);
        assert_eq!(
            parsed.get("scalar_ver").unwrap().to_scalar_i32().unwrap(),
            3
        );
        assert_eq!(parsed.get("scalar_ver").unwrap().shape(), &[] as &[usize]);
        assert!(
            (parsed.get("scalar_scale").unwrap().to_scalar_f32().unwrap() - 1.5).abs() < 1e-6
        );
        assert_eq!(
            parsed.get("vec_i64").unwrap().as_i64().unwrap(),
            &[10, -20, 30, 40, 50]
        );
        assert_eq!(parsed.get("mat_u8").unwrap().shape(), &[3, 4]);
        assert_eq!(parsed.get("tensor_f32").unwrap().shape(), &[2, 3, 4]);

        let base = make_temp_dir("roundtrip");
        let path = base.join("sample.npz");
        archive.write_file_atomic(&path).expect("write_file_atomic");
        assert!(NpzArchive::validate_npz_file(&path));

        let single = NpzArchive::read_single_from_file(&path, "vec_i64").expect("single read");
        assert_eq!(single.as_i64().unwrap(), &[10, -20, 30, 40, 50]);

        let _ = std::fs::remove_dir_all(&base);
    }

    #[test]
    fn parses_synthetic_v2_and_v3_headers() {
        for major in [2u8, 3u8] {
            let dict = "{'descr': '<i4', 'fortran_order': False, 'shape': (2,), }";
            let pad_len = (64 - ((12 + dict.len() + 1) % 64)) % 64;
            let header_len = (dict.len() + 1 + pad_len) as u32;
            let mut raw = Vec::new();
            raw.extend_from_slice(NPY_MAGIC);
            raw.extend_from_slice(&[major, 0]);
            raw.extend_from_slice(&header_len.to_le_bytes());
            raw.extend_from_slice(dict.as_bytes());
            raw.resize(raw.len() + pad_len, b' ');
            raw.push(b'\n');
            raw.extend_from_slice(&42i32.to_le_bytes());
            raw.extend_from_slice(&(-7i32).to_le_bytes());

            let arr = NpyArray::from_npy_bytes(&raw).expect("parse v2/v3 npy");
            assert_eq!(arr.shape(), &[2]);
            assert_eq!(arr.as_i32().unwrap(), &[42, -7]);
        }
    }

    #[test]
    fn rejects_invalid_magic_dtypes_fortran_and_truncated_payload() {
        assert!(NpyArray::from_npy_bytes(b"NOTNPY\x01\x00").is_err());

        let make_hdr = |dict: &str, payload: &[u8]| {
            let mut raw = Vec::new();
            raw.extend_from_slice(NPY_MAGIC);
            raw.extend_from_slice(&[1, 0]);
            raw.extend_from_slice(&(dict.len() as u16).to_le_bytes());
            raw.extend_from_slice(dict.as_bytes());
            raw.extend_from_slice(payload);
            raw
        };

        // Big-endian
        assert!(
            NpyArray::from_npy_bytes(&make_hdr(
                "{'descr': '>f4', 'fortran_order': False, 'shape': (1,)}\n",
                &[0; 4]
            ))
            .is_err()
        );
        // Unsupported dtype
        assert!(
            NpyArray::from_npy_bytes(&make_hdr(
                "{'descr': '<f8', 'fortran_order': False, 'shape': (1,)}\n",
                &[0; 8]
            ))
            .is_err()
        );
        // 2D Fortran order
        assert!(
            NpyArray::from_npy_bytes(&make_hdr(
                "{'descr': '|u1', 'fortran_order': True, 'shape': (2, 2)}\n",
                &[0; 4]
            ))
            .is_err()
        );
        // Truncated payload
        assert!(
            NpyArray::from_npy_bytes(&make_hdr(
                "{'descr': '<i4', 'fortran_order': False, 'shape': (2,)}\n",
                &[0; 4]
            ))
            .is_err()
        );
    }

    #[test]
    fn rejects_huge_non_overflowing_shape_without_oom_abort() {
        let header = b"{'descr': '<f4', 'fortran_order': False, 'shape': (100000000000, 1)}\n";
        let mut raw = Vec::with_capacity(10 + header.len() + 32);
        raw.extend_from_slice(NPY_MAGIC);
        raw.extend_from_slice(&[1, 0]);
        raw.extend_from_slice(&(header.len() as u16).to_le_bytes());
        raw.extend_from_slice(header);
        raw.extend_from_slice(&[0u8; 32]);

        let err = NpyArray::from_npy_bytes(&raw)
            .expect_err("huge-shape truncated NPY must return Err without OOM abort");
        assert!(
            matches!(err, CoreError::Io(ref io) if io.kind() == std::io::ErrorKind::UnexpectedEof),
            "unexpected error variant: {err:?}"
        );
    }

    #[test]
    fn cross_reads_real_data_npz_fixtures_when_present() {
        let data_dir = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .join("..")
            .join("..")
            .join("..")
            .join("data");
        if !data_dir.is_dir() {
            return;
        }

        let terrain_path = data_dir.join("terrain_n2_v4.npz");
        if terrain_path.is_file() {
            assert!(NpzArchive::validate_npz_file(&terrain_path));
            let map2_gray =
                NpzArchive::read_single_from_file(&terrain_path, "map_2_gray").expect("map_2_gray");
            assert_eq!(map2_gray.dtype(), NpyDType::U8);
            assert_eq!(map2_gray.shape(), &[5120, 5632]);

            let map2_meta =
                NpzArchive::read_single_from_file(&terrain_path, "map_2_meta").expect("map_2_meta");
            assert_eq!(map2_meta.dtype(), NpyDType::I64);
            assert_eq!(map2_meta.shape(), &[5]);
        }

        let cave_path = data_dir.join("cave_masks_v4.npz");
        if cave_path.is_file() {
            let caves = NpzArchive::read_file(&cave_path).expect("cave_masks_v4.npz");
            assert_eq!(caves.len(), 356);
            for arr in caves.arrays.values() {
                assert_eq!(arr.dtype(), NpyDType::U8);
                assert_eq!(arr.shape().len(), 2);
            }
        }

        let ui_path = data_dir.join("ui_anchors_v4.npz");
        if ui_path.is_file() {
            let ui = NpzArchive::read_file(&ui_path).expect("ui_anchors_v4.npz");
            let tpl_gear = ui.get("tpl_gear").expect("tpl_gear");
            assert_eq!(tpl_gear.dtype(), NpyDType::U8);
            assert_eq!(tpl_gear.shape(), &[56, 56, 3]);

            let gear_ratio = ui.get("gear_ratio").expect("gear_ratio");
            assert_eq!(gear_ratio.dtype(), NpyDType::F32);
            assert_eq!(gear_ratio.shape(), &[2]);
        }
    }

    #[test]
    fn list_keys_from_file_matches_read_file_keys() {
        let tmp = std::env::temp_dir().join(format!("gh_core_npz_keys_{}.npz", std::process::id()));
        let mut npz = NpzArchive::new();
        npz.insert("zeta", NpyArray::from_u8(vec![2], vec![1, 2]).unwrap());
        npz.insert("alpha.npy", NpyArray::scalar_i32(42));
        npz.insert("map_2_gray", NpyArray::from_u8(vec![1, 1], vec![7]).unwrap());
        npz.write_file_atomic(&tmp).unwrap();

        let listed = NpzArchive::list_keys_from_file(&tmp).expect("list_keys_from_file");
        let full = NpzArchive::read_file(&tmp).expect("read_file");
        let expected: Vec<String> = full.keys().map(str::to_owned).collect();
        assert_eq!(listed, expected);
        assert_eq!(listed, vec!["alpha", "map_2_gray", "zeta"]);

        let _ = std::fs::remove_file(&tmp);
    }
}
