//! Core Fathom model inspection, capability routing, and narrow local runtime support.
//!
//! Public capability states stay separate from internal readiness scaffolding so
//! metadata-readable packages never imply runnable inference.

use candle_core::{DType, Device, IndexOp, Tensor, D};
use candle_nn::{embedding, layer_norm, ops, Embedding, LayerNorm, Module, VarBuilder};
use candle_transformers::models::{
    gemma::{Config as GemmaRuntimeConfig, Model as GemmaModel},
    llama::{Config as CandleLlamaRuntimeConfig, Llama},
    mistral::{Config as CandleMistralRuntimeConfig, Model as MistralModel},
    phi::{Config as PhiRuntimeConfig, Model as PhiModel},
    qwen2::ModelForCausalLM as Qwen2ModelForCausalLM,
};
use serde::{Deserialize, Serialize};
use std::{
    cmp::Ordering,
    collections::{BTreeMap, BTreeSet, BinaryHeap, HashMap, HashSet},
    fs,
    path::{Path, PathBuf},
    sync::{Arc, Mutex, MutexGuard, OnceLock},
    time::{Instant, SystemTime},
};

const GGUF_LLAMA3_BPE_EXPECTED_VOCAB_SIZE: usize = 128_256;

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub enum ModelFormat {
    Gguf,
    SafeTensors,
    SafeTensorsIndex,
    PyTorchBin,
    Onnx,
    Mlx,
    CoreMl,
    TensorRtPlan,
    TokenizerJson,
    TokenizerConfigJson,
    SentencePiece,
    ConfigJson,
    ChatTemplate,
    Unknown,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub enum SupportLevel {
    Detected,
    MetadataReadable,
    LoadPlanned,
    Runnable,
    Blocked,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelArtifact {
    pub path: PathBuf,
    pub format: ModelFormat,
    pub support: SupportLevel,
    pub runnable_today: bool,
    pub notes: Vec<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub gguf_metadata: Option<GgufMetadataSummary>,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct GgufMetadataSummary {
    pub version: u32,
    pub tensor_count: u64,
    pub metadata_kv_count: u64,
    pub metadata: Vec<GgufMetadataEntry>,
    #[serde(default)]
    pub tensors: Vec<GgufTensorInfoSummary>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub tensor_summary: Option<GgufTensorAggregateSummary>,
    #[serde(default)]
    pub hints: GgufMetadataHints,
    #[serde(default)]
    pub tokenizer_summary: GgufTokenizerSummary,
    #[serde(default)]
    pub architecture_summary: GgufArchitectureSummary,
    #[serde(default)]
    pub compatibility: GgufCompatibilitySummary,
    #[serde(default, skip)]
    pub tokenizer_spec: Option<GgufTokenizerSpec>,
    #[serde(default, skip)]
    pub payload_ranges: Vec<GgufTensorPayloadRange>,
    #[serde(default, skip)]
    pub payload_range_status: GgufTensorPayloadRangeStatus,
    pub notes: Vec<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct GgufTensorPayloadRange {
    pub name: String,
    pub ggml_type: u32,
    pub ggml_type_name: String,
    pub shape: Vec<u64>,
    pub element_count: u64,
    pub relative_offset: u64,
    pub absolute_start: u64,
    pub absolute_end: u64,
    pub byte_len: u64,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum GgufTensorPayloadRangeStatus {
    Ready,
    #[default]
    Empty,
    UnsupportedTypesPresent,
    PayloadBudgetExceeded {
        requested: u64,
        limit: u64,
    },
}

#[allow(dead_code)]
#[derive(Debug, Clone, PartialEq, Eq)]
enum GgufPayloadReadError {
    EmptyRange {
        name: String,
    },
    RangeLengthMismatch {
        name: String,
        start: u64,
        end: u64,
        byte_len: u64,
    },
    UnknownGgmlTypeForPayload {
        name: String,
        tag: u32,
    },
    PayloadByteLengthMismatch {
        name: String,
        expected: u64,
        actual: u64,
    },
    PerReadBudgetExceeded {
        name: String,
        requested: u64,
        limit: u64,
    },
    FileTruncatedOrMutated {
        name: String,
        required_end: u64,
        file_len: u64,
    },
    Io {
        message: String,
    },
}

impl std::fmt::Display for GgufPayloadReadError {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::EmptyRange { name } => write!(formatter, "GGUF tensor {name} has an empty payload range"),
            Self::RangeLengthMismatch { name, start, end, byte_len } => write!(
                formatter,
                "GGUF tensor {name} range {start}..{end} does not match byte_len {byte_len}"
            ),
            Self::UnknownGgmlTypeForPayload { name, tag } => write!(
                formatter,
                "GGUF tensor {name} uses unknown GGML type tag {tag}; payload bytes are not readable"
            ),
            Self::PayloadByteLengthMismatch { name, expected, actual } => write!(
                formatter,
                "GGUF tensor {name} payload length mismatch: expected {expected} bytes, got {actual}"
            ),
            Self::PerReadBudgetExceeded { name, requested, limit } => write!(
                formatter,
                "GGUF tensor {name} payload read of {requested} bytes exceeds per-read budget {limit}"
            ),
            Self::FileTruncatedOrMutated { name, required_end, file_len } => write!(
                formatter,
                "GGUF tensor {name} requires bytes through {required_end}, but file length is now {file_len}"
            ),
            Self::Io { message } => write!(formatter, "GGUF payload read failed: {message}"),
        }
    }
}

impl std::error::Error for GgufPayloadReadError {}

#[allow(dead_code)]
#[derive(Debug, Clone, PartialEq, Eq)]
enum GgufPayloadDecodeError {
    UnknownGgmlType {
        name: String,
        tag: u32,
    },
    UnsupportedGgmlType {
        name: String,
        tag: u32,
        type_name: String,
    },
    WrongLength {
        name: String,
        expected: usize,
        actual: usize,
    },
    ElementCountTooLarge {
        name: String,
        element_count: u64,
    },
}

impl std::fmt::Display for GgufPayloadDecodeError {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::UnknownGgmlType { name, tag } => write!(
                formatter,
                "GGUF tensor {name} uses unknown GGML type tag {tag}; no f32 decode is available"
            ),
            Self::UnsupportedGgmlType { name, tag, type_name } => write!(
                formatter,
                "GGUF tensor {name} uses unsupported GGML type {type_name} ({tag}); no f32 decode is available"
            ),
            Self::WrongLength { name, expected, actual } => write!(
                formatter,
                "GGUF tensor {name} payload decode length mismatch: expected {expected} bytes, got {actual}"
            ),
            Self::ElementCountTooLarge { name, element_count } => write!(
                formatter,
                "GGUF tensor {name} element count {element_count} is too large to decode in memory"
            ),
        }
    }
}

impl std::error::Error for GgufPayloadDecodeError {}

#[allow(dead_code)]
fn read_validated_gguf_payload_range(
    path: impl AsRef<Path>,
    range: &GgufTensorPayloadRange,
    max_bytes: u64,
) -> Result<Vec<u8>, GgufPayloadReadError> {
    if range.byte_len == 0 {
        return Err(GgufPayloadReadError::EmptyRange {
            name: range.name.clone(),
        });
    }
    let computed_len = range
        .absolute_end
        .checked_sub(range.absolute_start)
        .ok_or_else(|| GgufPayloadReadError::RangeLengthMismatch {
            name: range.name.clone(),
            start: range.absolute_start,
            end: range.absolute_end,
            byte_len: range.byte_len,
        })?;
    if computed_len != range.byte_len {
        return Err(GgufPayloadReadError::RangeLengthMismatch {
            name: range.name.clone(),
            start: range.absolute_start,
            end: range.absolute_end,
            byte_len: range.byte_len,
        });
    }
    let Some(expected_bytes) =
        gguf_payload_estimate_tensor_bytes(range.ggml_type, range.element_count)
    else {
        return Err(GgufPayloadReadError::UnknownGgmlTypeForPayload {
            name: range.name.clone(),
            tag: range.ggml_type,
        });
    };
    if expected_bytes != range.byte_len {
        return Err(GgufPayloadReadError::PayloadByteLengthMismatch {
            name: range.name.clone(),
            expected: expected_bytes,
            actual: range.byte_len,
        });
    }
    if range.byte_len > max_bytes {
        return Err(GgufPayloadReadError::PerReadBudgetExceeded {
            name: range.name.clone(),
            requested: range.byte_len,
            limit: max_bytes,
        });
    }
    let path = path.as_ref();
    let file_len = fs::metadata(path)
        .map_err(|error| GgufPayloadReadError::Io {
            message: error.to_string(),
        })?
        .len();
    if range.absolute_end > file_len {
        return Err(GgufPayloadReadError::FileTruncatedOrMutated {
            name: range.name.clone(),
            required_end: range.absolute_end,
            file_len,
        });
    }
    let read_len = usize::try_from(range.byte_len).map_err(|_| {
        GgufPayloadReadError::PerReadBudgetExceeded {
            name: range.name.clone(),
            requested: range.byte_len,
            limit: max_bytes,
        }
    })?;
    let mut file = fs::File::open(path).map_err(|error| GgufPayloadReadError::Io {
        message: error.to_string(),
    })?;
    std::io::Seek::seek(&mut file, std::io::SeekFrom::Start(range.absolute_start)).map_err(
        |error| GgufPayloadReadError::Io {
            message: error.to_string(),
        },
    )?;
    let mut bytes = vec![0u8; read_len];
    std::io::Read::read_exact(&mut file, &mut bytes).map_err(|error| {
        if error.kind() == std::io::ErrorKind::UnexpectedEof {
            GgufPayloadReadError::FileTruncatedOrMutated {
                name: range.name.clone(),
                required_end: range.absolute_end,
                file_len: fs::metadata(path)
                    .map(|metadata| metadata.len())
                    .unwrap_or(0),
            }
        } else {
            GgufPayloadReadError::Io {
                message: error.to_string(),
            }
        }
    })?;
    if bytes.len() as u64 != range.byte_len {
        return Err(GgufPayloadReadError::PayloadByteLengthMismatch {
            name: range.name.clone(),
            expected: range.byte_len,
            actual: bytes.len() as u64,
        });
    }
    let post_read_len = fs::metadata(path)
        .map_err(|error| GgufPayloadReadError::Io {
            message: error.to_string(),
        })?
        .len();
    if range.absolute_end > post_read_len || post_read_len != file_len {
        return Err(GgufPayloadReadError::FileTruncatedOrMutated {
            name: range.name.clone(),
            required_end: range.absolute_end,
            file_len: post_read_len,
        });
    }
    Ok(bytes)
}

#[allow(dead_code)]
fn decode_gguf_payload_to_f32(
    raw: &[u8],
    range: &GgufTensorPayloadRange,
) -> Result<Vec<f32>, GgufPayloadDecodeError> {
    match range.ggml_type {
        0 => decode_gguf_f32_payload_to_f32(raw, range),
        1 => decode_gguf_f16_payload_to_f32(raw, range),
        2 => decode_gguf_q4_0_payload_to_f32(raw, range),
        8 => decode_gguf_q8_0_payload_to_f32(raw, range),
        tag if gguf_payload_type_block(tag).is_none() => {
            Err(GgufPayloadDecodeError::UnknownGgmlType {
                name: range.name.clone(),
                tag,
            })
        }
        tag => Err(GgufPayloadDecodeError::UnsupportedGgmlType {
            name: range.name.clone(),
            tag,
            type_name: range.ggml_type_name.clone(),
        }),
    }
}

#[allow(dead_code)]
fn decode_gguf_f32_payload_to_f32(
    raw: &[u8],
    range: &GgufTensorPayloadRange,
) -> Result<Vec<f32>, GgufPayloadDecodeError> {
    let elements = usize::try_from(range.element_count).map_err(|_| {
        GgufPayloadDecodeError::ElementCountTooLarge {
            name: range.name.clone(),
            element_count: range.element_count,
        }
    })?;
    let expected =
        elements
            .checked_mul(4)
            .ok_or_else(|| GgufPayloadDecodeError::ElementCountTooLarge {
                name: range.name.clone(),
                element_count: range.element_count,
            })?;
    if raw.len() != expected {
        return Err(GgufPayloadDecodeError::WrongLength {
            name: range.name.clone(),
            expected,
            actual: raw.len(),
        });
    }
    Ok(raw
        .chunks_exact(4)
        .map(|bytes| f32::from_le_bytes([bytes[0], bytes[1], bytes[2], bytes[3]]))
        .collect())
}

#[allow(dead_code)]
fn decode_gguf_f16_payload_to_f32(
    raw: &[u8],
    range: &GgufTensorPayloadRange,
) -> Result<Vec<f32>, GgufPayloadDecodeError> {
    let elements = usize::try_from(range.element_count).map_err(|_| {
        GgufPayloadDecodeError::ElementCountTooLarge {
            name: range.name.clone(),
            element_count: range.element_count,
        }
    })?;
    let expected =
        elements
            .checked_mul(2)
            .ok_or_else(|| GgufPayloadDecodeError::ElementCountTooLarge {
                name: range.name.clone(),
                element_count: range.element_count,
            })?;
    if raw.len() != expected {
        return Err(GgufPayloadDecodeError::WrongLength {
            name: range.name.clone(),
            expected,
            actual: raw.len(),
        });
    }
    Ok(raw
        .chunks_exact(2)
        .map(|bytes| f16_bits_to_f32(u16::from_le_bytes([bytes[0], bytes[1]])))
        .collect())
}

#[allow(dead_code)]
fn decode_gguf_q8_0_payload_to_f32(
    raw: &[u8],
    range: &GgufTensorPayloadRange,
) -> Result<Vec<f32>, GgufPayloadDecodeError> {
    // GGML Q8_0 block layout (ggml type tag 8): 32 output elements per block.
    // Each block is 34 bytes: little-endian IEEE F16 scale `d`, followed by 32
    // signed i8 quantized values. Dequantization is `output[i] = d * qs[i]`.
    // GGUF payload bytes are little-endian; partial final tensor blocks still
    // occupy a full raw block and are truncated to `element_count` outputs.
    const BLOCK_ELEMENTS: usize = 32;
    const BLOCK_BYTES: usize = 34;
    let elements = checked_gguf_decode_elements(range)?;
    let expected = checked_gguf_quantized_raw_len(range, BLOCK_ELEMENTS, BLOCK_BYTES)?;
    if raw.len() != expected {
        return Err(GgufPayloadDecodeError::WrongLength {
            name: range.name.clone(),
            expected,
            actual: raw.len(),
        });
    }
    let mut decoded = Vec::with_capacity(elements);
    for block in raw.chunks_exact(BLOCK_BYTES) {
        let scale = f16_bits_to_f32(u16::from_le_bytes([block[0], block[1]]));
        for quantized in &block[2..] {
            if decoded.len() == elements {
                break;
            }
            decoded.push(scale * (*quantized as i8 as f32));
        }
    }
    Ok(decoded)
}

#[allow(dead_code)]
fn decode_gguf_q4_0_payload_to_f32(
    raw: &[u8],
    range: &GgufTensorPayloadRange,
) -> Result<Vec<f32>, GgufPayloadDecodeError> {
    // GGML Q4_0 block layout (ggml type tag 2): 32 output elements per block.
    // Each block is 18 bytes: little-endian IEEE F16 scale `d`, followed by 16
    // bytes holding two 4-bit values each. Nibbles are unsigned values biased by
    // -8. For byte `qs[j]`, the low nibble is output `j` and the high nibble is
    // output `j + 16`; dequantization is `d * (nibble - 8)`. GGUF payload bytes
    // are little-endian; partial final tensor blocks still occupy a full raw
    // block and are truncated to `element_count` outputs.
    const BLOCK_ELEMENTS: usize = 32;
    const BLOCK_BYTES: usize = 18;
    let elements = checked_gguf_decode_elements(range)?;
    let expected = checked_gguf_quantized_raw_len(range, BLOCK_ELEMENTS, BLOCK_BYTES)?;
    if raw.len() != expected {
        return Err(GgufPayloadDecodeError::WrongLength {
            name: range.name.clone(),
            expected,
            actual: raw.len(),
        });
    }
    let mut decoded = Vec::with_capacity(elements);
    for block in raw.chunks_exact(BLOCK_BYTES) {
        let scale = f16_bits_to_f32(u16::from_le_bytes([block[0], block[1]]));
        let qs = &block[2..];
        let remaining = elements - decoded.len();
        let mut block_values = [0f32; BLOCK_ELEMENTS];
        for (index, packed) in qs.iter().enumerate() {
            block_values[index] = scale * (((packed & 0x0f) as i8 - 8) as f32);
            block_values[index + 16] = scale * (((packed >> 4) as i8 - 8) as f32);
        }
        decoded.extend_from_slice(&block_values[..remaining.min(BLOCK_ELEMENTS)]);
    }
    Ok(decoded)
}

fn checked_gguf_decode_elements(
    range: &GgufTensorPayloadRange,
) -> Result<usize, GgufPayloadDecodeError> {
    usize::try_from(range.element_count).map_err(|_| GgufPayloadDecodeError::ElementCountTooLarge {
        name: range.name.clone(),
        element_count: range.element_count,
    })
}

fn checked_gguf_quantized_raw_len(
    range: &GgufTensorPayloadRange,
    block_elements: usize,
    block_bytes: usize,
) -> Result<usize, GgufPayloadDecodeError> {
    let elements = checked_gguf_decode_elements(range)?;
    let blocks = elements.checked_add(block_elements - 1).ok_or_else(|| {
        GgufPayloadDecodeError::ElementCountTooLarge {
            name: range.name.clone(),
            element_count: range.element_count,
        }
    })? / block_elements;
    blocks
        .checked_mul(block_bytes)
        .ok_or_else(|| GgufPayloadDecodeError::ElementCountTooLarge {
            name: range.name.clone(),
            element_count: range.element_count,
        })
}

#[allow(dead_code)]
fn gguf_payload_type_block(raw: u32) -> Option<(u64, u64)> {
    Some(match raw {
        0 => (1, 4),
        1 => (1, 2),
        2 => (32, 18),
        3 => (32, 20),
        6 => (32, 22),
        7 => (32, 24),
        8 => (32, 34),
        9 => (32, 40),
        10 => (256, 84),
        11 => (256, 110),
        12 => (256, 144),
        13 => (256, 176),
        14 => (256, 210),
        15 => (256, 292),
        16 => (256, 66),
        17 => (256, 74),
        18 => (256, 98),
        19 => (256, 34),
        20 => (32, 18),
        21 => (256, 110),
        22 => (256, 82),
        23 => (256, 136),
        24 => (1, 1),
        25 => (1, 2),
        26 => (1, 4),
        27 => (1, 8),
        28 => (1, 8),
        29 => (256, 56),
        30 => (1, 2),
        34 => (256, 54),
        35 => (256, 66),
        _ => return None,
    })
}

#[allow(dead_code)]
fn gguf_payload_estimate_tensor_bytes(ggml_type: u32, elements: u64) -> Option<u64> {
    let (block_size, type_size) = gguf_payload_type_block(ggml_type)?;
    let blocks = elements
        .checked_add(block_size - 1)?
        .checked_div(block_size)?;
    blocks.checked_mul(type_size)
}

#[allow(dead_code)]
fn f16_bits_to_f32(bits: u16) -> f32 {
    let sign = ((bits & 0x8000) as u32) << 16;
    let exponent = (bits >> 10) & 0x1f;
    let mantissa = (bits & 0x03ff) as u32;
    let f32_bits = match exponent {
        0 => {
            if mantissa == 0 {
                sign
            } else {
                let mut mantissa = mantissa;
                let mut exponent = -14i32;
                while (mantissa & 0x0400) == 0 {
                    mantissa <<= 1;
                    exponent -= 1;
                }
                mantissa &= 0x03ff;
                sign | (((exponent + 127) as u32) << 23) | (mantissa << 13)
            }
        }
        0x1f => sign | 0x7f80_0000 | (mantissa << 13),
        exponent => sign | (((exponent as i32 - 15 + 127) as u32) << 23) | (mantissa << 13),
    };
    f32::from_bits(f32_bits)
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq, Default)]
pub struct GgufMetadataHints {
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub architecture: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub context_length: Option<u64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub tokenizer_model: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub tokenizer_token_count: Option<u64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub file_type: Option<u64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub alignment: Option<u64>,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq, Default)]
pub struct GgufTokenizerSummary {
    pub status: GgufTokenizerMetadataStatus,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub model: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub token_count: Option<u64>,
    #[serde(default)]
    pub token_samples: Vec<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub merge_count: Option<u64>,
    #[serde(default)]
    pub merge_samples: Vec<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub added_token_count: Option<u64>,
    #[serde(default)]
    pub special_token_ids: BTreeMap<String, u64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub chat_template: Option<GgufBoundedTextSummary>,
    #[serde(default)]
    pub notes: Vec<String>,
}

#[derive(Debug, Clone, PartialEq)]
pub struct GgufTokenizerSpec {
    pub family: GgufTokenizerSpecFamily,
    pub model: String,
    pub tokens: Vec<String>,
    pub merges: Vec<String>,
    pub scores: Vec<f32>,
    pub token_types: Vec<i32>,
    pub special_token_ids: BTreeMap<String, u64>,
    pub has_byte_fallback: bool,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum GgufTokenizerSpecFamily {
    SyntheticGpt2ByteLevelBpe,
    Llama3Bpe,
    LlamaSentencePiece,
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq, Default)]
#[serde(rename_all = "snake_case")]
pub enum GgufTokenizerMetadataStatus {
    Present,
    Partial,
    #[default]
    Missing,
    UnsupportedShape,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct GgufBoundedTextSummary {
    pub byte_len: u64,
    pub hash: String,
    pub preview: String,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Default)]
pub struct GgufArchitectureSummary {
    pub status: GgufArchitectureMetadataStatus,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub architecture: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub family: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub context_length: Option<u64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub embedding_length: Option<u64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub block_count: Option<u64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub attention_head_count: Option<u64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub attention_kv_head_count: Option<u64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub feed_forward_length: Option<u64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub rope_dimension_count: Option<u64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub rope_freq_base: Option<f64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub file_type: Option<u64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub quantization_hint: Option<String>,
    #[serde(default)]
    pub notes: Vec<String>,
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq, Default)]
#[serde(rename_all = "snake_case")]
pub enum GgufArchitectureMetadataStatus {
    Recognized,
    Partial,
    #[default]
    Unknown,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq, Default)]
pub struct GgufCompatibilitySummary {
    pub metadata_readable: bool,
    pub tokenizer_metadata: GgufTokenizerMetadataStatus,
    pub architecture_metadata: GgufArchitectureMetadataStatus,
    #[serde(default)]
    pub categories: Vec<String>,
    #[serde(default)]
    pub runtime_blockers: Vec<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct GgufTensorInfoSummary {
    pub name: String,
    pub shape: Vec<u64>,
    pub ggml_type: u32,
    pub ggml_type_name: String,
    pub offset: u64,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub absolute_offset: Option<u64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub element_count: Option<u64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub estimated_bytes: Option<u64>,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct GgufTensorAggregateSummary {
    pub described_tensor_count: u64,
    pub tensors_omitted_from_summary: u64,
    pub type_counts: BTreeMap<String, u64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub total_estimated_tensor_bytes: Option<u64>,
    pub unknown_type_tags: Vec<u32>,
    pub largest_tensors: Vec<GgufTensorInfoSummary>,
    pub tensor_data_start: u64,
    pub file_size: u64,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct GgufMetadataEntry {
    pub key: String,
    pub value_type: GgufMetadataValueType,
    pub value: GgufMetadataValueSummary,
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum GgufMetadataValueType {
    Uint8,
    Int8,
    Uint16,
    Int16,
    Uint32,
    Int32,
    Float32,
    Bool,
    String,
    Array,
    Uint64,
    Int64,
    Float64,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum GgufMetadataValueSummary {
    Unsigned {
        value: u64,
    },
    Signed {
        value: i64,
    },
    Float {
        value: f64,
    },
    Bool {
        value: bool,
    },
    String {
        value: String,
    },
    Array {
        element_type: GgufMetadataValueType,
        len: u64,
        preview: Vec<GgufMetadataValueSummary>,
        #[serde(default, skip)]
        full_strings: Option<Vec<String>>,
        #[serde(default, skip)]
        full_floats: Option<Vec<f32>>,
        #[serde(default, skip)]
        full_i32s: Option<Vec<i32>>,
    },
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelPackage {
    pub root: PathBuf,
    pub artifacts: Vec<ModelArtifact>,
    pub tokenizer_files: Vec<PathBuf>,
    pub tokenizer: Option<TokenizerMetadata>,
    pub config_file: Option<PathBuf>,
    pub chat_template_file: Option<PathBuf>,
    pub model_type: Option<String>,
    pub architectures: Vec<String>,
    pub hf_validation: HfPackageValidation,
    pub notes: Vec<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct HfPackageValidation {
    pub has_safetensors_weights: bool,
    pub has_config: bool,
    pub has_tokenizer: bool,
    pub has_chat_template: bool,
    pub missing_required: Vec<String>,
    pub ready_for_loader_metadata: bool,
    pub notes: Vec<String>,
}

impl Default for HfPackageValidation {
    fn default() -> Self {
        Self {
            has_safetensors_weights: false,
            has_config: false,
            has_tokenizer: false,
            has_chat_template: false,
            missing_required: Vec::new(),
            ready_for_loader_metadata: false,
            notes: Vec::new(),
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub enum TokenizerKind {
    HuggingFaceTokenizerJson,
    SentencePiece,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct TokenizerMetadata {
    pub kind: TokenizerKind,
    pub files: Vec<PathBuf>,
    pub bos_token: Option<String>,
    pub eos_token: Option<String>,
    pub pad_token: Option<String>,
    pub unk_token: Option<String>,
    pub chat_template: Option<ChatTemplateMetadata>,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct ChatTemplateMetadata {
    pub source: PathBuf,
    pub template: String,
    pub format: ChatTemplateFormat,
    pub needs_template_engine: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub enum ChatTemplateFormat {
    HuggingFaceJinja,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct ChatMessage {
    pub role: String,
    pub content: String,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct PromptRenderOptions {
    pub add_generation_prompt: bool,
}

pub trait ChatPromptRenderer {
    fn render_chat_prompt(
        &self,
        messages: &[ChatMessage],
        options: &PromptRenderOptions,
    ) -> anyhow::Result<String>;
}

pub struct PlainRolePromptRenderer;

impl ChatPromptRenderer for PlainRolePromptRenderer {
    fn render_chat_prompt(
        &self,
        messages: &[ChatMessage],
        options: &PromptRenderOptions,
    ) -> anyhow::Result<String> {
        render_plain_role_prompt(messages, options)
    }
}

pub fn render_plain_role_prompt(
    messages: &[ChatMessage],
    options: &PromptRenderOptions,
) -> anyhow::Result<String> {
    if messages.is_empty() {
        anyhow::bail!("cannot render a chat prompt with no messages");
    }

    let mut prompt = String::new();
    for message in messages {
        let role = message.role.trim();
        if role.is_empty() {
            anyhow::bail!("chat message role cannot be empty");
        }
        prompt.push_str(role);
        prompt.push_str(": ");
        prompt.push_str(message.content.trim_end());
        prompt.push('\n');
    }
    if options.add_generation_prompt {
        prompt.push_str("assistant: ");
    }
    Ok(prompt)
}

pub fn render_hf_chat_template_prompt(
    template: &ChatTemplateMetadata,
    messages: &[ChatMessage],
    options: &PromptRenderOptions,
) -> anyhow::Result<String> {
    if messages.is_empty() {
        anyhow::bail!("cannot render a chat prompt with no messages");
    }
    if supports_chatml_hf_template(&template.template) {
        return render_chatml_prompt(messages, options);
    }
    if supports_inst_hf_template(&template.template) {
        return render_inst_prompt(messages, options);
    }
    if supports_llama3_header_hf_template(&template.template) {
        return render_llama3_header_prompt(messages, options);
    }
    if supports_gemma_hf_template(&template.template) {
        return render_gemma_prompt(messages, options);
    }

    anyhow::bail!(
        "chat_template_not_supported: chat template at {} uses {:?}; Fathom extracted it, but only a small tested set of ChatML/Qwen, [INST], Llama-3 header, and Gemma HF template patterns is supported for local rendering today",
        template.source.display(),
        template.format
    );
}

fn supports_chatml_hf_template(template: &str) -> bool {
    template.contains("<|im_start|>")
        && template.contains("<|im_end|>")
        && (template.contains("message['role']") || template.contains("message.role"))
        && (template.contains("message['content']") || template.contains("message.content"))
}

fn supports_inst_hf_template(template: &str) -> bool {
    template.contains("[INST]")
        && template.contains("[/INST]")
        && (template.contains("message['role']") || template.contains("message.role"))
        && (template.contains("message['content']") || template.contains("message.content"))
}

fn supports_llama3_header_hf_template(template: &str) -> bool {
    template.contains("<|start_header_id|>")
        && template.contains("<|end_header_id|>")
        && template.contains("<|eot_id|>")
        && template.contains("add_generation_prompt")
        && template.contains("assistant")
        && (template.contains("message['role']") || template.contains("message.role"))
        && (template.contains("message['content']") || template.contains("message.content"))
}

fn supports_gemma_hf_template(template: &str) -> bool {
    let has_role = template.contains("message['role']") || template.contains("message.role");
    let has_content =
        template.contains("message['content']") || template.contains("message.content");
    let has_literal_turns =
        template.contains("<start_of_turn>user") && template.contains("<start_of_turn>model");
    let has_dynamic_model_role = template.contains("<start_of_turn>")
        && template.contains("assistant")
        && template.contains("model");

    template.contains("<end_of_turn>")
        && has_role
        && has_content
        && (has_literal_turns || has_dynamic_model_role)
}

fn supported_hf_chat_role(message: &ChatMessage) -> anyhow::Result<&str> {
    let role = message.role.trim();
    match role {
        "system" | "user" | "assistant" => Ok(role),
        "" => anyhow::bail!("chat_template_not_supported: chat message role cannot be empty"),
        other => anyhow::bail!(
            "chat_template_not_supported: unsupported chat message role {:?}; supported HF template roles are system, user, and assistant",
            other
        ),
    }
}

fn render_chatml_prompt(
    messages: &[ChatMessage],
    options: &PromptRenderOptions,
) -> anyhow::Result<String> {
    let mut prompt = String::new();
    for message in messages {
        let role = supported_hf_chat_role(message)?;
        prompt.push_str("<|im_start|>");
        prompt.push_str(role);
        prompt.push('\n');
        prompt.push_str(message.content.trim_end());
        prompt.push_str("<|im_end|>\n");
    }
    if options.add_generation_prompt {
        prompt.push_str("<|im_start|>assistant\n");
    }
    Ok(prompt)
}

fn render_inst_prompt(
    messages: &[ChatMessage],
    options: &PromptRenderOptions,
) -> anyhow::Result<String> {
    let mut index = 0;
    let mut system = None;
    if supported_hf_chat_role(&messages[0])? == "system" {
        system = Some(messages[0].content.trim());
        index = 1;
    }

    if index >= messages.len() {
        anyhow::bail!(
            "chat_template_not_supported: [INST] template requires at least one user message after an optional system message"
        );
    }

    let mut prompt = String::new();
    let mut turn = 0;
    while index < messages.len() {
        let user = &messages[index];
        if supported_hf_chat_role(user)? != "user" {
            anyhow::bail!(
                "chat_template_not_supported: [INST] templates require user/assistant turns after an optional leading system message"
            );
        }

        prompt.push_str("<s>[INST] ");
        if turn == 0 {
            if let Some(system) = system {
                prompt.push_str("<<SYS>>\n");
                prompt.push_str(system);
                prompt.push_str("\n<</SYS>>\n\n");
            }
        }
        prompt.push_str(user.content.trim());
        prompt.push_str(" [/INST]");
        index += 1;

        if index < messages.len() {
            let assistant = &messages[index];
            if supported_hf_chat_role(assistant)? != "assistant" {
                anyhow::bail!(
                    "chat_template_not_supported: [INST] templates require user/assistant turns after an optional leading system message"
                );
            }
            prompt.push(' ');
            prompt.push_str(assistant.content.trim());
            prompt.push_str(" </s>");
            index += 1;
        } else if !options.add_generation_prompt {
            // A trailing user turn is a complete prompt for generation when no explicit
            // assistant header is available in the [INST] pattern.
        }
        turn += 1;
    }

    Ok(prompt)
}

fn render_llama3_header_prompt(
    messages: &[ChatMessage],
    options: &PromptRenderOptions,
) -> anyhow::Result<String> {
    let mut prompt = String::new();
    for message in messages {
        let role = supported_hf_chat_role(message)?;
        prompt.push_str("<|start_header_id|>");
        prompt.push_str(role);
        prompt.push_str("<|end_header_id|>\n\n");
        prompt.push_str(message.content.trim_end());
        prompt.push_str("<|eot_id|>");
    }
    if options.add_generation_prompt {
        prompt.push_str("<|start_header_id|>assistant<|end_header_id|>\n\n");
    }
    Ok(prompt)
}

fn render_gemma_prompt(
    messages: &[ChatMessage],
    options: &PromptRenderOptions,
) -> anyhow::Result<String> {
    let mut prompt = String::new();
    for message in messages {
        let template_role = match supported_hf_chat_role(message)? {
            "user" => "user",
            "assistant" => "model",
            "system" => anyhow::bail!(
                "chat_template_not_supported: Gemma chat templates support user/model turns; system messages are not rendered by this narrow pattern"
            ),
            _ => unreachable!(),
        };
        prompt.push_str("<start_of_turn>");
        prompt.push_str(template_role);
        prompt.push('\n');
        prompt.push_str(message.content.trim());
        prompt.push_str("<end_of_turn>\n");
    }
    if options.add_generation_prompt {
        prompt.push_str("<start_of_turn>model\n");
    }
    Ok(prompt)
}

#[derive(Debug, Clone, Serialize, PartialEq, Eq)]
pub enum BackendLaneKind {
    SafeTensorsHf,
    Gguf,
    Onnx,
    LocalEmbeddingsRetrieval,
    AppleMlxCoreMl,
    NvidiaTensorRt,
    PyTorchTrustedImport,
    ExternalOpenAiCompatible,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum ModelTaskKind {
    TextGeneration,
    TextEmbedding,
    RetrievalContext,
    Unknown,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub enum CapabilityStatus {
    Runnable,
    Planned,
    MetadataOnly,
    Blocked,
    Unsupported,
}

#[derive(Debug, Clone, Serialize)]
pub struct BackendLane {
    pub id: &'static str,
    pub name: &'static str,
    pub kind: BackendLaneKind,
    pub status: CapabilityStatus,
    pub summary: &'static str,
    pub formats: Vec<ModelFormat>,
    pub blockers: Vec<&'static str>,
}

#[derive(Debug, Clone, Serialize)]
pub struct MachineProfile {
    pub os: String,
    pub arch: String,
    pub apple_platform: bool,
    pub nvidia_requested: bool,
}

#[derive(Debug, Clone, Serialize)]
pub struct PackageCapabilityReport {
    pub package: ModelPackage,
    pub matching_lanes: Vec<BackendLane>,
    pub best_status: CapabilityStatus,
    pub runnable: bool,
    pub summary: String,
}

#[derive(Debug, Clone, Serialize)]
pub struct EmbeddingModelStatus {
    pub task: ModelTaskKind,
    pub status: CapabilityStatus,
    pub runtime_lane: &'static str,
    pub runtime_installed: bool,
    pub runnable: bool,
    pub summary: String,
    pub embedding_dimension: Option<usize>,
    pub blockers: Vec<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct RetrievalIndexSummary {
    pub schema_version: u32,
    pub id: String,
    pub embedding_model_id: String,
    pub embedding_dimension: usize,
    pub document_count: usize,
    pub chunk_count: usize,
    pub status: CapabilityStatus,
}

pub const VECTOR_INDEX_SCHEMA_VERSION: u32 = 1;
pub const MAX_VECTOR_INDEX_DIMENSION: usize = 8192;
pub const MAX_VECTOR_INDEX_CHUNKS: usize = 50_000;
pub const MAX_VECTOR_CHUNK_TEXT_CHARS: usize = 20_000;

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct EmbeddingVector {
    pub values: Vec<f32>,
}

#[derive(Debug, Clone, Copy)]
pub struct EmbeddingRequest<'a> {
    pub inputs: &'a [String],
    pub normalize: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct EmbeddingRuntimeMetrics {
    pub tokenization_ms: u128,
    pub inference_ms: u128,
    pub pooling_ms: u128,
    pub total_ms: u128,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct EmbeddingOutput {
    pub vectors: Vec<EmbeddingVector>,
    pub dimension: usize,
    pub metrics: EmbeddingRuntimeMetrics,
}

impl EmbeddingVector {
    pub fn new(values: Vec<f32>) -> anyhow::Result<Self> {
        validate_vector_values(&values)?;
        Ok(Self { values })
    }

    pub fn dimension(&self) -> usize {
        self.values.len()
    }
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct VectorIndexChunk {
    pub id: String,
    pub document_id: String,
    pub text: String,
    pub byte_start: usize,
    pub byte_end: usize,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct VectorIndexEntry {
    pub chunk: VectorIndexChunk,
    pub vector: EmbeddingVector,
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum VectorSearchMetric {
    Cosine,
    DotProduct,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct VectorSearchHit {
    pub chunk: VectorIndexChunk,
    pub score: f32,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct VectorIndex {
    pub schema_version: u32,
    pub id: String,
    pub embedding_model_id: String,
    pub embedding_dimension: usize,
    pub entries: Vec<VectorIndexEntry>,
}

impl VectorIndex {
    pub fn new(
        id: impl Into<String>,
        embedding_model_id: impl Into<String>,
        embedding_dimension: usize,
    ) -> anyhow::Result<Self> {
        let id = id.into();
        let embedding_model_id = embedding_model_id.into();
        validate_state_file_id(&id)?;
        if embedding_model_id.trim().is_empty() {
            anyhow::bail!("embedding model id cannot be empty");
        }
        if embedding_dimension == 0 {
            anyhow::bail!("embedding dimension must be greater than zero");
        }
        if embedding_dimension > MAX_VECTOR_INDEX_DIMENSION {
            anyhow::bail!(
                "embedding dimension {} exceeds maximum {}",
                embedding_dimension,
                MAX_VECTOR_INDEX_DIMENSION
            );
        }

        Ok(Self {
            schema_version: VECTOR_INDEX_SCHEMA_VERSION,
            id,
            embedding_model_id,
            embedding_dimension,
            entries: Vec::new(),
        })
    }

    pub fn add_chunk(
        &mut self,
        chunk: VectorIndexChunk,
        vector: EmbeddingVector,
    ) -> anyhow::Result<()> {
        validate_chunk(&chunk)?;
        if vector.dimension() != self.embedding_dimension {
            anyhow::bail!(
                "vector dimension {} does not match index dimension {}",
                vector.dimension(),
                self.embedding_dimension
            );
        }
        if self.entries.iter().any(|entry| entry.chunk.id == chunk.id) {
            anyhow::bail!(
                "chunk id '{}' already exists in index '{}'",
                chunk.id,
                self.id
            );
        }
        if self.entries.len() >= MAX_VECTOR_INDEX_CHUNKS {
            anyhow::bail!(
                "retrieval index '{}' already has the maximum {} chunks",
                self.id,
                MAX_VECTOR_INDEX_CHUNKS
            );
        }
        self.entries.push(VectorIndexEntry { chunk, vector });
        Ok(())
    }

    pub fn search(
        &self,
        query: &EmbeddingVector,
        top_k: usize,
        metric: VectorSearchMetric,
    ) -> anyhow::Result<Vec<VectorSearchHit>> {
        self.validate_schema_and_dimensions()?;
        if query.dimension() != self.embedding_dimension {
            anyhow::bail!(
                "query dimension {} does not match index dimension {}",
                query.dimension(),
                self.embedding_dimension
            );
        }
        if top_k == 0 || self.entries.is_empty() {
            return Ok(Vec::new());
        }

        let mut hits: Vec<VectorSearchHit> = self
            .entries
            .iter()
            .map(|entry| VectorSearchHit {
                chunk: entry.chunk.clone(),
                score: match metric {
                    VectorSearchMetric::Cosine => cosine_similarity(query, &entry.vector),
                    VectorSearchMetric::DotProduct => dot_product(query, &entry.vector),
                },
            })
            .collect();

        hits.sort_by(|a, b| {
            b.score
                .total_cmp(&a.score)
                .then_with(|| a.chunk.id.cmp(&b.chunk.id))
        });
        hits.truncate(top_k);
        Ok(hits)
    }

    pub fn summary(&self) -> RetrievalIndexSummary {
        let document_count = self
            .entries
            .iter()
            .map(|entry| entry.chunk.document_id.clone())
            .collect::<BTreeSet<_>>()
            .len();

        RetrievalIndexSummary {
            schema_version: self.schema_version,
            id: self.id.clone(),
            embedding_model_id: self.embedding_model_id.clone(),
            embedding_dimension: self.embedding_dimension,
            document_count,
            chunk_count: self.entries.len(),
            status: CapabilityStatus::Runnable,
        }
    }

    pub fn save_to_state_dir(&self, state_dir: &Path) -> anyhow::Result<PathBuf> {
        self.validate_schema_and_dimensions()?;
        let path = vector_index_state_path(state_dir, &self.id)?;
        let parent = path
            .parent()
            .ok_or_else(|| anyhow::anyhow!("index state path has no parent"))?;
        fs::create_dir_all(parent)?;
        let tmp_path = path.with_extension("json.tmp");
        let bytes = serde_json::to_vec_pretty(self)?;
        fs::write(&tmp_path, bytes)?;
        fs::rename(&tmp_path, &path)?;
        Ok(path)
    }

    pub fn load_from_state_dir(state_dir: &Path, id: &str) -> anyhow::Result<Self> {
        let path = vector_index_state_path(state_dir, id)?;
        let bytes = fs::read(&path)?;
        let index: Self = serde_json::from_slice(&bytes)?;
        if index.id != id {
            anyhow::bail!(
                "vector index file id '{}' does not match requested id '{}'",
                index.id,
                id
            );
        }
        index.validate_schema_and_dimensions()?;
        Ok(index)
    }

    fn validate_schema_and_dimensions(&self) -> anyhow::Result<()> {
        if self.schema_version != VECTOR_INDEX_SCHEMA_VERSION {
            anyhow::bail!(
                "unsupported vector index schema version {}; expected {}",
                self.schema_version,
                VECTOR_INDEX_SCHEMA_VERSION
            );
        }
        validate_state_file_id(&self.id)?;
        if self.embedding_model_id.trim().is_empty() {
            anyhow::bail!("embedding model id cannot be empty");
        }
        if self.embedding_dimension == 0 {
            anyhow::bail!("embedding dimension must be greater than zero");
        }
        if self.embedding_dimension > MAX_VECTOR_INDEX_DIMENSION {
            anyhow::bail!(
                "embedding dimension {} exceeds maximum {}",
                self.embedding_dimension,
                MAX_VECTOR_INDEX_DIMENSION
            );
        }
        if self.entries.len() > MAX_VECTOR_INDEX_CHUNKS {
            anyhow::bail!(
                "retrieval index '{}' has {} chunks, exceeding maximum {}",
                self.id,
                self.entries.len(),
                MAX_VECTOR_INDEX_CHUNKS
            );
        }
        for entry in &self.entries {
            validate_chunk(&entry.chunk)?;
            validate_vector_values(&entry.vector.values)?;
            if entry.vector.dimension() != self.embedding_dimension {
                anyhow::bail!(
                    "chunk '{}' vector dimension {} does not match index dimension {}",
                    entry.chunk.id,
                    entry.vector.dimension(),
                    self.embedding_dimension
                );
            }
        }
        Ok(())
    }
}

pub fn dot_product(left: &EmbeddingVector, right: &EmbeddingVector) -> f32 {
    if left.dimension() != right.dimension() {
        return f32::NAN;
    }
    left.values
        .iter()
        .zip(right.values.iter())
        .map(|(a, b)| a * b)
        .sum()
}

pub fn cosine_similarity(left: &EmbeddingVector, right: &EmbeddingVector) -> f32 {
    if left.dimension() != right.dimension() {
        return f32::NAN;
    }
    let dot = dot_product(left, right);
    let left_norm = left
        .values
        .iter()
        .map(|value| value * value)
        .sum::<f32>()
        .sqrt();
    let right_norm = right
        .values
        .iter()
        .map(|value| value * value)
        .sum::<f32>()
        .sqrt();
    if left_norm == 0.0 || right_norm == 0.0 {
        return 0.0;
    }
    dot / (left_norm * right_norm)
}

fn vector_index_state_path(state_dir: &Path, id: &str) -> anyhow::Result<PathBuf> {
    validate_state_file_id(id)?;
    Ok(state_dir
        .join("retrieval-indexes")
        .join(format!("{id}.json")))
}

fn validate_state_file_id(id: &str) -> anyhow::Result<()> {
    if id.is_empty()
        || !id
            .chars()
            .all(|ch| ch.is_ascii_alphanumeric() || ch == '-' || ch == '_')
    {
        anyhow::bail!("state file id must contain only ASCII letters, numbers, '-' or '_'");
    }
    Ok(())
}

fn validate_chunk(chunk: &VectorIndexChunk) -> anyhow::Result<()> {
    if chunk.id.trim().is_empty() {
        anyhow::bail!("chunk id cannot be empty");
    }
    if chunk.document_id.trim().is_empty() {
        anyhow::bail!("chunk document id cannot be empty");
    }
    if chunk.byte_end < chunk.byte_start {
        anyhow::bail!("chunk byte_end cannot be before byte_start");
    }
    let text_chars = chunk.text.chars().count();
    if text_chars > MAX_VECTOR_CHUNK_TEXT_CHARS {
        anyhow::bail!(
            "chunk text has {} characters, exceeding maximum {}",
            text_chars,
            MAX_VECTOR_CHUNK_TEXT_CHARS
        );
    }
    Ok(())
}

fn validate_vector_values(values: &[f32]) -> anyhow::Result<()> {
    if values.is_empty() {
        anyhow::bail!("embedding vector cannot be empty");
    }
    if values.iter().any(|value| !value.is_finite()) {
        anyhow::bail!("embedding vector values must be finite");
    }
    Ok(())
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum ContextEngineRecommendation {
    InlinePrompt,
    LocalMemorySearch,
    RetrievalIndex,
    CodeIndex,
    ExternalManaged,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct ContextStrategyAdvice {
    pub label: String,
    pub engine: ContextEngineRecommendation,
    pub summary: String,
    pub max_context_tokens: Option<usize>,
    pub reserve_output_tokens: usize,
    pub recommended_chunk_tokens: usize,
    pub recommended_overlap_tokens: usize,
    pub top_k: usize,
    pub needs_retrieval: bool,
    pub caveats: Vec<String>,
    pub suggested_workflow: Vec<String>,
}

pub trait InferenceRuntime: Send + Sync {
    fn runtime_name(&self) -> &'static str;
    fn supports(&self, artifact: &ModelArtifact) -> bool;
    fn generate(&self, request: GenerationRequest) -> anyhow::Result<GenerationResponse>;
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GenerationRequest {
    pub model_path: PathBuf,
    pub prompt: String,
    pub max_tokens: usize,
    #[serde(default)]
    pub options: GenerationOptions,
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub struct GenerationOptions {
    #[serde(default = "default_temperature")]
    pub temperature: f32,
    #[serde(default)]
    pub top_k: Option<usize>,
    #[serde(default)]
    pub top_p: Option<f32>,
}

impl Default for GenerationOptions {
    fn default() -> Self {
        Self {
            temperature: default_temperature(),
            top_k: None,
            top_p: None,
        }
    }
}

fn default_temperature() -> f32 {
    1.0
}

impl GenerationOptions {
    pub fn validate(self) -> anyhow::Result<Self> {
        anyhow::ensure!(
            self.temperature.is_finite() && self.temperature >= 0.0,
            "temperature must be a finite number >= 0"
        );
        if let Some(top_k) = self.top_k {
            anyhow::ensure!(top_k > 0, "top_k must be greater than 0 when provided");
        }
        if let Some(top_p) = self.top_p {
            anyhow::ensure!(
                top_p.is_finite() && top_p > 0.0 && top_p <= 1.0,
                "top_p must be in the range (0, 1] when provided"
            );
        }
        Ok(self)
    }
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum GenerationFinishReason {
    Stop,
    Length,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GenerationResponse {
    pub text: String,
    pub prompt_tokens: usize,
    pub completion_tokens: usize,
    pub finish_reason: GenerationFinishReason,
    pub metrics: GenerationMetrics,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GenerationMetrics {
    pub model_load_ms: u128,
    pub generation_ms: u128,
    pub total_ms: u128,
    pub tokens_per_second: Option<f64>,
    pub ttft_ms: Option<u128>,
    pub prefill_ms: Option<u128>,
    pub decode_ms: Option<u128>,
    pub prefill_tokens_per_second: Option<f64>,
    pub decode_tokens_per_second: Option<f64>,
    pub runtime_cache_hit: bool,
    pub runtime_cache_lookup_ms: u128,
    pub runtime_residency: Option<String>,
    pub runtime_family: Option<String>,
}

fn generation_rate(tokens: usize, seconds: f64) -> Option<f64> {
    if tokens > 0 && seconds > 0.0 {
        Some(tokens as f64 / seconds)
    } else {
        None
    }
}

fn build_generation_metrics(
    model_load_ms: u128,
    generation_elapsed: std::time::Duration,
    total_ms: u128,
    prompt_tokens: usize,
    completion_tokens: usize,
    first_token_elapsed: Option<std::time::Duration>,
) -> GenerationMetrics {
    let generation_ms = generation_elapsed.as_millis();
    let tokens_per_second = generation_rate(completion_tokens, generation_elapsed.as_secs_f64());
    let ttft_ms = first_token_elapsed.map(|elapsed| elapsed.as_millis());
    let prefill_ms = ttft_ms;
    let prefill_tokens_per_second = first_token_elapsed
        .and_then(|elapsed| generation_rate(prompt_tokens, elapsed.as_secs_f64()));
    let decode_elapsed =
        first_token_elapsed.and_then(|first| generation_elapsed.checked_sub(first));
    let decode_tokens = completion_tokens.saturating_sub(1);
    let decode_ms = decode_elapsed.map(|elapsed| elapsed.as_millis());
    let decode_tokens_per_second =
        decode_elapsed.and_then(|elapsed| generation_rate(decode_tokens, elapsed.as_secs_f64()));

    GenerationMetrics {
        model_load_ms,
        generation_ms,
        total_ms,
        tokens_per_second,
        ttft_ms,
        prefill_ms,
        decode_ms,
        prefill_tokens_per_second,
        decode_tokens_per_second,
        runtime_cache_hit: false,
        runtime_cache_lookup_ms: 0,
        runtime_residency: Some("not_cached".to_string()),
        runtime_family: None,
    }
}

struct Conv1D {
    weight: Tensor,
    bias: Tensor,
}

impl Conv1D {
    fn load(in_dim: usize, out_dim: usize, vb: VarBuilder<'_>) -> anyhow::Result<Self> {
        Ok(Self {
            weight: vb.get((in_dim, out_dim), "weight")?,
            bias: vb.get(out_dim, "bias")?,
        })
    }

    fn forward(&self, xs: &Tensor) -> candle_core::Result<Tensor> {
        let (b_sz, seq_len, in_dim) = xs.dims3()?;
        let out_dim = self.bias.dim(0)?;
        xs.reshape((b_sz * seq_len, in_dim))?
            .matmul(&self.weight)?
            .broadcast_add(&self.bias)?
            .reshape((b_sz, seq_len, out_dim))
    }
}

struct Attention {
    c_attn: Conv1D,
    c_proj: Conv1D,
    n_head: usize,
    head_dim: usize,
}

impl Attention {
    fn load(config: &CandleGpt2Config, vb: VarBuilder<'_>) -> anyhow::Result<Self> {
        anyhow::ensure!(
            config.n_embd % config.n_head == 0,
            "GPT-2 hidden size must be divisible by attention heads"
        );
        Ok(Self {
            c_attn: Conv1D::load(config.n_embd, 3 * config.n_embd, vb.pp("c_attn"))?,
            c_proj: Conv1D::load(config.n_embd, config.n_embd, vb.pp("c_proj"))?,
            n_head: config.n_head,
            head_dim: config.n_embd / config.n_head,
        })
    }

    fn split_heads(&self, xs: Tensor) -> candle_core::Result<Tensor> {
        let (b_sz, seq_len, _) = xs.dims3()?;
        xs.reshape((b_sz, seq_len, self.n_head, self.head_dim))?
            .transpose(1, 2)
    }

    fn forward(
        &self,
        xs: &Tensor,
        past_key: &mut Option<Tensor>,
        past_value: &mut Option<Tensor>,
        index_pos: usize,
    ) -> candle_core::Result<Tensor> {
        let (_, seq_len, hidden) = xs.dims3()?;
        let qkv = self.c_attn.forward(xs)?;
        let chunks = qkv.chunk(3, D::Minus1)?;
        let query = self.split_heads(chunks[0].clone())?;
        let key = self.split_heads(chunks[1].clone())?;
        let value = self.split_heads(chunks[2].clone())?;
        let key = match past_key.as_ref() {
            Some(cached_key) => Tensor::cat(&[cached_key, &key], 2)?,
            None => key,
        };
        let value = match past_value.as_ref() {
            Some(cached_value) => Tensor::cat(&[cached_value, &value], 2)?,
            None => value,
        };
        *past_key = Some(key.clone());
        *past_value = Some(value.clone());
        let kv_len = key.dim(2)?;
        let scale = (self.head_dim as f64).sqrt();
        let scores = (query.matmul(&key.transpose(2, 3)?)? / scale)?;
        let mask = causal_mask(seq_len, kv_len, index_pos, xs.device())?;
        let scores = scores.broadcast_add(&mask)?;
        let probs = ops::softmax(&scores, D::Minus1)?;
        let context = probs
            .matmul(&value)?
            .transpose(1, 2)?
            .reshape(((), seq_len, hidden))?;
        self.c_proj.forward(&context)
    }
}

struct Mlp {
    c_fc: Conv1D,
    c_proj: Conv1D,
}

impl Mlp {
    fn load(config: &CandleGpt2Config, vb: VarBuilder<'_>) -> anyhow::Result<Self> {
        let inner = config.n_inner.unwrap_or(4 * config.n_embd);
        Ok(Self {
            c_fc: Conv1D::load(config.n_embd, inner, vb.pp("c_fc"))?,
            c_proj: Conv1D::load(inner, config.n_embd, vb.pp("c_proj"))?,
        })
    }

    fn forward(&self, xs: &Tensor) -> candle_core::Result<Tensor> {
        self.c_proj.forward(&self.c_fc.forward(xs)?.gelu()?)
    }
}

struct Block {
    ln_1: LayerNorm,
    attn: Attention,
    ln_2: LayerNorm,
    mlp: Mlp,
}

impl Block {
    fn load(config: &CandleGpt2Config, vb: VarBuilder<'_>) -> anyhow::Result<Self> {
        Ok(Self {
            ln_1: layer_norm(config.n_embd, config.layer_norm_epsilon, vb.pp("ln_1"))?,
            attn: Attention::load(config, vb.pp("attn"))?,
            ln_2: layer_norm(config.n_embd, config.layer_norm_epsilon, vb.pp("ln_2"))?,
            mlp: Mlp::load(config, vb.pp("mlp"))?,
        })
    }

    fn forward(
        &self,
        xs: &Tensor,
        past_key: &mut Option<Tensor>,
        past_value: &mut Option<Tensor>,
        index_pos: usize,
    ) -> candle_core::Result<Tensor> {
        let attn = self
            .attn
            .forward(&self.ln_1.forward(xs)?, past_key, past_value, index_pos)?;
        let xs = (xs + attn)?;
        let mlp = self.mlp.forward(&self.ln_2.forward(&xs)?)?;
        xs + mlp
    }
}

struct Gpt2KvCache {
    keys: Vec<Option<Tensor>>,
    values: Vec<Option<Tensor>>,
}

impl Gpt2KvCache {
    fn new(layer_count: usize) -> Self {
        Self {
            keys: (0..layer_count).map(|_| None).collect(),
            values: (0..layer_count).map(|_| None).collect(),
        }
    }
}

struct Gpt2Model {
    wte: Embedding,
    wpe: Embedding,
    blocks: Vec<Block>,
    ln_f: LayerNorm,
    lm_head: Option<Tensor>,
    vocab_size: usize,
}

impl Gpt2Model {
    fn load(config: CandleGpt2Config, vb: VarBuilder<'_>) -> anyhow::Result<Self> {
        let transformer = vb.pp("transformer");
        let wte = embedding(config.vocab_size, config.n_embd, transformer.pp("wte"))?;
        let wpe = embedding(config.n_positions, config.n_embd, transformer.pp("wpe"))?;
        let mut blocks = Vec::with_capacity(config.n_layer);
        for layer_idx in 0..config.n_layer {
            blocks.push(Block::load(&config, transformer.pp("h").pp(layer_idx))?);
        }
        let ln_f = layer_norm(
            config.n_embd,
            config.layer_norm_epsilon,
            transformer.pp("ln_f"),
        )?;
        let lm_head = if vb.contains_tensor("lm_head.weight") {
            Some(vb.get((config.vocab_size, config.n_embd), "lm_head.weight")?)
        } else {
            None
        };
        Ok(Self {
            wte,
            wpe,
            blocks,
            ln_f,
            lm_head,
            vocab_size: config.vocab_size,
        })
    }

    fn forward(
        &self,
        input_ids: &Tensor,
        index_pos: usize,
        cache: &mut Gpt2KvCache,
    ) -> candle_core::Result<Tensor> {
        let (_, seq_len) = input_ids.dims2()?;
        let positions = Tensor::arange(
            index_pos as u32,
            (index_pos + seq_len) as u32,
            input_ids.device(),
        )?
        .unsqueeze(0)?;
        let mut xs = self
            .wte
            .forward(input_ids)?
            .broadcast_add(&self.wpe.forward(&positions)?)?;
        for (layer_idx, block) in self.blocks.iter().enumerate() {
            xs = block.forward(
                &xs,
                &mut cache.keys[layer_idx],
                &mut cache.values[layer_idx],
                index_pos,
            )?;
        }
        let xs = self.ln_f.forward(&xs)?;
        let (b_sz, seq_len, hidden) = xs.dims3()?;
        let lm_head = self.lm_head.as_ref().unwrap_or(self.wte.embeddings());
        xs.reshape((b_sz * seq_len, hidden))?
            .matmul(&lm_head.t()?)?
            .reshape((b_sz, seq_len, self.vocab_size))
    }
}

fn causal_mask(
    query_len: usize,
    kv_len: usize,
    index_pos: usize,
    device: &Device,
) -> candle_core::Result<Tensor> {
    let mut values = vec![0f32; query_len * kv_len];
    for row in 0..query_len {
        let absolute_position = index_pos + row;
        for col in (absolute_position + 1)..kv_len {
            values[row * kv_len + col] = f32::NEG_INFINITY;
        }
    }
    Tensor::from_vec(values, (1, 1, query_len, kv_len), device)
}

pub struct CandleGpt2Runtime;

impl InferenceRuntime for CandleGpt2Runtime {
    fn runtime_name(&self) -> &'static str {
        "candle-gpt2"
    }

    fn supports(&self, artifact: &ModelArtifact) -> bool {
        artifact.format == ModelFormat::SafeTensors
    }

    fn generate(&self, request: GenerationRequest) -> anyhow::Result<GenerationResponse> {
        generate_with_candle_gpt2_options(
            &request.model_path,
            &request.prompt,
            request.max_tokens,
            request.options,
        )
    }
}

#[derive(Debug, Clone, Deserialize)]
struct CandleGpt2Config {
    vocab_size: usize,
    #[serde(alias = "max_position_embeddings")]
    n_positions: usize,
    #[serde(alias = "hidden_size")]
    n_embd: usize,
    #[serde(alias = "num_hidden_layers")]
    n_layer: usize,
    #[serde(alias = "num_attention_heads")]
    n_head: usize,
    #[serde(default)]
    n_inner: Option<usize>,
    #[serde(default = "default_gpt2_layer_norm_epsilon")]
    layer_norm_epsilon: f64,
    #[serde(default)]
    model_type: Option<String>,
    #[serde(default)]
    architectures: Vec<String>,
    #[serde(default)]
    eos_token_id: Option<serde_json::Value>,
}

fn default_gpt2_layer_norm_epsilon() -> f64 {
    1e-5
}

#[derive(Debug, Clone, Deserialize)]
struct CandleLlamaConfig {
    vocab_size: usize,
    hidden_size: usize,
    intermediate_size: usize,
    num_hidden_layers: usize,
    num_attention_heads: usize,
    #[serde(default)]
    num_key_value_heads: Option<usize>,
    #[serde(default = "default_llama_max_position_embeddings")]
    max_position_embeddings: usize,
    #[serde(default = "default_llama_rms_norm_eps")]
    rms_norm_eps: f64,
    #[serde(default)]
    model_type: Option<String>,
    #[serde(default)]
    architectures: Vec<String>,
    #[serde(default)]
    tie_word_embeddings: bool,
}

fn default_llama_max_position_embeddings() -> usize {
    4096
}

fn default_llama_rms_norm_eps() -> f64 {
    1e-6
}

#[derive(Debug, Clone, Deserialize)]
struct CandleQwen2Config {
    vocab_size: usize,
    hidden_size: usize,
    intermediate_size: usize,
    num_hidden_layers: usize,
    num_attention_heads: usize,
    num_key_value_heads: usize,
    max_position_embeddings: usize,
    sliding_window: usize,
    max_window_layers: usize,
    tie_word_embeddings: bool,
    rope_theta: f64,
    rms_norm_eps: f64,
    use_sliding_window: bool,
    hidden_act: candle_nn::Activation,
    #[serde(default)]
    model_type: Option<String>,
    #[serde(default)]
    architectures: Vec<String>,
    #[serde(default)]
    eos_token_id: Option<serde_json::Value>,
}

#[derive(Debug, Clone, Deserialize)]
struct CandlePhiConfig {
    vocab_size: usize,
    hidden_size: usize,
    intermediate_size: usize,
    num_hidden_layers: usize,
    num_attention_heads: usize,
    #[serde(default)]
    num_key_value_heads: Option<usize>,
    max_position_embeddings: usize,
    layer_norm_eps: f64,
    tie_word_embeddings: bool,
    rope_theta: f32,
    partial_rotary_factor: f64,
    qk_layernorm: bool,
    hidden_act: candle_nn::Activation,
    #[serde(default)]
    model_type: Option<String>,
    #[serde(default)]
    architectures: Vec<String>,
    #[serde(default)]
    eos_token_id: Option<serde_json::Value>,
}

#[derive(Debug, Clone, Deserialize)]
struct CandleMistralConfig {
    vocab_size: usize,
    hidden_size: usize,
    intermediate_size: usize,
    num_hidden_layers: usize,
    num_attention_heads: usize,
    num_key_value_heads: usize,
    max_position_embeddings: usize,
    rms_norm_eps: f64,
    rope_theta: f64,
    #[serde(default)]
    sliding_window: Option<usize>,
    #[serde(default)]
    head_dim: Option<usize>,
    #[serde(default)]
    model_type: Option<String>,
    #[serde(default)]
    architectures: Vec<String>,
    #[serde(default)]
    eos_token_id: Option<serde_json::Value>,
}

#[derive(Debug, Clone, Deserialize)]
struct CandleGemmaConfig {
    vocab_size: usize,
    hidden_size: usize,
    intermediate_size: usize,
    num_hidden_layers: usize,
    num_attention_heads: usize,
    num_key_value_heads: usize,
    head_dim: usize,
    max_position_embeddings: usize,
    rms_norm_eps: f64,
    rope_theta: f64,
    attention_bias: bool,
    #[serde(default)]
    hidden_act: Option<candle_nn::Activation>,
    #[serde(default)]
    hidden_activation: Option<candle_nn::Activation>,
    #[serde(default)]
    model_type: Option<String>,
    #[serde(default)]
    architectures: Vec<String>,
    #[serde(default)]
    eos_token_id: Option<serde_json::Value>,
}

#[derive(Debug, Clone, Deserialize)]
struct CandleBertEmbeddingConfig {
    vocab_size: usize,
    hidden_size: usize,
    num_hidden_layers: usize,
    num_attention_heads: usize,
    intermediate_size: usize,
    max_position_embeddings: usize,
    type_vocab_size: usize,
    #[serde(default = "default_bert_layer_norm_eps")]
    layer_norm_eps: f64,
    #[serde(default)]
    model_type: Option<String>,
    #[serde(default)]
    architectures: Vec<String>,
}

fn default_bert_layer_norm_eps() -> f64 {
    1e-12
}

const ARTIFACT_PATH_ESCAPE_BLOCKER: &str = "artifact resolves outside the registered model package";

fn ensure_required_artifacts_within_package_root(
    package: &ModelPackage,
    filenames: &[&str],
) -> anyhow::Result<()> {
    let root = package.root.canonicalize().map_err(|_| {
        anyhow::anyhow!("{ARTIFACT_PATH_ESCAPE_BLOCKER}: could not canonicalize package root")
    })?;
    for filename in filenames {
        ensure_artifact_within_package_root(&package.root, &root, &package.root.join(filename))?;
    }
    Ok(())
}

fn ensure_artifact_within_package_root(
    package_root: &Path,
    canonical_package_root: &Path,
    artifact_path: &Path,
) -> anyhow::Result<()> {
    let canonical_artifact_path = artifact_path.canonicalize().map_err(|_| {
        anyhow::anyhow!(
            "{ARTIFACT_PATH_ESCAPE_BLOCKER}: could not canonicalize {}",
            artifact_path.display()
        )
    })?;
    anyhow::ensure!(
        canonical_artifact_path.starts_with(canonical_package_root),
        "{ARTIFACT_PATH_ESCAPE_BLOCKER}"
    );
    anyhow::ensure!(
        canonical_artifact_path.is_file(),
        "{ARTIFACT_PATH_ESCAPE_BLOCKER}: {} is not a regular file under {}",
        artifact_path.display(),
        package_root.display()
    );
    Ok(())
}

pub fn is_candle_gpt2_supported_package(package: &ModelPackage) -> bool {
    validate_candle_gpt2_supported_package(package).is_ok()
}

pub fn validate_candle_gpt2_supported_package(package: &ModelPackage) -> anyhow::Result<()> {
    anyhow::ensure!(
        package.hf_validation.ready_for_loader_metadata,
        "SafeTensors/HF package is missing required loader metadata"
    );
    ensure_required_artifacts_within_package_root(
        package,
        &["config.json", "tokenizer.json", "model.safetensors"],
    )?;
    anyhow::ensure!(
        package.model_type.as_deref() == Some("gpt2"),
        "Candle GPT-2 lane only supports model_type=gpt2 today"
    );
    anyhow::ensure!(
        package
            .architectures
            .iter()
            .any(|architecture| architecture == "GPT2LMHeadModel"),
        "Candle GPT-2 lane only supports GPT2LMHeadModel today"
    );

    let config_path = package
        .config_file
        .as_ref()
        .ok_or_else(|| anyhow::anyhow!("missing config.json"))?;
    let config = read_candle_gpt2_config(config_path)?;
    validate_candle_gpt2_config(&config)?;

    let tokenizer_path = package.root.join("tokenizer.json");
    anyhow::ensure!(
        tokenizer_path.exists(),
        "Candle GPT-2 lane requires tokenizer.json at the package root"
    );
    load_tokenizer_cached(&tokenizer_path)?;

    let weights_path = package.root.join("model.safetensors");
    anyhow::ensure!(
        weights_path.exists(),
        "Candle GPT-2 lane requires a single model.safetensors file at the package root"
    );
    let tensor_headers = read_safetensors_header_tensors(&weights_path)?;
    validate_candle_gpt2_tensor_headers(&config, &tensor_headers)?;
    Ok(())
}

pub fn is_candle_llama_supported_package(package: &ModelPackage) -> bool {
    validate_candle_llama_supported_package(package).is_ok()
}

pub fn is_candle_qwen2_supported_package(package: &ModelPackage) -> bool {
    validate_candle_qwen2_supported_package(package).is_ok()
}

pub fn is_candle_phi_supported_package(package: &ModelPackage) -> bool {
    validate_candle_phi_supported_package(package).is_ok()
}

pub fn is_candle_mistral_supported_package(package: &ModelPackage) -> bool {
    validate_candle_mistral_supported_package(package).is_ok()
}

pub fn is_candle_gemma_supported_package(package: &ModelPackage) -> bool {
    validate_candle_gemma_supported_package(package).is_ok()
}

pub fn is_candle_bert_embedding_supported_package(package: &ModelPackage) -> bool {
    validate_candle_bert_embedding_supported_package(package).is_ok()
}

pub fn validate_candle_bert_embedding_supported_package(
    package: &ModelPackage,
) -> anyhow::Result<()> {
    anyhow::ensure!(
        package.hf_validation.ready_for_loader_metadata,
        "SafeTensors/HF embedding package is missing required loader metadata"
    );
    ensure_required_artifacts_within_package_root(
        package,
        &["config.json", "tokenizer.json", "model.safetensors"],
    )?;
    anyhow::ensure!(
        package.model_type.as_deref() == Some("bert"),
        "Candle BERT embedding lane only supports model_type=bert today"
    );
    anyhow::ensure!(
        package
            .architectures
            .iter()
            .any(|architecture| architecture == "BertModel"),
        "Candle BERT embedding lane only supports BertModel today"
    );

    let config_path = package
        .config_file
        .as_ref()
        .ok_or_else(|| anyhow::anyhow!("missing config.json"))?;
    let config = read_candle_bert_embedding_config(config_path)?;
    validate_candle_bert_embedding_config(&config)?;

    let tokenizer_path = package.root.join("tokenizer.json");
    anyhow::ensure!(
        tokenizer_path.exists(),
        "Candle BERT embedding lane requires tokenizer.json at the package root"
    );
    load_tokenizer_cached(&tokenizer_path)?;

    let weights_path = package.root.join("model.safetensors");
    anyhow::ensure!(
        weights_path.exists(),
        "Candle BERT embedding lane requires a single model.safetensors file at the package root"
    );
    let tensor_headers = read_safetensors_header_tensors(&weights_path)?;
    validate_candle_bert_embedding_tensor_headers(&config, &tensor_headers)?;
    Ok(())
}

pub fn validate_candle_llama_supported_package(package: &ModelPackage) -> anyhow::Result<()> {
    anyhow::ensure!(
        package.hf_validation.ready_for_loader_metadata,
        "SafeTensors/HF package is missing required loader metadata"
    );
    ensure_required_artifacts_within_package_root(
        package,
        &["config.json", "tokenizer.json", "model.safetensors"],
    )?;
    anyhow::ensure!(
        package.model_type.as_deref() == Some("llama"),
        "Candle Llama lane only supports model_type=llama today"
    );
    anyhow::ensure!(
        package
            .architectures
            .iter()
            .any(|architecture| architecture == "LlamaForCausalLM"),
        "Candle Llama lane only supports LlamaForCausalLM today"
    );

    let config_path = package
        .config_file
        .as_ref()
        .ok_or_else(|| anyhow::anyhow!("missing config.json"))?;
    let config = read_candle_llama_config(config_path)?;
    validate_candle_llama_config(&config)?;

    let tokenizer_path = package.root.join("tokenizer.json");
    anyhow::ensure!(
        tokenizer_path.exists(),
        "Candle Llama lane requires tokenizer.json at the package root"
    );
    load_tokenizer_cached(&tokenizer_path)?;

    let weights_path = package.root.join("model.safetensors");
    anyhow::ensure!(
        weights_path.exists(),
        "Candle Llama lane requires a single model.safetensors file at the package root"
    );
    let tensor_headers = read_safetensors_header_tensors(&weights_path)?;
    validate_candle_llama_tensor_headers(&config, &tensor_headers)?;
    Ok(())
}

pub fn validate_candle_qwen2_supported_package(package: &ModelPackage) -> anyhow::Result<()> {
    anyhow::ensure!(
        package.hf_validation.ready_for_loader_metadata,
        "SafeTensors/HF package is missing required loader metadata"
    );
    ensure_required_artifacts_within_package_root(
        package,
        &["config.json", "tokenizer.json", "model.safetensors"],
    )?;
    anyhow::ensure!(
        package.model_type.as_deref() == Some("qwen2"),
        "Candle Qwen2 lane only supports model_type=qwen2 today"
    );
    anyhow::ensure!(
        package
            .architectures
            .iter()
            .any(|architecture| architecture == "Qwen2ForCausalLM"),
        "Candle Qwen2 lane only supports Qwen2ForCausalLM today"
    );

    let config_path = package
        .config_file
        .as_ref()
        .ok_or_else(|| anyhow::anyhow!("missing config.json"))?;
    let config = read_candle_qwen2_config(config_path)?;
    validate_candle_qwen2_config(&config)?;

    let tokenizer_path = package.root.join("tokenizer.json");
    anyhow::ensure!(
        tokenizer_path.exists(),
        "Candle Qwen2 lane requires tokenizer.json at the package root"
    );
    load_tokenizer_cached(&tokenizer_path)?;

    let weights_path = package.root.join("model.safetensors");
    anyhow::ensure!(
        weights_path.exists(),
        "Candle Qwen2 lane requires a single model.safetensors file at the package root"
    );
    let tensor_headers = read_safetensors_header_tensors(&weights_path)?;
    validate_candle_qwen2_tensor_headers(&config, &tensor_headers)?;
    Ok(())
}

pub fn validate_candle_phi_supported_package(package: &ModelPackage) -> anyhow::Result<()> {
    anyhow::ensure!(
        package.hf_validation.ready_for_loader_metadata,
        "SafeTensors/HF package is missing required loader metadata"
    );
    ensure_required_artifacts_within_package_root(
        package,
        &["config.json", "tokenizer.json", "model.safetensors"],
    )?;
    anyhow::ensure!(
        package.model_type.as_deref() == Some("phi"),
        "Candle Phi lane only supports model_type=phi today"
    );
    anyhow::ensure!(
        package
            .architectures
            .iter()
            .any(|architecture| architecture == "PhiForCausalLM"),
        "Candle Phi lane only supports PhiForCausalLM today"
    );

    let config_path = package
        .config_file
        .as_ref()
        .ok_or_else(|| anyhow::anyhow!("missing config.json"))?;
    let config = read_candle_phi_config(config_path)?;
    validate_candle_phi_config(&config)?;

    let tokenizer_path = package.root.join("tokenizer.json");
    anyhow::ensure!(
        tokenizer_path.exists(),
        "Candle Phi lane requires tokenizer.json at the package root"
    );
    load_tokenizer_cached(&tokenizer_path)?;

    let weights_path = package.root.join("model.safetensors");
    anyhow::ensure!(
        weights_path.exists(),
        "Candle Phi lane requires a single model.safetensors file at the package root"
    );
    let tensor_headers = read_safetensors_header_tensors(&weights_path)?;
    validate_candle_phi_tensor_headers(&config, &tensor_headers)?;
    Ok(())
}

pub fn validate_candle_mistral_supported_package(package: &ModelPackage) -> anyhow::Result<()> {
    anyhow::ensure!(
        package.hf_validation.ready_for_loader_metadata,
        "SafeTensors/HF package is missing required loader metadata"
    );
    ensure_required_artifacts_within_package_root(
        package,
        &["config.json", "tokenizer.json", "model.safetensors"],
    )?;
    anyhow::ensure!(
        package.model_type.as_deref() == Some("mistral"),
        "Candle Mistral lane only supports model_type=mistral today"
    );
    anyhow::ensure!(
        package
            .architectures
            .iter()
            .any(|architecture| architecture == "MistralForCausalLM"),
        "Candle Mistral lane only supports MistralForCausalLM today"
    );

    let config_path = package
        .config_file
        .as_ref()
        .ok_or_else(|| anyhow::anyhow!("missing config.json"))?;
    let config = read_candle_mistral_config(config_path)?;
    validate_candle_mistral_config(&config)?;

    let tokenizer_path = package.root.join("tokenizer.json");
    anyhow::ensure!(
        tokenizer_path.exists(),
        "Candle Mistral lane requires tokenizer.json at the package root"
    );
    load_tokenizer_cached(&tokenizer_path)?;

    let weights_path = package.root.join("model.safetensors");
    anyhow::ensure!(
        weights_path.exists(),
        "Candle Mistral lane requires a single model.safetensors file at the package root"
    );
    let tensor_headers = read_safetensors_header_tensors(&weights_path)?;
    validate_candle_mistral_tensor_headers(&config, &tensor_headers)?;
    Ok(())
}

pub fn validate_candle_gemma_supported_package(package: &ModelPackage) -> anyhow::Result<()> {
    anyhow::ensure!(
        package.hf_validation.ready_for_loader_metadata,
        "SafeTensors/HF package is missing required loader metadata"
    );
    ensure_required_artifacts_within_package_root(
        package,
        &["config.json", "tokenizer.json", "model.safetensors"],
    )?;
    anyhow::ensure!(
        package.model_type.as_deref() == Some("gemma"),
        "Candle Gemma lane only supports model_type=gemma today"
    );
    anyhow::ensure!(
        package
            .architectures
            .iter()
            .any(|architecture| architecture == "GemmaForCausalLM"),
        "Candle Gemma lane only supports GemmaForCausalLM today"
    );

    let config_path = package
        .config_file
        .as_ref()
        .ok_or_else(|| anyhow::anyhow!("missing config.json"))?;
    let config = read_candle_gemma_config(config_path)?;
    validate_candle_gemma_config(&config)?;

    let tokenizer_path = package.root.join("tokenizer.json");
    anyhow::ensure!(
        tokenizer_path.exists(),
        "Candle Gemma lane requires tokenizer.json at the package root"
    );
    load_tokenizer_cached(&tokenizer_path)?;

    let weights_path = package.root.join("model.safetensors");
    anyhow::ensure!(
        weights_path.exists(),
        "Candle Gemma lane requires a single model.safetensors file at the package root"
    );
    let tensor_headers = read_safetensors_header_tensors(&weights_path)?;
    validate_candle_gemma_tensor_headers(&config, &tensor_headers)?;
    Ok(())
}

fn read_candle_gpt2_config(path: &Path) -> anyhow::Result<CandleGpt2Config> {
    serde_json::from_slice(&fs::read(path)?)
        .map_err(|error| anyhow::anyhow!("invalid GPT-2 config at {}: {error}", path.display()))
}

fn validate_candle_gpt2_config(config: &CandleGpt2Config) -> anyhow::Result<()> {
    anyhow::ensure!(
        config.model_type.as_deref().unwrap_or("gpt2") == "gpt2",
        "config model_type is not gpt2"
    );
    if !config.architectures.is_empty() {
        anyhow::ensure!(
            config
                .architectures
                .iter()
                .any(|architecture| architecture == "GPT2LMHeadModel"),
            "config architectures do not include GPT2LMHeadModel"
        );
    }
    anyhow::ensure!(config.vocab_size > 0, "invalid GPT-2 vocab_size");
    anyhow::ensure!(config.n_positions > 0, "invalid GPT-2 context window");
    anyhow::ensure!(config.n_embd > 0, "invalid GPT-2 hidden size");
    anyhow::ensure!(config.n_layer > 0, "invalid GPT-2 layer count");
    anyhow::ensure!(config.n_head > 0, "invalid GPT-2 attention head count");
    anyhow::ensure!(
        config.n_embd % config.n_head == 0,
        "GPT-2 hidden size must be divisible by attention heads"
    );
    Ok(())
}

fn read_candle_llama_config(path: &Path) -> anyhow::Result<CandleLlamaConfig> {
    serde_json::from_slice(&fs::read(path)?)
        .map_err(|error| anyhow::anyhow!("invalid Llama config at {}: {error}", path.display()))
}

fn read_candle_qwen2_config(path: &Path) -> anyhow::Result<CandleQwen2Config> {
    serde_json::from_slice(&fs::read(path)?)
        .map_err(|error| anyhow::anyhow!("invalid Qwen2 config at {}: {error}", path.display()))
}

fn read_candle_phi_config(path: &Path) -> anyhow::Result<CandlePhiConfig> {
    serde_json::from_slice(&fs::read(path)?)
        .map_err(|error| anyhow::anyhow!("invalid Phi config at {}: {error}", path.display()))
}

fn read_candle_mistral_config(path: &Path) -> anyhow::Result<CandleMistralConfig> {
    serde_json::from_slice(&fs::read(path)?)
        .map_err(|error| anyhow::anyhow!("invalid Mistral config at {}: {error}", path.display()))
}

fn read_candle_gemma_config(path: &Path) -> anyhow::Result<CandleGemmaConfig> {
    serde_json::from_slice(&fs::read(path)?)
        .map_err(|error| anyhow::anyhow!("invalid Gemma config at {}: {error}", path.display()))
}

fn read_candle_bert_embedding_config(path: &Path) -> anyhow::Result<CandleBertEmbeddingConfig> {
    serde_json::from_slice(&fs::read(path)?).map_err(|error| {
        anyhow::anyhow!(
            "invalid BERT embedding config at {}: {error}",
            path.display()
        )
    })
}

fn validate_candle_bert_embedding_config(config: &CandleBertEmbeddingConfig) -> anyhow::Result<()> {
    anyhow::ensure!(
        config.model_type.as_deref().unwrap_or("bert") == "bert",
        "config model_type is not bert"
    );
    if !config.architectures.is_empty() {
        anyhow::ensure!(
            config
                .architectures
                .iter()
                .any(|architecture| architecture == "BertModel"),
            "config architectures do not include BertModel"
        );
    }
    anyhow::ensure!(config.vocab_size > 0, "invalid BERT vocab_size");
    anyhow::ensure!(config.hidden_size > 0, "invalid BERT hidden size");
    anyhow::ensure!(
        config.intermediate_size > 0,
        "invalid BERT intermediate size"
    );
    anyhow::ensure!(config.num_hidden_layers > 0, "invalid BERT layer count");
    anyhow::ensure!(
        config.num_attention_heads > 0,
        "invalid BERT attention head count"
    );
    anyhow::ensure!(
        config.hidden_size % config.num_attention_heads == 0,
        "BERT hidden size must be divisible by attention heads"
    );
    anyhow::ensure!(
        config.max_position_embeddings > 0,
        "invalid BERT max_position_embeddings"
    );
    anyhow::ensure!(config.type_vocab_size > 0, "invalid BERT type_vocab_size");
    anyhow::ensure!(
        config.layer_norm_eps.is_finite() && config.layer_norm_eps > 0.0,
        "invalid BERT layer_norm_eps"
    );
    Ok(())
}

fn validate_candle_llama_config(config: &CandleLlamaConfig) -> anyhow::Result<()> {
    anyhow::ensure!(
        config.model_type.as_deref().unwrap_or("llama") == "llama",
        "config model_type is not llama"
    );
    if !config.architectures.is_empty() {
        anyhow::ensure!(
            config
                .architectures
                .iter()
                .any(|architecture| architecture == "LlamaForCausalLM"),
            "config architectures do not include LlamaForCausalLM"
        );
    }
    anyhow::ensure!(config.vocab_size > 0, "invalid Llama vocab_size");
    anyhow::ensure!(config.hidden_size > 0, "invalid Llama hidden size");
    anyhow::ensure!(
        config.intermediate_size > 0,
        "invalid Llama intermediate size"
    );
    anyhow::ensure!(config.num_hidden_layers > 0, "invalid Llama layer count");
    anyhow::ensure!(
        config.num_attention_heads > 0,
        "invalid Llama attention head count"
    );
    anyhow::ensure!(
        config.hidden_size % config.num_attention_heads == 0,
        "Llama hidden size must be divisible by attention heads"
    );
    if let Some(num_key_value_heads) = config.num_key_value_heads {
        anyhow::ensure!(
            num_key_value_heads > 0,
            "invalid Llama key/value head count"
        );
        anyhow::ensure!(
            config.num_attention_heads % num_key_value_heads == 0,
            "Llama attention heads must be divisible by key/value heads"
        );
    }
    anyhow::ensure!(
        config.max_position_embeddings > 0,
        "invalid Llama context window"
    );
    anyhow::ensure!(
        config.rms_norm_eps.is_finite() && config.rms_norm_eps > 0.0,
        "invalid Llama RMS norm epsilon"
    );
    Ok(())
}

fn validate_candle_qwen2_config(config: &CandleQwen2Config) -> anyhow::Result<()> {
    anyhow::ensure!(
        config.model_type.as_deref().unwrap_or("qwen2") == "qwen2",
        "config model_type is not qwen2"
    );
    if !config.architectures.is_empty() {
        anyhow::ensure!(
            config
                .architectures
                .iter()
                .any(|architecture| architecture == "Qwen2ForCausalLM"),
            "config architectures do not include Qwen2ForCausalLM"
        );
    }
    anyhow::ensure!(config.vocab_size > 0, "invalid Qwen2 vocab_size");
    anyhow::ensure!(config.hidden_size > 0, "invalid Qwen2 hidden size");
    anyhow::ensure!(
        config.intermediate_size > 0,
        "invalid Qwen2 intermediate size"
    );
    anyhow::ensure!(config.num_hidden_layers > 0, "invalid Qwen2 layer count");
    anyhow::ensure!(
        config.num_attention_heads > 0,
        "invalid Qwen2 attention head count"
    );
    anyhow::ensure!(
        config.num_key_value_heads > 0,
        "invalid Qwen2 key/value head count"
    );
    anyhow::ensure!(
        config.hidden_size % config.num_attention_heads == 0,
        "Qwen2 hidden size must be divisible by attention heads"
    );
    anyhow::ensure!(
        config.num_attention_heads % config.num_key_value_heads == 0,
        "Qwen2 attention heads must be divisible by key/value heads"
    );
    anyhow::ensure!(
        config.max_position_embeddings > 0,
        "invalid Qwen2 context window"
    );
    anyhow::ensure!(config.sliding_window > 0, "invalid Qwen2 sliding window");
    anyhow::ensure!(
        config.max_window_layers > 0,
        "invalid Qwen2 max window layer count"
    );
    anyhow::ensure!(
        config.rope_theta.is_finite() && config.rope_theta > 0.0,
        "invalid Qwen2 rope theta"
    );
    anyhow::ensure!(
        config.rms_norm_eps.is_finite() && config.rms_norm_eps > 0.0,
        "invalid Qwen2 RMS norm epsilon"
    );
    let _ = (
        config.tie_word_embeddings,
        config.use_sliding_window,
        config.hidden_act,
    );
    Ok(())
}

fn validate_candle_phi_config(config: &CandlePhiConfig) -> anyhow::Result<()> {
    anyhow::ensure!(
        config.model_type.as_deref().unwrap_or("phi") == "phi",
        "config model_type is not phi"
    );
    if !config.architectures.is_empty() {
        anyhow::ensure!(
            config
                .architectures
                .iter()
                .any(|architecture| architecture == "PhiForCausalLM"),
            "config architectures do not include PhiForCausalLM"
        );
    }
    anyhow::ensure!(config.vocab_size > 0, "invalid Phi vocab_size");
    anyhow::ensure!(config.hidden_size > 0, "invalid Phi hidden size");
    anyhow::ensure!(
        config.intermediate_size > 0,
        "invalid Phi intermediate size"
    );
    anyhow::ensure!(config.num_hidden_layers > 0, "invalid Phi layer count");
    anyhow::ensure!(
        config.num_attention_heads > 0,
        "invalid Phi attention head count"
    );
    anyhow::ensure!(
        config.hidden_size % config.num_attention_heads == 0,
        "Phi hidden size must be divisible by attention heads"
    );
    if let Some(num_key_value_heads) = config.num_key_value_heads {
        anyhow::ensure!(num_key_value_heads > 0, "invalid Phi key/value head count");
        anyhow::ensure!(
            config.num_attention_heads % num_key_value_heads == 0,
            "Phi attention heads must be divisible by key/value heads"
        );
    }
    anyhow::ensure!(
        config.max_position_embeddings > 0,
        "invalid Phi context window"
    );
    anyhow::ensure!(
        config.layer_norm_eps.is_finite() && config.layer_norm_eps > 0.0,
        "invalid Phi layer norm epsilon"
    );
    anyhow::ensure!(
        config.rope_theta.is_finite() && config.rope_theta > 0.0,
        "invalid Phi rope theta"
    );
    anyhow::ensure!(
        config.partial_rotary_factor.is_finite()
            && config.partial_rotary_factor > 0.0
            && config.partial_rotary_factor <= 1.0,
        "invalid Phi partial rotary factor"
    );
    let _ = (
        config.tie_word_embeddings,
        config.qk_layernorm,
        config.hidden_act,
    );
    Ok(())
}

fn qwen2_eos_token_id(config: &CandleQwen2Config) -> Option<u32> {
    match config.eos_token_id.as_ref()? {
        serde_json::Value::Number(number) => number.as_u64().map(|value| value as u32),
        serde_json::Value::Array(values) => values
            .first()
            .and_then(|value| value.as_u64())
            .map(|value| value as u32),
        _ => None,
    }
}

fn validate_candle_mistral_config(config: &CandleMistralConfig) -> anyhow::Result<()> {
    anyhow::ensure!(
        config.model_type.as_deref().unwrap_or("mistral") == "mistral",
        "config model_type is not mistral"
    );
    if !config.architectures.is_empty() {
        anyhow::ensure!(
            config
                .architectures
                .iter()
                .any(|architecture| architecture == "MistralForCausalLM"),
            "config architectures do not include MistralForCausalLM"
        );
    }
    anyhow::ensure!(config.vocab_size > 0, "invalid Mistral vocab_size");
    anyhow::ensure!(config.hidden_size > 0, "invalid Mistral hidden size");
    anyhow::ensure!(
        config.intermediate_size > 0,
        "invalid Mistral intermediate size"
    );
    anyhow::ensure!(config.num_hidden_layers > 0, "invalid Mistral layer count");
    anyhow::ensure!(
        config.num_attention_heads > 0,
        "invalid Mistral attention head count"
    );
    anyhow::ensure!(
        config.num_key_value_heads > 0,
        "invalid Mistral key/value head count"
    );
    anyhow::ensure!(
        config.num_attention_heads % config.num_key_value_heads == 0,
        "Mistral attention heads must be divisible by key/value heads"
    );
    let head_dim = config
        .head_dim
        .unwrap_or(config.hidden_size / config.num_attention_heads);
    anyhow::ensure!(head_dim > 0, "invalid Mistral head_dim");
    anyhow::ensure!(
        config.hidden_size == config.num_attention_heads * head_dim,
        "Mistral hidden size must match attention heads * head_dim"
    );
    anyhow::ensure!(
        config.max_position_embeddings > 0,
        "invalid Mistral context window"
    );
    anyhow::ensure!(
        config.rms_norm_eps > 0.0,
        "invalid Mistral RMS norm epsilon"
    );
    anyhow::ensure!(config.rope_theta > 0.0, "invalid Mistral rope theta");
    if let Some(sliding_window) = config.sliding_window {
        anyhow::ensure!(sliding_window > 0, "invalid Mistral sliding window");
    }
    Ok(())
}

fn phi_eos_token_id(config: &CandlePhiConfig) -> Option<u32> {
    match config.eos_token_id.as_ref()? {
        serde_json::Value::Number(number) => number.as_u64().map(|value| value as u32),
        serde_json::Value::Array(values) => values
            .first()
            .and_then(|value| value.as_u64())
            .map(|value| value as u32),
        _ => None,
    }
}

fn validate_candle_gemma_config(config: &CandleGemmaConfig) -> anyhow::Result<()> {
    anyhow::ensure!(
        config.model_type.as_deref().unwrap_or("gemma") == "gemma",
        "config model_type is not gemma"
    );
    if !config.architectures.is_empty() {
        anyhow::ensure!(
            config
                .architectures
                .iter()
                .any(|architecture| architecture == "GemmaForCausalLM"),
            "config architectures do not include GemmaForCausalLM"
        );
    }
    anyhow::ensure!(config.vocab_size > 0, "invalid Gemma vocab_size");
    anyhow::ensure!(config.hidden_size > 0, "invalid Gemma hidden size");
    anyhow::ensure!(
        config.intermediate_size > 0,
        "invalid Gemma intermediate size"
    );
    anyhow::ensure!(config.num_hidden_layers > 0, "invalid Gemma layer count");
    anyhow::ensure!(
        config.num_attention_heads > 0,
        "invalid Gemma attention head count"
    );
    anyhow::ensure!(
        config.num_key_value_heads > 0,
        "invalid Gemma key/value head count"
    );
    anyhow::ensure!(
        config.num_attention_heads % config.num_key_value_heads == 0,
        "Gemma attention heads must be divisible by key/value heads"
    );
    anyhow::ensure!(config.head_dim > 0, "invalid Gemma head_dim");
    anyhow::ensure!(
        config.num_attention_heads * config.head_dim > 0,
        "invalid Gemma attention projection size"
    );
    anyhow::ensure!(
        config.max_position_embeddings > 0,
        "invalid Gemma context window"
    );
    anyhow::ensure!(
        config.rms_norm_eps.is_finite() && config.rms_norm_eps > 0.0,
        "invalid Gemma RMS norm epsilon"
    );
    anyhow::ensure!(
        config.rope_theta.is_finite() && config.rope_theta > 0.0,
        "invalid Gemma rope theta"
    );
    anyhow::ensure!(
        matches!(
            (config.hidden_act, config.hidden_activation),
            (Some(_), None) | (None, Some(_))
        ),
        "Gemma config must set exactly one of hidden_act or hidden_activation for Candle"
    );
    let _ = config.attention_bias;
    Ok(())
}

fn gemma_eos_token_id(config: &CandleGemmaConfig) -> Option<u32> {
    match config.eos_token_id.as_ref()? {
        serde_json::Value::Number(number) => number.as_u64().map(|value| value as u32),
        serde_json::Value::Array(values) => values
            .first()
            .and_then(|value| value.as_u64())
            .map(|value| value as u32),
        _ => None,
    }
}

fn validate_candle_bert_embedding_tensor_headers(
    config: &CandleBertEmbeddingConfig,
    tensor_headers: &std::collections::HashMap<String, serde_json::Value>,
) -> anyhow::Result<()> {
    let mut required = vec![
        (
            "embeddings.word_embeddings.weight".to_string(),
            vec![config.vocab_size, config.hidden_size],
        ),
        (
            "embeddings.position_embeddings.weight".to_string(),
            vec![config.max_position_embeddings, config.hidden_size],
        ),
        (
            "embeddings.token_type_embeddings.weight".to_string(),
            vec![config.type_vocab_size, config.hidden_size],
        ),
        (
            "embeddings.LayerNorm.weight".to_string(),
            vec![config.hidden_size],
        ),
        (
            "embeddings.LayerNorm.bias".to_string(),
            vec![config.hidden_size],
        ),
    ];
    for layer_idx in 0..config.num_hidden_layers {
        required.extend([
            (
                format!("encoder.layer.{layer_idx}.attention.self.query.weight"),
                vec![config.hidden_size, config.hidden_size],
            ),
            (
                format!("encoder.layer.{layer_idx}.attention.self.query.bias"),
                vec![config.hidden_size],
            ),
            (
                format!("encoder.layer.{layer_idx}.attention.self.key.weight"),
                vec![config.hidden_size, config.hidden_size],
            ),
            (
                format!("encoder.layer.{layer_idx}.attention.self.key.bias"),
                vec![config.hidden_size],
            ),
            (
                format!("encoder.layer.{layer_idx}.attention.self.value.weight"),
                vec![config.hidden_size, config.hidden_size],
            ),
            (
                format!("encoder.layer.{layer_idx}.attention.self.value.bias"),
                vec![config.hidden_size],
            ),
            (
                format!("encoder.layer.{layer_idx}.attention.output.dense.weight"),
                vec![config.hidden_size, config.hidden_size],
            ),
            (
                format!("encoder.layer.{layer_idx}.attention.output.dense.bias"),
                vec![config.hidden_size],
            ),
            (
                format!("encoder.layer.{layer_idx}.attention.output.LayerNorm.weight"),
                vec![config.hidden_size],
            ),
            (
                format!("encoder.layer.{layer_idx}.attention.output.LayerNorm.bias"),
                vec![config.hidden_size],
            ),
            (
                format!("encoder.layer.{layer_idx}.intermediate.dense.weight"),
                vec![config.intermediate_size, config.hidden_size],
            ),
            (
                format!("encoder.layer.{layer_idx}.intermediate.dense.bias"),
                vec![config.intermediate_size],
            ),
            (
                format!("encoder.layer.{layer_idx}.output.dense.weight"),
                vec![config.hidden_size, config.intermediate_size],
            ),
            (
                format!("encoder.layer.{layer_idx}.output.dense.bias"),
                vec![config.hidden_size],
            ),
            (
                format!("encoder.layer.{layer_idx}.output.LayerNorm.weight"),
                vec![config.hidden_size],
            ),
            (
                format!("encoder.layer.{layer_idx}.output.LayerNorm.bias"),
                vec![config.hidden_size],
            ),
        ]);
    }

    let mut errors = Vec::new();
    for (name, expected_shape) in required {
        let header = tensor_headers.get(&name).or_else(|| {
            config
                .model_type
                .as_deref()
                .and_then(|model_type| tensor_headers.get(&format!("{model_type}.{name}")))
        });
        let Some(header) = header else {
            errors.push(format!("missing {name}"));
            continue;
        };
        let dtype = header.get("dtype").and_then(|value| value.as_str());
        if !matches!(dtype, Some("F32" | "F16" | "BF16")) {
            errors.push(format!(
                "{name} has dtype {:?}, expected F32, F16, or BF16",
                dtype
            ));
        }
        let shape = header
            .get("shape")
            .and_then(|value| value.as_array())
            .map(|values| {
                values
                    .iter()
                    .filter_map(|value| value.as_u64().map(|value| value as usize))
                    .collect::<Vec<_>>()
            });
        if shape.as_deref() != Some(expected_shape.as_slice()) {
            errors.push(format!(
                "{name} has shape {:?}, expected {:?}",
                shape, expected_shape
            ));
        }
    }
    anyhow::ensure!(
        errors.is_empty(),
        "model.safetensors is not a loadable BertModel checkpoint for Fathom's current Candle embedding lane: {}",
        errors.join("; ")
    );
    Ok(())
}

fn validate_candle_gemma_tensor_headers(
    config: &CandleGemmaConfig,
    tensor_headers: &std::collections::HashMap<String, serde_json::Value>,
) -> anyhow::Result<()> {
    let kv_dim = config.num_key_value_heads * config.head_dim;
    let mut required = vec![
        (
            "model.embed_tokens.weight".to_string(),
            vec![config.vocab_size, config.hidden_size],
        ),
        ("model.norm.weight".to_string(), vec![config.hidden_size]),
    ];
    for layer_idx in 0..config.num_hidden_layers {
        required.extend([
            (
                format!("model.layers.{layer_idx}.input_layernorm.weight"),
                vec![config.hidden_size],
            ),
            (
                format!("model.layers.{layer_idx}.post_attention_layernorm.weight"),
                vec![config.hidden_size],
            ),
            (
                format!("model.layers.{layer_idx}.self_attn.q_proj.weight"),
                vec![
                    config.num_attention_heads * config.head_dim,
                    config.hidden_size,
                ],
            ),
            (
                format!("model.layers.{layer_idx}.self_attn.k_proj.weight"),
                vec![kv_dim, config.hidden_size],
            ),
            (
                format!("model.layers.{layer_idx}.self_attn.v_proj.weight"),
                vec![kv_dim, config.hidden_size],
            ),
            (
                format!("model.layers.{layer_idx}.self_attn.o_proj.weight"),
                vec![
                    config.hidden_size,
                    config.num_attention_heads * config.head_dim,
                ],
            ),
            (
                format!("model.layers.{layer_idx}.mlp.gate_proj.weight"),
                vec![config.intermediate_size, config.hidden_size],
            ),
            (
                format!("model.layers.{layer_idx}.mlp.up_proj.weight"),
                vec![config.intermediate_size, config.hidden_size],
            ),
            (
                format!("model.layers.{layer_idx}.mlp.down_proj.weight"),
                vec![config.hidden_size, config.intermediate_size],
            ),
        ]);
        if config.attention_bias {
            required.extend([
                (
                    format!("model.layers.{layer_idx}.self_attn.q_proj.bias"),
                    vec![config.num_attention_heads * config.head_dim],
                ),
                (
                    format!("model.layers.{layer_idx}.self_attn.k_proj.bias"),
                    vec![kv_dim],
                ),
                (
                    format!("model.layers.{layer_idx}.self_attn.v_proj.bias"),
                    vec![kv_dim],
                ),
                (
                    format!("model.layers.{layer_idx}.self_attn.o_proj.bias"),
                    vec![config.hidden_size],
                ),
            ]);
        }
    }

    let mut errors = Vec::new();
    for (name, expected_shape) in required {
        let Some(header) = tensor_headers.get(&name) else {
            errors.push(format!("missing {name}"));
            continue;
        };
        let dtype = header.get("dtype").and_then(|value| value.as_str());
        if !matches!(dtype, Some("F32" | "F16" | "BF16")) {
            errors.push(format!(
                "{name} has dtype {:?}, expected F32, F16, or BF16",
                dtype
            ));
        }
        let shape = header
            .get("shape")
            .and_then(|value| value.as_array())
            .map(|values| {
                values
                    .iter()
                    .filter_map(|value| value.as_u64().map(|value| value as usize))
                    .collect::<Vec<_>>()
            });
        if shape.as_deref() != Some(expected_shape.as_slice()) {
            errors.push(format!(
                "{name} has shape {:?}, expected {:?}",
                shape, expected_shape
            ));
        }
    }
    anyhow::ensure!(
        errors.is_empty(),
        "model.safetensors is not a loadable GemmaForCausalLM checkpoint for Fathom's current Candle lane: {}",
        errors.join("; ")
    );
    Ok(())
}

fn validate_candle_phi_tensor_headers(
    config: &CandlePhiConfig,
    tensor_headers: &std::collections::HashMap<String, serde_json::Value>,
) -> anyhow::Result<()> {
    let num_key_value_heads = config
        .num_key_value_heads
        .unwrap_or(config.num_attention_heads);
    let head_dim = config.hidden_size / config.num_attention_heads;
    let kv_dim = num_key_value_heads * head_dim;
    let mut required = vec![
        (
            "model.embed_tokens.weight".to_string(),
            vec![config.vocab_size, config.hidden_size],
        ),
        (
            "model.final_layernorm.weight".to_string(),
            vec![config.hidden_size],
        ),
        (
            "model.final_layernorm.bias".to_string(),
            vec![config.hidden_size],
        ),
        (
            "lm_head.weight".to_string(),
            vec![config.vocab_size, config.hidden_size],
        ),
        ("lm_head.bias".to_string(), vec![config.vocab_size]),
    ];
    for layer_idx in 0..config.num_hidden_layers {
        required.extend([
            (
                format!("model.layers.{layer_idx}.input_layernorm.weight"),
                vec![config.hidden_size],
            ),
            (
                format!("model.layers.{layer_idx}.input_layernorm.bias"),
                vec![config.hidden_size],
            ),
            (
                format!("model.layers.{layer_idx}.self_attn.q_proj.weight"),
                vec![config.num_attention_heads * head_dim, config.hidden_size],
            ),
            (
                format!("model.layers.{layer_idx}.self_attn.q_proj.bias"),
                vec![config.num_attention_heads * head_dim],
            ),
            (
                format!("model.layers.{layer_idx}.self_attn.k_proj.weight"),
                vec![kv_dim, config.hidden_size],
            ),
            (
                format!("model.layers.{layer_idx}.self_attn.k_proj.bias"),
                vec![kv_dim],
            ),
            (
                format!("model.layers.{layer_idx}.self_attn.v_proj.weight"),
                vec![kv_dim, config.hidden_size],
            ),
            (
                format!("model.layers.{layer_idx}.self_attn.v_proj.bias"),
                vec![kv_dim],
            ),
            (
                format!("model.layers.{layer_idx}.self_attn.dense.weight"),
                vec![config.hidden_size, config.num_attention_heads * head_dim],
            ),
            (
                format!("model.layers.{layer_idx}.self_attn.dense.bias"),
                vec![config.hidden_size],
            ),
            (
                format!("model.layers.{layer_idx}.mlp.fc1.weight"),
                vec![config.intermediate_size, config.hidden_size],
            ),
            (
                format!("model.layers.{layer_idx}.mlp.fc1.bias"),
                vec![config.intermediate_size],
            ),
            (
                format!("model.layers.{layer_idx}.mlp.fc2.weight"),
                vec![config.hidden_size, config.intermediate_size],
            ),
            (
                format!("model.layers.{layer_idx}.mlp.fc2.bias"),
                vec![config.hidden_size],
            ),
        ]);
        if config.qk_layernorm {
            required.extend([
                (
                    format!("model.layers.{layer_idx}.self_attn.q_layernorm.weight"),
                    vec![head_dim],
                ),
                (
                    format!("model.layers.{layer_idx}.self_attn.q_layernorm.bias"),
                    vec![head_dim],
                ),
                (
                    format!("model.layers.{layer_idx}.self_attn.k_layernorm.weight"),
                    vec![head_dim],
                ),
                (
                    format!("model.layers.{layer_idx}.self_attn.k_layernorm.bias"),
                    vec![head_dim],
                ),
            ]);
        }
    }

    let mut errors = Vec::new();
    for (name, expected_shape) in required {
        let Some(header) = tensor_headers.get(&name) else {
            errors.push(format!("missing {name}"));
            continue;
        };
        let dtype = header.get("dtype").and_then(|value| value.as_str());
        if !matches!(dtype, Some("F32" | "F16" | "BF16")) {
            errors.push(format!(
                "{name} has dtype {:?}, expected F32, F16, or BF16",
                dtype
            ));
        }
        let shape = header
            .get("shape")
            .and_then(|value| value.as_array())
            .map(|values| {
                values
                    .iter()
                    .filter_map(|value| value.as_u64().map(|value| value as usize))
                    .collect::<Vec<_>>()
            });
        if shape.as_deref() != Some(expected_shape.as_slice()) {
            errors.push(format!(
                "{name} has shape {:?}, expected {:?}",
                shape, expected_shape
            ));
        }
    }
    anyhow::ensure!(
        errors.is_empty(),
        "model.safetensors is not a loadable PhiForCausalLM checkpoint for Fathom's current Candle lane: {}",
        errors.join("; ")
    );
    Ok(())
}

fn validate_candle_qwen2_tensor_headers(
    config: &CandleQwen2Config,
    tensor_headers: &std::collections::HashMap<String, serde_json::Value>,
) -> anyhow::Result<()> {
    let head_dim = config.hidden_size / config.num_attention_heads;
    let kv_dim = config.num_key_value_heads * head_dim;
    let mut required = vec![
        (
            "model.embed_tokens.weight".to_string(),
            vec![config.vocab_size, config.hidden_size],
        ),
        ("model.norm.weight".to_string(), vec![config.hidden_size]),
    ];
    if !config.tie_word_embeddings {
        required.push((
            "lm_head.weight".to_string(),
            vec![config.vocab_size, config.hidden_size],
        ));
    }
    for layer_idx in 0..config.num_hidden_layers {
        required.extend([
            (
                format!("model.layers.{layer_idx}.input_layernorm.weight"),
                vec![config.hidden_size],
            ),
            (
                format!("model.layers.{layer_idx}.post_attention_layernorm.weight"),
                vec![config.hidden_size],
            ),
            (
                format!("model.layers.{layer_idx}.self_attn.q_proj.weight"),
                vec![config.num_attention_heads * head_dim, config.hidden_size],
            ),
            (
                format!("model.layers.{layer_idx}.self_attn.q_proj.bias"),
                vec![config.num_attention_heads * head_dim],
            ),
            (
                format!("model.layers.{layer_idx}.self_attn.k_proj.weight"),
                vec![kv_dim, config.hidden_size],
            ),
            (
                format!("model.layers.{layer_idx}.self_attn.k_proj.bias"),
                vec![kv_dim],
            ),
            (
                format!("model.layers.{layer_idx}.self_attn.v_proj.weight"),
                vec![kv_dim, config.hidden_size],
            ),
            (
                format!("model.layers.{layer_idx}.self_attn.v_proj.bias"),
                vec![kv_dim],
            ),
            (
                format!("model.layers.{layer_idx}.self_attn.o_proj.weight"),
                vec![config.hidden_size, config.num_attention_heads * head_dim],
            ),
            (
                format!("model.layers.{layer_idx}.mlp.gate_proj.weight"),
                vec![config.intermediate_size, config.hidden_size],
            ),
            (
                format!("model.layers.{layer_idx}.mlp.up_proj.weight"),
                vec![config.intermediate_size, config.hidden_size],
            ),
            (
                format!("model.layers.{layer_idx}.mlp.down_proj.weight"),
                vec![config.hidden_size, config.intermediate_size],
            ),
        ]);
    }

    let mut errors = Vec::new();
    for (name, expected_shape) in required {
        let Some(header) = tensor_headers.get(&name) else {
            errors.push(format!("missing {name}"));
            continue;
        };
        let dtype = header.get("dtype").and_then(|value| value.as_str());
        if !matches!(dtype, Some("F32" | "F16" | "BF16")) {
            errors.push(format!(
                "{name} has dtype {:?}, expected F32, F16, or BF16",
                dtype
            ));
        }
        let shape = header
            .get("shape")
            .and_then(|value| value.as_array())
            .map(|values| {
                values
                    .iter()
                    .filter_map(|value| value.as_u64().map(|value| value as usize))
                    .collect::<Vec<_>>()
            });
        if shape.as_deref() != Some(expected_shape.as_slice()) {
            errors.push(format!(
                "{name} has shape {:?}, expected {:?}",
                shape, expected_shape
            ));
        }
    }
    anyhow::ensure!(
        errors.is_empty(),
        "model.safetensors is not a loadable Qwen2ForCausalLM checkpoint for Fathom's current Candle lane: {}",
        errors.join("; ")
    );
    Ok(())
}

fn validate_candle_llama_tensor_headers(
    config: &CandleLlamaConfig,
    tensor_headers: &std::collections::HashMap<String, serde_json::Value>,
) -> anyhow::Result<()> {
    let num_key_value_heads = config
        .num_key_value_heads
        .unwrap_or(config.num_attention_heads);
    let head_dim = config.hidden_size / config.num_attention_heads;
    let kv_dim = num_key_value_heads * head_dim;
    let mut required = vec![
        (
            "model.embed_tokens.weight".to_string(),
            vec![config.vocab_size, config.hidden_size],
        ),
        ("model.norm.weight".to_string(), vec![config.hidden_size]),
    ];
    if !config.tie_word_embeddings {
        required.push((
            "lm_head.weight".to_string(),
            vec![config.vocab_size, config.hidden_size],
        ));
    }
    for layer_idx in 0..config.num_hidden_layers {
        required.extend([
            (
                format!("model.layers.{layer_idx}.input_layernorm.weight"),
                vec![config.hidden_size],
            ),
            (
                format!("model.layers.{layer_idx}.post_attention_layernorm.weight"),
                vec![config.hidden_size],
            ),
            (
                format!("model.layers.{layer_idx}.self_attn.q_proj.weight"),
                vec![config.num_attention_heads * head_dim, config.hidden_size],
            ),
            (
                format!("model.layers.{layer_idx}.self_attn.k_proj.weight"),
                vec![kv_dim, config.hidden_size],
            ),
            (
                format!("model.layers.{layer_idx}.self_attn.v_proj.weight"),
                vec![kv_dim, config.hidden_size],
            ),
            (
                format!("model.layers.{layer_idx}.self_attn.o_proj.weight"),
                vec![config.hidden_size, config.num_attention_heads * head_dim],
            ),
            (
                format!("model.layers.{layer_idx}.mlp.gate_proj.weight"),
                vec![config.intermediate_size, config.hidden_size],
            ),
            (
                format!("model.layers.{layer_idx}.mlp.up_proj.weight"),
                vec![config.intermediate_size, config.hidden_size],
            ),
            (
                format!("model.layers.{layer_idx}.mlp.down_proj.weight"),
                vec![config.hidden_size, config.intermediate_size],
            ),
        ]);
    }

    let mut errors = Vec::new();
    for (name, expected_shape) in required {
        let Some(header) = tensor_headers.get(&name) else {
            errors.push(format!("missing {name}"));
            continue;
        };
        let dtype = header.get("dtype").and_then(|value| value.as_str());
        if !matches!(dtype, Some("F32" | "F16" | "BF16")) {
            errors.push(format!(
                "{name} has dtype {:?}, expected F32, F16, or BF16",
                dtype
            ));
        }
        let shape = header
            .get("shape")
            .and_then(|value| value.as_array())
            .map(|values| {
                values
                    .iter()
                    .filter_map(|value| value.as_u64().map(|value| value as usize))
                    .collect::<Vec<_>>()
            });
        if shape.as_deref() != Some(expected_shape.as_slice()) {
            errors.push(format!(
                "{name} has shape {:?}, expected {:?}",
                shape, expected_shape
            ));
        }
    }
    anyhow::ensure!(
        errors.is_empty(),
        "model.safetensors is not a loadable LlamaForCausalLM checkpoint for Fathom's current Candle lane: {}",
        errors.join("; ")
    );
    Ok(())
}

fn validate_candle_mistral_tensor_headers(
    config: &CandleMistralConfig,
    tensor_headers: &std::collections::HashMap<String, serde_json::Value>,
) -> anyhow::Result<()> {
    let head_dim = config
        .head_dim
        .unwrap_or(config.hidden_size / config.num_attention_heads);
    let kv_dim = config.num_key_value_heads * head_dim;
    let mut required = vec![
        (
            "model.embed_tokens.weight".to_string(),
            vec![config.vocab_size, config.hidden_size],
        ),
        ("model.norm.weight".to_string(), vec![config.hidden_size]),
        (
            "lm_head.weight".to_string(),
            vec![config.vocab_size, config.hidden_size],
        ),
    ];
    for layer_idx in 0..config.num_hidden_layers {
        required.extend([
            (
                format!("model.layers.{layer_idx}.input_layernorm.weight"),
                vec![config.hidden_size],
            ),
            (
                format!("model.layers.{layer_idx}.post_attention_layernorm.weight"),
                vec![config.hidden_size],
            ),
            (
                format!("model.layers.{layer_idx}.self_attn.q_proj.weight"),
                vec![config.num_attention_heads * head_dim, config.hidden_size],
            ),
            (
                format!("model.layers.{layer_idx}.self_attn.k_proj.weight"),
                vec![kv_dim, config.hidden_size],
            ),
            (
                format!("model.layers.{layer_idx}.self_attn.v_proj.weight"),
                vec![kv_dim, config.hidden_size],
            ),
            (
                format!("model.layers.{layer_idx}.self_attn.o_proj.weight"),
                vec![config.hidden_size, config.num_attention_heads * head_dim],
            ),
            (
                format!("model.layers.{layer_idx}.mlp.gate_proj.weight"),
                vec![config.intermediate_size, config.hidden_size],
            ),
            (
                format!("model.layers.{layer_idx}.mlp.up_proj.weight"),
                vec![config.intermediate_size, config.hidden_size],
            ),
            (
                format!("model.layers.{layer_idx}.mlp.down_proj.weight"),
                vec![config.hidden_size, config.intermediate_size],
            ),
        ]);
    }

    let mut errors = Vec::new();
    for (name, expected_shape) in required {
        let Some(header) = tensor_headers.get(&name) else {
            errors.push(format!("missing {name}"));
            continue;
        };
        let dtype = header.get("dtype").and_then(|value| value.as_str());
        if !matches!(dtype, Some("F32" | "F16" | "BF16")) {
            errors.push(format!(
                "{name} has dtype {:?}, expected F32, F16, or BF16",
                dtype
            ));
        }
        let shape = header
            .get("shape")
            .and_then(|value| value.as_array())
            .map(|values| {
                values
                    .iter()
                    .filter_map(|value| value.as_u64().map(|value| value as usize))
                    .collect::<Vec<_>>()
            });
        if shape.as_deref() != Some(expected_shape.as_slice()) {
            errors.push(format!(
                "{name} has shape {:?}, expected {:?}",
                shape, expected_shape
            ));
        }
    }
    anyhow::ensure!(
        errors.is_empty(),
        "model.safetensors is not a loadable MistralForCausalLM checkpoint for Fathom's current Candle lane: {}",
        errors.join("; ")
    );
    Ok(())
}

fn read_safetensors_header_tensors(
    path: &Path,
) -> anyhow::Result<std::collections::HashMap<String, serde_json::Value>> {
    use std::io::Read;

    let mut file = fs::File::open(path)?;
    let mut header_len_bytes = [0u8; 8];
    file.read_exact(&mut header_len_bytes)?;
    let header_len = u64::from_le_bytes(header_len_bytes) as usize;
    anyhow::ensure!(
        header_len > 0 && header_len <= 100 * 1024 * 1024,
        "invalid SafeTensors header length in {}",
        path.display()
    );
    let file_len = file.metadata()?.len() as usize;
    anyhow::ensure!(
        file_len >= 8 + header_len,
        "SafeTensors file is shorter than its header in {}",
        path.display()
    );
    let mut header = vec![0u8; header_len];
    file.read_exact(&mut header)?;
    let value: serde_json::Value = serde_json::from_slice(&header).map_err(|error| {
        anyhow::anyhow!("invalid SafeTensors header in {}: {error}", path.display())
    })?;
    let object = value.as_object().ok_or_else(|| {
        anyhow::anyhow!("SafeTensors header is not an object in {}", path.display())
    })?;

    let mut tensors = std::collections::HashMap::new();
    for (name, value) in object {
        if name == "__metadata__" {
            continue;
        }
        if let Some(end) = value
            .get("data_offsets")
            .and_then(|offsets| offsets.as_array())
            .and_then(|offsets| offsets.get(1))
            .and_then(|offset| offset.as_u64())
        {
            anyhow::ensure!(
                8 + header_len + end as usize <= file_len,
                "SafeTensors tensor {name} points past the end of {}",
                path.display()
            );
        }
        tensors.insert(name.clone(), value.clone());
    }
    Ok(tensors)
}

fn validate_candle_gpt2_tensor_headers(
    config: &CandleGpt2Config,
    tensor_headers: &std::collections::HashMap<String, serde_json::Value>,
) -> anyhow::Result<()> {
    let inner = config.n_inner.unwrap_or(4 * config.n_embd);
    let mut required = vec![
        (
            "transformer.wte.weight".to_string(),
            vec![config.vocab_size, config.n_embd],
        ),
        (
            "transformer.wpe.weight".to_string(),
            vec![config.n_positions, config.n_embd],
        ),
        ("transformer.ln_f.weight".to_string(), vec![config.n_embd]),
        ("transformer.ln_f.bias".to_string(), vec![config.n_embd]),
    ];
    for layer_idx in 0..config.n_layer {
        required.extend([
            (
                format!("transformer.h.{layer_idx}.ln_1.weight"),
                vec![config.n_embd],
            ),
            (
                format!("transformer.h.{layer_idx}.ln_1.bias"),
                vec![config.n_embd],
            ),
            (
                format!("transformer.h.{layer_idx}.attn.c_attn.weight"),
                vec![config.n_embd, 3 * config.n_embd],
            ),
            (
                format!("transformer.h.{layer_idx}.attn.c_attn.bias"),
                vec![3 * config.n_embd],
            ),
            (
                format!("transformer.h.{layer_idx}.attn.c_proj.weight"),
                vec![config.n_embd, config.n_embd],
            ),
            (
                format!("transformer.h.{layer_idx}.attn.c_proj.bias"),
                vec![config.n_embd],
            ),
            (
                format!("transformer.h.{layer_idx}.ln_2.weight"),
                vec![config.n_embd],
            ),
            (
                format!("transformer.h.{layer_idx}.ln_2.bias"),
                vec![config.n_embd],
            ),
            (
                format!("transformer.h.{layer_idx}.mlp.c_fc.weight"),
                vec![config.n_embd, inner],
            ),
            (
                format!("transformer.h.{layer_idx}.mlp.c_fc.bias"),
                vec![inner],
            ),
            (
                format!("transformer.h.{layer_idx}.mlp.c_proj.weight"),
                vec![inner, config.n_embd],
            ),
            (
                format!("transformer.h.{layer_idx}.mlp.c_proj.bias"),
                vec![config.n_embd],
            ),
        ]);
    }

    let mut errors = Vec::new();
    for (name, expected_shape) in required {
        let Some(header) = tensor_headers.get(&name) else {
            errors.push(format!("missing {name}"));
            continue;
        };
        let dtype = header.get("dtype").and_then(|value| value.as_str());
        if dtype != Some("F32") {
            errors.push(format!("{name} has dtype {:?}, expected F32", dtype));
        }
        let shape = header
            .get("shape")
            .and_then(|value| value.as_array())
            .map(|values| {
                values
                    .iter()
                    .filter_map(|value| value.as_u64().map(|value| value as usize))
                    .collect::<Vec<_>>()
            });
        if shape.as_deref() != Some(expected_shape.as_slice()) {
            errors.push(format!(
                "{name} has shape {:?}, expected {:?}",
                shape, expected_shape
            ));
        }
    }
    anyhow::ensure!(
        errors.is_empty(),
        "model.safetensors is not a loadable F32 GPT-2 checkpoint for Fathom's current Candle lane: {}",
        errors.join("; ")
    );
    Ok(())
}

fn mistral_eos_token_id(config: &CandleMistralConfig) -> Option<u32> {
    match config.eos_token_id.as_ref()? {
        serde_json::Value::Number(number) => number.as_u64().map(|value| value as u32),
        serde_json::Value::Array(values) => values
            .first()
            .and_then(|value| value.as_u64())
            .map(|value| value as u32),
        _ => None,
    }
}

fn gpt2_eos_token_id(config: &CandleGpt2Config) -> Option<u32> {
    match config.eos_token_id.as_ref()? {
        serde_json::Value::Number(number) => number.as_u64().map(|value| value as u32),
        serde_json::Value::Array(values) => values
            .first()
            .and_then(|value| value.as_u64())
            .map(|value| value as u32),
        _ => None,
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
struct RuntimeFileFingerprint {
    path: PathBuf,
    len: u64,
    modified: Option<SystemTime>,
}

impl RuntimeFileFingerprint {
    fn for_path(path: &Path) -> anyhow::Result<Self> {
        let metadata = fs::metadata(path)?;
        Ok(Self {
            path: path.canonicalize().unwrap_or_else(|_| path.to_path_buf()),
            len: metadata.len(),
            modified: metadata.modified().ok(),
        })
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
struct RuntimePackageKey {
    root: PathBuf,
    runtime_family: String,
    device: String,
    dtype: String,
    tokenizer: RuntimeFileFingerprint,
    config: Option<RuntimeFileFingerprint>,
    weights: RuntimeFileFingerprint,
    revision: Option<String>,
}

impl RuntimePackageKey {
    fn for_hf_safetensors(
        model_path: &Path,
        runtime_family: &str,
        dtype: &str,
        device: &str,
    ) -> anyhow::Result<Self> {
        Ok(Self {
            root: model_path
                .canonicalize()
                .unwrap_or_else(|_| model_path.to_path_buf()),
            runtime_family: runtime_family.to_string(),
            device: device.to_string(),
            dtype: dtype.to_string(),
            tokenizer: RuntimeFileFingerprint::for_path(&model_path.join("tokenizer.json"))?,
            config: RuntimeFileFingerprint::for_path(&model_path.join("config.json")).ok(),
            weights: RuntimeFileFingerprint::for_path(&model_path.join("model.safetensors"))?,
            revision: download_manifest_revision(model_path),
        })
    }

    #[cfg(feature = "onnx-embeddings-ort")]
    fn for_onnx_embedding(model_path: &Path, weights_path: &Path) -> anyhow::Result<Self> {
        Ok(Self {
            root: model_path
                .canonicalize()
                .unwrap_or_else(|_| model_path.to_path_buf()),
            runtime_family: "onnx_embedding".to_string(),
            device: "cpu".to_string(),
            dtype: "runtime_default".to_string(),
            tokenizer: RuntimeFileFingerprint::for_path(&model_path.join("tokenizer.json"))?,
            config: RuntimeFileFingerprint::for_path(&model_path.join("config.json")).ok(),
            weights: RuntimeFileFingerprint::for_path(weights_path)?,
            revision: download_manifest_revision(model_path),
        })
    }
}

fn download_manifest_revision(model_path: &Path) -> Option<String> {
    let manifest_path = model_path.join("fathom-download-manifest.json");
    let manifest = fs::read_to_string(manifest_path).ok()?;
    let value: serde_json::Value = serde_json::from_str(&manifest).ok()?;
    value
        .get("revision")
        .and_then(|revision| revision.as_str())
        .map(str::to_owned)
}

static TOKENIZER_CACHE: OnceLock<
    Mutex<HashMap<RuntimeFileFingerprint, Arc<tokenizers::Tokenizer>>>,
> = OnceLock::new();

fn load_tokenizer_cached(tokenizer_path: &Path) -> anyhow::Result<Arc<tokenizers::Tokenizer>> {
    let key = RuntimeFileFingerprint::for_path(tokenizer_path)?;
    let cache = TOKENIZER_CACHE.get_or_init(|| Mutex::new(HashMap::new()));
    if let Some(tokenizer) = cache
        .lock()
        .map_err(|_| anyhow::anyhow!("tokenizer cache lock was poisoned"))?
        .get(&key)
        .cloned()
    {
        return Ok(tokenizer);
    }
    let tokenizer = Arc::new(
        tokenizers::Tokenizer::from_file(tokenizer_path)
            .map_err(|error| anyhow::anyhow!("could not load tokenizer.json: {error}"))?,
    );
    cache
        .lock()
        .map_err(|_| anyhow::anyhow!("tokenizer cache lock was poisoned"))?
        .insert(key, tokenizer.clone());
    Ok(tokenizer)
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum CandleRuntimeCacheStatus {
    ColdLoaded,
    WarmReused,
}

impl CandleRuntimeCacheStatus {
    fn cache_hit(self) -> bool {
        matches!(self, Self::WarmReused)
    }

    fn residency(self) -> &'static str {
        match self {
            Self::ColdLoaded => "cold_loaded",
            Self::WarmReused => "warm_reused",
        }
    }
}

struct Gpt2RuntimeEntry {
    tokenizer: Arc<tokenizers::Tokenizer>,
    config: CandleGpt2Config,
    model: Gpt2Model,
    device: Device,
    eos_token_id: Option<u32>,
}

struct LlamaRuntimeEntry {
    tokenizer: Arc<tokenizers::Tokenizer>,
    config: CandleLlamaRuntimeConfig,
    model: Mutex<Llama>,
    device: Device,
    eos_token_ids: Vec<u32>,
}

struct Qwen2RuntimeEntry {
    tokenizer: Arc<tokenizers::Tokenizer>,
    config: CandleQwen2Config,
    model: Mutex<Qwen2ModelForCausalLM>,
    device: Device,
    eos_token_id: Option<u32>,
}

struct PhiRuntimeEntry {
    tokenizer: Arc<tokenizers::Tokenizer>,
    config: CandlePhiConfig,
    model: Mutex<PhiModel>,
    device: Device,
    eos_token_id: Option<u32>,
}

struct MistralRuntimeEntry {
    tokenizer: Arc<tokenizers::Tokenizer>,
    config: CandleMistralConfig,
    model: Mutex<MistralModel>,
    device: Device,
    eos_token_id: Option<u32>,
}

struct GemmaRuntimeEntry {
    tokenizer: Arc<tokenizers::Tokenizer>,
    config: CandleGemmaConfig,
    model: Mutex<GemmaModel>,
    device: Device,
    eos_token_id: Option<u32>,
}

struct Qwen2ModelResetGuard<'a> {
    model: MutexGuard<'a, Qwen2ModelForCausalLM>,
}

impl<'a> Qwen2ModelResetGuard<'a> {
    fn new(mut model: MutexGuard<'a, Qwen2ModelForCausalLM>) -> Self {
        model.clear_kv_cache();
        Self { model }
    }

    fn model_mut(&mut self) -> &mut Qwen2ModelForCausalLM {
        &mut self.model
    }
}

impl Drop for Qwen2ModelResetGuard<'_> {
    fn drop(&mut self) {
        self.model.clear_kv_cache();
    }
}

struct PhiModelResetGuard<'a> {
    model: MutexGuard<'a, PhiModel>,
}

impl<'a> PhiModelResetGuard<'a> {
    fn new(mut model: MutexGuard<'a, PhiModel>) -> Self {
        model.clear_kv_cache();
        Self { model }
    }

    fn model_mut(&mut self) -> &mut PhiModel {
        &mut self.model
    }
}

impl Drop for PhiModelResetGuard<'_> {
    fn drop(&mut self) {
        self.model.clear_kv_cache();
    }
}

struct MistralModelResetGuard<'a> {
    model: MutexGuard<'a, MistralModel>,
}

impl<'a> MistralModelResetGuard<'a> {
    fn new(mut model: MutexGuard<'a, MistralModel>) -> Self {
        model.clear_kv_cache();
        Self { model }
    }

    fn model_mut(&mut self) -> &mut MistralModel {
        &mut self.model
    }
}

impl Drop for MistralModelResetGuard<'_> {
    fn drop(&mut self) {
        self.model.clear_kv_cache();
    }
}

struct GemmaModelResetGuard<'a> {
    model: MutexGuard<'a, GemmaModel>,
}

impl<'a> GemmaModelResetGuard<'a> {
    fn new(mut model: MutexGuard<'a, GemmaModel>) -> Self {
        model.clear_kv_cache();
        Self { model }
    }

    fn model_mut(&mut self) -> &mut GemmaModel {
        &mut self.model
    }
}

impl Drop for GemmaModelResetGuard<'_> {
    fn drop(&mut self) {
        self.model.clear_kv_cache();
    }
}

enum CandleChatRuntimeEntry {
    Gpt2(Gpt2RuntimeEntry),
    Llama(LlamaRuntimeEntry),
    Qwen2(Qwen2RuntimeEntry),
    Phi(PhiRuntimeEntry),
    Mistral(MistralRuntimeEntry),
    Gemma(GemmaRuntimeEntry),
}

static CANDLE_CHAT_RUNTIME_CACHE: OnceLock<
    Mutex<HashMap<RuntimePackageKey, Arc<CandleChatRuntimeEntry>>>,
> = OnceLock::new();

#[cfg(test)]
fn candle_chat_runtime_cache_len() -> anyhow::Result<usize> {
    let cache = CANDLE_CHAT_RUNTIME_CACHE.get_or_init(|| Mutex::new(HashMap::new()));
    Ok(cache
        .lock()
        .map_err(|_| anyhow::anyhow!("Candle chat runtime cache lock was poisoned"))?
        .len())
}

#[cfg(test)]
fn clear_candle_chat_runtime_cache_for_tests() -> anyhow::Result<()> {
    let cache = CANDLE_CHAT_RUNTIME_CACHE.get_or_init(|| Mutex::new(HashMap::new()));
    cache
        .lock()
        .map_err(|_| anyhow::anyhow!("Candle chat runtime cache lock was poisoned"))?
        .clear();
    Ok(())
}

fn load_candle_gpt2_runtime_cached(
    model_path: &Path,
) -> anyhow::Result<(Arc<CandleChatRuntimeEntry>, CandleRuntimeCacheStatus, u128)> {
    let lookup_started = Instant::now();
    let key = RuntimePackageKey::for_hf_safetensors(model_path, "gpt2", "f32", "cpu")?;
    let cache = CANDLE_CHAT_RUNTIME_CACHE.get_or_init(|| Mutex::new(HashMap::new()));
    if let Some(runtime) = cache
        .lock()
        .map_err(|_| anyhow::anyhow!("Candle chat runtime cache lock was poisoned"))?
        .get(&key)
        .cloned()
    {
        return Ok((
            runtime,
            CandleRuntimeCacheStatus::WarmReused,
            lookup_started.elapsed().as_millis(),
        ));
    }
    let lookup_ms = lookup_started.elapsed().as_millis();

    let package = inspect_model_package(model_path)?;
    validate_candle_gpt2_supported_package(&package)?;
    let config_path = model_path.join("config.json");
    let tokenizer_path = model_path.join("tokenizer.json");
    let weights_path = model_path.join("model.safetensors");
    let config = read_candle_gpt2_config(&config_path)?;
    validate_candle_gpt2_config(&config)?;
    let tokenizer = load_tokenizer_cached(&tokenizer_path)?;
    let device = Device::Cpu;
    let vb = unsafe { VarBuilder::from_mmaped_safetensors(&[weights_path], DType::F32, &device)? };
    let model = Gpt2Model::load(config.clone(), vb)?;
    let eos_token_id = gpt2_eos_token_id(&config);
    let runtime = Arc::new(CandleChatRuntimeEntry::Gpt2(Gpt2RuntimeEntry {
        tokenizer,
        config,
        model,
        device,
        eos_token_id,
    }));

    let mut guard = cache
        .lock()
        .map_err(|_| anyhow::anyhow!("Candle chat runtime cache lock was poisoned"))?;
    if guard.len() >= 2 && !guard.contains_key(&key) {
        guard.clear();
    }
    let runtime = guard.entry(key).or_insert_with(|| runtime.clone()).clone();
    Ok((runtime, CandleRuntimeCacheStatus::ColdLoaded, lookup_ms))
}

fn llama_eos_token_ids(config: &candle_transformers::models::llama::LlamaConfig) -> Vec<u32> {
    use candle_transformers::models::llama::LlamaEosToks;

    match config.eos_token_id.as_ref() {
        Some(LlamaEosToks::Single(token)) => vec![*token],
        Some(LlamaEosToks::Multiple(tokens)) => tokens.clone(),
        None => Vec::new(),
    }
}

fn load_candle_llama_runtime_cached(
    model_path: &Path,
) -> anyhow::Result<(Arc<CandleChatRuntimeEntry>, CandleRuntimeCacheStatus, u128)> {
    let lookup_started = Instant::now();
    let key = RuntimePackageKey::for_hf_safetensors(model_path, "llama", "f32", "cpu")?;
    let cache = CANDLE_CHAT_RUNTIME_CACHE.get_or_init(|| Mutex::new(HashMap::new()));
    if let Some(runtime) = cache
        .lock()
        .map_err(|_| anyhow::anyhow!("Candle chat runtime cache lock was poisoned"))?
        .get(&key)
        .cloned()
    {
        return Ok((
            runtime,
            CandleRuntimeCacheStatus::WarmReused,
            lookup_started.elapsed().as_millis(),
        ));
    }
    let lookup_ms = lookup_started.elapsed().as_millis();

    let package = inspect_model_package(model_path)?;
    validate_candle_llama_supported_package(&package)?;
    let config_path = model_path.join("config.json");
    let tokenizer_path = model_path.join("tokenizer.json");
    let weights_path = model_path.join("model.safetensors");
    let raw_config: candle_transformers::models::llama::LlamaConfig =
        serde_json::from_slice(&fs::read(&config_path)?)?;
    let eos_token_ids = llama_eos_token_ids(&raw_config);
    let config = raw_config.into_config(false);
    let tokenizer = load_tokenizer_cached(&tokenizer_path)?;
    let device = Device::Cpu;
    let vb = unsafe { VarBuilder::from_mmaped_safetensors(&[weights_path], DType::F32, &device)? };
    let model = Llama::load(vb, &config)?;
    let runtime = Arc::new(CandleChatRuntimeEntry::Llama(LlamaRuntimeEntry {
        tokenizer,
        config,
        model: Mutex::new(model),
        device,
        eos_token_ids,
    }));

    let mut guard = cache
        .lock()
        .map_err(|_| anyhow::anyhow!("Candle chat runtime cache lock was poisoned"))?;
    if guard.len() >= 2 && !guard.contains_key(&key) {
        guard.clear();
    }
    let runtime = guard.entry(key).or_insert_with(|| runtime.clone()).clone();
    Ok((runtime, CandleRuntimeCacheStatus::ColdLoaded, lookup_ms))
}

fn load_candle_qwen2_runtime_cached(
    model_path: &Path,
) -> anyhow::Result<(Arc<CandleChatRuntimeEntry>, CandleRuntimeCacheStatus, u128)> {
    let lookup_started = Instant::now();
    let key = RuntimePackageKey::for_hf_safetensors(model_path, "qwen2", "f32", "cpu")?;
    let cache = CANDLE_CHAT_RUNTIME_CACHE.get_or_init(|| Mutex::new(HashMap::new()));
    if let Some(runtime) = cache
        .lock()
        .map_err(|_| anyhow::anyhow!("Candle chat runtime cache lock was poisoned"))?
        .get(&key)
        .cloned()
    {
        return Ok((
            runtime,
            CandleRuntimeCacheStatus::WarmReused,
            lookup_started.elapsed().as_millis(),
        ));
    }
    let lookup_ms = lookup_started.elapsed().as_millis();

    let package = inspect_model_package(model_path)?;
    validate_candle_qwen2_supported_package(&package)?;
    let config_path = model_path.join("config.json");
    let tokenizer_path = model_path.join("tokenizer.json");
    let weights_path = model_path.join("model.safetensors");
    let raw_config: candle_transformers::models::qwen2::Config =
        serde_json::from_slice(&fs::read(&config_path)?)?;
    let config = read_candle_qwen2_config(&config_path)?;
    validate_candle_qwen2_config(&config)?;
    let eos_token_id = qwen2_eos_token_id(&config);
    let tokenizer = load_tokenizer_cached(&tokenizer_path)?;
    let device = Device::Cpu;
    let vb = unsafe { VarBuilder::from_mmaped_safetensors(&[weights_path], DType::F32, &device)? };
    let model = Qwen2ModelForCausalLM::new(&raw_config, vb)?;
    let runtime = Arc::new(CandleChatRuntimeEntry::Qwen2(Qwen2RuntimeEntry {
        tokenizer,
        config,
        model: Mutex::new(model),
        device,
        eos_token_id,
    }));

    let mut guard = cache
        .lock()
        .map_err(|_| anyhow::anyhow!("Candle chat runtime cache lock was poisoned"))?;
    if guard.len() >= 2 && !guard.contains_key(&key) {
        guard.clear();
    }
    let runtime = guard.entry(key).or_insert_with(|| runtime.clone()).clone();
    Ok((runtime, CandleRuntimeCacheStatus::ColdLoaded, lookup_ms))
}

fn load_candle_phi_runtime_cached(
    model_path: &Path,
) -> anyhow::Result<(Arc<CandleChatRuntimeEntry>, CandleRuntimeCacheStatus, u128)> {
    let lookup_started = Instant::now();
    let key = RuntimePackageKey::for_hf_safetensors(model_path, "phi", "f32", "cpu")?;
    let cache = CANDLE_CHAT_RUNTIME_CACHE.get_or_init(|| Mutex::new(HashMap::new()));
    if let Some(runtime) = cache
        .lock()
        .map_err(|_| anyhow::anyhow!("Candle chat runtime cache lock was poisoned"))?
        .get(&key)
        .cloned()
    {
        return Ok((
            runtime,
            CandleRuntimeCacheStatus::WarmReused,
            lookup_started.elapsed().as_millis(),
        ));
    }
    let lookup_ms = lookup_started.elapsed().as_millis();

    let package = inspect_model_package(model_path)?;
    validate_candle_phi_supported_package(&package)?;
    let config_path = model_path.join("config.json");
    let tokenizer_path = model_path.join("tokenizer.json");
    let weights_path = model_path.join("model.safetensors");
    let raw_config: PhiRuntimeConfig = serde_json::from_slice(&fs::read(&config_path)?)?;
    let config = read_candle_phi_config(&config_path)?;
    validate_candle_phi_config(&config)?;
    let eos_token_id = phi_eos_token_id(&config);
    let tokenizer = load_tokenizer_cached(&tokenizer_path)?;
    let device = Device::Cpu;
    let vb = unsafe { VarBuilder::from_mmaped_safetensors(&[weights_path], DType::F32, &device)? };
    let model = PhiModel::new(&raw_config, vb)?;
    let runtime = Arc::new(CandleChatRuntimeEntry::Phi(PhiRuntimeEntry {
        tokenizer,
        config,
        model: Mutex::new(model),
        device,
        eos_token_id,
    }));

    let mut guard = cache
        .lock()
        .map_err(|_| anyhow::anyhow!("Candle chat runtime cache lock was poisoned"))?;
    if guard.len() >= 2 && !guard.contains_key(&key) {
        guard.clear();
    }
    let runtime = guard.entry(key).or_insert_with(|| runtime.clone()).clone();
    Ok((runtime, CandleRuntimeCacheStatus::ColdLoaded, lookup_ms))
}

fn load_candle_gemma_runtime_cached(
    model_path: &Path,
) -> anyhow::Result<(Arc<CandleChatRuntimeEntry>, CandleRuntimeCacheStatus, u128)> {
    let lookup_started = Instant::now();
    let key = RuntimePackageKey::for_hf_safetensors(model_path, "gemma", "f32", "cpu")?;
    let cache = CANDLE_CHAT_RUNTIME_CACHE.get_or_init(|| Mutex::new(HashMap::new()));
    if let Some(runtime) = cache
        .lock()
        .map_err(|_| anyhow::anyhow!("Candle chat runtime cache lock was poisoned"))?
        .get(&key)
        .cloned()
    {
        return Ok((
            runtime,
            CandleRuntimeCacheStatus::WarmReused,
            lookup_started.elapsed().as_millis(),
        ));
    }
    let lookup_ms = lookup_started.elapsed().as_millis();

    let package = inspect_model_package(model_path)?;
    validate_candle_gemma_supported_package(&package)?;
    let config_path = model_path.join("config.json");
    let tokenizer_path = model_path.join("tokenizer.json");
    let weights_path = model_path.join("model.safetensors");
    let raw_config: GemmaRuntimeConfig = serde_json::from_slice(&fs::read(&config_path)?)?;
    let config = read_candle_gemma_config(&config_path)?;
    validate_candle_gemma_config(&config)?;
    let eos_token_id = gemma_eos_token_id(&config);
    let tokenizer = load_tokenizer_cached(&tokenizer_path)?;
    let device = Device::Cpu;
    let vb = unsafe { VarBuilder::from_mmaped_safetensors(&[weights_path], DType::F32, &device)? };
    let model = GemmaModel::new(false, &raw_config, vb)?;
    let runtime = Arc::new(CandleChatRuntimeEntry::Gemma(GemmaRuntimeEntry {
        tokenizer,
        config,
        model: Mutex::new(model),
        device,
        eos_token_id,
    }));

    let mut guard = cache
        .lock()
        .map_err(|_| anyhow::anyhow!("Candle chat runtime cache lock was poisoned"))?;
    if guard.len() >= 2 && !guard.contains_key(&key) {
        guard.clear();
    }
    let runtime = guard.entry(key).or_insert_with(|| runtime.clone()).clone();
    Ok((runtime, CandleRuntimeCacheStatus::ColdLoaded, lookup_ms))
}

fn load_candle_mistral_runtime_cached(
    model_path: &Path,
) -> anyhow::Result<(Arc<CandleChatRuntimeEntry>, CandleRuntimeCacheStatus, u128)> {
    let lookup_started = Instant::now();
    let key = RuntimePackageKey::for_hf_safetensors(model_path, "mistral", "f32", "cpu")?;
    let cache = CANDLE_CHAT_RUNTIME_CACHE.get_or_init(|| Mutex::new(HashMap::new()));
    if let Some(runtime) = cache
        .lock()
        .map_err(|_| anyhow::anyhow!("Candle chat runtime cache lock was poisoned"))?
        .get(&key)
        .cloned()
    {
        return Ok((
            runtime,
            CandleRuntimeCacheStatus::WarmReused,
            lookup_started.elapsed().as_millis(),
        ));
    }
    let lookup_ms = lookup_started.elapsed().as_millis();

    let package = inspect_model_package(model_path)?;
    validate_candle_mistral_supported_package(&package)?;
    let config_path = model_path.join("config.json");
    let tokenizer_path = model_path.join("tokenizer.json");
    let weights_path = model_path.join("model.safetensors");
    let raw_config: CandleMistralRuntimeConfig = serde_json::from_slice(&fs::read(&config_path)?)?;
    let config = read_candle_mistral_config(&config_path)?;
    validate_candle_mistral_config(&config)?;
    let eos_token_id = mistral_eos_token_id(&config);
    let tokenizer = load_tokenizer_cached(&tokenizer_path)?;
    let device = Device::Cpu;
    let vb = unsafe { VarBuilder::from_mmaped_safetensors(&[weights_path], DType::F32, &device)? };
    let model = MistralModel::new(&raw_config, vb)?;
    let runtime = Arc::new(CandleChatRuntimeEntry::Mistral(MistralRuntimeEntry {
        tokenizer,
        config,
        model: Mutex::new(model),
        device,
        eos_token_id,
    }));

    let mut guard = cache
        .lock()
        .map_err(|_| anyhow::anyhow!("Candle chat runtime cache lock was poisoned"))?;
    if guard.len() >= 2 && !guard.contains_key(&key) {
        guard.clear();
    }
    let runtime = guard.entry(key).or_insert_with(|| runtime.clone()).clone();
    Ok((runtime, CandleRuntimeCacheStatus::ColdLoaded, lookup_ms))
}

#[cfg(feature = "onnx-embeddings-ort")]
struct OnnxEmbeddingRuntimeEntry {
    tokenizer: Arc<tokenizers::Tokenizer>,
    session: Mutex<ort::session::Session>,
    input_names: BTreeSet<String>,
}

#[cfg(feature = "onnx-embeddings-ort")]
static ONNX_EMBEDDING_RUNTIME_CACHE: OnceLock<
    Mutex<HashMap<RuntimePackageKey, Arc<OnnxEmbeddingRuntimeEntry>>>,
> = OnceLock::new();

#[cfg(feature = "onnx-embeddings-ort")]
fn load_onnx_embedding_runtime_cached(
    model_dir: &Path,
    model_path: &Path,
) -> anyhow::Result<Arc<OnnxEmbeddingRuntimeEntry>> {
    let key = RuntimePackageKey::for_onnx_embedding(model_dir, model_path)?;
    let cache = ONNX_EMBEDDING_RUNTIME_CACHE.get_or_init(|| Mutex::new(HashMap::new()));
    if let Some(runtime) = cache
        .lock()
        .map_err(|_| anyhow::anyhow!("ONNX embedding runtime cache lock was poisoned"))?
        .get(&key)
        .cloned()
    {
        return Ok(runtime);
    }

    let tokenizer = load_tokenizer_cached(&model_dir.join("tokenizer.json"))?;
    let session = ort::session::Session::builder()
        .map_err(|error| anyhow::anyhow!("could not create ONNX Runtime session builder: {error}"))?
        .with_intra_threads(1)
        .map_err(|error| {
            anyhow::anyhow!("could not configure ONNX Runtime session threads: {error}")
        })?
        .commit_from_file(model_path)
        .map_err(|error| anyhow::anyhow!("could not load ONNX embedding model: {error}"))?;
    let input_names = session
        .inputs()
        .iter()
        .map(|input| input.name().to_owned())
        .collect();
    let runtime = Arc::new(OnnxEmbeddingRuntimeEntry {
        tokenizer,
        session: Mutex::new(session),
        input_names,
    });
    cache
        .lock()
        .map_err(|_| anyhow::anyhow!("ONNX embedding runtime cache lock was poisoned"))?
        .insert(key, runtime.clone());
    Ok(runtime)
}

pub fn generate_with_candle_hf_options(
    model_path: impl AsRef<Path>,
    prompt: &str,
    max_tokens: usize,
    options: GenerationOptions,
) -> anyhow::Result<GenerationResponse> {
    let model_path = model_path.as_ref();
    let package = inspect_model_package(model_path)?;
    if ["config.json", "tokenizer.json", "model.safetensors"]
        .iter()
        .all(|filename| model_path.join(filename).exists())
    {
        ensure_required_artifacts_within_package_root(
            &package,
            &["config.json", "tokenizer.json", "model.safetensors"],
        )?;
    }
    if is_candle_gpt2_supported_package(&package) {
        generate_with_candle_gpt2_options(model_path, prompt, max_tokens, options)
    } else if is_candle_llama_supported_package(&package) {
        generate_with_candle_llama_options(model_path, prompt, max_tokens, options)
    } else if is_candle_qwen2_supported_package(&package) {
        generate_with_candle_qwen2_options(model_path, prompt, max_tokens, options)
    } else if is_candle_phi_supported_package(&package) {
        generate_with_candle_phi_options(model_path, prompt, max_tokens, options)
    } else if is_candle_mistral_supported_package(&package) {
        generate_with_candle_mistral_options(model_path, prompt, max_tokens, options)
    } else if is_candle_gemma_supported_package(&package) {
        generate_with_candle_gemma_options(model_path, prompt, max_tokens, options)
    } else {
        anyhow::bail!(
            "SafeTensors/HF package is not runnable by Fathom's current Candle GPT-2, Llama, Qwen2, Phi, Mistral, or Gemma lanes"
        )
    }
}

pub fn generate_with_candle_hf(
    model_path: impl AsRef<Path>,
    prompt: &str,
    max_tokens: usize,
) -> anyhow::Result<GenerationResponse> {
    generate_with_candle_hf_options(model_path, prompt, max_tokens, GenerationOptions::default())
}

pub fn generate_with_candle_gpt2(
    model_path: impl AsRef<Path>,
    prompt: &str,
    max_tokens: usize,
) -> anyhow::Result<GenerationResponse> {
    generate_with_candle_gpt2_options(model_path, prompt, max_tokens, GenerationOptions::default())
}

pub fn generate_with_candle_gpt2_options(
    model_path: impl AsRef<Path>,
    prompt: &str,
    max_tokens: usize,
    options: GenerationOptions,
) -> anyhow::Result<GenerationResponse> {
    let total_started = Instant::now();
    let load_started = Instant::now();
    let model_path = model_path.as_ref();
    let options = options.validate()?;
    for required in ["config.json", "tokenizer.json", "model.safetensors"] {
        let path = model_path.join(required);
        anyhow::ensure!(
            path.exists(),
            "missing {required} in {}",
            model_path.display()
        );
    }

    let (runtime, cache_status, runtime_cache_lookup_ms) =
        load_candle_gpt2_runtime_cached(model_path)?;
    let model_load_ms = if cache_status.cache_hit() {
        0
    } else {
        load_started.elapsed().as_millis()
    };
    let CandleChatRuntimeEntry::Gpt2(runtime) = runtime.as_ref() else {
        anyhow::bail!("cached runtime entry was not a GPT-2 runtime")
    };
    let tokenizer = runtime.tokenizer.clone();
    let config = &runtime.config;
    let model = &runtime.model;
    let device = &runtime.device;
    let eos_token_id = runtime.eos_token_id;

    let encoding = tokenizer
        .encode(prompt, true)
        .map_err(|error| anyhow::anyhow!("could not tokenize prompt: {error}"))?;
    let mut token_ids = encoding.get_ids().to_vec();
    anyhow::ensure!(!token_ids.is_empty(), "prompt encoded to zero tokens");
    let prompt_tokens = token_ids.len();
    let mut cache = Gpt2KvCache::new(config.n_layer);

    let generation_started = Instant::now();
    let requested_max_tokens = max_tokens.min(128);
    let mut finish_reason = GenerationFinishReason::Length;
    let mut first_token_elapsed = None;
    let mut prefilling_prompt = true;
    for _ in 0..requested_max_tokens {
        anyhow::ensure!(
            token_ids.len() < config.n_positions,
            "prompt plus generated tokens exceeded GPT-2 context window ({})",
            config.n_positions
        );
        let (step_tokens, index_pos) = cached_decode_step(&token_ids, prefilling_prompt);
        let input = Tensor::new(step_tokens, device)?.unsqueeze(0)?;
        let logits = model.forward(&input, index_pos, &mut cache)?;
        prefilling_prompt = false;
        let last = logits.i((0, step_tokens.len() - 1))?;
        let logits = last.to_vec1::<f32>()?;
        let generated = &token_ids[prompt_tokens..];
        let visible_so_far = tokenizer
            .decode(generated, true)
            .map(|text| !text.trim().is_empty())
            .unwrap_or(false);
        let next_token = if visible_so_far {
            select_next_token_from_logits(&logits, options)?
        } else {
            select_next_visible_token_from_logits(&logits, options, &tokenizer, generated)?
        };
        token_ids.push(next_token);
        if first_token_elapsed.is_none() {
            first_token_elapsed = Some(generation_started.elapsed());
        }
        if Some(next_token) == eos_token_id {
            finish_reason = GenerationFinishReason::Stop;
            break;
        }
    }

    let generated = &token_ids[prompt_tokens..];
    let generation_elapsed = generation_started.elapsed();
    let text = tokenizer
        .decode(generated, true)
        .map_err(|error| anyhow::anyhow!("could not decode generated tokens: {error}"))?;
    let completion_tokens = generated.len();
    let mut metrics = build_generation_metrics(
        model_load_ms,
        generation_elapsed,
        total_started.elapsed().as_millis(),
        prompt_tokens,
        completion_tokens,
        first_token_elapsed,
    );
    metrics.runtime_cache_hit = cache_status.cache_hit();
    metrics.runtime_cache_lookup_ms = runtime_cache_lookup_ms;
    metrics.runtime_residency = Some(cache_status.residency().to_string());
    metrics.runtime_family = Some("gpt2".to_string());
    Ok(GenerationResponse {
        text,
        prompt_tokens,
        completion_tokens,
        finish_reason,
        metrics,
    })
}

pub fn generate_with_candle_llama(
    model_path: impl AsRef<Path>,
    prompt: &str,
    max_tokens: usize,
) -> anyhow::Result<GenerationResponse> {
    generate_with_candle_llama_options(model_path, prompt, max_tokens, GenerationOptions::default())
}

pub fn generate_with_candle_llama_options(
    model_path: impl AsRef<Path>,
    prompt: &str,
    max_tokens: usize,
    options: GenerationOptions,
) -> anyhow::Result<GenerationResponse> {
    use candle_transformers::models::llama::Cache;

    let total_started = Instant::now();
    let load_started = Instant::now();
    let model_path = model_path.as_ref();
    let options = options.validate()?;
    for required in ["config.json", "tokenizer.json", "model.safetensors"] {
        let path = model_path.join(required);
        anyhow::ensure!(
            path.exists(),
            "missing {required} in {}",
            model_path.display()
        );
    }

    let (runtime, cache_status, runtime_cache_lookup_ms) =
        load_candle_llama_runtime_cached(model_path)?;
    let model_load_ms = if cache_status.cache_hit() {
        0
    } else {
        load_started.elapsed().as_millis()
    };
    let CandleChatRuntimeEntry::Llama(runtime) = runtime.as_ref() else {
        anyhow::bail!("cached runtime entry was not a Llama runtime")
    };
    let tokenizer = runtime.tokenizer.clone();
    let config = &runtime.config;
    let device = &runtime.device;
    let eos_token_ids = &runtime.eos_token_ids;
    let encoding = tokenizer
        .encode(prompt, true)
        .map_err(|error| anyhow::anyhow!("could not tokenize prompt: {error}"))?;
    let mut token_ids = encoding.get_ids().to_vec();
    anyhow::ensure!(!token_ids.is_empty(), "prompt encoded to zero tokens");
    let prompt_tokens = token_ids.len();
    let mut cache = Cache::new(true, DType::F32, config, device)?;
    let model = runtime
        .model
        .lock()
        .map_err(|_| anyhow::anyhow!("Llama runtime model lock was poisoned"))?;

    let generation_started = Instant::now();
    let requested_max_tokens = max_tokens.min(128);
    let mut finish_reason = GenerationFinishReason::Length;
    let mut first_token_elapsed = None;
    let mut prefilling_prompt = true;
    for _ in 0..requested_max_tokens {
        anyhow::ensure!(
            token_ids.len() < config.max_position_embeddings,
            "prompt plus generated tokens exceeded Llama context window ({})",
            config.max_position_embeddings
        );
        let (step_tokens, index_pos) = cached_decode_step(&token_ids, prefilling_prompt);
        let input = Tensor::new(step_tokens, device)?.unsqueeze(0)?;
        let logits = model.forward(&input, index_pos, &mut cache)?;
        prefilling_prompt = false;
        let logits = logits.i(0)?.to_vec1::<f32>()?;
        let generated = &token_ids[prompt_tokens..];
        let visible_so_far = tokenizer
            .decode(generated, true)
            .map(|text| !text.trim().is_empty())
            .unwrap_or(false);
        let next_token = if visible_so_far {
            select_next_token_from_logits(&logits, options)?
        } else {
            select_next_visible_token_from_logits(&logits, options, &tokenizer, generated)?
        };
        token_ids.push(next_token);
        if first_token_elapsed.is_none() {
            first_token_elapsed = Some(generation_started.elapsed());
        }
        if eos_token_ids.contains(&next_token) {
            finish_reason = GenerationFinishReason::Stop;
            break;
        }
    }

    drop(model);
    let generated = &token_ids[prompt_tokens..];
    let generation_elapsed = generation_started.elapsed();
    let text = tokenizer
        .decode(generated, true)
        .map_err(|error| anyhow::anyhow!("could not decode generated tokens: {error}"))?;
    let completion_tokens = generated.len();
    let mut metrics = build_generation_metrics(
        model_load_ms,
        generation_elapsed,
        total_started.elapsed().as_millis(),
        prompt_tokens,
        completion_tokens,
        first_token_elapsed,
    );
    metrics.runtime_cache_hit = cache_status.cache_hit();
    metrics.runtime_cache_lookup_ms = runtime_cache_lookup_ms;
    metrics.runtime_residency = Some(cache_status.residency().to_string());
    metrics.runtime_family = Some("llama".to_string());
    Ok(GenerationResponse {
        text,
        prompt_tokens,
        completion_tokens,
        finish_reason,
        metrics,
    })
}

pub fn generate_with_candle_phi(
    model_path: impl AsRef<Path>,
    prompt: &str,
    max_tokens: usize,
) -> anyhow::Result<GenerationResponse> {
    generate_with_candle_phi_options(model_path, prompt, max_tokens, GenerationOptions::default())
}

pub fn generate_with_candle_phi_options(
    model_path: impl AsRef<Path>,
    prompt: &str,
    max_tokens: usize,
    options: GenerationOptions,
) -> anyhow::Result<GenerationResponse> {
    let total_started = Instant::now();
    let load_started = Instant::now();
    let model_path = model_path.as_ref();
    let options = options.validate()?;
    for required in ["config.json", "tokenizer.json", "model.safetensors"] {
        let path = model_path.join(required);
        anyhow::ensure!(
            path.exists(),
            "missing {required} in {}",
            model_path.display()
        );
    }

    let (runtime, cache_status, runtime_cache_lookup_ms) =
        load_candle_phi_runtime_cached(model_path)?;
    let model_load_ms = if cache_status.cache_hit() {
        0
    } else {
        load_started.elapsed().as_millis()
    };
    let CandleChatRuntimeEntry::Phi(runtime) = runtime.as_ref() else {
        anyhow::bail!("cached runtime entry was not a Phi runtime")
    };
    let tokenizer = runtime.tokenizer.clone();
    let config = &runtime.config;
    let device = &runtime.device;
    let eos_token_id = runtime.eos_token_id;
    let encoding = tokenizer
        .encode(prompt, true)
        .map_err(|error| anyhow::anyhow!("could not tokenize prompt: {error}"))?;
    let mut token_ids = encoding.get_ids().to_vec();
    anyhow::ensure!(!token_ids.is_empty(), "prompt encoded to zero tokens");
    let prompt_tokens = token_ids.len();
    let mut model = PhiModelResetGuard::new(
        runtime
            .model
            .lock()
            .map_err(|_| anyhow::anyhow!("Phi runtime model lock was poisoned"))?,
    );

    let generation_started = Instant::now();
    let requested_max_tokens = max_tokens.min(128);
    let mut finish_reason = GenerationFinishReason::Length;
    let mut first_token_elapsed = None;
    let mut prefilling_prompt = true;
    for _ in 0..requested_max_tokens {
        anyhow::ensure!(
            token_ids.len() < config.max_position_embeddings,
            "prompt plus generated tokens exceeded Phi context window ({})",
            config.max_position_embeddings
        );
        let (step_tokens, _) = cached_decode_step(&token_ids, prefilling_prompt);
        let input = Tensor::new(step_tokens, device)?.unsqueeze(0)?;
        let logits = model.model_mut().forward(&input)?;
        prefilling_prompt = false;
        let logits = logits.i(0)?.to_vec1::<f32>()?;
        let generated = &token_ids[prompt_tokens..];
        let visible_so_far = tokenizer
            .decode(generated, true)
            .map(|text| !text.trim().is_empty())
            .unwrap_or(false);
        let next_token = if visible_so_far {
            select_next_token_from_logits(&logits, options)?
        } else {
            select_next_visible_token_from_logits(&logits, options, &tokenizer, generated)?
        };
        token_ids.push(next_token);
        if first_token_elapsed.is_none() {
            first_token_elapsed = Some(generation_started.elapsed());
        }
        if Some(next_token) == eos_token_id {
            finish_reason = GenerationFinishReason::Stop;
            break;
        }
    }

    drop(model);
    let generated = &token_ids[prompt_tokens..];
    let generation_elapsed = generation_started.elapsed();
    let text = tokenizer
        .decode(generated, true)
        .map_err(|error| anyhow::anyhow!("could not decode generated tokens: {error}"))?;
    let completion_tokens = generated.len();
    let mut metrics = build_generation_metrics(
        model_load_ms,
        generation_elapsed,
        total_started.elapsed().as_millis(),
        prompt_tokens,
        completion_tokens,
        first_token_elapsed,
    );
    metrics.runtime_cache_hit = cache_status.cache_hit();
    metrics.runtime_cache_lookup_ms = runtime_cache_lookup_ms;
    metrics.runtime_residency = Some(cache_status.residency().to_string());
    metrics.runtime_family = Some("phi".to_string());
    Ok(GenerationResponse {
        text,
        prompt_tokens,
        completion_tokens,
        finish_reason,
        metrics,
    })
}

pub fn generate_with_candle_qwen2(
    model_path: impl AsRef<Path>,
    prompt: &str,
    max_tokens: usize,
) -> anyhow::Result<GenerationResponse> {
    generate_with_candle_qwen2_options(model_path, prompt, max_tokens, GenerationOptions::default())
}

pub fn generate_with_candle_qwen2_options(
    model_path: impl AsRef<Path>,
    prompt: &str,
    max_tokens: usize,
    options: GenerationOptions,
) -> anyhow::Result<GenerationResponse> {
    let total_started = Instant::now();
    let load_started = Instant::now();
    let model_path = model_path.as_ref();
    let options = options.validate()?;
    for required in ["config.json", "tokenizer.json", "model.safetensors"] {
        let path = model_path.join(required);
        anyhow::ensure!(
            path.exists(),
            "missing {required} in {}",
            model_path.display()
        );
    }

    let (runtime, cache_status, runtime_cache_lookup_ms) =
        load_candle_qwen2_runtime_cached(model_path)?;
    let model_load_ms = if cache_status.cache_hit() {
        0
    } else {
        load_started.elapsed().as_millis()
    };
    let CandleChatRuntimeEntry::Qwen2(runtime) = runtime.as_ref() else {
        anyhow::bail!("cached runtime entry was not a Qwen2 runtime")
    };
    let tokenizer = runtime.tokenizer.clone();
    let config = &runtime.config;
    let device = &runtime.device;
    let eos_token_id = runtime.eos_token_id;
    let encoding = tokenizer
        .encode(prompt, true)
        .map_err(|error| anyhow::anyhow!("could not tokenize prompt: {error}"))?;
    let mut token_ids = encoding.get_ids().to_vec();
    anyhow::ensure!(!token_ids.is_empty(), "prompt encoded to zero tokens");
    let prompt_tokens = token_ids.len();
    let mut model = Qwen2ModelResetGuard::new(
        runtime
            .model
            .lock()
            .map_err(|_| anyhow::anyhow!("Qwen2 runtime model lock was poisoned"))?,
    );

    let generation_started = Instant::now();
    let requested_max_tokens = max_tokens.min(128);
    let mut finish_reason = GenerationFinishReason::Length;
    let mut first_token_elapsed = None;
    let mut prefilling_prompt = true;
    for _ in 0..requested_max_tokens {
        anyhow::ensure!(
            token_ids.len() < config.max_position_embeddings,
            "prompt plus generated tokens exceeded Qwen2 context window ({})",
            config.max_position_embeddings
        );
        let (step_tokens, index_pos) = cached_decode_step(&token_ids, prefilling_prompt);
        let input = Tensor::new(step_tokens, device)?.unsqueeze(0)?;
        let logits = model.model_mut().forward(&input, index_pos)?;
        prefilling_prompt = false;
        let logits = logits.i((0, 0))?.to_vec1::<f32>()?;
        let generated = &token_ids[prompt_tokens..];
        let visible_so_far = tokenizer
            .decode(generated, true)
            .map(|text| !text.trim().is_empty())
            .unwrap_or(false);
        let next_token = if visible_so_far {
            select_next_token_from_logits(&logits, options)?
        } else {
            select_next_visible_token_from_logits(&logits, options, &tokenizer, generated)?
        };
        token_ids.push(next_token);
        if first_token_elapsed.is_none() {
            first_token_elapsed = Some(generation_started.elapsed());
        }
        if Some(next_token) == eos_token_id {
            finish_reason = GenerationFinishReason::Stop;
            break;
        }
    }

    drop(model);
    let generated = &token_ids[prompt_tokens..];
    let generation_elapsed = generation_started.elapsed();
    let text = tokenizer
        .decode(generated, true)
        .map_err(|error| anyhow::anyhow!("could not decode generated tokens: {error}"))?;
    let completion_tokens = generated.len();
    let mut metrics = build_generation_metrics(
        model_load_ms,
        generation_elapsed,
        total_started.elapsed().as_millis(),
        prompt_tokens,
        completion_tokens,
        first_token_elapsed,
    );
    metrics.runtime_cache_hit = cache_status.cache_hit();
    metrics.runtime_cache_lookup_ms = runtime_cache_lookup_ms;
    metrics.runtime_residency = Some(cache_status.residency().to_string());
    metrics.runtime_family = Some("qwen2".to_string());
    Ok(GenerationResponse {
        text,
        prompt_tokens,
        completion_tokens,
        finish_reason,
        metrics,
    })
}

pub fn generate_with_candle_mistral(
    model_path: impl AsRef<Path>,
    prompt: &str,
    max_tokens: usize,
) -> anyhow::Result<GenerationResponse> {
    generate_with_candle_mistral_options(
        model_path,
        prompt,
        max_tokens,
        GenerationOptions::default(),
    )
}

pub fn generate_with_candle_mistral_options(
    model_path: impl AsRef<Path>,
    prompt: &str,
    max_tokens: usize,
    options: GenerationOptions,
) -> anyhow::Result<GenerationResponse> {
    let total_started = Instant::now();
    let load_started = Instant::now();
    let model_path = model_path.as_ref();
    let options = options.validate()?;
    for required in ["config.json", "tokenizer.json", "model.safetensors"] {
        let path = model_path.join(required);
        anyhow::ensure!(
            path.exists(),
            "missing {required} in {}",
            model_path.display()
        );
    }

    let (runtime, cache_status, runtime_cache_lookup_ms) =
        load_candle_mistral_runtime_cached(model_path)?;
    let model_load_ms = if cache_status.cache_hit() {
        0
    } else {
        load_started.elapsed().as_millis()
    };
    let CandleChatRuntimeEntry::Mistral(runtime) = runtime.as_ref() else {
        anyhow::bail!("cached runtime entry was not a Mistral runtime")
    };
    let tokenizer = runtime.tokenizer.clone();
    let config = &runtime.config;
    let device = &runtime.device;
    let eos_token_id = runtime.eos_token_id;
    let encoding = tokenizer
        .encode(prompt, true)
        .map_err(|error| anyhow::anyhow!("could not tokenize prompt: {error}"))?;
    let mut token_ids = encoding.get_ids().to_vec();
    anyhow::ensure!(!token_ids.is_empty(), "prompt encoded to zero tokens");
    let prompt_tokens = token_ids.len();
    let mut model = MistralModelResetGuard::new(
        runtime
            .model
            .lock()
            .map_err(|_| anyhow::anyhow!("Mistral runtime model lock was poisoned"))?,
    );

    let generation_started = Instant::now();
    let requested_max_tokens = max_tokens.min(128);
    let mut finish_reason = GenerationFinishReason::Length;
    let mut first_token_elapsed = None;
    let mut prefilling_prompt = true;
    for _ in 0..requested_max_tokens {
        anyhow::ensure!(
            token_ids.len() < config.max_position_embeddings,
            "prompt plus generated tokens exceeded Mistral context window ({})",
            config.max_position_embeddings
        );
        let (step_tokens, index_pos) = cached_decode_step(&token_ids, prefilling_prompt);
        let input = Tensor::new(step_tokens, device)?.unsqueeze(0)?;
        let logits = model.model_mut().forward(&input, index_pos)?;
        prefilling_prompt = false;
        let logits = logits.i((0, 0))?.to_vec1::<f32>()?;
        let generated = &token_ids[prompt_tokens..];
        let visible_so_far = tokenizer
            .decode(generated, true)
            .map(|text| !text.trim().is_empty())
            .unwrap_or(false);
        let next_token = if visible_so_far {
            select_next_token_from_logits(&logits, options)?
        } else {
            select_next_visible_token_from_logits(&logits, options, &tokenizer, generated)?
        };
        token_ids.push(next_token);
        if first_token_elapsed.is_none() {
            first_token_elapsed = Some(generation_started.elapsed());
        }
        if Some(next_token) == eos_token_id {
            finish_reason = GenerationFinishReason::Stop;
            break;
        }
    }

    drop(model);
    let generated = &token_ids[prompt_tokens..];
    let generation_elapsed = generation_started.elapsed();
    let text = tokenizer
        .decode(generated, true)
        .map_err(|error| anyhow::anyhow!("could not decode generated tokens: {error}"))?;
    let completion_tokens = generated.len();
    let mut metrics = build_generation_metrics(
        model_load_ms,
        generation_elapsed,
        total_started.elapsed().as_millis(),
        prompt_tokens,
        completion_tokens,
        first_token_elapsed,
    );
    metrics.runtime_cache_hit = cache_status.cache_hit();
    metrics.runtime_cache_lookup_ms = runtime_cache_lookup_ms;
    metrics.runtime_residency = Some(cache_status.residency().to_string());
    metrics.runtime_family = Some("mistral".to_string());
    Ok(GenerationResponse {
        text,
        prompt_tokens,
        completion_tokens,
        finish_reason,
        metrics,
    })
}

pub fn generate_with_candle_gemma(
    model_path: impl AsRef<Path>,
    prompt: &str,
    max_tokens: usize,
) -> anyhow::Result<GenerationResponse> {
    generate_with_candle_gemma_options(model_path, prompt, max_tokens, GenerationOptions::default())
}

pub fn generate_with_candle_gemma_options(
    model_path: impl AsRef<Path>,
    prompt: &str,
    max_tokens: usize,
    options: GenerationOptions,
) -> anyhow::Result<GenerationResponse> {
    let total_started = Instant::now();
    let load_started = Instant::now();
    let model_path = model_path.as_ref();
    let options = options.validate()?;
    for required in ["config.json", "tokenizer.json", "model.safetensors"] {
        let path = model_path.join(required);
        anyhow::ensure!(
            path.exists(),
            "missing {required} in {}",
            model_path.display()
        );
    }

    let (runtime, cache_status, runtime_cache_lookup_ms) =
        load_candle_gemma_runtime_cached(model_path)?;
    let model_load_ms = if cache_status.cache_hit() {
        0
    } else {
        load_started.elapsed().as_millis()
    };
    let CandleChatRuntimeEntry::Gemma(runtime) = runtime.as_ref() else {
        anyhow::bail!("cached runtime entry was not a Gemma runtime")
    };
    let tokenizer = runtime.tokenizer.clone();
    let config = &runtime.config;
    let device = &runtime.device;
    let eos_token_id = runtime.eos_token_id;
    let encoding = tokenizer
        .encode(prompt, true)
        .map_err(|error| anyhow::anyhow!("could not tokenize prompt: {error}"))?;
    let mut token_ids = encoding.get_ids().to_vec();
    anyhow::ensure!(!token_ids.is_empty(), "prompt encoded to zero tokens");
    let prompt_tokens = token_ids.len();
    let mut model = GemmaModelResetGuard::new(
        runtime
            .model
            .lock()
            .map_err(|_| anyhow::anyhow!("Gemma runtime model lock was poisoned"))?,
    );

    let generation_started = Instant::now();
    let requested_max_tokens = max_tokens.min(128);
    let mut finish_reason = GenerationFinishReason::Length;
    let mut first_token_elapsed = None;
    let mut prefilling_prompt = true;
    for _ in 0..requested_max_tokens {
        anyhow::ensure!(
            token_ids.len() < config.max_position_embeddings,
            "prompt plus generated tokens exceeded Gemma context window ({})",
            config.max_position_embeddings
        );
        let (step_tokens, index_pos) = cached_decode_step(&token_ids, prefilling_prompt);
        let input = Tensor::new(step_tokens, device)?.unsqueeze(0)?;
        let logits = model.model_mut().forward(&input, index_pos)?;
        prefilling_prompt = false;
        let logits = logits.i((0, 0))?.to_vec1::<f32>()?;
        let generated = &token_ids[prompt_tokens..];
        let visible_so_far = tokenizer
            .decode(generated, true)
            .map(|text| !text.trim().is_empty())
            .unwrap_or(false);
        let next_token = if visible_so_far {
            select_next_token_from_logits(&logits, options)?
        } else {
            select_next_visible_token_from_logits(&logits, options, &tokenizer, generated)?
        };
        token_ids.push(next_token);
        if first_token_elapsed.is_none() {
            first_token_elapsed = Some(generation_started.elapsed());
        }
        if Some(next_token) == eos_token_id {
            finish_reason = GenerationFinishReason::Stop;
            break;
        }
    }

    drop(model);
    let generated = &token_ids[prompt_tokens..];
    let generation_elapsed = generation_started.elapsed();
    let text = tokenizer
        .decode(generated, true)
        .map_err(|error| anyhow::anyhow!("could not decode generated tokens: {error}"))?;
    let completion_tokens = generated.len();
    let mut metrics = build_generation_metrics(
        model_load_ms,
        generation_elapsed,
        total_started.elapsed().as_millis(),
        prompt_tokens,
        completion_tokens,
        first_token_elapsed,
    );
    metrics.runtime_cache_hit = cache_status.cache_hit();
    metrics.runtime_cache_lookup_ms = runtime_cache_lookup_ms;
    metrics.runtime_residency = Some(cache_status.residency().to_string());
    metrics.runtime_family = Some("gemma".to_string());
    Ok(GenerationResponse {
        text,
        prompt_tokens,
        completion_tokens,
        finish_reason,
        metrics,
    })
}

fn cached_decode_step(token_ids: &[u32], prefilling_prompt: bool) -> (&[u32], usize) {
    if prefilling_prompt {
        (token_ids, 0)
    } else {
        let next_position = token_ids.len().saturating_sub(1);
        (&token_ids[next_position..], next_position)
    }
}

fn select_next_token_from_logits(
    logits: &[f32],
    options: GenerationOptions,
) -> anyhow::Result<u32> {
    Ok(ranked_token_candidates(logits, options)?[0])
}

fn select_next_visible_token_from_logits(
    logits: &[f32],
    options: GenerationOptions,
    tokenizer: &tokenizers::Tokenizer,
    generated: &[u32],
) -> anyhow::Result<u32> {
    let candidates = ranked_token_candidates(logits, options)?;
    for candidate in &candidates {
        let mut probe = generated.to_vec();
        probe.push(*candidate);
        let text = tokenizer.decode(&probe, true).map_err(|error| {
            anyhow::anyhow!("could not decode generated token candidate: {error}")
        })?;
        if !text.trim().is_empty() {
            return Ok(*candidate);
        }
    }
    Ok(candidates[0])
}

fn ranked_token_candidates(logits: &[f32], options: GenerationOptions) -> anyhow::Result<Vec<u32>> {
    let options = options.validate()?;
    anyhow::ensure!(!logits.is_empty(), "cannot sample from empty logits");
    let mut candidates = logits
        .iter()
        .enumerate()
        .filter_map(|(idx, value)| value.is_finite().then_some((idx, *value)))
        .collect::<Vec<_>>();
    anyhow::ensure!(!candidates.is_empty(), "logits contained no finite values");
    candidates.sort_by(|left, right| {
        right
            .1
            .partial_cmp(&left.1)
            .unwrap_or(std::cmp::Ordering::Equal)
    });

    if let Some(top_k) = options.top_k {
        candidates.truncate(top_k.min(candidates.len()));
    }

    if options.temperature == 0.0 {
        return Ok(candidates.into_iter().map(|(idx, _)| idx as u32).collect());
    }

    let temperature = options.temperature.max(1e-6);
    let max_logit = candidates[0].1;
    let mut weighted = candidates
        .into_iter()
        .map(|(idx, logit)| (idx, ((logit - max_logit) / temperature).exp()))
        .collect::<Vec<_>>();
    let total_weight = weighted.iter().map(|(_, weight)| *weight).sum::<f32>();
    anyhow::ensure!(
        total_weight.is_finite() && total_weight > 0.0,
        "invalid sampling weights"
    );

    for (_, weight) in &mut weighted {
        *weight /= total_weight;
    }

    if let Some(top_p) = options.top_p {
        let mut cumulative = 0.0f32;
        let mut keep = 0usize;
        for (_, probability) in &weighted {
            cumulative += *probability;
            keep += 1;
            if cumulative >= top_p {
                break;
            }
        }
        weighted.truncate(keep.max(1));
    }

    // Deterministic for reproducible local demos/tests: apply the OpenAI-style
    // temperature/top_k/top_p candidate restrictions, then choose the most likely
    // remaining token instead of introducing RNG into the backend lane.
    Ok(weighted.into_iter().map(|(idx, _)| idx as u32).collect())
}

pub fn current_machine_profile() -> MachineProfile {
    MachineProfile {
        os: std::env::consts::OS.to_string(),
        arch: std::env::consts::ARCH.to_string(),
        apple_platform: cfg!(target_os = "macos") || cfg!(target_os = "ios"),
        nvidia_requested: std::env::var("FATHOM_NVIDIA")
            .map(|value| value == "1" || value.eq_ignore_ascii_case("true"))
            .unwrap_or(false),
    }
}

pub fn backend_lanes_for_machine(machine: &MachineProfile) -> Vec<BackendLane> {
    vec![
        BackendLane {
            id: "safetensors-hf",
            name: "Custom Rust SafeTensors / HF runtime",
            kind: BackendLaneKind::SafeTensorsHf,
            status: CapabilityStatus::Runnable,
            summary: "Fathom's from-scratch Rust runtime can run verified HF SafeTensors causal-LM packages for GPT-2, Llama/Llama-style tied-embedding fixtures, Qwen2 including tied-output checkpoints when validation passes, Phi, Mistral, Gemma, and TinyStories GPT-2. Other HF/SafeTensors packages are only detected or planned until they pass a lane-specific loader and generation test.",
            formats: vec![ModelFormat::SafeTensors, ModelFormat::SafeTensorsIndex, ModelFormat::ConfigJson, ModelFormat::TokenizerJson, ModelFormat::TokenizerConfigJson, ModelFormat::SentencePiece, ModelFormat::ChatTemplate],
            blockers: vec![],
        },
        BackendLane {
            id: "gguf-native",
            name: "Native GGUF",
            kind: BackendLaneKind::Gguf,
            status: CapabilityStatus::MetadataOnly,
            summary: "Single-file GGUF lane can read safe header/key-value metadata, tensor descriptors, validated internal payload ranges, compatibility hints, and bounded internal tokenizer metadata for narrow synthetic GPT-2/BPE and Llama/SentencePiece shapes, with private fixture-scoped Llama/SentencePiece encode/decode parity helpers. It is metadata-only; public/runtime tokenizer execution, runtime weight loading, dequantization/kernels, architecture runtime, and generation are not implemented.",
            formats: vec![ModelFormat::Gguf],
            blockers: vec!["native_gguf_tokenizer_not_implemented", "gguf_runtime_weight_loading_not_implemented", "gguf_general_dequantization_not_implemented", "gguf_quantized_kernels_not_implemented", "architecture_runtime_not_implemented", "gguf_generation_not_implemented"],
        },
        BackendLane {
            id: "onnx",
            name: "ONNX graph runtime",
            kind: BackendLaneKind::Onnx,
            status: CapabilityStatus::Planned,
            summary: "General ONNX graph loading is planned, but Fathom does not claim ONNX chat/generation support yet.",
            formats: vec![ModelFormat::Onnx],
            blockers: vec!["ort/tract adapter decision", "tokenizer wrapper for text models", "generation loop adapter"],
        },
        BackendLane {
            id: "local-embeddings-retrieval",
            name: "Local embeddings + retrieval",
            kind: BackendLaneKind::LocalEmbeddingsRetrieval,
            status: CapabilityStatus::Planned,
            summary: "Bounded retrieval lane for verified local embedding models, a small vector index, and opt-in context assembly. Runtime targets are default-build Candle/SafeTensors MiniLM embeddings and feature-gated ONNX MiniLM embeddings; this is not chat/generation support.",
            formats: vec![ModelFormat::SafeTensors, ModelFormat::Onnx],
            blockers: vec!["general embedding model coverage", "document ingestion", "automatic query embedding in retrieval requests"],
        },
        BackendLane {
            id: "apple-mlx-coreml",
            name: "Apple MLX / Core ML",
            kind: BackendLaneKind::AppleMlxCoreMl,
            status: if machine.apple_platform { CapabilityStatus::Planned } else { CapabilityStatus::Unsupported },
            summary: "Apple-specific lane for MLX and Core ML artifacts.",
            formats: vec![ModelFormat::Mlx, ModelFormat::CoreMl],
            blockers: if machine.apple_platform { vec!["MLX/Core ML adapter", "platform-specific model execution contract"] } else { vec!["requires Apple platform"] },
        },
        BackendLane {
            id: "nvidia-tensorrt",
            name: "NVIDIA TensorRT",
            kind: BackendLaneKind::NvidiaTensorRt,
            status: if machine.nvidia_requested { CapabilityStatus::Planned } else { CapabilityStatus::Unsupported },
            summary: "NVIDIA-specific lane for prebuilt TensorRT plan/engine artifacts.",
            formats: vec![ModelFormat::TensorRtPlan],
            blockers: if machine.nvidia_requested { vec!["TensorRT runtime adapter", "GPU/runtime compatibility checks"] } else { vec!["NVIDIA runtime not requested/detected"] },
        },
        BackendLane {
            id: "pytorch-trusted-import",
            name: "PyTorch trusted import",
            kind: BackendLaneKind::PyTorchTrustedImport,
            status: CapabilityStatus::Blocked,
            summary: "Legacy PyTorch .bin artifacts require a strict trusted-import policy because pickle can execute code.",
            formats: vec![ModelFormat::PyTorchBin],
            blockers: vec!["pickle security policy", "prefer SafeTensors when available"],
        },
        BackendLane {
            id: "external-openai",
            name: "External OpenAI-compatible API",
            kind: BackendLaneKind::ExternalOpenAiCompatible,
            status: CapabilityStatus::Runnable,
            summary: "Compatibility lane for remote or already-running OpenAI-compatible APIs; not a local model artifact runtime.",
            formats: vec![],
            blockers: vec![],
        },
    ]
}

pub fn external_context_strategy_advice() -> ContextStrategyAdvice {
    ContextStrategyAdvice {
        label: "Endpoint-managed context".into(),
        engine: ContextEngineRecommendation::ExternalManaged,
        summary: "This model is reached through an OpenAI-compatible endpoint, so Fathom cannot truthfully inspect its real context window or retrieval stack yet. Keep local memories small before proxy support exposes provider metadata.".into(),
        max_context_tokens: None,
        reserve_output_tokens: 1024,
        recommended_chunk_tokens: 700,
        recommended_overlap_tokens: 120,
        top_k: 4,
        needs_retrieval: true,
        caveats: vec!["Context limit is unknown until the external endpoint reports it.".into()],
        suggested_workflow: vec![
            "Use concise pinned memories for profile/preferences.".into(),
            "For offline knowledge or coding, prefer a local indexed model package when available.".into(),
        ],
    }
}

pub fn recommend_context_strategy_for_package(package: &ModelPackage) -> ContextStrategyAdvice {
    let max_context_tokens = package
        .config_file
        .as_deref()
        .and_then(read_hf_context_window_hint);
    let model_type = package.model_type.as_deref().unwrap_or("unknown");
    let has_chat_template = package.hf_validation.has_chat_template;
    let has_tokenizer = package.hf_validation.has_tokenizer;
    let runnable_today = is_candle_gpt2_supported_package(package)
        || is_candle_llama_supported_package(package)
        || is_candle_qwen2_supported_package(package)
        || is_candle_phi_supported_package(package)
        || is_candle_mistral_supported_package(package)
        || is_candle_gemma_supported_package(package);

    let window = max_context_tokens.unwrap_or(4096);
    let reserve_output_tokens = if window <= 1024 {
        192
    } else if window <= 4096 {
        512
    } else {
        1024
    };
    let usable = window.saturating_sub(reserve_output_tokens).max(256);
    let small_context_needs_retrieval = usable < 12_000;
    let engine = if model_type.contains("qwen")
        || package
            .architectures
            .iter()
            .any(|arch| arch.to_lowercase().contains("coder"))
    {
        ContextEngineRecommendation::CodeIndex
    } else if small_context_needs_retrieval {
        ContextEngineRecommendation::RetrievalIndex
    } else if usable <= 2048 {
        ContextEngineRecommendation::LocalMemorySearch
    } else {
        ContextEngineRecommendation::InlinePrompt
    };
    let needs_retrieval = matches!(
        engine,
        ContextEngineRecommendation::RetrievalIndex | ContextEngineRecommendation::CodeIndex
    );
    let recommended_chunk_tokens = if usable <= 1024 {
        220
    } else if usable <= 4096 {
        500
    } else {
        900
    };
    let recommended_overlap_tokens = (recommended_chunk_tokens / 5).max(40);
    let top_k = if usable <= 1024 {
        3
    } else if usable <= 4096 {
        5
    } else {
        8
    };

    let mut caveats = Vec::new();
    if max_context_tokens.is_none() {
        caveats.push("No context-window metadata was found in config.json; recommendations use a conservative 4k-token assumption.".into());
    }
    if !has_tokenizer {
        caveats.push("Tokenizer metadata is missing, so token budgets are approximate.".into());
    }
    if !has_chat_template {
        caveats.push("No chat template is detected; chat-style context should stay plain until prompt formatting is faithful.".into());
    }
    if !runnable_today {
        caveats.push("The model is not runnable in Fathom yet; this is a planning recommendation, not a claim that retrieval is wired into generation.".into());
    }

    let label = match engine {
        ContextEngineRecommendation::CodeIndex => "Code-aware retrieval".to_string(),
        ContextEngineRecommendation::RetrievalIndex => "Small-window retrieval".to_string(),
        ContextEngineRecommendation::LocalMemorySearch => "Tiny local memory set".to_string(),
        ContextEngineRecommendation::InlinePrompt => "Inline context first".to_string(),
        ContextEngineRecommendation::ExternalManaged => "Endpoint-managed context".to_string(),
    };
    let summary = match engine {
        ContextEngineRecommendation::CodeIndex => "Use a local code index plus retrieval snippets; reserve most of the prompt for the active file, symbols, and the user task.".to_string(),
        ContextEngineRecommendation::RetrievalIndex => "Use chunked local retrieval for offline knowledge/coding; do not paste whole documents into this context window.".to_string(),
        ContextEngineRecommendation::LocalMemorySearch => "Keep only a few short pinned memories inline; this context window is too small for broad document retrieval.".to_string(),
        ContextEngineRecommendation::InlinePrompt => "The context window is large enough for curated inline context plus light retrieval, but still reserve output budget.".to_string(),
        ContextEngineRecommendation::ExternalManaged => unreachable!(),
    };

    ContextStrategyAdvice {
        label,
        engine,
        summary,
        max_context_tokens,
        reserve_output_tokens,
        recommended_chunk_tokens,
        recommended_overlap_tokens,
        top_k,
        needs_retrieval,
        caveats,
        suggested_workflow: vec![
            format!("Reserve ~{reserve_output_tokens} tokens for the answer."),
            format!("Index source material in ~{recommended_chunk_tokens}-token chunks with ~{recommended_overlap_tokens}-token overlap."),
            format!("Retrieve the top {top_k} chunks, then rerank by recency/path/task relevance before prompt assembly."),
        ],
    }
}

pub fn onnx_embeddings_ort_compiled() -> bool {
    cfg!(feature = "onnx-embeddings-ort")
}

pub fn generate_candle_bert_embeddings(
    model_dir: impl AsRef<Path>,
    request: &EmbeddingRequest<'_>,
) -> anyhow::Result<EmbeddingOutput> {
    use candle_core::{DType, Device, Tensor};
    use candle_nn::VarBuilder;
    use candle_transformers::models::bert::{BertModel, Config as BertConfig};

    let model_dir = model_dir.as_ref();
    if request.inputs.is_empty() {
        anyhow::bail!("embedding input cannot be empty");
    }
    if request.inputs.iter().any(|input| input.trim().is_empty()) {
        anyhow::bail!("embedding input strings cannot be empty");
    }

    let package = inspect_model_package(model_dir)?;
    validate_candle_bert_embedding_supported_package(&package)?;
    let config_path = model_dir.join("config.json");
    let tokenizer_path = model_dir.join("tokenizer.json");
    let weights_path = model_dir.join("model.safetensors");

    let total_start = Instant::now();
    let tokenizer_start = Instant::now();
    let tokenizer = load_tokenizer_cached(&tokenizer_path)?;
    let config: BertConfig = serde_json::from_slice(&fs::read(&config_path)?)?;
    let mut encodings = request
        .inputs
        .iter()
        .map(|input| {
            tokenizer
                .encode(input.as_str(), true)
                .map_err(|error| anyhow::anyhow!("could not tokenize embedding input: {error}"))
        })
        .collect::<anyhow::Result<Vec<_>>>()?;
    for encoding in &mut encodings {
        encoding.truncate(
            config.max_position_embeddings,
            0,
            tokenizers::TruncationDirection::Right,
        );
    }
    let batch = encodings.len();
    let sequence_len = encodings
        .iter()
        .map(|encoding| encoding.get_ids().len())
        .max()
        .ok_or_else(|| anyhow::anyhow!("embedding input cannot be empty"))?;
    anyhow::ensure!(sequence_len > 0, "embedding input encoded to zero tokens");

    let pad_id = tokenizer
        .get_padding()
        .map(|params| params.pad_id)
        .unwrap_or(config.pad_token_id as u32);
    let mut input_ids = vec![pad_id; batch * sequence_len];
    let mut attention_mask = vec![0_u32; batch * sequence_len];
    let mut token_type_ids = vec![0_u32; batch * sequence_len];

    for (row, encoding) in encodings.iter().enumerate() {
        for (col, token_id) in encoding.get_ids().iter().enumerate() {
            let offset = row * sequence_len + col;
            input_ids[offset] = *token_id;
        }
        for (col, mask) in encoding.get_attention_mask().iter().enumerate() {
            let offset = row * sequence_len + col;
            attention_mask[offset] = *mask;
        }
        for (col, type_id) in encoding.get_type_ids().iter().enumerate() {
            let offset = row * sequence_len + col;
            token_type_ids[offset] = *type_id;
        }
    }
    let tokenization_ms = tokenizer_start.elapsed().as_millis();

    let inference_start = Instant::now();
    let device = Device::Cpu;
    let input_ids_tensor = Tensor::from_vec(input_ids, (batch, sequence_len), &device)?;
    let token_type_ids_tensor = Tensor::from_vec(token_type_ids, (batch, sequence_len), &device)?;
    let attention_mask_tensor =
        Tensor::from_vec(attention_mask.clone(), (batch, sequence_len), &device)?;
    let vb = unsafe { VarBuilder::from_mmaped_safetensors(&[weights_path], DType::F32, &device)? };
    let model = BertModel::load(vb, &config)?;
    let hidden_states = model.forward(
        &input_ids_tensor,
        &token_type_ids_tensor,
        Some(&attention_mask_tensor),
    )?;
    let (out_batch, out_sequence_len, hidden_dimension) = hidden_states.dims3()?;
    let hidden_values = hidden_states.flatten_all()?.to_vec1::<f32>()?;
    let inference_ms = inference_start.elapsed().as_millis();

    let pooling_start = Instant::now();
    let attention_mask_i64 = attention_mask
        .into_iter()
        .map(i64::from)
        .collect::<Vec<_>>();
    let vectors = mean_pool_last_hidden_state_values(
        &hidden_values,
        out_batch,
        out_sequence_len,
        hidden_dimension,
        &attention_mask_i64,
        batch,
        sequence_len,
        config.hidden_size,
        request.normalize,
    )?;
    let pooling_ms = pooling_start.elapsed().as_millis();

    Ok(EmbeddingOutput {
        vectors,
        dimension: config.hidden_size,
        metrics: EmbeddingRuntimeMetrics {
            tokenization_ms,
            inference_ms,
            pooling_ms,
            total_ms: total_start.elapsed().as_millis(),
        },
    })
}

pub fn generate_onnx_embeddings(
    model_dir: impl AsRef<Path>,
    request: &EmbeddingRequest<'_>,
) -> anyhow::Result<EmbeddingOutput> {
    let model_dir = model_dir.as_ref();
    if request.inputs.is_empty() {
        anyhow::bail!("embedding input cannot be empty");
    }
    if request.inputs.iter().any(|input| input.trim().is_empty()) {
        anyhow::bail!("embedding input strings cannot be empty");
    }
    generate_onnx_embeddings_impl(model_dir, request)
}

#[cfg(not(feature = "onnx-embeddings-ort"))]
fn generate_onnx_embeddings_impl(
    _model_dir: &Path,
    _request: &EmbeddingRequest<'_>,
) -> anyhow::Result<EmbeddingOutput> {
    anyhow::bail!("ONNX embedding runtime was not compiled into this Fathom build")
}

#[cfg(feature = "onnx-embeddings-ort")]
fn generate_onnx_embeddings_impl(
    model_dir: &Path,
    request: &EmbeddingRequest<'_>,
) -> anyhow::Result<EmbeddingOutput> {
    let package = inspect_model_package(model_dir)?;
    let status = embedding_model_status_for_package(&package)
        .ok_or_else(|| anyhow::anyhow!("model package does not contain ONNX embedding metadata"))?;
    if status.task != ModelTaskKind::TextEmbedding {
        anyhow::bail!("ONNX package is not classified as a text-embedding model");
    }
    if package.tokenizer.is_none() {
        anyhow::bail!("ONNX embedding package is missing tokenizer metadata");
    }
    if status.embedding_dimension.is_none() {
        anyhow::bail!("ONNX embedding dimension was not found in config metadata");
    }
    if !status.runnable {
        let blocker = status.blockers.first().cloned().unwrap_or_else(|| {
            "ONNX embedding package is not runnable in this Fathom build".to_string()
        });
        anyhow::bail!(blocker);
    }
    run_onnx_embedding_session(model_dir, request, status.embedding_dimension.unwrap())
}

#[cfg(feature = "onnx-embeddings-ort")]
fn run_onnx_embedding_session(
    model_dir: &Path,
    request: &EmbeddingRequest<'_>,
    expected_dimension: usize,
) -> anyhow::Result<EmbeddingOutput> {
    use ort::value::TensorRef;

    let total_start = Instant::now();
    let model_path = find_onnx_embedding_model_path(model_dir)?;
    let runtime = load_onnx_embedding_runtime_cached(model_dir, &model_path)?;

    let tokenizer_start = Instant::now();
    let encodings = request
        .inputs
        .iter()
        .map(|input| {
            runtime
                .tokenizer
                .encode(input.as_str(), true)
                .map_err(|error| anyhow::anyhow!("could not tokenize embedding input: {error}"))
        })
        .collect::<anyhow::Result<Vec<_>>>()?;
    let batch = encodings.len();
    let sequence_len = encodings
        .iter()
        .map(|encoding| encoding.get_ids().len())
        .max()
        .ok_or_else(|| anyhow::anyhow!("embedding input cannot be empty"))?;
    anyhow::ensure!(sequence_len > 0, "embedding input encoded to zero tokens");

    let pad_id = runtime
        .tokenizer
        .get_padding()
        .map(|params| i64::from(params.pad_id))
        .unwrap_or(0);
    let mut input_ids = vec![pad_id; batch * sequence_len];
    let mut attention_mask = vec![0_i64; batch * sequence_len];
    let mut token_type_ids = vec![0_i64; batch * sequence_len];

    for (row, encoding) in encodings.iter().enumerate() {
        for (col, token_id) in encoding.get_ids().iter().enumerate() {
            let offset = row * sequence_len + col;
            input_ids[offset] = i64::from(*token_id);
        }
        for (col, mask) in encoding.get_attention_mask().iter().enumerate() {
            let offset = row * sequence_len + col;
            attention_mask[offset] = i64::from(*mask);
        }
        for (col, type_id) in encoding.get_type_ids().iter().enumerate() {
            let offset = row * sequence_len + col;
            token_type_ids[offset] = i64::from(*type_id);
        }
    }
    let tokenization_ms = tokenizer_start.elapsed().as_millis();

    let inference_start = Instant::now();
    let input_shape = [batch, sequence_len];
    let mut inputs: Vec<(
        std::borrow::Cow<'_, str>,
        ort::session::SessionInputValue<'_>,
    )> = vec![
        (
            "input_ids".into(),
            TensorRef::from_array_view((input_shape, &input_ids[..]))?.into(),
        ),
        (
            "attention_mask".into(),
            TensorRef::from_array_view((input_shape, &attention_mask[..]))?.into(),
        ),
    ];
    if runtime.input_names.contains("token_type_ids") {
        inputs.push((
            "token_type_ids".into(),
            TensorRef::from_array_view((input_shape, &token_type_ids[..]))?.into(),
        ));
    }

    let mut session = runtime
        .session
        .lock()
        .map_err(|_| anyhow::anyhow!("ONNX embedding session cache lock was poisoned"))?;
    let outputs = session
        .run(inputs)
        .map_err(|error| anyhow::anyhow!("ONNX embedding inference failed: {error}"))?;
    let inference_ms = inference_start.elapsed().as_millis();

    let pooling_start = Instant::now();
    let output = outputs
        .get("last_hidden_state")
        .or_else(|| outputs.get("output_0"))
        .unwrap_or(&outputs[0]);
    let (shape, hidden_states) = output
        .try_extract_tensor::<f32>()
        .map_err(|error| anyhow::anyhow!("ONNX embedding output was not an f32 tensor: {error}"))?;
    let vectors = mean_pool_last_hidden_state(
        hidden_states,
        &shape[..],
        &attention_mask,
        batch,
        sequence_len,
        expected_dimension,
        request.normalize,
    )?;
    let pooling_ms = pooling_start.elapsed().as_millis();

    Ok(EmbeddingOutput {
        vectors,
        dimension: expected_dimension,
        metrics: EmbeddingRuntimeMetrics {
            tokenization_ms,
            inference_ms,
            pooling_ms,
            total_ms: total_start.elapsed().as_millis(),
        },
    })
}

#[cfg(feature = "onnx-embeddings-ort")]
fn find_onnx_embedding_model_path(model_dir: &Path) -> anyhow::Result<PathBuf> {
    for filename in ["model_quantized.onnx", "model.onnx"] {
        let candidate = model_dir.join(filename);
        if candidate.exists() {
            return Ok(candidate);
        }
    }
    fs::read_dir(model_dir)?
        .filter_map(Result::ok)
        .map(|entry| entry.path())
        .find(|path| {
            path.extension()
                .and_then(|extension| extension.to_str())
                .is_some_and(|extension| extension.eq_ignore_ascii_case("onnx"))
        })
        .ok_or_else(|| {
            anyhow::anyhow!("ONNX embedding package is missing model_quantized.onnx or model.onnx")
        })
}

#[cfg(any(test, feature = "onnx-embeddings-ort"))]
fn mean_pool_last_hidden_state(
    hidden_states: &[f32],
    shape: &[i64],
    attention_mask: &[i64],
    batch: usize,
    sequence_len: usize,
    expected_dimension: usize,
    normalize: bool,
) -> anyhow::Result<Vec<EmbeddingVector>> {
    anyhow::ensure!(
        shape.len() == 3,
        "ONNX embedding output must be rank-3 [batch, sequence, hidden], got {shape:?}"
    );
    mean_pool_last_hidden_state_values(
        hidden_states,
        shape[0] as usize,
        shape[1] as usize,
        shape[2] as usize,
        attention_mask,
        batch,
        sequence_len,
        expected_dimension,
        normalize,
    )
}

fn mean_pool_last_hidden_state_values(
    hidden_states: &[f32],
    output_batch: usize,
    output_sequence_len: usize,
    hidden_dimension: usize,
    attention_mask: &[i64],
    batch: usize,
    sequence_len: usize,
    expected_dimension: usize,
    normalize: bool,
) -> anyhow::Result<Vec<EmbeddingVector>> {
    anyhow::ensure!(
        output_batch == batch && output_sequence_len == sequence_len,
        "embedding output shape [{output_batch}, {output_sequence_len}, {hidden_dimension}] does not match tokenized batch [{batch}, {sequence_len}]"
    );
    anyhow::ensure!(
        hidden_dimension == expected_dimension,
        "embedding dimension {hidden_dimension} did not match expected {expected_dimension}"
    );
    anyhow::ensure!(
        hidden_states.len() == batch * sequence_len * hidden_dimension,
        "embedding output tensor length did not match its shape"
    );
    anyhow::ensure!(
        attention_mask.len() == batch * sequence_len,
        "attention mask length did not match tokenized batch"
    );

    let mut vectors = Vec::with_capacity(batch);
    for row in 0..batch {
        let mut values = vec![0_f32; hidden_dimension];
        let mut token_count = 0_f32;
        for token in 0..sequence_len {
            if attention_mask[row * sequence_len + token] == 0 {
                continue;
            }
            token_count += 1.0;
            let base = (row * sequence_len + token) * hidden_dimension;
            for dim in 0..hidden_dimension {
                values[dim] += hidden_states[base + dim];
            }
        }
        anyhow::ensure!(
            token_count > 0.0,
            "embedding input had no attended tokens after tokenization"
        );
        for value in &mut values {
            *value /= token_count;
        }
        if normalize {
            l2_normalize(&mut values)?;
        }
        vectors.push(EmbeddingVector::new(values)?);
    }
    Ok(vectors)
}

fn l2_normalize(values: &mut [f32]) -> anyhow::Result<()> {
    let norm = values.iter().map(|value| value * value).sum::<f32>().sqrt();
    anyhow::ensure!(
        norm.is_finite() && norm > 0.0,
        "embedding vector cannot be L2-normalized because its norm is zero or non-finite"
    );
    for value in values {
        *value /= norm;
    }
    Ok(())
}

const PINNED_MINILM_ONNX_EMBEDDING_FILENAME: &str = "model_quantized.onnx";
const PINNED_MINILM_ONNX_EMBEDDING_BYTES: u64 = 22_972_869;
const ONNX_EMBEDDING_PINNED_FIXTURE_BLOCKER: &str =
    "ONNX embedding lane only accepts the pinned MiniLM model_quantized.onnx fixture";
const ONNX_EMBEDDING_EXTERNAL_DATA_BLOCKER: &str =
    "ONNX embedding lane does not accept external-data sidecars";
const ONNX_EMBEDDING_CUSTOM_OP_BLOCKER: &str =
    "ONNX embedding lane does not accept custom-op, plugin, or shared-library configuration";

#[derive(Debug, Clone)]
struct OnnxEmbeddingPackagePolicy;

impl OnnxEmbeddingPackagePolicy {
    fn validate(package: &ModelPackage) -> anyhow::Result<Self> {
        // This is intentionally a narrow package preflight for the current
        // feature-gated MiniLM embedding lane. Full ONNX graph/opset metadata
        // parsing is deferred because robust protobuf ONNX inspection would add
        // dependency surface; the lane instead enforces the single-file pinned
        // fixture contract before ORT sees any path.
        let selected_artifact = selected_pinned_minilm_onnx_embedding_artifact(package)
            .ok_or_else(|| anyhow::anyhow!(ONNX_EMBEDDING_PINNED_FIXTURE_BLOCKER))?
            .to_path_buf();
        ensure_onnx_embedding_artifacts_within_package_root(package, &selected_artifact)?;
        ensure_no_onnx_embedding_external_data_sidecars(package, &selected_artifact)?;
        ensure_no_onnx_embedding_custom_op_configuration(package)?;
        Ok(Self)
    }
}

fn selected_pinned_minilm_onnx_embedding_artifact(package: &ModelPackage) -> Option<&Path> {
    package
        .artifacts
        .iter()
        .find(|artifact| {
            artifact.format == ModelFormat::Onnx
                && artifact
                    .path
                    .file_name()
                    .and_then(|name| name.to_str())
                    .is_some_and(|name| name == PINNED_MINILM_ONNX_EMBEDDING_FILENAME)
                && fs::metadata(&artifact.path)
                    .map(|metadata| metadata.len() == PINNED_MINILM_ONNX_EMBEDDING_BYTES)
                    .unwrap_or(false)
        })
        .map(|artifact| artifact.path.as_path())
}

fn ensure_onnx_embedding_artifacts_within_package_root(
    package: &ModelPackage,
    selected_artifact: &Path,
) -> anyhow::Result<()> {
    ensure_required_artifacts_within_package_root(package, &["config.json", "tokenizer.json"])?;
    let root = package.root.canonicalize().map_err(|_| {
        anyhow::anyhow!("{ARTIFACT_PATH_ESCAPE_BLOCKER}: could not canonicalize package root")
    })?;
    ensure_artifact_within_package_root(&package.root, &root, selected_artifact)
}

fn ensure_no_onnx_embedding_external_data_sidecars(
    package: &ModelPackage,
    selected_artifact: &Path,
) -> anyhow::Result<()> {
    let selected_name = selected_artifact
        .file_name()
        .and_then(|name| name.to_str())
        .unwrap_or(PINNED_MINILM_ONNX_EMBEDDING_FILENAME);
    let selected_stem = selected_artifact
        .file_stem()
        .and_then(|name| name.to_str())
        .unwrap_or("model_quantized");
    for entry in fs::read_dir(&package.root)? {
        let entry = entry?;
        let name = entry.file_name();
        let Some(name) = name.to_str() else {
            continue;
        };
        let lowered = name.to_ascii_lowercase();
        let looks_like_external_data_file = lowered.ends_with(".onnx.data")
            || lowered == format!("{selected_name}.data").to_ascii_lowercase()
            || lowered == format!("{selected_stem}.data").to_ascii_lowercase();
        let looks_like_external_data_dir = entry.path().is_dir()
            && matches!(
                lowered.as_str(),
                "data" | "external_data" | "external-data" | "onnx_data" | "onnx-data"
            );
        if looks_like_external_data_file || looks_like_external_data_dir {
            anyhow::bail!(ONNX_EMBEDDING_EXTERNAL_DATA_BLOCKER);
        }
    }
    Ok(())
}

fn ensure_no_onnx_embedding_custom_op_configuration(package: &ModelPackage) -> anyhow::Result<()> {
    for entry in fs::read_dir(&package.root)? {
        let entry = entry?;
        let path = entry.path();
        let name = entry.file_name();
        let lowered = name.to_string_lossy().to_ascii_lowercase();
        let extension = path
            .extension()
            .and_then(|extension| extension.to_str())
            .unwrap_or_default()
            .to_ascii_lowercase();
        if path.is_dir()
            && matches!(
                lowered.as_str(),
                "custom_ops" | "custom-ops" | "plugins" | "onnx_plugins" | "onnx-plugins"
            )
        {
            anyhow::bail!(ONNX_EMBEDDING_CUSTOM_OP_BLOCKER);
        }
        if matches!(extension.as_str(), "so" | "dylib" | "dll")
            || lowered.contains("custom_op")
            || lowered.contains("custom-ops")
            || lowered.contains("onnxruntime_extensions")
        {
            anyhow::bail!(ONNX_EMBEDDING_CUSTOM_OP_BLOCKER);
        }
    }

    for path in [
        &package.root.join("config.json"),
        &package.root.join("onnx_config.json"),
    ] {
        if path.exists() && json_mentions_onnx_custom_op_configuration(path)? {
            anyhow::bail!(ONNX_EMBEDDING_CUSTOM_OP_BLOCKER);
        }
    }
    Ok(())
}

fn json_mentions_onnx_custom_op_configuration(path: &Path) -> anyhow::Result<bool> {
    let value: serde_json::Value = serde_json::from_slice(&fs::read(path)?)?;
    Ok(json_value_mentions_onnx_custom_op_configuration(&value))
}

fn json_value_mentions_onnx_custom_op_configuration(value: &serde_json::Value) -> bool {
    match value {
        serde_json::Value::Object(map) => map.iter().any(|(key, value)| {
            let lowered = key.to_ascii_lowercase();
            matches!(
                lowered.as_str(),
                "custom_op"
                    | "custom_ops"
                    | "custom_operator"
                    | "custom_operators"
                    | "custom_op_library"
                    | "custom_op_library_path"
                    | "shared_library"
                    | "shared_libraries"
                    | "plugin"
                    | "plugins"
                    | "library_path"
                    | "onnxruntime_extensions"
            ) || json_value_mentions_onnx_custom_op_configuration(value)
        }),
        serde_json::Value::Array(values) => values
            .iter()
            .any(json_value_mentions_onnx_custom_op_configuration),
        serde_json::Value::String(value) => {
            let lowered = value.to_ascii_lowercase();
            lowered.ends_with(".so")
                || lowered.ends_with(".dylib")
                || lowered.ends_with(".dll")
                || lowered.contains("custom_op")
                || lowered.contains("onnxruntime_extensions")
        }
        _ => false,
    }
}

pub fn embedding_model_status_for_package(package: &ModelPackage) -> Option<EmbeddingModelStatus> {
    let has_onnx = package
        .artifacts
        .iter()
        .any(|artifact| artifact.format == ModelFormat::Onnx);
    let has_safetensors = package.artifacts.iter().any(|artifact| {
        matches!(
            artifact.format,
            ModelFormat::SafeTensors | ModelFormat::SafeTensorsIndex
        )
    });
    let task = infer_model_task_kind(package);
    let dimension = package
        .config_file
        .as_deref()
        .and_then(read_embedding_dimension_hint);

    if has_safetensors
        && package.model_type.as_deref() == Some("bert")
        && package
            .architectures
            .iter()
            .any(|architecture| architecture == "BertModel")
    {
        let has_tokenizer = package.tokenizer.is_some();
        let validation_error = validate_candle_bert_embedding_supported_package(package)
            .err()
            .map(|error| error.to_string());
        let runnable = validation_error.is_none() && task == ModelTaskKind::TextEmbedding;
        let mut blockers = Vec::new();
        if !has_tokenizer {
            blockers.push("Embedding package is missing tokenizer.json metadata.".to_string());
        }
        if task != ModelTaskKind::TextEmbedding {
            blockers.push("Package metadata does not identify a text-embedding task.".to_string());
        }
        if let Some(error) = validation_error {
            blockers.push(error);
        }
        let summary = if runnable {
            "Fathom can run this verified BertModel/MiniLM SafeTensors/HF text-embedding package through its default custom Rust/Candle embedding lane and return pooled normalized vectors. This is not chat/generation support.".to_string()
        } else {
            "Fathom detects this as a BERT/MiniLM SafeTensors/HF embedding package, but it is not runnable until config, tokenizer, task metadata, and SafeTensors headers match the verified embedding lane.".to_string()
        };
        return Some(EmbeddingModelStatus {
            task,
            status: if runnable {
                CapabilityStatus::Runnable
            } else if has_tokenizer {
                CapabilityStatus::MetadataOnly
            } else {
                CapabilityStatus::Planned
            },
            runtime_lane: "candle-bert-embeddings",
            runtime_installed: true,
            runnable,
            summary,
            embedding_dimension: dimension,
            blockers,
        });
    }

    if !has_onnx {
        return None;
    }

    let has_tokenizer = package.tokenizer.is_some();
    let runtime_installed = onnx_embeddings_ort_compiled();
    let policy_result =
        if has_tokenizer && dimension == Some(384) && task == ModelTaskKind::TextEmbedding {
            Some(OnnxEmbeddingPackagePolicy::validate(package))
        } else {
            None
        };
    let policy_error = policy_result
        .as_ref()
        .and_then(|result| result.as_ref().err().map(|error| error.to_string()));
    let policy_valid = policy_result.as_ref().is_some_and(|result| result.is_ok());
    let policy_failed = policy_error.is_some();
    let runnable = runtime_installed
        && has_tokenizer
        && dimension == Some(384)
        && task == ModelTaskKind::TextEmbedding
        && policy_valid;
    let mut blockers = Vec::new();
    if !runtime_installed {
        blockers.push(
            "ONNX embedding runtime adapter is not installed in this Fathom build.".to_string(),
        );
    }
    if !has_tokenizer {
        blockers.push(
            "Embedding package is missing tokenizer.json or SentencePiece tokenizer metadata."
                .to_string(),
        );
    }
    if dimension.is_none() {
        blockers.push("Embedding dimension was not found in config.json; runtime verification is required before indexing.".to_string());
    }
    if let Some(error) = policy_error {
        blockers.push(error);
    }

    let status = if runnable {
        CapabilityStatus::Runnable
    } else if policy_failed {
        CapabilityStatus::Planned
    } else if has_tokenizer {
        CapabilityStatus::MetadataOnly
    } else {
        CapabilityStatus::Planned
    };
    let summary = if runnable {
        "Fathom can run this verified ONNX text-embedding package through the non-default onnx-embeddings-ort lane and return pooled embedding vectors. This is not ONNX chat/LLM support.".to_string()
    } else if task == ModelTaskKind::TextEmbedding {
        "Fathom detects this as a likely ONNX text-embedding package and can inspect its metadata, but it is not runnable unless the non-default ONNX embeddings runtime is compiled and metadata validates. This is not ONNX chat/LLM support.".to_string()
    } else {
        "Fathom detects ONNX artifacts, but cannot truthfully classify this package as a runnable embedding model yet. ONNX chat/generation is not supported.".to_string()
    };

    Some(EmbeddingModelStatus {
        task,
        status,
        runtime_lane: "onnx-embeddings",
        runtime_installed,
        runnable,
        summary,
        embedding_dimension: dimension,
        blockers,
    })
}

pub fn capability_report_for_package(
    package: ModelPackage,
    machine: &MachineProfile,
) -> PackageCapabilityReport {
    let lanes = backend_lanes_for_machine(machine);
    let artifact_formats: Vec<ModelFormat> = package
        .artifacts
        .iter()
        .map(|artifact| artifact.format.clone())
        .collect();
    let mut matching_lanes: Vec<BackendLane> = lanes
        .iter()
        .cloned()
        .filter(|lane| lane_matches_artifacts(lane, &artifact_formats))
        .collect();
    let verified_embedding_lane = is_candle_bert_embedding_supported_package(&package);
    if verified_embedding_lane
        && !matching_lanes
            .iter()
            .any(|lane| lane.kind == BackendLaneKind::LocalEmbeddingsRetrieval)
    {
        if let Some(lane) = lanes
            .iter()
            .find(|lane| lane.kind == BackendLaneKind::LocalEmbeddingsRetrieval)
            .cloned()
        {
            matching_lanes.push(lane);
        }
    }

    let mut best_status = best_capability_status(&matching_lanes);
    let has_safetensors_artifact = package.artifacts.iter().any(|artifact| {
        matches!(
            artifact.format,
            ModelFormat::SafeTensors | ModelFormat::SafeTensorsIndex
        )
    });
    let has_gguf_metadata = package
        .artifacts
        .iter()
        .any(|artifact| artifact.format == ModelFormat::Gguf && artifact.gguf_metadata.is_some());
    let verified_custom_rust_lane = is_candle_gpt2_supported_package(&package)
        || is_candle_llama_supported_package(&package)
        || is_candle_qwen2_supported_package(&package)
        || is_candle_phi_supported_package(&package)
        || is_candle_mistral_supported_package(&package)
        || is_candle_gemma_supported_package(&package);
    let matches_safetensors_hf_lane = matching_lanes
        .iter()
        .any(|lane| lane.kind == BackendLaneKind::SafeTensorsHf);
    if verified_custom_rust_lane || verified_embedding_lane {
        best_status = CapabilityStatus::Runnable;
    } else if has_gguf_metadata {
        best_status = CapabilityStatus::MetadataOnly;
    } else if has_safetensors_artifact && !package.hf_validation.ready_for_loader_metadata {
        best_status = CapabilityStatus::MetadataOnly;
    } else if matches_safetensors_hf_lane && best_status == CapabilityStatus::Runnable {
        best_status = CapabilityStatus::Planned;
    }
    let runnable = verified_custom_rust_lane
        || verified_embedding_lane
        || (best_status == CapabilityStatus::Runnable && !matches_safetensors_hf_lane);
    let summary = if matching_lanes.is_empty() {
        "Fathom can inspect this path, but no backend lane recognizes the model artifact yet."
            .to_string()
    } else if is_candle_gpt2_supported_package(&package) {
        "Fathom can run this verified GPT2LMHeadModel SafeTensors/HF package through its custom Rust GPT-2 lane (Candle GPT-2 runtime).".to_string()
    } else if is_candle_llama_supported_package(&package) {
        "Fathom can run this verified LlamaForCausalLM SafeTensors/HF package through its custom Rust Llama lane, including tied-embedding fixtures when the checkpoint validates.".to_string()
    } else if is_candle_qwen2_supported_package(&package) {
        "Fathom can run this verified Qwen2ForCausalLM SafeTensors/HF package through its custom Rust Qwen2 lane, including tied-output checkpoints when the checkpoint validates.".to_string()
    } else if is_candle_phi_supported_package(&package) {
        "Fathom can run this verified PhiForCausalLM SafeTensors/HF package through its custom Rust Phi lane.".to_string()
    } else if is_candle_mistral_supported_package(&package) {
        "Fathom can run this verified MistralForCausalLM SafeTensors/HF package through its custom Rust Mistral lane.".to_string()
    } else if is_candle_gemma_supported_package(&package) {
        "Fathom can run this verified GemmaForCausalLM SafeTensors/HF package through its custom Rust Gemma lane with tied output projection.".to_string()
    } else if is_candle_bert_embedding_supported_package(&package) {
        "Fathom can run this verified BertModel/MiniLM SafeTensors/HF text-embedding package through its default custom Rust/Candle embedding lane. It is retrieval/embedding-only and is not a chat generation model.".to_string()
    } else if runnable {
        "Fathom has at least one verified runnable backend lane for this package.".to_string()
    } else if has_gguf_metadata {
        "Fathom can read this GGUF file's header, key/value metadata, tensor descriptors, validated internal payload ranges, tokenizer hints, architecture compatibility signals, and bounded internal tokenizer metadata for narrow synthetic GPT-2/BPE or Llama/SentencePiece shapes when present, with private fixture-scoped Llama/SentencePiece encode/decode parity helpers. GGUF remains metadata-only and not runnable; public/runtime tokenizer execution, runtime weight loading, general dequantization, quantized kernels, an architecture runtime, and generation are still required.".to_string()
    } else if !package.hf_validation.missing_required.is_empty() {
        format!(
            "Fathom recognizes SafeTensors/HF artifacts, but loader metadata is incomplete: missing {}.",
            package.hf_validation.missing_required.join(", ")
        )
    } else {
        let lane_names = matching_lanes
            .iter()
            .map(|lane| lane.name)
            .collect::<Vec<_>>()
            .join(", ");
        format!("Fathom recognizes this package and mapped it to: {lane_names}. It is not runnable until it matches a verified GPT-2, Llama, Qwen2, Phi, Mistral, Gemma, or BERT/MiniLM embedding custom Rust lane.")
    };

    PackageCapabilityReport {
        package,
        matching_lanes,
        best_status,
        runnable,
        summary,
    }
}

fn lane_matches_artifacts(lane: &BackendLane, artifact_formats: &[ModelFormat]) -> bool {
    match lane.kind {
        BackendLaneKind::SafeTensorsHf => artifact_formats.iter().any(|format| {
            matches!(
                format,
                ModelFormat::SafeTensors | ModelFormat::SafeTensorsIndex
            )
        }),
        BackendLaneKind::Onnx | BackendLaneKind::LocalEmbeddingsRetrieval => artifact_formats
            .iter()
            .any(|format| matches!(format, ModelFormat::Onnx)),
        _ => lane
            .formats
            .iter()
            .any(|format| artifact_formats.contains(format)),
    }
}

fn best_capability_status(lanes: &[BackendLane]) -> CapabilityStatus {
    if lanes
        .iter()
        .any(|lane| lane.status == CapabilityStatus::Runnable)
    {
        CapabilityStatus::Runnable
    } else if lanes
        .iter()
        .any(|lane| lane.status == CapabilityStatus::Planned)
    {
        CapabilityStatus::Planned
    } else if lanes
        .iter()
        .any(|lane| lane.status == CapabilityStatus::MetadataOnly)
    {
        CapabilityStatus::MetadataOnly
    } else if lanes
        .iter()
        .any(|lane| lane.status == CapabilityStatus::Blocked)
    {
        CapabilityStatus::Blocked
    } else {
        CapabilityStatus::Unsupported
    }
}

pub fn detect_model_artifact(path: impl AsRef<Path>) -> ModelArtifact {
    let path = path.as_ref().to_path_buf();
    let lowered = path.to_string_lossy().to_ascii_lowercase();
    let filename = path
        .file_name()
        .and_then(|value| value.to_str())
        .unwrap_or_default()
        .to_ascii_lowercase();

    let (format, support, note) = if lowered.ends_with(".gguf") {
        (ModelFormat::Gguf, SupportLevel::LoadPlanned, "GGUF artifact detected. Fathom can inspect metadata when safe, but public/runtime tokenizer execution, runtime weight loading, dequantization/kernels, an architecture runtime, and generation are required before this is runnable.")
    } else if filename == "model.safetensors.index.json"
        || lowered.ends_with(".safetensors.index.json")
    {
        (ModelFormat::SafeTensorsIndex, SupportLevel::MetadataReadable, "Hugging Face sharded SafeTensors index detected; needs tokenizer/config plus architecture-specific graph runtime support.")
    } else if lowered.ends_with(".safetensors") {
        (ModelFormat::SafeTensors, SupportLevel::LoadPlanned, "SafeTensors weights detected; needs config/tokenizer and architecture-specific loader/runtime support.")
    } else if lowered.ends_with(".bin") || lowered.ends_with("pytorch_model.bin") {
        (ModelFormat::PyTorchBin, SupportLevel::Blocked, "PyTorch pickle/bin weights detected; safest path is metadata support plus explicit trusted import policy because pickle can execute code.")
    } else if lowered.ends_with(".onnx") {
        (
            ModelFormat::Onnx,
            SupportLevel::LoadPlanned,
            "ONNX model detected; likely runtime lane via ort or tract with tokenizer adapter.",
        )
    } else if lowered.ends_with(".mlmodel") || lowered.ends_with(".mlpackage") {
        (
            ModelFormat::CoreMl,
            SupportLevel::LoadPlanned,
            "CoreML artifact detected; macOS backend lane, not a portable server path yet.",
        )
    } else if lowered.ends_with(".engine") || lowered.ends_with(".plan") {
        (ModelFormat::TensorRtPlan, SupportLevel::LoadPlanned, "TensorRT plan/engine detected; NVIDIA-specific backend lane and hardware/version sensitive.")
    } else if filename == "weights.npz" || lowered.contains("/mlx") {
        (
            ModelFormat::Mlx,
            SupportLevel::LoadPlanned,
            "MLX-style artifact detected; Apple Silicon backend lane.",
        )
    } else if filename == "tokenizer.json" {
        (
            ModelFormat::TokenizerJson,
            SupportLevel::MetadataReadable,
            "Hugging Face tokenizer.json detected.",
        )
    } else if filename == "tokenizer_config.json" {
        (
            ModelFormat::TokenizerConfigJson,
            SupportLevel::MetadataReadable,
            "Hugging Face tokenizer_config.json detected; may contain chat_template metadata.",
        )
    } else if filename == "chat_template.jinja" || filename == "chat_template.json" {
        (
            ModelFormat::ChatTemplate,
            SupportLevel::MetadataReadable,
            "Hugging Face chat template file detected.",
        )
    } else if lowered.ends_with(".model")
        && (filename.contains("tokenizer")
            || filename.contains("sentencepiece")
            || filename == "spiece.model")
    {
        (
            ModelFormat::SentencePiece,
            SupportLevel::MetadataReadable,
            "SentencePiece tokenizer detected.",
        )
    } else if filename == "config.json" {
        (
            ModelFormat::ConfigJson,
            SupportLevel::MetadataReadable,
            "Hugging Face config.json detected.",
        )
    } else {
        (
            ModelFormat::Unknown,
            SupportLevel::Detected,
            "Unknown artifact format.",
        )
    };

    let mut artifact = ModelArtifact {
        path,
        format,
        support,
        runnable_today: false,
        notes: vec![note.to_string()],
        gguf_metadata: None,
    };

    if artifact.format == ModelFormat::Gguf && artifact.path.is_file() {
        match read_gguf_metadata_summary(&artifact.path) {
            Ok(metadata) => {
                artifact.support = SupportLevel::MetadataReadable;
                artifact.notes = vec![format!(
                    "GGUF artifact detected; metadata header, tensor descriptors, and compatibility hints are readable (version {}, {} tensors, {} metadata entries). It remains metadata-only and not runnable; public/runtime tokenizer execution, runtime weight loading, dequantization/kernels, architecture runtime, and generation are not implemented.",
                    metadata.version, metadata.tensor_count, metadata.metadata_kv_count
                )];
                artifact.gguf_metadata = Some(metadata);
            }
            Err(error) => artifact.notes.push(format!(
                "GGUF metadata could not be read safely: {error}. GGUF remains metadata-only/not runnable; public/runtime tokenizer execution, runtime weight loading, dequantization/kernels, architecture runtime, and generation are not implemented."
            )),
        }
    }

    artifact
}

pub fn read_gguf_metadata_summary(path: impl AsRef<Path>) -> anyhow::Result<GgufMetadataSummary> {
    const MAX_GGUF_METADATA_ENTRIES: u64 = 4096;
    const MAX_GGUF_TENSORS: u64 = 200_000;
    const MAX_GGUF_TENSOR_SUMMARY_ENTRIES: u64 = 4096;
    const MAX_GGUF_TENSOR_RANK: u32 = 16;
    const MAX_GGUF_STRING_BYTES: u64 = 1_048_576;
    const MAX_GGUF_TENSOR_NAME_BYTES: u64 = 4096;
    const MAX_GGUF_ARRAY_PREVIEW: usize = 8;
    const MAX_GGUF_ARRAY_ELEMENTS: u64 = 1_000_000;
    const MAX_GGUF_ARRAY_BYTES_TO_SKIP: u64 = 64 * 1024 * 1024;
    const MAX_GGUF_BPE_TOKENIZER_SPEC_MERGES: u64 = 512 * 1024;
    const MAX_GGUF_LLAMA_TOKENIZER_SPEC_TOKENS: u64 = GGUF_LLAMA3_BPE_EXPECTED_VOCAB_SIZE as u64;
    const MAX_GGUF_TOKENIZER_SPEC_STRING_BYTES: u64 = 64 * 1024 * 1024;
    const MAX_GGUF_INTERNAL_PAYLOAD_RANGE_BYTES: u64 = 512 * 1024 * 1024;

    struct Reader {
        file: fs::File,
        offset: u64,
    }

    impl Reader {
        fn read_exact<const N: usize>(&mut self) -> anyhow::Result<[u8; N]> {
            let mut bytes = [0u8; N];
            std::io::Read::read_exact(&mut self.file, &mut bytes)
                .map_err(|error| anyhow::anyhow!("GGUF metadata ended unexpectedly: {error}"))?;
            self.offset = self
                .offset
                .checked_add(N as u64)
                .ok_or_else(|| anyhow::anyhow!("GGUF offset overflow"))?;
            Ok(bytes)
        }

        fn read_u8(&mut self) -> anyhow::Result<u8> {
            Ok(self.read_exact::<1>()?[0])
        }
        fn read_i8(&mut self) -> anyhow::Result<i8> {
            Ok(self.read_u8()? as i8)
        }
        fn read_u16(&mut self) -> anyhow::Result<u16> {
            Ok(u16::from_le_bytes(self.read_exact()?))
        }
        fn read_i16(&mut self) -> anyhow::Result<i16> {
            Ok(i16::from_le_bytes(self.read_exact()?))
        }
        fn read_u32(&mut self) -> anyhow::Result<u32> {
            Ok(u32::from_le_bytes(self.read_exact()?))
        }
        fn read_i32(&mut self) -> anyhow::Result<i32> {
            Ok(i32::from_le_bytes(self.read_exact()?))
        }
        fn read_u64(&mut self) -> anyhow::Result<u64> {
            Ok(u64::from_le_bytes(self.read_exact()?))
        }
        fn read_i64(&mut self) -> anyhow::Result<i64> {
            Ok(i64::from_le_bytes(self.read_exact()?))
        }
        fn read_f32(&mut self) -> anyhow::Result<f32> {
            Ok(f32::from_le_bytes(self.read_exact()?))
        }
        fn read_f64(&mut self) -> anyhow::Result<f64> {
            Ok(f64::from_le_bytes(self.read_exact()?))
        }
        fn read_bool(&mut self) -> anyhow::Result<bool> {
            match self.read_u8()? {
                0 => Ok(false),
                1 => Ok(true),
                other => anyhow::bail!("invalid GGUF bool value {other}"),
            }
        }
        fn read_string_with_limit(&mut self, limit: u64, label: &str) -> anyhow::Result<String> {
            let len = self.read_u64()?;
            if len > limit {
                anyhow::bail!("GGUF {label} length {len} exceeds safe parser limit {limit}");
            }
            let len = usize::try_from(len)?;
            let mut bytes = vec![0u8; len];
            std::io::Read::read_exact(&mut self.file, &mut bytes)
                .map_err(|error| anyhow::anyhow!("GGUF string ended unexpectedly: {error}"))?;
            self.offset = self
                .offset
                .checked_add(len as u64)
                .ok_or_else(|| anyhow::anyhow!("GGUF string offset overflow"))?;
            Ok(std::str::from_utf8(&bytes)?.to_string())
        }
        fn read_string(&mut self) -> anyhow::Result<String> {
            self.read_string_with_limit(MAX_GGUF_STRING_BYTES, "string")
        }
        fn read_tensor_name(&mut self) -> anyhow::Result<String> {
            self.read_string_with_limit(MAX_GGUF_TENSOR_NAME_BYTES, "tensor name")
        }
        fn skip_bytes(&mut self, bytes: u64) -> anyhow::Result<()> {
            if bytes > MAX_GGUF_ARRAY_BYTES_TO_SKIP {
                anyhow::bail!("GGUF array payload {bytes} bytes exceeds safe metadata skip limit");
            }
            std::io::Seek::seek(
                &mut self.file,
                std::io::SeekFrom::Current(i64::try_from(bytes)?),
            )?;
            self.offset = self
                .offset
                .checked_add(bytes)
                .ok_or_else(|| anyhow::anyhow!("GGUF skip offset overflow"))?;
            Ok(())
        }
    }

    fn value_type(raw: u32) -> anyhow::Result<GgufMetadataValueType> {
        Ok(match raw {
            0 => GgufMetadataValueType::Uint8,
            1 => GgufMetadataValueType::Int8,
            2 => GgufMetadataValueType::Uint16,
            3 => GgufMetadataValueType::Int16,
            4 => GgufMetadataValueType::Uint32,
            5 => GgufMetadataValueType::Int32,
            6 => GgufMetadataValueType::Float32,
            7 => GgufMetadataValueType::Bool,
            8 => GgufMetadataValueType::String,
            9 => GgufMetadataValueType::Array,
            10 => GgufMetadataValueType::Uint64,
            11 => GgufMetadataValueType::Int64,
            12 => GgufMetadataValueType::Float64,
            other => anyhow::bail!("unsupported GGUF metadata value type {other}"),
        })
    }

    fn fixed_width(ty: GgufMetadataValueType) -> Option<u64> {
        match ty {
            GgufMetadataValueType::Uint8
            | GgufMetadataValueType::Int8
            | GgufMetadataValueType::Bool => Some(1),
            GgufMetadataValueType::Uint16 | GgufMetadataValueType::Int16 => Some(2),
            GgufMetadataValueType::Uint32
            | GgufMetadataValueType::Int32
            | GgufMetadataValueType::Float32 => Some(4),
            GgufMetadataValueType::Uint64
            | GgufMetadataValueType::Int64
            | GgufMetadataValueType::Float64 => Some(8),
            GgufMetadataValueType::String | GgufMetadataValueType::Array => None,
        }
    }

    fn read_scalar(
        reader: &mut Reader,
        ty: GgufMetadataValueType,
    ) -> anyhow::Result<GgufMetadataValueSummary> {
        Ok(match ty {
            GgufMetadataValueType::Uint8 => GgufMetadataValueSummary::Unsigned {
                value: reader.read_u8()? as u64,
            },
            GgufMetadataValueType::Int8 => GgufMetadataValueSummary::Signed {
                value: reader.read_i8()? as i64,
            },
            GgufMetadataValueType::Uint16 => GgufMetadataValueSummary::Unsigned {
                value: reader.read_u16()? as u64,
            },
            GgufMetadataValueType::Int16 => GgufMetadataValueSummary::Signed {
                value: reader.read_i16()? as i64,
            },
            GgufMetadataValueType::Uint32 => GgufMetadataValueSummary::Unsigned {
                value: reader.read_u32()? as u64,
            },
            GgufMetadataValueType::Int32 => GgufMetadataValueSummary::Signed {
                value: reader.read_i32()? as i64,
            },
            GgufMetadataValueType::Float32 => GgufMetadataValueSummary::Float {
                value: reader.read_f32()? as f64,
            },
            GgufMetadataValueType::Bool => GgufMetadataValueSummary::Bool {
                value: reader.read_bool()?,
            },
            GgufMetadataValueType::String => GgufMetadataValueSummary::String {
                value: reader.read_string()?,
            },
            GgufMetadataValueType::Uint64 => GgufMetadataValueSummary::Unsigned {
                value: reader.read_u64()?,
            },
            GgufMetadataValueType::Int64 => GgufMetadataValueSummary::Signed {
                value: reader.read_i64()?,
            },
            GgufMetadataValueType::Float64 => GgufMetadataValueSummary::Float {
                value: reader.read_f64()?,
            },
            GgufMetadataValueType::Array => {
                anyhow::bail!("nested GGUF metadata arrays are not supported")
            }
        })
    }

    fn is_tokenizer_spec_string_array_key(key: &str) -> bool {
        matches!(key, "tokenizer.ggml.tokens" | "tokenizer.ggml.merges")
    }

    fn is_tokenizer_spec_numeric_array_key(key: &str) -> bool {
        matches!(key, "tokenizer.ggml.scores" | "tokenizer.ggml.token_type")
    }

    fn read_value(
        reader: &mut Reader,
        ty: GgufMetadataValueType,
        key: &str,
    ) -> anyhow::Result<GgufMetadataValueSummary> {
        if ty != GgufMetadataValueType::Array {
            return read_scalar(reader, ty);
        }
        let element_type = value_type(reader.read_u32()?)?;
        if element_type == GgufMetadataValueType::Array {
            anyhow::bail!("nested GGUF metadata arrays are not supported");
        }
        let len = reader.read_u64()?;
        if len > MAX_GGUF_ARRAY_ELEMENTS {
            anyhow::bail!(
                "GGUF array length {len} exceeds safe parser limit {MAX_GGUF_ARRAY_ELEMENTS}"
            );
        }
        let tokenizer_spec_string_array_key = is_tokenizer_spec_string_array_key(key);
        let tokenizer_spec_numeric_array_key = is_tokenizer_spec_numeric_array_key(key);
        let string_retention_limit = match key {
            "tokenizer.ggml.tokens" => MAX_GGUF_LLAMA_TOKENIZER_SPEC_TOKENS,
            "tokenizer.ggml.merges" => MAX_GGUF_BPE_TOKENIZER_SPEC_MERGES,
            _ => 0,
        };
        let retain_full_strings = tokenizer_spec_string_array_key
            && element_type == GgufMetadataValueType::String
            && len <= string_retention_limit;
        let retain_full_floats = tokenizer_spec_numeric_array_key
            && element_type == GgufMetadataValueType::Float32
            && len <= MAX_GGUF_LLAMA_TOKENIZER_SPEC_TOKENS;
        let retain_full_i32s = tokenizer_spec_numeric_array_key
            && element_type == GgufMetadataValueType::Int32
            && len <= MAX_GGUF_LLAMA_TOKENIZER_SPEC_TOKENS;
        if tokenizer_spec_string_array_key && element_type != GgufMetadataValueType::String {
            anyhow::bail!("GGUF tokenizer metadata {key} must be an array of strings");
        }
        let read_len = if retain_full_strings || retain_full_floats || retain_full_i32s {
            len
        } else {
            len.min(MAX_GGUF_ARRAY_PREVIEW as u64)
        };
        let mut preview = Vec::new();
        let mut full_strings = retain_full_strings.then(Vec::new);
        let mut full_floats = retain_full_floats.then(Vec::new);
        let mut full_i32s = retain_full_i32s.then(Vec::new);
        let string_array_start = reader.offset;
        for index in 0..read_len {
            let value = read_scalar(reader, element_type)?;
            if let (Some(strings), GgufMetadataValueSummary::String { value }) =
                (&mut full_strings, &value)
            {
                strings.push(value.clone());
                if reader.offset - string_array_start > MAX_GGUF_TOKENIZER_SPEC_STRING_BYTES {
                    anyhow::bail!(
                        "GGUF tokenizer metadata {key} payload exceeds safe tokenizer-spec byte limit {MAX_GGUF_TOKENIZER_SPEC_STRING_BYTES}"
                    );
                }
            }
            if let (Some(floats), GgufMetadataValueSummary::Float { value }) =
                (&mut full_floats, &value)
            {
                floats.push(*value as f32);
            }
            if let (Some(i32s), GgufMetadataValueSummary::Signed { value }) =
                (&mut full_i32s, &value)
            {
                i32s.push(i32::try_from(*value)?);
            }
            if index < MAX_GGUF_ARRAY_PREVIEW as u64 {
                preview.push(value);
            }
        }
        let remaining = len - read_len;
        if remaining > 0 {
            if element_type == GgufMetadataValueType::String {
                let string_array_start = reader.offset;
                for _ in 0..remaining {
                    let _ = reader.read_string()?;
                    if reader.offset - string_array_start > MAX_GGUF_ARRAY_BYTES_TO_SKIP {
                        anyhow::bail!("GGUF string array payload exceeds safe metadata skip limit {MAX_GGUF_ARRAY_BYTES_TO_SKIP}");
                    }
                }
            } else {
                let width = fixed_width(element_type).ok_or_else(|| {
                    anyhow::anyhow!("GGUF array element type cannot be skipped safely")
                })?;
                reader.skip_bytes(
                    remaining
                        .checked_mul(width)
                        .ok_or_else(|| anyhow::anyhow!("GGUF array byte size overflow"))?,
                )?;
            }
        }
        Ok(GgufMetadataValueSummary::Array {
            element_type,
            len,
            preview,
            full_strings,
            full_floats,
            full_i32s,
        })
    }

    fn metadata_string(metadata: &[GgufMetadataEntry], key: &str) -> Option<String> {
        metadata.iter().find_map(|entry| {
            if entry.key == key {
                if let GgufMetadataValueSummary::String { value } = &entry.value {
                    return Some(value.clone());
                }
            }
            None
        })
    }

    fn metadata_unsigned(metadata: &[GgufMetadataEntry], key: &str) -> Option<u64> {
        metadata.iter().find_map(|entry| {
            if entry.key == key {
                if let GgufMetadataValueSummary::Unsigned { value } = entry.value {
                    return Some(value);
                }
            }
            None
        })
    }

    fn metadata_array_len(metadata: &[GgufMetadataEntry], key: &str) -> Option<u64> {
        metadata.iter().find_map(|entry| {
            if entry.key == key {
                if let GgufMetadataValueSummary::Array { len, .. } = entry.value {
                    return Some(len);
                }
            }
            None
        })
    }

    fn metadata_array_preview_strings(
        metadata: &[GgufMetadataEntry],
        key: &str,
    ) -> Option<Result<Vec<String>, GgufMetadataValueType>> {
        metadata.iter().find_map(|entry| {
            if entry.key != key {
                return None;
            }
            match &entry.value {
                GgufMetadataValueSummary::Array {
                    element_type,
                    preview,
                    ..
                } => {
                    if *element_type != GgufMetadataValueType::String {
                        return Some(Err(*element_type));
                    }
                    Some(Ok(preview
                        .iter()
                        .filter_map(|value| match value {
                            GgufMetadataValueSummary::String { value } => Some(value.clone()),
                            _ => None,
                        })
                        .collect()))
                }
                _ => None,
            }
        })
    }

    fn metadata_array_full_strings(
        metadata: &[GgufMetadataEntry],
        key: &str,
    ) -> Option<Vec<String>> {
        metadata.iter().find_map(|entry| {
            if entry.key != key {
                return None;
            }
            match &entry.value {
                GgufMetadataValueSummary::Array { full_strings, .. } => full_strings.clone(),
                _ => None,
            }
        })
    }

    fn metadata_array_full_floats(metadata: &[GgufMetadataEntry], key: &str) -> Option<Vec<f32>> {
        metadata.iter().find_map(|entry| {
            if entry.key != key {
                return None;
            }
            match &entry.value {
                GgufMetadataValueSummary::Array { full_floats, .. } => full_floats.clone(),
                _ => None,
            }
        })
    }

    fn metadata_array_full_i32s(metadata: &[GgufMetadataEntry], key: &str) -> Option<Vec<i32>> {
        metadata.iter().find_map(|entry| {
            if entry.key != key {
                return None;
            }
            match &entry.value {
                GgufMetadataValueSummary::Array { full_i32s, .. } => full_i32s.clone(),
                _ => None,
            }
        })
    }

    fn metadata_has_key(metadata: &[GgufMetadataEntry], key: &str) -> bool {
        metadata.iter().any(|entry| entry.key == key)
    }

    fn metadata_float(metadata: &[GgufMetadataEntry], key: &str) -> Option<f64> {
        metadata.iter().find_map(|entry| {
            if entry.key == key {
                if let GgufMetadataValueSummary::Float { value } = entry.value {
                    return Some(value);
                }
            }
            None
        })
    }

    fn first_metadata_unsigned(metadata: &[GgufMetadataEntry], keys: &[String]) -> Option<u64> {
        keys.iter().find_map(|key| metadata_unsigned(metadata, key))
    }

    fn first_metadata_float(metadata: &[GgufMetadataEntry], keys: &[String]) -> Option<f64> {
        keys.iter().find_map(|key| metadata_float(metadata, key))
    }

    fn arch_keys(architecture: Option<&str>, suffix: &str) -> Vec<String> {
        let mut keys = Vec::new();
        if let Some(architecture) = architecture {
            keys.push(format!("{architecture}.{suffix}"));
        }
        for prefix in ["llama", "qwen2", "gemma", "mistral", "phi", "gpt2"] {
            let key = format!("{prefix}.{suffix}");
            if !keys.contains(&key) {
                keys.push(key);
            }
        }
        keys.push(format!("general.{suffix}"));
        keys
    }

    fn bounded_text_summary(value: &str) -> GgufBoundedTextSummary {
        let mut hash = 0xcbf29ce484222325u64;
        for byte in value.as_bytes() {
            hash ^= *byte as u64;
            hash = hash.wrapping_mul(0x100000001b3);
        }
        let preview: String = value.chars().take(240).collect();
        GgufBoundedTextSummary {
            byte_len: value.len() as u64,
            hash: format!("{hash:016x}"),
            preview,
        }
    }

    fn build_gguf_tokenizer_spec(
        metadata: &[GgufMetadataEntry],
    ) -> (Option<GgufTokenizerSpec>, Vec<String>) {
        let Some(model) = metadata_string(metadata, "tokenizer.ggml.model") else {
            return (
                None,
                vec!["GGUF tokenizer spec not built: tokenizer.ggml.model is missing.".into()],
            );
        };
        let normalized_model = model.to_ascii_lowercase();
        match normalized_model.as_str() {
            "gpt2" | "gpt-2"
                if metadata_string(metadata, "tokenizer.ggml.pre")
                    .is_some_and(|pre| pre == "llama-bpe") =>
            {
                build_llama3_bpe_tokenizer_spec(metadata, model)
            }
            "gpt2" | "gpt-2" => build_gpt2_tokenizer_spec(metadata, model),
            "llama" => build_llama_sentencepiece_tokenizer_spec(metadata, model),
            _ => (
                None,
                vec![format!(
                    "GGUF tokenizer spec not built: tokenizer family {model:?} is outside the narrow synthetic GPT-2/ByteLevel-BPE, Llama 3 BPE, and Llama/SentencePiece metadata-retention proofs."
                )],
            ),
        }
    }

    fn build_llama3_bpe_tokenizer_spec(
        metadata: &[GgufMetadataEntry],
        model: String,
    ) -> (Option<GgufTokenizerSpec>, Vec<String>) {
        let Some(tokens) = metadata_array_full_strings(metadata, "tokenizer.ggml.tokens") else {
            return (None, vec!["GGUF Llama 3 BPE tokenizer spec not built: tokenizer.ggml.tokens is missing, malformed, or over the internal Llama 3 retention limit.".into()]);
        };
        let Some(merges) = metadata_array_full_strings(metadata, "tokenizer.ggml.merges") else {
            return (None, vec!["GGUF Llama 3 BPE tokenizer spec not built: tokenizer.ggml.merges is missing, malformed, or over the internal Llama 3 retention limit.".into()]);
        };
        if tokens.len() != GGUF_LLAMA3_BPE_EXPECTED_VOCAB_SIZE {
            return (None, vec![format!(
                "GGUF Llama 3 BPE tokenizer spec not built: tokenizer.ggml.tokens length {} does not match expected Llama 3 vocab size {GGUF_LLAMA3_BPE_EXPECTED_VOCAB_SIZE}.",
                tokens.len()
            )]);
        }
        if merges.is_empty() {
            return (
                None,
                vec!["GGUF Llama 3 BPE tokenizer spec not built: merge array is empty.".into()],
            );
        }
        let mut seen_tokens = HashSet::new();
        for (index, token) in tokens.iter().enumerate() {
            if token.is_empty() {
                return (
                    None,
                    vec![format!(
                        "GGUF Llama 3 BPE tokenizer spec not built: token at index {index} is empty."
                    )],
                );
            }
            if !seen_tokens.insert(token.clone()) {
                return (
                    None,
                    vec![format!(
                        "GGUF Llama 3 BPE tokenizer spec not built: duplicate token {token:?}."
                    )],
                );
            }
        }
        for (index, merge) in merges.iter().enumerate() {
            if gguf_bpe_split_merge(merge).is_none() {
                return (None, vec![format!("GGUF Llama 3 BPE tokenizer spec not built: merge at index {index} is not a two-token BPE merge.")]);
            }
        }
        let mut special_token_ids = BTreeMap::new();
        for (label, key) in [
            ("bos", "tokenizer.ggml.bos_token_id"),
            ("eos", "tokenizer.ggml.eos_token_id"),
            ("unk", "tokenizer.ggml.unknown_token_id"),
            ("pad", "tokenizer.ggml.padding_token_id"),
        ] {
            if let Some(value) = metadata_unsigned(metadata, key) {
                if value >= tokens.len() as u64 {
                    return (None, vec![format!("GGUF Llama 3 BPE tokenizer spec not built: {key} value {value} is outside token array length {}.", tokens.len())]);
                }
                special_token_ids.insert(label.to_string(), value);
            }
        }
        (
            Some(GgufTokenizerSpec {
                family: GgufTokenizerSpecFamily::Llama3Bpe,
                model,
                tokens,
                merges,
                scores: Vec::new(),
                token_types: Vec::new(),
                special_token_ids,
                has_byte_fallback: false,
            }),
            vec!["Internal tokenizer spec retained for exact-size Llama 3 BPE GGUF metadata only; no public/runtime tokenizer execution, weight loading, or inference readiness is claimed.".into()],
        )
    }

    fn build_gpt2_tokenizer_spec(
        metadata: &[GgufMetadataEntry],
        model: String,
    ) -> (Option<GgufTokenizerSpec>, Vec<String>) {
        let Some(tokens) = metadata_array_full_strings(metadata, "tokenizer.ggml.tokens") else {
            return (None, vec!["GGUF tokenizer spec not built: tokenizer.ggml.tokens is missing, malformed, or over the internal retention limit.".into()]);
        };
        let Some(merges) = metadata_array_full_strings(metadata, "tokenizer.ggml.merges") else {
            return (None, vec!["GGUF tokenizer spec not built: tokenizer.ggml.merges is missing, malformed, or over the internal retention limit.".into()]);
        };
        if tokens.is_empty() {
            return (
                None,
                vec!["GGUF tokenizer spec not built: token array is empty.".into()],
            );
        }
        if merges.is_empty() {
            return (
                None,
                vec!["GGUF tokenizer spec not built: merge array is empty.".into()],
            );
        }
        let mut seen_tokens = HashSet::new();
        for (index, token) in tokens.iter().enumerate() {
            if token.is_empty() {
                return (
                    None,
                    vec![format!(
                        "GGUF tokenizer spec not built: token at index {index} is empty."
                    )],
                );
            }
            if !seen_tokens.insert(token.clone()) {
                return (
                    None,
                    vec![format!(
                        "GGUF tokenizer spec not built: duplicate token {token:?}."
                    )],
                );
            }
        }
        for (index, merge) in merges.iter().enumerate() {
            let parts: Vec<&str> = merge.split_whitespace().collect();
            if parts.len() != 2 || parts.iter().any(|part| part.is_empty()) {
                return (None, vec![format!("GGUF tokenizer spec not built: merge at index {index} is not a two-token BPE merge.")]);
            }
        }
        let mut special_token_ids = BTreeMap::new();
        for (label, key) in [
            ("bos", "tokenizer.ggml.bos_token_id"),
            ("eos", "tokenizer.ggml.eos_token_id"),
            ("unk", "tokenizer.ggml.unknown_token_id"),
            ("pad", "tokenizer.ggml.padding_token_id"),
        ] {
            if let Some(value) = metadata_unsigned(metadata, key) {
                if value >= tokens.len() as u64 {
                    return (None, vec![format!("GGUF tokenizer spec not built: {key} value {value} is outside token array length {}.", tokens.len())]);
                }
                special_token_ids.insert(label.to_string(), value);
            }
        }
        (
            Some(GgufTokenizerSpec {
                family: GgufTokenizerSpecFamily::SyntheticGpt2ByteLevelBpe,
                model,
                tokens,
                merges,
                scores: Vec::new(),
                token_types: Vec::new(),
                special_token_ids,
                has_byte_fallback: false,
            }),
            vec!["Internal tokenizer spec built for synthetic GPT-2/ByteLevel-BPE-shaped GGUF metadata only; GGUF remains metadata-only and non-runnable.".into()],
        )
    }

    fn build_llama_sentencepiece_tokenizer_spec(
        metadata: &[GgufMetadataEntry],
        model: String,
    ) -> (Option<GgufTokenizerSpec>, Vec<String>) {
        let Some(tokens) = metadata_array_full_strings(metadata, "tokenizer.ggml.tokens") else {
            return (None, vec!["GGUF Llama/SentencePiece tokenizer spec not built: tokenizer.ggml.tokens is missing, malformed, or over the internal Llama retention limit.".into()]);
        };
        let Some(scores) = metadata_array_full_floats(metadata, "tokenizer.ggml.scores") else {
            return (None, vec!["GGUF Llama/SentencePiece tokenizer spec not built: tokenizer.ggml.scores is missing, malformed, or over the internal Llama retention limit.".into()]);
        };
        let Some(token_types) = metadata_array_full_i32s(metadata, "tokenizer.ggml.token_type")
        else {
            return (None, vec!["GGUF Llama/SentencePiece tokenizer spec not built: tokenizer.ggml.token_type is missing, malformed, or over the internal Llama retention limit.".into()]);
        };
        if tokens.is_empty() {
            return (
                None,
                vec![
                    "GGUF Llama/SentencePiece tokenizer spec not built: token array is empty."
                        .into(),
                ],
            );
        }
        if scores.len() != tokens.len() {
            return (None, vec![format!("GGUF Llama/SentencePiece tokenizer spec not built: tokenizer.ggml.scores length {} does not match token length {}.", scores.len(), tokens.len())]);
        }
        if token_types.len() != tokens.len() {
            return (None, vec![format!("GGUF Llama/SentencePiece tokenizer spec not built: tokenizer.ggml.token_type length {} does not match token length {}.", token_types.len(), tokens.len())]);
        }
        let mut special_token_ids = BTreeMap::new();
        for (label, key) in [
            ("bos", "tokenizer.ggml.bos_token_id"),
            ("eos", "tokenizer.ggml.eos_token_id"),
            ("unk", "tokenizer.ggml.unknown_token_id"),
            ("pad", "tokenizer.ggml.padding_token_id"),
        ] {
            if let Some(value) = metadata_unsigned(metadata, key) {
                if value >= tokens.len() as u64 {
                    return (None, vec![format!("GGUF Llama/SentencePiece tokenizer spec not built: {key} value {value} is outside token array length {}.", tokens.len())]);
                }
                special_token_ids.insert(label.to_string(), value);
            }
        }
        let byte_fallback_count = tokens
            .iter()
            .zip(token_types.iter())
            .filter(|(token, token_type)| **token_type == 6 && is_gguf_hex_byte_token(token))
            .count();
        let has_byte_fallback = byte_fallback_count >= 256;
        let mut notes = vec!["Internal tokenizer spec retained for bounded Llama/SentencePiece-shaped GGUF metadata only; no public/runtime tokenizer execution or runtime readiness is claimed.".into()];
        if has_byte_fallback {
            notes.push(
                "Detected GGUF Llama byte-fallback evidence from <0xNN> tokens with token_type=6."
                    .into(),
            );
        } else {
            notes.push("GGUF Llama byte-fallback evidence is incomplete or absent; public/runtime tokenizer execution remains unsupported.".into());
        }
        for (key, label) in [
            ("tokenizer.ggml.add_bos_token", "add_bos_token"),
            ("tokenizer.ggml.add_eos_token", "add_eos_token"),
            ("tokenizer.ggml.pre", "normalizer/pre-tokenizer"),
            ("tokenizer.chat_template", "chat template"),
        ] {
            if !metadata_has_key(metadata, key) {
                notes.push(format!(
                    "GGUF Llama/SentencePiece metadata does not include explicit {label}; future execution must rely on conventions or external proof."
                ));
            }
        }
        (
            Some(GgufTokenizerSpec {
                family: GgufTokenizerSpecFamily::LlamaSentencePiece,
                model,
                tokens,
                merges: Vec::new(),
                scores,
                token_types,
                special_token_ids,
                has_byte_fallback,
            }),
            notes,
        )
    }

    fn gguf_file_type_name(file_type: u64) -> Option<String> {
        Some(
            match file_type {
                0 => "mostly_f32",
                1 => "mostly_f16",
                2 => "mostly_q4_0",
                3 => "mostly_q4_1",
                7 => "mostly_q8_0",
                8 => "mostly_q5_0",
                9 => "mostly_q5_1",
                10 => "mostly_q2_k",
                11 => "mostly_q3_k_s",
                12 => "mostly_q3_k_m",
                13 => "mostly_q3_k_l",
                14 => "mostly_q4_k_s",
                15 => "mostly_q4_k_m",
                16 => "mostly_q5_k_s",
                17 => "mostly_q5_k_m",
                18 => "mostly_q6_k",
                _ => return None,
            }
            .to_string(),
        )
    }

    fn summarize_gguf_tokenizer(metadata: &[GgufMetadataEntry]) -> GgufTokenizerSummary {
        let model = metadata_string(metadata, "tokenizer.ggml.model");
        let token_count = metadata_array_len(metadata, "tokenizer.ggml.tokens");
        let mut token_samples = Vec::new();
        let mut notes = Vec::new();
        let mut unsupported_shape = false;
        if let Some(result) = metadata_array_preview_strings(metadata, "tokenizer.ggml.tokens") {
            match result {
                Ok(samples) => token_samples = samples,
                Err(element_type) => {
                    unsupported_shape = true;
                    notes.push(format!(
                        "tokenizer.ggml.tokens uses unsupported non-string element type {element_type:?}"
                    ));
                }
            }
        }
        let merge_count = metadata_array_len(metadata, "tokenizer.ggml.merges");
        let mut merge_samples = Vec::new();
        if let Some(result) = metadata_array_preview_strings(metadata, "tokenizer.ggml.merges") {
            match result {
                Ok(samples) => merge_samples = samples,
                Err(element_type) => {
                    unsupported_shape = true;
                    notes.push(format!(
                        "tokenizer.ggml.merges uses unsupported non-string element type {element_type:?}"
                    ));
                }
            }
        }
        let added_token_count = metadata_array_len(metadata, "tokenizer.ggml.added_tokens");
        let mut special_token_ids = BTreeMap::new();
        for (label, key) in [
            ("bos", "tokenizer.ggml.bos_token_id"),
            ("eos", "tokenizer.ggml.eos_token_id"),
            ("unk", "tokenizer.ggml.unknown_token_id"),
            ("pad", "tokenizer.ggml.padding_token_id"),
            ("sep", "tokenizer.ggml.separator_token_id"),
            ("sep_legacy", "tokenizer.ggml.seperator_token_id"),
            ("cls", "tokenizer.ggml.cls_token_id"),
            ("mask", "tokenizer.ggml.mask_token_id"),
        ] {
            if let Some(value) = metadata_unsigned(metadata, key) {
                special_token_ids.insert(label.to_string(), value);
            }
        }
        let chat_template = metadata_string(metadata, "tokenizer.chat_template")
            .or_else(|| metadata_string(metadata, "tokenizer.ggml.chat_template"))
            .map(|value| bounded_text_summary(&value));
        let any_metadata = model.is_some()
            || token_count.is_some()
            || merge_count.is_some()
            || added_token_count.is_some()
            || !special_token_ids.is_empty()
            || chat_template.is_some();
        let selected_bpe_family = model
            .as_deref()
            .map(|value| matches!(value.to_ascii_lowercase().as_str(), "gpt2" | "gpt-2"))
            .unwrap_or(false);
        let selected_bpe_complete = !selected_bpe_family || merge_count.is_some();
        let status = if unsupported_shape {
            GgufTokenizerMetadataStatus::UnsupportedShape
        } else if model.is_some() && token_count.is_some() && selected_bpe_complete {
            GgufTokenizerMetadataStatus::Present
        } else if any_metadata {
            GgufTokenizerMetadataStatus::Partial
        } else {
            GgufTokenizerMetadataStatus::Missing
        };
        if status == GgufTokenizerMetadataStatus::Present {
            notes.push(
                "Tokenizer metadata is present, but Fathom does not execute GGUF tokenizers yet."
                    .into(),
            );
        } else if status == GgufTokenizerMetadataStatus::Partial {
            notes.push(
                "Tokenizer metadata is partial; native GGUF tokenization remains blocked.".into(),
            );
        } else if status == GgufTokenizerMetadataStatus::Missing {
            notes.push("No bounded GGUF tokenizer metadata was found.".into());
        }
        GgufTokenizerSummary {
            status,
            model,
            token_count,
            token_samples,
            merge_count,
            merge_samples,
            added_token_count,
            special_token_ids,
            chat_template,
            notes,
        }
    }

    fn summarize_gguf_architecture(metadata: &[GgufMetadataEntry]) -> GgufArchitectureSummary {
        let architecture = metadata_string(metadata, "general.architecture");
        let architecture_ref = architecture.as_deref();
        let normalized = architecture_ref.map(|value| value.to_ascii_lowercase());
        let family = normalized.as_ref().and_then(|value| {
            if value.contains("llama") || value.contains("mistral") || value.contains("qwen2") {
                Some("llama_like_metadata".to_string())
            } else if value.contains("gemma") {
                Some("gemma_like_metadata".to_string())
            } else if value.contains("phi") {
                Some("phi_like_metadata".to_string())
            } else if value.contains("gpt") {
                Some("gpt_like_metadata".to_string())
            } else {
                None
            }
        });
        let context_length =
            first_metadata_unsigned(metadata, &arch_keys(architecture_ref, "context_length"));
        let embedding_length =
            first_metadata_unsigned(metadata, &arch_keys(architecture_ref, "embedding_length"));
        let block_count =
            first_metadata_unsigned(metadata, &arch_keys(architecture_ref, "block_count"));
        let attention_head_count = first_metadata_unsigned(
            metadata,
            &arch_keys(architecture_ref, "attention.head_count"),
        );
        let attention_kv_head_count = first_metadata_unsigned(
            metadata,
            &arch_keys(architecture_ref, "attention.head_count_kv"),
        );
        let feed_forward_length = first_metadata_unsigned(
            metadata,
            &arch_keys(architecture_ref, "feed_forward_length"),
        );
        let rope_dimension_count = first_metadata_unsigned(
            metadata,
            &arch_keys(architecture_ref, "rope.dimension_count"),
        );
        let rope_freq_base =
            first_metadata_float(metadata, &arch_keys(architecture_ref, "rope.freq_base"));
        let file_type = metadata_unsigned(metadata, "general.file_type");
        let quantization_hint = file_type.and_then(gguf_file_type_name);
        let has_dimensions = context_length.is_some()
            || embedding_length.is_some()
            || block_count.is_some()
            || attention_head_count.is_some();
        let status = if family.is_some() && has_dimensions {
            GgufArchitectureMetadataStatus::Recognized
        } else if architecture.is_some() || has_dimensions || file_type.is_some() {
            GgufArchitectureMetadataStatus::Partial
        } else {
            GgufArchitectureMetadataStatus::Unknown
        };
        let mut notes = Vec::new();
        if status == GgufArchitectureMetadataStatus::Recognized {
            notes.push("Architecture metadata is recognizable, but no native GGUF architecture runtime is implemented.".into());
        } else if status == GgufArchitectureMetadataStatus::Partial {
            notes.push(
                "Architecture metadata is partial; native GGUF runtime selection remains blocked."
                    .into(),
            );
        } else {
            notes.push("No recognized GGUF architecture metadata was found.".into());
        }
        GgufArchitectureSummary {
            status,
            architecture,
            family,
            context_length,
            embedding_length,
            block_count,
            attention_head_count,
            attention_kv_head_count,
            feed_forward_length,
            rope_dimension_count,
            rope_freq_base,
            file_type,
            quantization_hint,
            notes,
        }
    }

    fn classify_gguf_compatibility(
        tokenizer: &GgufTokenizerSummary,
        architecture: &GgufArchitectureSummary,
    ) -> GgufCompatibilitySummary {
        let mut categories = vec!["metadata_readable".to_string()];
        categories.push(
            match tokenizer.status {
                GgufTokenizerMetadataStatus::Present => "tokenizer_metadata_present",
                GgufTokenizerMetadataStatus::Partial => "tokenizer_metadata_partial",
                GgufTokenizerMetadataStatus::Missing => "tokenizer_metadata_missing",
                GgufTokenizerMetadataStatus::UnsupportedShape => {
                    "tokenizer_metadata_unsupported_shape"
                }
            }
            .to_string(),
        );
        categories.push(
            match architecture.status {
                GgufArchitectureMetadataStatus::Recognized => "architecture_metadata_recognized",
                GgufArchitectureMetadataStatus::Partial => "architecture_metadata_partial",
                GgufArchitectureMetadataStatus::Unknown => "architecture_metadata_unknown",
            }
            .to_string(),
        );
        if let Some(family) = &architecture.family {
            categories.push(family.clone());
        }
        let mut runtime_blockers = vec![
            "native_gguf_tokenizer_not_implemented".to_string(),
            "gguf_runtime_weight_loading_not_implemented".to_string(),
            "gguf_general_dequantization_not_implemented".to_string(),
            "gguf_quantized_kernels_not_implemented".to_string(),
            "architecture_runtime_not_implemented".to_string(),
            "gguf_generation_not_implemented".to_string(),
        ];
        if tokenizer.status != GgufTokenizerMetadataStatus::Present {
            runtime_blockers.push("complete_tokenizer_metadata_not_verified".to_string());
        }
        if architecture.status != GgufArchitectureMetadataStatus::Recognized {
            runtime_blockers.push("recognized_architecture_metadata_not_verified".to_string());
        }
        GgufCompatibilitySummary {
            metadata_readable: true,
            tokenizer_metadata: tokenizer.status,
            architecture_metadata: architecture.status,
            categories,
            runtime_blockers,
        }
    }

    fn ggml_type_name(raw: u32) -> String {
        match raw {
            0 => "F32",
            1 => "F16",
            2 => "Q4_0",
            3 => "Q4_1",
            6 => "Q5_0",
            7 => "Q5_1",
            8 => "Q8_0",
            9 => "Q8_1",
            10 => "Q2_K",
            11 => "Q3_K",
            12 => "Q4_K",
            13 => "Q5_K",
            14 => "Q6_K",
            15 => "Q8_K",
            16 => "IQ2_XXS",
            17 => "IQ2_XS",
            18 => "IQ3_XXS",
            19 => "IQ1_S",
            20 => "IQ4_NL",
            21 => "IQ3_S",
            22 => "IQ2_S",
            23 => "IQ4_XS",
            24 => "I8",
            25 => "I16",
            26 => "I32",
            27 => "I64",
            28 => "F64",
            29 => "IQ1_M",
            30 => "BF16",
            34 => "TQ1_0",
            35 => "TQ2_0",
            other => return format!("UNKNOWN_{other}"),
        }
        .to_string()
    }

    fn ggml_type_block(raw: u32) -> Option<(u64, u64)> {
        Some(match raw {
            0 => (1, 4),
            1 => (1, 2),
            2 => (32, 18),
            3 => (32, 20),
            6 => (32, 22),
            7 => (32, 24),
            8 => (32, 34),
            9 => (32, 40),
            10 => (256, 84),
            11 => (256, 110),
            12 => (256, 144),
            13 => (256, 176),
            14 => (256, 210),
            15 => (256, 292),
            16 => (256, 66),
            17 => (256, 74),
            18 => (256, 98),
            19 => (256, 34),
            20 => (32, 18),
            21 => (256, 110),
            22 => (256, 82),
            23 => (256, 136),
            24 => (1, 1),
            25 => (1, 2),
            26 => (1, 4),
            27 => (1, 8),
            28 => (1, 8),
            29 => (256, 56),
            30 => (1, 2),
            34 => (256, 54),
            35 => (256, 66),
            _ => return None,
        })
    }

    fn element_count(shape: &[u64]) -> Option<u64> {
        shape
            .iter()
            .try_fold(1u64, |acc, dim| acc.checked_mul(*dim))
    }

    fn estimate_tensor_bytes(ggml_type: u32, elements: u64) -> Option<u64> {
        let (block_size, type_size) = ggml_type_block(ggml_type)?;
        let blocks = elements
            .checked_add(block_size - 1)?
            .checked_div(block_size)?;
        blocks.checked_mul(type_size)
    }

    fn align_offset(offset: u64, alignment: u64) -> anyhow::Result<u64> {
        if alignment == 0 {
            anyhow::bail!("GGUF alignment cannot be zero");
        }
        let remainder = offset % alignment;
        if remainder == 0 {
            Ok(offset)
        } else {
            offset
                .checked_add(alignment - remainder)
                .ok_or_else(|| anyhow::anyhow!("GGUF aligned tensor data offset overflow"))
        }
    }

    #[derive(Debug)]
    struct PendingTensorPayloadRange {
        name: String,
        ggml_type: u32,
        ggml_type_name: String,
        shape: Vec<u64>,
        element_count: u64,
        relative_offset: u64,
        estimated_bytes: Option<u64>,
    }

    fn derive_payload_ranges(
        pending: &[PendingTensorPayloadRange],
        tensor_data_start: u64,
        file_size: u64,
    ) -> anyhow::Result<(Vec<GgufTensorPayloadRange>, GgufTensorPayloadRangeStatus)> {
        let mut ranges = Vec::new();
        let mut unsupported_types_present = false;
        let mut requested = 0u64;
        for tensor in pending {
            let absolute_start = tensor_data_start
                .checked_add(tensor.relative_offset)
                .ok_or_else(|| anyhow::anyhow!("GGUF tensor absolute offset overflow"))?;
            if absolute_start > file_size {
                anyhow::bail!(
                    "GGUF tensor {} offset {absolute_start} is beyond file size {file_size}",
                    tensor.name
                );
            }
            let Some(byte_len) = tensor.estimated_bytes else {
                unsupported_types_present = true;
                continue;
            };
            let absolute_end = absolute_start.checked_add(byte_len).ok_or_else(|| {
                anyhow::anyhow!("GGUF tensor {} byte range overflow", tensor.name)
            })?;
            if absolute_end > file_size {
                anyhow::bail!(
                    "GGUF tensor {} estimated byte range ends at {absolute_end}, beyond file size {file_size}",
                    tensor.name
                );
            }
            requested = requested
                .checked_add(byte_len)
                .ok_or_else(|| anyhow::anyhow!("GGUF known tensor payload byte budget overflow"))?;
            ranges.push(GgufTensorPayloadRange {
                name: tensor.name.clone(),
                ggml_type: tensor.ggml_type,
                ggml_type_name: tensor.ggml_type_name.clone(),
                shape: tensor.shape.clone(),
                element_count: tensor.element_count,
                relative_offset: tensor.relative_offset,
                absolute_start,
                absolute_end,
                byte_len,
            });
        }
        ranges.sort_by(|left, right| {
            left.absolute_start
                .cmp(&right.absolute_start)
                .then_with(|| left.absolute_end.cmp(&right.absolute_end))
                .then_with(|| left.name.cmp(&right.name))
        });
        for pair in ranges.windows(2) {
            let left = &pair[0];
            let right = &pair[1];
            if left.absolute_end > right.absolute_start {
                anyhow::bail!(
                    "GGUF tensor payload ranges overlap: {} ends at {}, {} starts at {}",
                    left.name,
                    left.absolute_end,
                    right.name,
                    right.absolute_start
                );
            }
        }
        let status = if unsupported_types_present {
            GgufTensorPayloadRangeStatus::UnsupportedTypesPresent
        } else if ranges.is_empty() {
            GgufTensorPayloadRangeStatus::Empty
        } else if requested > MAX_GGUF_INTERNAL_PAYLOAD_RANGE_BYTES {
            GgufTensorPayloadRangeStatus::PayloadBudgetExceeded {
                requested,
                limit: MAX_GGUF_INTERNAL_PAYLOAD_RANGE_BYTES,
            }
        } else {
            GgufTensorPayloadRangeStatus::Ready
        };
        if matches!(
            status,
            GgufTensorPayloadRangeStatus::PayloadBudgetExceeded { .. }
        ) {
            ranges.clear();
        }
        Ok((ranges, status))
    }

    let path = path.as_ref();
    let file_size = fs::metadata(path)?.len();
    let mut reader = Reader {
        file: fs::File::open(path)?,
        offset: 0,
    };
    if reader.read_exact::<4>()? != *b"GGUF" {
        anyhow::bail!("not a GGUF file: missing GGUF magic");
    }
    let version = reader.read_u32()?;
    let tensor_count = reader.read_u64()?;
    let metadata_kv_count = reader.read_u64()?;
    if metadata_kv_count > MAX_GGUF_METADATA_ENTRIES {
        anyhow::bail!("GGUF metadata entry count {metadata_kv_count} exceeds safe parser limit {MAX_GGUF_METADATA_ENTRIES}");
    }
    if tensor_count > MAX_GGUF_TENSORS {
        anyhow::bail!(
            "GGUF tensor count {tensor_count} exceeds safe parser limit {MAX_GGUF_TENSORS}"
        );
    }
    let mut metadata = Vec::with_capacity(usize::try_from(metadata_kv_count)?);
    for _ in 0..metadata_kv_count {
        let key = reader.read_string()?;
        if key.trim().is_empty() {
            anyhow::bail!("GGUF metadata key cannot be empty");
        }
        let ty = value_type(reader.read_u32()?)?;
        let value = read_value(&mut reader, ty, &key)?;
        metadata.push(GgufMetadataEntry {
            key,
            value_type: ty,
            value,
        });
    }

    let alignment = metadata_unsigned(&metadata, "general.alignment").unwrap_or(32);
    if alignment == 0 {
        anyhow::bail!("GGUF alignment cannot be zero");
    }
    let hints = GgufMetadataHints {
        architecture: metadata_string(&metadata, "general.architecture"),
        context_length: metadata_unsigned(&metadata, "llama.context_length")
            .or_else(|| metadata_unsigned(&metadata, "qwen2.context_length"))
            .or_else(|| metadata_unsigned(&metadata, "gemma.context_length"))
            .or_else(|| metadata_unsigned(&metadata, "general.context_length")),
        tokenizer_model: metadata_string(&metadata, "tokenizer.ggml.model"),
        tokenizer_token_count: metadata_array_len(&metadata, "tokenizer.ggml.tokens"),
        file_type: metadata_unsigned(&metadata, "general.file_type"),
        alignment: Some(alignment),
    };

    let mut tensors = Vec::new();
    let mut seen_names = HashSet::new();
    let summary_limit = tensor_count.min(MAX_GGUF_TENSOR_SUMMARY_ENTRIES);
    let mut type_counts: BTreeMap<String, u64> = BTreeMap::new();
    let mut total_estimated_tensor_bytes = Some(0u64);
    let mut unknown_type_tags = BTreeSet::new();
    let mut tensor_ranges =
        Vec::with_capacity(usize::try_from(tensor_count.min(MAX_GGUF_TENSORS))?);

    for index in 0..tensor_count {
        let name = reader.read_tensor_name()?;
        if name.trim().is_empty() {
            anyhow::bail!("GGUF tensor name at index {index} cannot be empty");
        }
        if !seen_names.insert(name.clone()) {
            anyhow::bail!("duplicate GGUF tensor name {name}");
        }
        let rank = reader.read_u32()?;
        if rank == 0 || rank > MAX_GGUF_TENSOR_RANK {
            anyhow::bail!(
                "GGUF tensor {name} rank {rank} exceeds safe parser limit {MAX_GGUF_TENSOR_RANK}"
            );
        }
        let mut shape = Vec::with_capacity(usize::try_from(rank)?);
        for _ in 0..rank {
            let dim = reader.read_u64()?;
            if dim == 0 {
                anyhow::bail!("GGUF tensor {name} contains zero dimension");
            }
            shape.push(dim);
        }
        let ggml_type = reader.read_u32()?;
        let ggml_type_name = ggml_type_name(ggml_type);
        let offset = reader.read_u64()?;
        if offset % alignment != 0 {
            anyhow::bail!("GGUF tensor {name} offset {offset} is not aligned to {alignment}");
        }
        let elements = element_count(&shape)
            .ok_or_else(|| anyhow::anyhow!("GGUF tensor {name} element count overflow"))?;
        let estimated_bytes = estimate_tensor_bytes(ggml_type, elements);
        if estimated_bytes.is_none() {
            unknown_type_tags.insert(ggml_type);
            total_estimated_tensor_bytes = None;
        } else if let (Some(total), Some(bytes)) = (total_estimated_tensor_bytes, estimated_bytes) {
            total_estimated_tensor_bytes = Some(
                total
                    .checked_add(bytes)
                    .ok_or_else(|| anyhow::anyhow!("GGUF tensor byte estimate overflow"))?,
            );
        }
        *type_counts.entry(ggml_type_name.clone()).or_insert(0) += 1;
        tensor_ranges.push(PendingTensorPayloadRange {
            name: name.clone(),
            ggml_type,
            ggml_type_name: ggml_type_name.clone(),
            shape: shape.clone(),
            element_count: elements,
            relative_offset: offset,
            estimated_bytes,
        });
        if index < summary_limit {
            tensors.push(GgufTensorInfoSummary {
                name,
                shape,
                ggml_type,
                ggml_type_name,
                offset,
                absolute_offset: None,
                element_count: Some(elements),
                estimated_bytes,
            });
        }
    }

    let tensor_data_start = align_offset(reader.offset, alignment)?;
    if tensor_count > 0 && tensor_data_start > file_size {
        anyhow::bail!(
            "GGUF tensor data starts at {tensor_data_start}, beyond file size {file_size}"
        );
    }
    let (payload_ranges, payload_range_status) =
        derive_payload_ranges(&tensor_ranges, tensor_data_start, file_size)?;
    for tensor in &mut tensors {
        tensor.absolute_offset = Some(
            tensor_data_start
                .checked_add(tensor.offset)
                .ok_or_else(|| anyhow::anyhow!("GGUF tensor absolute offset overflow"))?,
        );
    }

    let mut largest_tensors = tensors.clone();
    largest_tensors.sort_by(|left, right| {
        right
            .estimated_bytes
            .unwrap_or(0)
            .cmp(&left.estimated_bytes.unwrap_or(0))
            .then_with(|| left.name.cmp(&right.name))
    });
    largest_tensors.truncate(8);

    let tensor_summary = Some(GgufTensorAggregateSummary {
        described_tensor_count: tensors.len() as u64,
        tensors_omitted_from_summary: tensor_count.saturating_sub(tensors.len() as u64),
        type_counts,
        total_estimated_tensor_bytes,
        unknown_type_tags: unknown_type_tags.into_iter().collect(),
        largest_tensors,
        tensor_data_start,
        file_size,
    });
    let (tokenizer_spec, tokenizer_spec_notes) = build_gguf_tokenizer_spec(&metadata);
    let mut tokenizer_summary = summarize_gguf_tokenizer(&metadata);
    tokenizer_summary.notes.extend(tokenizer_spec_notes);
    let architecture_summary = summarize_gguf_architecture(&metadata);
    let compatibility = classify_gguf_compatibility(&tokenizer_summary, &architecture_summary);

    Ok(GgufMetadataSummary {
        version,
        tensor_count,
        metadata_kv_count,
        metadata,
        tensors,
        tensor_summary,
        hints,
        tokenizer_summary,
        architecture_summary,
        compatibility,
        tokenizer_spec,
        payload_ranges,
        payload_range_status,
        notes: vec!["Metadata-only GGUF inspection; header, key/value metadata, tensor descriptors, tokenizer hints, architecture compatibility signals, and bounded internal tokenizer metadata for narrow synthetic GPT-2/BPE or Llama/SentencePiece shapes may be retained privately, with private fixture-scoped Llama/SentencePiece encode/decode parity helpers. No public/runtime GGUF tokenizer execution is implemented or runnable; runtime weight loading, dequantization/kernels, architecture runtime, and generation are also absent.".into()],
    })
}

#[allow(dead_code)]
#[derive(Debug, Clone, PartialEq, Eq)]
enum GgufBpeError {
    UnsupportedFamily { family: GgufTokenizerSpecFamily },
    VocabSizeMismatch { expected: usize, actual: usize },
    MalformedMerge { index: usize, merge: String },
    DuplicateToken { token: String },
    DuplicateMerge { merge: String },
    UnknownToken { token: String },
    TokenIdTooLarge { index: usize },
}

#[allow(dead_code)]
#[derive(Debug, Clone, Eq, PartialEq)]
struct GgufBpeMergeCandidate {
    rank: usize,
    position: usize,
    left: String,
    right: String,
}

impl Ord for GgufBpeMergeCandidate {
    fn cmp(&self, other: &Self) -> Ordering {
        other
            .rank
            .cmp(&self.rank)
            .then_with(|| other.position.cmp(&self.position))
            .then_with(|| other.left.cmp(&self.left))
            .then_with(|| other.right.cmp(&self.right))
    }
}

impl PartialOrd for GgufBpeMergeCandidate {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

#[allow(dead_code)]
#[derive(Debug, Clone)]
struct GgufBpeRegistry {
    token_to_id: HashMap<String, u32>,
    merge_ranks: HashMap<(String, String), usize>,
}

impl GgufBpeRegistry {
    #[allow(dead_code)]
    fn from_llama3_spec(spec: &GgufTokenizerSpec) -> Result<Self, GgufBpeError> {
        if spec.family != GgufTokenizerSpecFamily::Llama3Bpe {
            return Err(GgufBpeError::UnsupportedFamily {
                family: spec.family,
            });
        }
        if spec.tokens.len() != GGUF_LLAMA3_BPE_EXPECTED_VOCAB_SIZE {
            return Err(GgufBpeError::VocabSizeMismatch {
                expected: GGUF_LLAMA3_BPE_EXPECTED_VOCAB_SIZE,
                actual: spec.tokens.len(),
            });
        }
        Self::from_tokens_and_merges(&spec.tokens, &spec.merges)
    }

    fn from_tokens_and_merges(tokens: &[String], merges: &[String]) -> Result<Self, GgufBpeError> {
        let mut token_to_id = HashMap::new();
        for (index, token) in tokens.iter().enumerate() {
            let token_id =
                u32::try_from(index).map_err(|_| GgufBpeError::TokenIdTooLarge { index })?;
            if token_to_id.insert(token.clone(), token_id).is_some() {
                return Err(GgufBpeError::DuplicateToken {
                    token: token.clone(),
                });
            }
        }

        let mut merge_ranks = HashMap::new();
        for (index, merge) in merges.iter().enumerate() {
            let Some((left, right)) = gguf_bpe_split_merge(merge) else {
                return Err(GgufBpeError::MalformedMerge {
                    index,
                    merge: merge.clone(),
                });
            };
            if merge_ranks
                .insert((left.to_string(), right.to_string()), index)
                .is_some()
            {
                return Err(GgufBpeError::DuplicateMerge {
                    merge: merge.clone(),
                });
            }
        }

        Ok(Self {
            token_to_id,
            merge_ranks,
        })
    }

    fn encode_piece(&self, piece: &str) -> Result<Vec<u32>, GgufBpeError> {
        if piece.is_empty() {
            return Ok(Vec::new());
        }
        let mut pieces = piece.chars().map(|ch| ch.to_string()).collect::<Vec<_>>();

        while pieces.len() > 1 {
            let mut heap = BinaryHeap::new();
            for position in 0..pieces.len() - 1 {
                if let Some(rank) = self
                    .merge_ranks
                    .get(&(pieces[position].clone(), pieces[position + 1].clone()))
                    .copied()
                {
                    heap.push(GgufBpeMergeCandidate {
                        rank,
                        position,
                        left: pieces[position].clone(),
                        right: pieces[position + 1].clone(),
                    });
                }
            }
            let Some(best) = heap.pop() else {
                break;
            };
            let merged = format!("{}{}", pieces[best.position], pieces[best.position + 1]);
            pieces.splice(best.position..=best.position + 1, [merged]);
        }

        pieces
            .iter()
            .map(|token| {
                self.token_to_id
                    .get(token)
                    .copied()
                    .ok_or_else(|| GgufBpeError::UnknownToken {
                        token: token.clone(),
                    })
            })
            .collect()
    }

    fn encode_pretokenized(&self, pieces: &[String]) -> Result<Vec<u32>, GgufBpeError> {
        let mut token_ids = Vec::new();
        for piece in pieces {
            token_ids.extend(self.encode_piece(piece)?);
        }
        Ok(token_ids)
    }
}

#[allow(dead_code)]
fn gguf_bpe_split_merge(merge: &str) -> Option<(&str, &str)> {
    let (left, right) = merge.split_once(' ')?;
    if left.is_empty() || right.is_empty() || right.contains(' ') {
        return None;
    }
    Some((left, right))
}

#[allow(dead_code)]
#[derive(Debug, Clone, PartialEq, Eq)]
enum GgufLlama3Pretoken {
    Text(String),
    Special { text: String, token_id: u32 },
}

#[allow(dead_code)]
fn gguf_llama3_bpe_pretokenize(
    text: &str,
    special_fragments: &[(String, u32)],
    parse_special: bool,
) -> Vec<GgufLlama3Pretoken> {
    let mut specials = special_fragments.to_vec();
    specials.sort_by(|(left_text, left_id), (right_text, right_id)| {
        right_text
            .len()
            .cmp(&left_text.len())
            .then_with(|| left_text.cmp(right_text))
            .then_with(|| left_id.cmp(right_id))
    });

    let mut tokens = Vec::new();
    let mut offset = 0usize;
    while offset < text.len() {
        if parse_special {
            if let Some((special_text, token_id)) = specials
                .iter()
                .find(|(special_text, _)| text[offset..].starts_with(special_text.as_str()))
            {
                tokens.push(GgufLlama3Pretoken::Special {
                    text: special_text.clone(),
                    token_id: *token_id,
                });
                offset += special_text.len();
                continue;
            }
        }

        let next_offset = gguf_llama3_bpe_next_pretoken_end(text, offset);
        tokens.push(GgufLlama3Pretoken::Text(
            text[offset..next_offset].to_string(),
        ));
        offset = next_offset;
    }
    tokens
}

#[allow(dead_code)]
fn gguf_llama3_bpe_next_pretoken_end(text: &str, offset: usize) -> usize {
    if let Some(end) = gguf_llama3_bpe_contraction_end(text, offset) {
        return end;
    }
    let current = gguf_char_at(text, offset).expect("offset is inside text");
    if current != '\r' && current != '\n' && !current.is_alphabetic() && !current.is_numeric() {
        let next_offset = offset + current.len_utf8();
        if next_offset < text.len()
            && gguf_char_at(text, next_offset).is_some_and(|ch| ch.is_alphabetic())
        {
            return gguf_consume_while(text, next_offset, |ch| ch.is_alphabetic());
        }
    }
    if current.is_alphabetic() {
        return gguf_consume_while(text, offset, |ch| ch.is_alphabetic());
    }
    if current.is_numeric() {
        return gguf_consume_up_to(text, offset, 3, |ch| ch.is_numeric());
    }
    if current == ' ' {
        let next_offset = offset + current.len_utf8();
        if next_offset < text.len()
            && gguf_char_at(text, next_offset)
                .is_some_and(|ch| !ch.is_whitespace() && !ch.is_alphabetic() && !ch.is_numeric())
        {
            return gguf_llama3_bpe_consume_punctuation(text, next_offset);
        }
    }
    if !current.is_whitespace() && !current.is_alphabetic() && !current.is_numeric() {
        return gguf_llama3_bpe_consume_punctuation(text, offset);
    }
    if current.is_whitespace() {
        return gguf_llama3_bpe_consume_whitespace(text, offset);
    }
    offset + current.len_utf8()
}

#[allow(dead_code)]
fn gguf_llama3_bpe_contraction_end(text: &str, offset: usize) -> Option<usize> {
    const CONTRACTIONS: [&str; 7] = ["'s", "'t", "'re", "'ve", "'m", "'ll", "'d"];
    let remaining = &text[offset..];
    CONTRACTIONS.iter().find_map(|contraction| {
        if remaining.len() >= contraction.len()
            && remaining[..contraction.len()].eq_ignore_ascii_case(contraction)
        {
            Some(offset + contraction.len())
        } else {
            None
        }
    })
}

#[allow(dead_code)]
fn gguf_llama3_bpe_consume_punctuation(text: &str, offset: usize) -> usize {
    let mut end = gguf_consume_while(text, offset, |ch| {
        !ch.is_whitespace() && !ch.is_alphabetic() && !ch.is_numeric()
    });
    while end < text.len() {
        let ch = gguf_char_at(text, end).expect("end is inside text");
        if ch == '\r' || ch == '\n' {
            end += ch.len_utf8();
        } else {
            break;
        }
    }
    end
}

#[allow(dead_code)]
fn gguf_llama3_bpe_consume_whitespace(text: &str, offset: usize) -> usize {
    let mut end = offset;
    let mut saw_newline = false;
    while end < text.len() {
        let ch = gguf_char_at(text, end).expect("end is inside text");
        if !ch.is_whitespace() {
            break;
        }
        saw_newline |= ch == '\r' || ch == '\n';
        end += ch.len_utf8();
    }
    if saw_newline || end == text.len() {
        end
    } else {
        gguf_consume_while(text, offset, |ch| ch.is_whitespace())
    }
}

#[allow(dead_code)]
fn gguf_char_at(text: &str, offset: usize) -> Option<char> {
    text[offset..].chars().next()
}

#[allow(dead_code)]
fn gguf_consume_while(text: &str, offset: usize, mut predicate: impl FnMut(char) -> bool) -> usize {
    let mut end = offset;
    while end < text.len() {
        let ch = gguf_char_at(text, end).expect("end is inside text");
        if !predicate(ch) {
            break;
        }
        end += ch.len_utf8();
    }
    end
}

#[allow(dead_code)]
fn gguf_consume_up_to(
    text: &str,
    offset: usize,
    max_chars: usize,
    mut predicate: impl FnMut(char) -> bool,
) -> usize {
    let mut end = offset;
    let mut count = 0usize;
    while end < text.len() && count < max_chars {
        let ch = gguf_char_at(text, end).expect("end is inside text");
        if !predicate(ch) {
            break;
        }
        end += ch.len_utf8();
        count += 1;
    }
    end
}

#[allow(dead_code)]
#[derive(Debug, Clone, PartialEq, Eq)]
enum GgufLlamaTokenizerError {
    UnsupportedFamily { family: GgufTokenizerSpecFamily },
    TokenIdOutOfRange { token_id: i32, vocab_size: usize },
    UnsupportedTokenType { token_id: usize, token_type: i32 },
    MissingSpecialToken { label: &'static str },
    IncompleteByteFallback,
    DuplicateOrdinaryPiece { piece: String },
    EmptyOrdinaryPiece { token_id: usize },
    InvalidScore { token_id: usize },
    IncompleteTokenMetadata,
}

#[allow(dead_code)]
#[derive(Debug, Clone)]
struct GgufLlamaSpmPiece {
    text: String,
    left: Option<Box<GgufLlamaSpmPiece>>,
    right: Option<Box<GgufLlamaSpmPiece>>,
}

impl GgufLlamaSpmPiece {
    fn seed(text: String) -> Self {
        Self {
            text,
            left: None,
            right: None,
        }
    }

    fn merged(left: GgufLlamaSpmPiece, right: GgufLlamaSpmPiece) -> Self {
        let text = format!("{}{}", left.text, right.text);
        Self {
            text,
            left: Some(Box::new(left)),
            right: Some(Box::new(right)),
        }
    }
}

#[allow(dead_code)]
#[derive(Debug)]
struct GgufLlamaSpmVocabulary {
    ordinary_tokens: HashMap<String, i32>,
    ordinary_scores: HashMap<String, f32>,
    byte_tokens: HashMap<u8, i32>,
    bos_token_id: i32,
    eos_token_id: Option<i32>,
    unk_token_id: Option<i32>,
}

#[allow(dead_code)]
fn gguf_llama_sentencepiece_token_to_piece_bytes(
    spec: &GgufTokenizerSpec,
    token_id: i32,
    special: bool,
) -> Result<Vec<u8>, GgufLlamaTokenizerError> {
    if spec.family != GgufTokenizerSpecFamily::LlamaSentencePiece {
        return Err(GgufLlamaTokenizerError::UnsupportedFamily {
            family: spec.family,
        });
    }
    let token_index =
        usize::try_from(token_id).map_err(|_| GgufLlamaTokenizerError::TokenIdOutOfRange {
            token_id,
            vocab_size: spec.tokens.len(),
        })?;
    let token = spec
        .tokens
        .get(token_index)
        .ok_or(GgufLlamaTokenizerError::TokenIdOutOfRange {
            token_id,
            vocab_size: spec.tokens.len(),
        })?;
    let token_type = spec
        .token_types
        .get(token_index)
        .copied()
        .unwrap_or_default();

    if token_type == 6 {
        if let Some(byte) = gguf_hex_byte_token_value(token) {
            return Ok(vec![byte]);
        }
    }

    match token_type {
        1 => {}
        2 | 3 if !special => return Ok(Vec::new()),
        2 | 3 => {}
        6 => {}
        // Fail closed for richer llama.cpp token classes until Fathom has
        // pinned goldens for user-defined partitioning, unused/undefined
        // semantics, and any future strip/normalized attribute behavior.
        other => {
            return Err(GgufLlamaTokenizerError::UnsupportedTokenType {
                token_id: token_index,
                token_type: other,
            })
        }
    }

    Ok(gguf_llama_sentencepiece_piece_bytes(token))
}

#[allow(dead_code)]
fn gguf_llama_sentencepiece_detokenize_bytes(
    spec: &GgufTokenizerSpec,
    token_ids: &[i32],
    remove_special: bool,
    unparse_special: bool,
) -> Result<Vec<u8>, GgufLlamaTokenizerError> {
    let bos_token_id = spec
        .special_token_ids
        .get("bos")
        .and_then(|value| i32::try_from(*value).ok());
    let mut output = Vec::new();

    for (position, token_id) in token_ids.iter().copied().enumerate() {
        if remove_special && position == 0 && Some(token_id) == bos_token_id {
            continue;
        }

        let mut piece =
            gguf_llama_sentencepiece_token_to_piece_bytes(spec, token_id, unparse_special)?;
        if position == 0 && piece.first() == Some(&b' ') {
            piece.remove(0);
        }
        output.extend(piece);
    }

    Ok(output)
}

#[allow(dead_code)]
fn gguf_llama_sentencepiece_encode(
    spec: &GgufTokenizerSpec,
    text: &str,
    add_bos: bool,
    parse_special: bool,
) -> Result<Vec<i32>, GgufLlamaTokenizerError> {
    let vocab = gguf_llama_sentencepiece_build_vocab(spec)?;
    let mut token_ids = Vec::new();
    if add_bos {
        token_ids.push(vocab.bos_token_id);
    }

    let special_fragments = if parse_special {
        gguf_llama_sentencepiece_special_fragments(spec, &vocab)
    } else {
        Vec::new()
    };
    let mut offset = 0usize;
    let mut raw_start = 0usize;
    let mut is_prev_special = true;

    while offset < text.len() {
        if let Some((special_text, token_id)) = special_fragments
            .iter()
            .find(|(special_text, _)| text[offset..].starts_with(special_text.as_str()))
        {
            if raw_start < offset {
                gguf_llama_sentencepiece_encode_raw_fragment(
                    &vocab,
                    &text[raw_start..offset],
                    is_prev_special,
                    &mut token_ids,
                )?;
            }
            token_ids.push(*token_id);
            offset += special_text.len();
            raw_start = offset;
            is_prev_special = true;
            continue;
        }
        let ch = text[offset..]
            .chars()
            .next()
            .expect("offset is inside a non-empty UTF-8 string");
        offset += ch.len_utf8();
    }

    if raw_start < text.len() {
        gguf_llama_sentencepiece_encode_raw_fragment(
            &vocab,
            &text[raw_start..],
            is_prev_special,
            &mut token_ids,
        )?;
    }

    Ok(token_ids)
}

#[allow(dead_code)]
fn gguf_llama_sentencepiece_build_vocab(
    spec: &GgufTokenizerSpec,
) -> Result<GgufLlamaSpmVocabulary, GgufLlamaTokenizerError> {
    if spec.family != GgufTokenizerSpecFamily::LlamaSentencePiece {
        return Err(GgufLlamaTokenizerError::UnsupportedFamily {
            family: spec.family,
        });
    }
    if spec.tokens.len() != spec.scores.len() || spec.tokens.len() != spec.token_types.len() {
        return Err(GgufLlamaTokenizerError::IncompleteTokenMetadata);
    }

    let bos_token_id = gguf_llama_sentencepiece_special_id(spec, "bos")?
        .ok_or(GgufLlamaTokenizerError::MissingSpecialToken { label: "bos" })?;
    let eos_token_id = gguf_llama_sentencepiece_special_id(spec, "eos")?;
    let unk_token_id = gguf_llama_sentencepiece_special_id(spec, "unk")?;

    let mut ordinary_tokens = HashMap::new();
    let mut ordinary_scores = HashMap::new();
    let mut byte_tokens = HashMap::new();
    for (token_id, ((token, score), token_type)) in spec
        .tokens
        .iter()
        .zip(spec.scores.iter())
        .zip(spec.token_types.iter())
        .enumerate()
    {
        if !score.is_finite() {
            return Err(GgufLlamaTokenizerError::InvalidScore { token_id });
        }
        if !matches!(*token_type, 1 | 2 | 3 | 6) {
            return Err(GgufLlamaTokenizerError::UnsupportedTokenType {
                token_id,
                token_type: *token_type,
            });
        }
        let token_id_i32 =
            i32::try_from(token_id).map_err(|_| GgufLlamaTokenizerError::TokenIdOutOfRange {
                token_id: i32::MAX,
                vocab_size: spec.tokens.len(),
            })?;
        if *token_type == 6 {
            if let Some(byte) = gguf_hex_byte_token_value(token) {
                byte_tokens.insert(byte, token_id_i32);
            }
            continue;
        }
        if *token_type == 2 || *token_type == 3 {
            continue;
        }
        if token.is_empty() {
            return Err(GgufLlamaTokenizerError::EmptyOrdinaryPiece { token_id });
        }
        if ordinary_tokens
            .insert(token.clone(), token_id_i32)
            .is_some()
        {
            return Err(GgufLlamaTokenizerError::DuplicateOrdinaryPiece {
                piece: token.clone(),
            });
        }
        ordinary_scores.insert(token.clone(), *score);
    }
    if byte_tokens.len() != 256 || !(u8::MIN..=u8::MAX).all(|byte| byte_tokens.contains_key(&byte))
    {
        return Err(GgufLlamaTokenizerError::IncompleteByteFallback);
    }

    Ok(GgufLlamaSpmVocabulary {
        ordinary_tokens,
        ordinary_scores,
        byte_tokens,
        bos_token_id,
        eos_token_id,
        unk_token_id,
    })
}

#[allow(dead_code)]
fn gguf_llama_sentencepiece_special_id(
    spec: &GgufTokenizerSpec,
    label: &'static str,
) -> Result<Option<i32>, GgufLlamaTokenizerError> {
    spec.special_token_ids
        .get(label)
        .map(|value| {
            let token_id =
                i32::try_from(*value).map_err(|_| GgufLlamaTokenizerError::TokenIdOutOfRange {
                    token_id: i32::MAX,
                    vocab_size: spec.tokens.len(),
                })?;
            if *value >= spec.tokens.len() as u64 {
                return Err(GgufLlamaTokenizerError::TokenIdOutOfRange {
                    token_id,
                    vocab_size: spec.tokens.len(),
                });
            }
            Ok(token_id)
        })
        .transpose()
}

#[allow(dead_code)]
fn gguf_llama_sentencepiece_special_fragments(
    spec: &GgufTokenizerSpec,
    vocab: &GgufLlamaSpmVocabulary,
) -> Vec<(String, i32)> {
    let mut fragments = Vec::new();
    for token_id in [
        Some(vocab.bos_token_id),
        vocab.eos_token_id,
        vocab.unk_token_id,
    ]
    .into_iter()
    .flatten()
    {
        if let Some(token) = usize::try_from(token_id)
            .ok()
            .and_then(|index| spec.tokens.get(index))
            .filter(|token| !token.is_empty())
        {
            fragments.push((token.clone(), token_id));
        }
    }
    fragments.sort_by(|(left, _), (right, _)| right.len().cmp(&left.len()).then(left.cmp(right)));
    fragments.dedup_by(|left, right| left.0 == right.0);
    fragments
}

#[allow(dead_code)]
fn gguf_llama_sentencepiece_encode_raw_fragment(
    vocab: &GgufLlamaSpmVocabulary,
    text: &str,
    add_space_prefix: bool,
    token_ids: &mut Vec<i32>,
) -> Result<(), GgufLlamaTokenizerError> {
    if text.is_empty() {
        return Ok(());
    }
    let mut normalized = String::new();
    if add_space_prefix {
        normalized.push(' ');
    }
    normalized.push_str(text);
    let escaped = normalized.replace(' ', "▁");
    gguf_llama_sentencepiece_encode_escaped(vocab, &escaped, token_ids)
}

#[allow(dead_code)]
fn gguf_llama_sentencepiece_encode_escaped(
    vocab: &GgufLlamaSpmVocabulary,
    escaped: &str,
    token_ids: &mut Vec<i32>,
) -> Result<(), GgufLlamaTokenizerError> {
    let mut pieces = escaped
        .chars()
        .map(|ch| GgufLlamaSpmPiece::seed(ch.to_string()))
        .collect::<Vec<_>>();

    while pieces.len() > 1 {
        let mut best: Option<(usize, f32)> = None;
        for index in 0..pieces.len() - 1 {
            let pair = format!("{}{}", pieces[index].text, pieces[index + 1].text);
            let Some(score) = vocab.ordinary_scores.get(&pair).copied() else {
                continue;
            };
            if best
                .map(|(best_index, best_score)| {
                    score > best_score || (score == best_score && index < best_index)
                })
                .unwrap_or(true)
            {
                best = Some((index, score));
            }
        }
        let Some((index, _score)) = best else {
            break;
        };
        let left = pieces.remove(index);
        let right = pieces.remove(index);
        pieces.insert(index, GgufLlamaSpmPiece::merged(left, right));
    }

    for piece in &pieces {
        gguf_llama_sentencepiece_emit_piece(vocab, piece, token_ids)?;
    }
    Ok(())
}

#[allow(dead_code)]
fn gguf_llama_sentencepiece_emit_piece(
    vocab: &GgufLlamaSpmVocabulary,
    piece: &GgufLlamaSpmPiece,
    token_ids: &mut Vec<i32>,
) -> Result<(), GgufLlamaTokenizerError> {
    if let Some(token_id) = vocab.ordinary_tokens.get(&piece.text) {
        token_ids.push(*token_id);
        return Ok(());
    }
    if let (Some(left), Some(right)) = (&piece.left, &piece.right) {
        gguf_llama_sentencepiece_emit_piece(vocab, left, token_ids)?;
        gguf_llama_sentencepiece_emit_piece(vocab, right, token_ids)?;
        return Ok(());
    }
    for byte in piece.text.as_bytes() {
        let token_id = vocab
            .byte_tokens
            .get(byte)
            .copied()
            .ok_or(GgufLlamaTokenizerError::IncompleteByteFallback)?;
        token_ids.push(token_id);
    }
    Ok(())
}

#[allow(dead_code)]
fn gguf_llama_sentencepiece_piece_bytes(piece: &str) -> Vec<u8> {
    piece.replace('▁', " ").into_bytes()
}

#[allow(dead_code)]
fn gguf_hex_byte_token_value(token: &str) -> Option<u8> {
    if !is_gguf_hex_byte_token(token) {
        return None;
    }
    u8::from_str_radix(&token[3..5], 16).ok()
}

fn is_gguf_hex_byte_token(token: &str) -> bool {
    let bytes = token.as_bytes();
    bytes.len() == 6
        && bytes[0] == b'<'
        && bytes[1] == b'0'
        && bytes[2] == b'x'
        && bytes[5] == b'>'
        && bytes[3].is_ascii_hexdigit()
        && bytes[4].is_ascii_hexdigit()
}

pub fn inspect_model_package(path: impl AsRef<Path>) -> anyhow::Result<ModelPackage> {
    let root = path.as_ref().to_path_buf();
    let mut package = ModelPackage {
        root: root.clone(),
        artifacts: Vec::new(),
        tokenizer_files: Vec::new(),
        tokenizer: None,
        config_file: None,
        chat_template_file: None,
        model_type: None,
        architectures: Vec::new(),
        hf_validation: HfPackageValidation::default(),
        notes: Vec::new(),
    };

    if root.is_file() {
        let artifact = detect_model_artifact(&root);
        classify_package_artifact(&mut package, &artifact);
        package.artifacts.push(artifact);
        finalize_hf_package_validation(&mut package);
        return Ok(package);
    }

    if !root.exists() {
        package
            .notes
            .push("Path does not exist yet; recorded as planned import target.".into());
        finalize_hf_package_validation(&mut package);
        return Ok(package);
    }

    for entry in fs::read_dir(&root)? {
        let entry = entry?;
        let path = entry.path();
        if path.is_dir() {
            // Keep initial scan shallow and predictable; Hugging Face repos usually keep key files at root.
            continue;
        }
        let artifact = detect_model_artifact(&path);
        classify_package_artifact(&mut package, &artifact);
        if artifact.format != ModelFormat::Unknown {
            package.artifacts.push(artifact);
        }
    }

    if package.artifacts.is_empty() {
        package
            .notes
            .push("No known model artifacts were found at the package root.".into());
    }
    finalize_hf_package_validation(&mut package);

    Ok(package)
}

fn classify_package_artifact(package: &mut ModelPackage, artifact: &ModelArtifact) {
    match artifact.format {
        ModelFormat::TokenizerJson
        | ModelFormat::TokenizerConfigJson
        | ModelFormat::SentencePiece => {
            package.tokenizer_files.push(artifact.path.clone());
            if artifact.format == ModelFormat::TokenizerConfigJson
                && tokenizer_config_has_chat_template(&artifact.path).unwrap_or(false)
            {
                package.chat_template_file = Some(artifact.path.clone());
            }
        }
        ModelFormat::ConfigJson => {
            package.config_file = Some(artifact.path.clone());
            if let Ok((model_type, architectures)) = read_hf_config_summary(&artifact.path) {
                package.model_type = model_type;
                package.architectures = architectures;
            }
        }
        ModelFormat::ChatTemplate => package.chat_template_file = Some(artifact.path.clone()),
        _ => {}
    }
}

fn finalize_hf_package_validation(package: &mut ModelPackage) {
    package.tokenizer = read_tokenizer_metadata(
        &package.tokenizer_files,
        package.chat_template_file.as_deref(),
    );
    let has_safetensors_weights = package.artifacts.iter().any(|artifact| {
        matches!(
            artifact.format,
            ModelFormat::SafeTensors | ModelFormat::SafeTensorsIndex
        )
    });
    let has_config = package.config_file.is_some();
    let has_tokenizer = package.tokenizer_files.iter().any(|path| {
        let filename = path
            .file_name()
            .and_then(|value| value.to_str())
            .unwrap_or_default();
        matches!(filename, "tokenizer.json" | "spiece.model") || filename.ends_with(".model")
    });
    let has_chat_template = package.chat_template_file.is_some();

    let mut missing_required = Vec::new();
    if has_safetensors_weights {
        if !has_config {
            missing_required.push("config.json".to_string());
        }
        if !has_tokenizer {
            missing_required.push("tokenizer.json or SentencePiece model".to_string());
        }
    }

    let mut notes = Vec::new();
    if has_safetensors_weights && !has_chat_template {
        notes.push("No chat template detected yet; instruct/chat models may need tokenizer_config.json chat_template or chat_template.jinja before chat completions can be faithful.".to_string());
    }
    if !missing_required.is_empty() {
        notes.push(format!(
            "SafeTensors/HF package is not ready for loader metadata: missing {}.",
            missing_required.join(", ")
        ));
    }

    package.hf_validation = HfPackageValidation {
        has_safetensors_weights,
        has_config,
        has_tokenizer,
        has_chat_template,
        ready_for_loader_metadata: has_safetensors_weights && missing_required.is_empty(),
        missing_required,
        notes,
    };
    package.tokenizer = read_tokenizer_metadata(
        &package.tokenizer_files,
        package.chat_template_file.as_deref(),
    );
}

fn read_tokenizer_metadata(
    tokenizer_files: &[PathBuf],
    chat_template_file: Option<&Path>,
) -> Option<TokenizerMetadata> {
    let kind = if tokenizer_files.iter().any(|path| {
        path.file_name()
            .and_then(|value| value.to_str())
            .map(|name| name == "tokenizer.json")
            .unwrap_or(false)
    }) {
        TokenizerKind::HuggingFaceTokenizerJson
    } else if tokenizer_files.iter().any(|path| {
        path.file_name()
            .and_then(|value| value.to_str())
            .map(|name| name == "spiece.model" || name.ends_with(".model"))
            .unwrap_or(false)
    }) {
        TokenizerKind::SentencePiece
    } else {
        return None;
    };

    let mut metadata = TokenizerMetadata {
        kind,
        files: tokenizer_files.to_vec(),
        bos_token: None,
        eos_token: None,
        pad_token: None,
        unk_token: None,
        chat_template: None,
    };

    for path in tokenizer_files {
        if path
            .file_name()
            .and_then(|value| value.to_str())
            .map(|name| name == "tokenizer_config.json")
            .unwrap_or(false)
        {
            if let Ok(summary) = read_tokenizer_config_summary(path) {
                metadata.bos_token = metadata.bos_token.or(summary.bos_token);
                metadata.eos_token = metadata.eos_token.or(summary.eos_token);
                metadata.pad_token = metadata.pad_token.or(summary.pad_token);
                metadata.unk_token = metadata.unk_token.or(summary.unk_token);
                metadata.chat_template = metadata.chat_template.or(summary.chat_template);
            }
        }
    }

    if metadata.chat_template.is_none() {
        if let Some(path) = chat_template_file {
            if let Ok(template) = fs::read_to_string(path) {
                if !template.trim().is_empty() {
                    metadata.chat_template = Some(ChatTemplateMetadata {
                        source: path.to_path_buf(),
                        template,
                        format: ChatTemplateFormat::HuggingFaceJinja,
                        needs_template_engine: true,
                    });
                }
            }
        }
    }

    Some(metadata)
}

struct TokenizerConfigSummary {
    bos_token: Option<String>,
    eos_token: Option<String>,
    pad_token: Option<String>,
    unk_token: Option<String>,
    chat_template: Option<ChatTemplateMetadata>,
}

fn read_tokenizer_config_summary(path: &Path) -> anyhow::Result<TokenizerConfigSummary> {
    let value: serde_json::Value = serde_json::from_slice(&fs::read(path)?)?;
    let chat_template = value
        .get("chat_template")
        .and_then(|value| value.as_str())
        .filter(|value| !value.trim().is_empty())
        .map(|template| ChatTemplateMetadata {
            source: path.to_path_buf(),
            template: template.to_string(),
            format: ChatTemplateFormat::HuggingFaceJinja,
            needs_template_engine: true,
        });

    Ok(TokenizerConfigSummary {
        bos_token: token_value(&value, "bos_token"),
        eos_token: token_value(&value, "eos_token"),
        pad_token: token_value(&value, "pad_token"),
        unk_token: token_value(&value, "unk_token"),
        chat_template,
    })
}

fn token_value(value: &serde_json::Value, key: &str) -> Option<String> {
    value.get(key).and_then(|token| {
        token.as_str().map(str::to_string).or_else(|| {
            token
                .get("content")
                .and_then(|content| content.as_str())
                .map(str::to_string)
        })
    })
}

fn infer_model_task_kind(package: &ModelPackage) -> ModelTaskKind {
    let architecture_text = package.architectures.join(" ").to_ascii_lowercase();
    let model_type = package
        .model_type
        .as_deref()
        .unwrap_or_default()
        .to_ascii_lowercase();
    let root_text = package.root.to_string_lossy().to_ascii_lowercase();

    if architecture_text.contains("causallm")
        || architecture_text.contains("conditionalgeneration")
        || architecture_text.contains("seq2seqlm")
    {
        return ModelTaskKind::TextGeneration;
    }
    if architecture_text.contains("sentence")
        || architecture_text.contains("embedding")
        || architecture_text.contains("featureextraction")
        || matches!(
            model_type.as_str(),
            "bert" | "roberta" | "mpnet" | "distilbert" | "minilm" | "xlm-roberta"
        )
        || root_text.contains("sentence-transformers")
        || root_text.contains("embedding")
    {
        return ModelTaskKind::TextEmbedding;
    }

    ModelTaskKind::Unknown
}

fn read_embedding_dimension_hint(path: &Path) -> Option<usize> {
    let value: serde_json::Value = serde_json::from_slice(&fs::read(path).ok()?).ok()?;
    [
        "sentence_embedding_dimension",
        "embedding_dimension",
        "hidden_size",
        "d_model",
    ]
    .into_iter()
    .filter_map(|key| value.get(key).and_then(|value| value.as_u64()))
    .filter_map(|value| usize::try_from(value).ok())
    .find(|value| *value > 0 && *value < 1_000_000)
}

fn read_hf_context_window_hint(path: &Path) -> Option<usize> {
    let value: serde_json::Value = serde_json::from_slice(&fs::read(path).ok()?).ok()?;
    [
        "max_position_embeddings",
        "n_positions",
        "seq_length",
        "model_max_length",
        "sliding_window",
    ]
    .into_iter()
    .filter_map(|key| value.get(key).and_then(|value| value.as_u64()))
    .filter_map(|value| usize::try_from(value).ok())
    .filter(|value| *value > 0 && *value < 10_000_000)
    .max()
}

fn read_hf_config_summary(path: &Path) -> anyhow::Result<(Option<String>, Vec<String>)> {
    let value: serde_json::Value = serde_json::from_slice(&fs::read(path)?)?;
    let model_type = value
        .get("model_type")
        .and_then(|value| value.as_str())
        .map(str::to_string);
    let architectures = value
        .get("architectures")
        .and_then(|value| value.as_array())
        .map(|items| {
            items
                .iter()
                .filter_map(|item| item.as_str().map(str::to_string))
                .collect()
        })
        .unwrap_or_default();
    Ok((model_type, architectures))
}

fn tokenizer_config_has_chat_template(path: &Path) -> anyhow::Result<bool> {
    let value: serde_json::Value = serde_json::from_slice(&fs::read(path)?)?;
    Ok(value
        .get("chat_template")
        .and_then(|value| value.as_str())
        .map(|value| !value.trim().is_empty())
        .unwrap_or(false))
}

#[cfg(test)]
mod tests {
    use super::*;
    use sha2::{Digest, Sha256};
    use std::{
        fs,
        path::{Path, PathBuf},
        process::Command,
        time::{SystemTime, UNIX_EPOCH},
    };

    #[test]
    fn runtime_file_fingerprint_changes_when_file_changes() {
        let unique = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_nanos();
        let path = std::env::temp_dir().join(format!("fathom-cache-fingerprint-{unique}.txt"));
        fs::write(&path, b"one").unwrap();
        let first = RuntimeFileFingerprint::for_path(&path).unwrap();
        fs::write(&path, b"one plus more").unwrap();
        let second = RuntimeFileFingerprint::for_path(&path).unwrap();
        assert_ne!(first, second);
        fs::remove_file(path).unwrap();
    }

    fn write_runtime_key_package(dir: &Path) {
        fs::create_dir_all(dir).unwrap();
        fs::write(dir.join("config.json"), br#"{"model_type":"gpt2"}"#).unwrap();
        fs::write(dir.join("tokenizer.json"), b"{}").unwrap();
        fs::write(dir.join("model.safetensors"), b"weights").unwrap();
    }

    #[derive(Debug, Clone, Copy)]
    struct GgufTokenizerEncodeGolden {
        label: &'static str,
        input: &'static str,
        no_bos_no_special_ids: &'static [u32],
        default_ids: &'static [u32],
    }

    #[derive(Debug, Clone, Copy)]
    struct GgufTokenizerPieceGolden {
        label: &'static str,
        token_id: i32,
        special_false_hex: &'static str,
        special_true_hex: &'static str,
    }

    #[derive(Debug, Clone, Copy)]
    struct GgufTokenizerDetokenizeGolden {
        label: &'static str,
        token_ids: &'static [i32],
        remove_special: bool,
        unparse_special: bool,
        expected_hex: &'static str,
    }

    const LLAMA_TINY_GGUF_FIXTURE_SHA256: &str =
        "81f226c62d28ed4a1a9b9fa080fcd9f0cc40e0f9d5680036583ff98fbcd035cb";
    const LLAMA_TINY_GGUF_FIXTURE_SIZE: u64 = 1_750_560;
    const LLAMA_TINY_GGUF_VOCAB_SIZE: i32 = 32_000;
    const LLAMA_CPP_TOKENIZE_REFERENCE_REVISION: &str = "15f786e65";
    const LLAMA_CPP_DECODE_REFERENCE_REVISION: &str = "15f786e65";
    const LLAMA_CPP_HOMEBREW_REFERENCE_VERSION: &str = "8680";
    const LLAMA3_8B_INSTRUCT_Q8_0_GGUF_SIZE: u64 = 8_540_770_880;
    const LLAMA_CPP_LLAMA3_TOKENIZE_REFERENCE_REVISION: &str = "665abc6";

    // Pinned llama.cpp references for the local Llama 3 8B Instruct Q8_0 GGUF.
    // These are external-oracle IDs only; they do not bless or expose a GGUF tokenizer path.
    const GGUF_TOKENIZER_LLAMA3_ENCODE_GOLDENS: &[GgufTokenizerEncodeGolden] = &[
        GgufTokenizerEncodeGolden {
            label: "quick brown fox",
            input: "The quick brown fox jumps over the lazy dog.",
            no_bos_no_special_ids: &[791, 4062, 14198, 39935, 35308, 927, 279, 16053, 5679, 13],
            default_ids: &[
                128000, 791, 4062, 14198, 39935, 35308, 927, 279, 16053, 5679, 13,
            ],
        },
        GgufTokenizerEncodeGolden {
            label: "begin text hello hows it going",
            input: "<|begin_of_text|>hello how's it going?",
            no_bos_no_special_ids: &[
                27, 91, 7413, 3659, 4424, 91, 29, 15339, 1268, 596, 433, 2133, 30,
            ],
            default_ids: &[128000, 128000, 15339, 1268, 596, 433, 2133, 30],
        },
    ];

    // These are internal reference goldens only. They intentionally do not expose a GGUF tokenizer.
    // GGUF remains metadata-only; private helpers must stay internal and match these pinned
    // llama.cpp encode, token-to-piece, and detokenize-byte references before any future promotion.
    const GGUF_TOKENIZER_LLAMA_TINY_PIECE_GOLDENS: &[GgufTokenizerPieceGolden] = &[
        GgufTokenizerPieceGolden {
            label: "hello piece",
            token_id: 22172,
            special_false_hex: "2068656c6c6f",
            special_true_hex: "2068656c6c6f",
        },
        GgufTokenizerPieceGolden {
            label: "world piece",
            token_id: 3186,
            special_false_hex: "20776f726c64",
            special_true_hex: "20776f726c64",
        },
        GgufTokenizerPieceGolden {
            label: "Hello piece",
            token_id: 15043,
            special_false_hex: "2048656c6c6f",
            special_true_hex: "2048656c6c6f",
        },
        GgufTokenizerPieceGolden {
            label: "comma",
            token_id: 29892,
            special_false_hex: "2c",
            special_true_hex: "2c",
        },
        GgufTokenizerPieceGolden {
            label: "bang",
            token_id: 29991,
            special_false_hex: "21",
            special_true_hex: "21",
        },
        GgufTokenizerPieceGolden {
            label: "double space",
            token_id: 259,
            special_false_hex: "2020",
            special_true_hex: "2020",
        },
        GgufTokenizerPieceGolden {
            label: "single space",
            token_id: 29871,
            special_false_hex: "20",
            special_true_hex: "20",
        },
        GgufTokenizerPieceGolden {
            label: "newline",
            token_id: 13,
            special_false_hex: "0a",
            special_true_hex: "0a",
        },
        GgufTokenizerPieceGolden {
            label: "tab",
            token_id: 12,
            special_false_hex: "09",
            special_true_hex: "09",
        },
        GgufTokenizerPieceGolden {
            label: "unk special",
            token_id: 0,
            special_false_hex: "",
            special_true_hex: "3c756e6b3e",
        },
        GgufTokenizerPieceGolden {
            label: "bos special",
            token_id: 1,
            special_false_hex: "",
            special_true_hex: "3c733e",
        },
        GgufTokenizerPieceGolden {
            label: "eos special",
            token_id: 2,
            special_false_hex: "",
            special_true_hex: "3c2f733e",
        },
        GgufTokenizerPieceGolden {
            label: "byte fallback nul",
            token_id: 3,
            special_false_hex: "00",
            special_true_hex: "00",
        },
        GgufTokenizerPieceGolden {
            label: "byte fallback space",
            token_id: 35,
            special_false_hex: "20",
            special_true_hex: "20",
        },
        GgufTokenizerPieceGolden {
            label: "byte fallback ff",
            token_id: 258,
            special_false_hex: "ff",
            special_true_hex: "ff",
        },
        GgufTokenizerPieceGolden {
            label: "lobster byte f0",
            token_id: 243,
            special_false_hex: "f0",
            special_true_hex: "f0",
        },
        GgufTokenizerPieceGolden {
            label: "lobster byte 9f",
            token_id: 162,
            special_false_hex: "9f",
            special_true_hex: "9f",
        },
        GgufTokenizerPieceGolden {
            label: "lobster byte a6",
            token_id: 169,
            special_false_hex: "a6",
            special_true_hex: "a6",
        },
        GgufTokenizerPieceGolden {
            label: "lobster byte 9e",
            token_id: 161,
            special_false_hex: "9e",
            special_true_hex: "9e",
        },
        GgufTokenizerPieceGolden {
            label: "cat byte e7",
            token_id: 234,
            special_false_hex: "e7",
            special_true_hex: "e7",
        },
        GgufTokenizerPieceGolden {
            label: "cat byte 8c",
            token_id: 143,
            special_false_hex: "8c",
            special_true_hex: "8c",
        },
        GgufTokenizerPieceGolden {
            label: "cat byte ab",
            token_id: 174,
            special_false_hex: "ab",
            special_true_hex: "ab",
        },
        GgufTokenizerPieceGolden {
            label: "cafe prefix",
            token_id: 274,
            special_false_hex: "2063",
            special_true_hex: "2063",
        },
        GgufTokenizerPieceGolden {
            label: "cafe suffix",
            token_id: 28059,
            special_false_hex: "6166c3a9",
            special_true_hex: "6166c3a9",
        },
        GgufTokenizerPieceGolden {
            label: "combining prefix",
            token_id: 321,
            special_false_hex: "2065",
            special_true_hex: "2065",
        },
        GgufTokenizerPieceGolden {
            label: "combining acute",
            token_id: 30103,
            special_false_hex: "cc81",
            special_true_hex: "cc81",
        },
    ];

    const GGUF_TOKENIZER_LLAMA_TINY_DETOKENIZE_GOLDENS: &[GgufTokenizerDetokenizeGolden] = &[
        GgufTokenizerDetokenizeGolden {
            label: "hello",
            token_ids: &[22172],
            remove_special: false,
            unparse_special: false,
            expected_hex: "68656c6c6f",
        },
        GgufTokenizerDetokenizeGolden {
            label: "hello world",
            token_ids: &[22172, 3186],
            remove_special: false,
            unparse_special: false,
            expected_hex: "68656c6c6f20776f726c64",
        },
        GgufTokenizerDetokenizeGolden {
            label: "leading space",
            token_ids: &[29871, 22172],
            remove_special: false,
            unparse_special: false,
            expected_hex: "2068656c6c6f",
        },
        GgufTokenizerDetokenizeGolden {
            label: "newline",
            token_ids: &[22172, 13, 11526],
            remove_special: false,
            unparse_special: false,
            expected_hex: "68656c6c6f0a776f726c64",
        },
        GgufTokenizerDetokenizeGolden {
            label: "tab",
            token_ids: &[22172, 12, 11526],
            remove_special: false,
            unparse_special: false,
            expected_hex: "68656c6c6f09776f726c64",
        },
        GgufTokenizerDetokenizeGolden {
            label: "lobster bytes",
            token_ids: &[243, 162, 169, 161],
            remove_special: false,
            unparse_special: false,
            expected_hex: "f09fa69e",
        },
        GgufTokenizerDetokenizeGolden {
            label: "cat bytes",
            token_ids: &[234, 143, 174],
            remove_special: false,
            unparse_special: false,
            expected_hex: "e78cab",
        },
        GgufTokenizerDetokenizeGolden {
            label: "nul byte",
            token_ids: &[3],
            remove_special: false,
            unparse_special: false,
            expected_hex: "00",
        },
        GgufTokenizerDetokenizeGolden {
            label: "space byte alone",
            token_ids: &[35],
            remove_special: false,
            unparse_special: false,
            expected_hex: "",
        },
        GgufTokenizerDetokenizeGolden {
            label: "ff byte",
            token_ids: &[258],
            remove_special: false,
            unparse_special: false,
            expected_hex: "ff",
        },
        GgufTokenizerDetokenizeGolden {
            label: "mixed byte fallback",
            token_ids: &[3, 35, 258],
            remove_special: false,
            unparse_special: false,
            expected_hex: "0020ff",
        },
        GgufTokenizerDetokenizeGolden {
            label: "incomplete utf8 bytes",
            token_ids: &[243, 162],
            remove_special: false,
            unparse_special: false,
            expected_hex: "f09f",
        },
        GgufTokenizerDetokenizeGolden {
            label: "specials hidden",
            token_ids: &[1, 22172, 2],
            remove_special: false,
            unparse_special: false,
            expected_hex: "2068656c6c6f",
        },
        GgufTokenizerDetokenizeGolden {
            label: "specials unparsed",
            token_ids: &[1, 22172, 2],
            remove_special: false,
            unparse_special: true,
            expected_hex: "3c733e2068656c6c6f3c2f733e",
        },
        GgufTokenizerDetokenizeGolden {
            label: "remove specials hidden",
            token_ids: &[1, 22172, 2],
            remove_special: true,
            unparse_special: false,
            expected_hex: "2068656c6c6f",
        },
        GgufTokenizerDetokenizeGolden {
            label: "remove specials unparsed",
            token_ids: &[1, 22172, 2],
            remove_special: true,
            unparse_special: true,
            expected_hex: "2068656c6c6f3c2f733e",
        },
    ];

    const GGUF_TOKENIZER_LLAMA_TINY_ENCODE_GOLDENS: &[GgufTokenizerEncodeGolden] = &[
        GgufTokenizerEncodeGolden {
            label: "empty string",
            input: "",
            no_bos_no_special_ids: &[],
            default_ids: &[1],
        },
        GgufTokenizerEncodeGolden {
            label: "single space",
            input: " ",
            no_bos_no_special_ids: &[259],
            default_ids: &[1, 259],
        },
        GgufTokenizerEncodeGolden {
            label: "hello",
            input: "hello",
            no_bos_no_special_ids: &[22172],
            default_ids: &[1, 22172],
        },
        GgufTokenizerEncodeGolden {
            label: "world",
            input: "world",
            no_bos_no_special_ids: &[3186],
            default_ids: &[1, 3186],
        },
        GgufTokenizerEncodeGolden {
            label: "hello world",
            input: "hello world",
            no_bos_no_special_ids: &[22172, 3186],
            default_ids: &[1, 22172, 3186],
        },
        GgufTokenizerEncodeGolden {
            label: "leading space",
            input: " hello",
            no_bos_no_special_ids: &[29871, 22172],
            default_ids: &[1, 29871, 22172],
        },
        GgufTokenizerEncodeGolden {
            label: "double space",
            input: "hello  world",
            no_bos_no_special_ids: &[22172, 29871, 3186],
            default_ids: &[1, 22172, 29871, 3186],
        },
        GgufTokenizerEncodeGolden {
            label: "newline",
            input: "hello\nworld",
            no_bos_no_special_ids: &[22172, 13, 11526],
            default_ids: &[1, 22172, 13, 11526],
        },
        GgufTokenizerEncodeGolden {
            label: "tab",
            input: "hello\tworld",
            no_bos_no_special_ids: &[22172, 12, 11526],
            default_ids: &[1, 22172, 12, 11526],
        },
        GgufTokenizerEncodeGolden {
            label: "Hello world punctuation",
            input: "Hello, world!",
            no_bos_no_special_ids: &[15043, 29892, 3186, 29991],
            default_ids: &[1, 15043, 29892, 3186, 29991],
        },
        GgufTokenizerEncodeGolden {
            label: "a dot b",
            input: "a.b",
            no_bos_no_special_ids: &[263, 29889, 29890],
            default_ids: &[1, 263, 29889, 29890],
        },
        GgufTokenizerEncodeGolden {
            label: "yes/no?",
            input: "yes/no?",
            no_bos_no_special_ids: &[4874, 29914, 1217, 29973],
            default_ids: &[1, 4874, 29914, 1217, 29973],
        },
        GgufTokenizerEncodeGolden {
            label: "literal <s>",
            input: "<s>",
            no_bos_no_special_ids: &[529, 29879, 29958],
            default_ids: &[1, 1],
        },
        GgufTokenizerEncodeGolden {
            label: "literal </s>",
            input: "</s>",
            no_bos_no_special_ids: &[1533, 29879, 29958],
            default_ids: &[1, 2],
        },
        GgufTokenizerEncodeGolden {
            label: "literal <unk>",
            input: "<unk>",
            no_bos_no_special_ids: &[529, 2960, 29958],
            default_ids: &[1, 0],
        },
        GgufTokenizerEncodeGolden {
            label: "lobster emoji byte fallback",
            input: "🦞",
            no_bos_no_special_ids: &[29871, 243, 162, 169, 161],
            default_ids: &[1, 29871, 243, 162, 169, 161],
        },
        GgufTokenizerEncodeGolden {
            label: "Chinese byte fallback mix",
            input: "给我猫",
            no_bos_no_special_ids: &[29871, 31999, 30672, 234, 143, 174],
            default_ids: &[1, 29871, 31999, 30672, 234, 143, 174],
        },
        GgufTokenizerEncodeGolden {
            label: "cafe accent",
            input: "café",
            no_bos_no_special_ids: &[274, 28059],
            default_ids: &[1, 274, 28059],
        },
        GgufTokenizerEncodeGolden {
            label: "combining accent regression probe",
            input: "e\u{0301}",
            no_bos_no_special_ids: &[321, 30103],
            default_ids: &[1, 321, 30103],
        },
        GgufTokenizerEncodeGolden {
            label: "single newline",
            input: "\n",
            no_bos_no_special_ids: &[29871, 13],
            default_ids: &[1, 29871, 13],
        },
        GgufTokenizerEncodeGolden {
            label: "single tab",
            input: "\t",
            no_bos_no_special_ids: &[29871, 12],
            default_ids: &[1, 29871, 12],
        },
        GgufTokenizerEncodeGolden {
            label: "leading newline",
            input: "\nhello",
            no_bos_no_special_ids: &[29871, 13, 12199],
            default_ids: &[1, 29871, 13, 12199],
        },
        GgufTokenizerEncodeGolden {
            label: "trailing newline",
            input: "hello\n",
            no_bos_no_special_ids: &[22172, 13],
            default_ids: &[1, 22172, 13],
        },
        GgufTokenizerEncodeGolden {
            label: "leading tab",
            input: "\thello",
            no_bos_no_special_ids: &[29871, 12, 12199],
            default_ids: &[1, 29871, 12, 12199],
        },
        GgufTokenizerEncodeGolden {
            label: "trailing tab",
            input: "hello\t",
            no_bos_no_special_ids: &[22172, 12],
            default_ids: &[1, 22172, 12],
        },
        GgufTokenizerEncodeGolden {
            label: "three spaces",
            input: "   ",
            no_bos_no_special_ids: &[268],
            default_ids: &[1, 268],
        },
        GgufTokenizerEncodeGolden {
            label: "hello three spaces world",
            input: "hello   world",
            no_bos_no_special_ids: &[22172, 259, 3186],
            default_ids: &[1, 22172, 259, 3186],
        },
        GgufTokenizerEncodeGolden {
            label: "hello space newline world",
            input: "hello \nworld",
            no_bos_no_special_ids: &[22172, 29871, 13, 11526],
            default_ids: &[1, 22172, 29871, 13, 11526],
        },
        GgufTokenizerEncodeGolden {
            label: "hello newline space world",
            input: "hello\n world",
            no_bos_no_special_ids: &[22172, 13, 3186],
            default_ids: &[1, 22172, 13, 3186],
        },
        GgufTokenizerEncodeGolden {
            label: "abcd ambiguous segmentation",
            input: "abcd",
            no_bos_no_special_ids: &[633, 2252],
            default_ids: &[1, 633, 2252],
        },
        GgufTokenizerEncodeGolden {
            label: "cats ambiguous segmentation",
            input: "cats",
            no_bos_no_special_ids: &[274, 1446],
            default_ids: &[1, 274, 1446],
        },
        GgufTokenizerEncodeGolden {
            label: "tokenization ambiguous segmentation",
            input: "tokenization",
            no_bos_no_special_ids: &[5993, 2133],
            default_ids: &[1, 5993, 2133],
        },
        GgufTokenizerEncodeGolden {
            label: "BOS special in context",
            input: "x<s>y",
            no_bos_no_special_ids: &[921, 29966, 29879, 29958, 29891],
            default_ids: &[1, 921, 1, 343],
        },
        GgufTokenizerEncodeGolden {
            label: "EOS special in context",
            input: "x</s>y",
            no_bos_no_special_ids: &[921, 829, 29879, 29958, 29891],
            default_ids: &[1, 921, 2, 343],
        },
        GgufTokenizerEncodeGolden {
            label: "UNK special in context",
            input: "x<unk>y",
            no_bos_no_special_ids: &[921, 29966, 2960, 29958, 29891],
            default_ids: &[1, 921, 0, 343],
        },
        GgufTokenizerEncodeGolden {
            label: "decomposed word",
            input: "e\u{0301}clair",
            no_bos_no_special_ids: &[321, 30103, 16398, 381],
            default_ids: &[1, 321, 30103, 16398, 381],
        },
        GgufTokenizerEncodeGolden {
            label: "emoji context byte fallback",
            input: "hi 🦞!",
            no_bos_no_special_ids: &[7251, 29871, 243, 162, 169, 161, 29991],
            default_ids: &[1, 7251, 29871, 243, 162, 169, 161, 29991],
        },
        GgufTokenizerEncodeGolden {
            label: "zero width joiner",
            input: "a\u{200D}b",
            no_bos_no_special_ids: &[263, 30722, 29890],
            default_ids: &[1, 263, 30722, 29890],
        },
        GgufTokenizerEncodeGolden {
            label: "non-breaking space",
            input: "a\u{00A0}b",
            no_bos_no_special_ids: &[263, 30081, 29890],
            default_ids: &[1, 263, 30081, 29890],
        },
    ];

    #[test]
    fn gguf_tokenizer_llama_tiny_encode_goldens_are_pinned_and_cover_reference_modes() {
        assert_eq!(LLAMA_CPP_TOKENIZE_REFERENCE_REVISION, "15f786e65");
        assert_eq!(LLAMA_TINY_GGUF_FIXTURE_SIZE, 1_750_560);
        assert_eq!(
            LLAMA_TINY_GGUF_FIXTURE_SHA256,
            "81f226c62d28ed4a1a9b9fa080fcd9f0cc40e0f9d5680036583ff98fbcd035cb"
        );
        assert!(GGUF_TOKENIZER_LLAMA_TINY_ENCODE_GOLDENS.len() >= 18);
        for golden in GGUF_TOKENIZER_LLAMA_TINY_ENCODE_GOLDENS {
            if golden.input.is_empty() {
                assert!(golden.no_bos_no_special_ids.is_empty(), "{}", golden.label);
            }
            assert_eq!(
                golden.default_ids.first().copied(),
                Some(1),
                "default llama.cpp mode should add BOS for {}",
                golden.label
            );
        }
    }

    #[test]
    fn gguf_tokenizer_llama3_encode_goldens_are_pinned_and_cover_reference_modes() {
        assert_eq!(LLAMA_CPP_LLAMA3_TOKENIZE_REFERENCE_REVISION, "665abc6");
        assert_eq!(LLAMA3_8B_INSTRUCT_Q8_0_GGUF_SIZE, 8_540_770_880);
        assert_eq!(GGUF_LLAMA3_BPE_EXPECTED_VOCAB_SIZE, 128_256);
        assert_eq!(GGUF_TOKENIZER_LLAMA3_ENCODE_GOLDENS.len(), 2);
        assert_eq!(
            GGUF_TOKENIZER_LLAMA3_ENCODE_GOLDENS[0].default_ids,
            &[128000, 791, 4062, 14198, 39935, 35308, 927, 279, 16053, 5679, 13]
        );
        assert_eq!(
            GGUF_TOKENIZER_LLAMA3_ENCODE_GOLDENS[1].default_ids,
            &[128000, 128000, 15339, 1268, 596, 433, 2133, 30]
        );
        for golden in GGUF_TOKENIZER_LLAMA3_ENCODE_GOLDENS {
            assert_eq!(
                golden.default_ids.first().copied(),
                Some(128000),
                "default llama.cpp mode should add Llama 3 BOS for {}",
                golden.label
            );
            assert!(
                !golden.no_bos_no_special_ids.contains(&128000),
                "no-BOS/no-special mode should not parse or prepend Llama 3 BOS for {}",
                golden.label
            );
        }
    }

    #[test]
    fn gguf_llama3_manual_pretokenizer_covers_contractions_punctuation_and_whitespace() {
        let tokens = gguf_llama3_bpe_pretokenize("Hello, world!\n\nIt's fine.", &[], false);
        assert_eq!(
            tokens,
            vec![
                GgufLlama3Pretoken::Text("Hello".into()),
                GgufLlama3Pretoken::Text(",".into()),
                GgufLlama3Pretoken::Text(" world".into()),
                GgufLlama3Pretoken::Text("!\n\n".into()),
                GgufLlama3Pretoken::Text("It".into()),
                GgufLlama3Pretoken::Text("'s".into()),
                GgufLlama3Pretoken::Text(" fine".into()),
                GgufLlama3Pretoken::Text(".".into()),
            ]
        );

        let numeric_and_trailing_ws = gguf_llama3_bpe_pretokenize("abc1234   ", &[], false);
        assert_eq!(
            numeric_and_trailing_ws,
            vec![
                GgufLlama3Pretoken::Text("abc".into()),
                GgufLlama3Pretoken::Text("123".into()),
                GgufLlama3Pretoken::Text("4".into()),
                GgufLlama3Pretoken::Text("   ".into()),
            ]
        );
    }

    #[test]
    fn gguf_llama3_manual_pretokenizer_handles_special_tokens_without_regex() {
        let specials = vec![("<|begin_of_text|>".to_string(), 128000)];
        let tokens =
            gguf_llama3_bpe_pretokenize("<|begin_of_text|>hello how's it going?", &specials, true);
        assert_eq!(
            tokens,
            vec![
                GgufLlama3Pretoken::Special {
                    text: "<|begin_of_text|>".into(),
                    token_id: 128000,
                },
                GgufLlama3Pretoken::Text("hello".into()),
                GgufLlama3Pretoken::Text(" how".into()),
                GgufLlama3Pretoken::Text("'s".into()),
                GgufLlama3Pretoken::Text(" it".into()),
                GgufLlama3Pretoken::Text(" going".into()),
                GgufLlama3Pretoken::Text("?".into()),
            ]
        );
    }

    fn exact_size_llama3_bpe_tokens_for_tests() -> Vec<String> {
        let mut tokens = (0..GGUF_LLAMA3_BPE_EXPECTED_VOCAB_SIZE)
            .map(|index| format!("tok{index}"))
            .collect::<Vec<_>>();
        tokens[128000] = "<|begin_of_text|>".into();
        tokens[128001] = "<|end_of_text|>".into();
        tokens
    }

    fn write_minimal_llama3_bpe_tokenizer_gguf(
        path: &Path,
        tokens: &[String],
        merges: &[&str],
        special_ids: &[(&str, u32)],
    ) {
        let mut bytes = Vec::new();
        bytes.extend_from_slice(b"GGUF");
        bytes.extend_from_slice(&3u32.to_le_bytes());
        bytes.extend_from_slice(&0u64.to_le_bytes());
        bytes.extend_from_slice(&(4u64 + special_ids.len() as u64).to_le_bytes());
        write_gguf_string_kv(&mut bytes, "tokenizer.ggml.model", "gpt2");
        write_gguf_string_kv(&mut bytes, "tokenizer.ggml.pre", "llama-bpe");
        let token_refs = tokens.iter().map(String::as_str).collect::<Vec<_>>();
        write_gguf_string_array_kv(&mut bytes, "tokenizer.ggml.tokens", &token_refs);
        write_gguf_string_array_kv(&mut bytes, "tokenizer.ggml.merges", merges);
        for (key, value) in special_ids {
            write_gguf_u32_kv(&mut bytes, key, *value);
        }
        fs::write(path, bytes).unwrap();
    }

    #[test]
    fn builds_exact_size_llama3_bpe_tokenizer_spec_as_internal_metadata_only() {
        let unique = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_nanos();
        let path = std::env::temp_dir().join(format!("fathom-gguf-llama3-bpe-{unique}.gguf"));
        write_minimal_llama3_bpe_tokenizer_gguf(
            &path,
            &exact_size_llama3_bpe_tokens_for_tests(),
            &["tok1 tok2", "tok3 tok4"],
            &[
                ("tokenizer.ggml.bos_token_id", 128000),
                ("tokenizer.ggml.eos_token_id", 128001),
            ],
        );

        let metadata = read_gguf_metadata_summary(&path).unwrap();
        let spec = metadata
            .tokenizer_spec
            .as_ref()
            .expect("exact-size Llama 3 BPE metadata should be retained internally");
        assert_eq!(spec.family, GgufTokenizerSpecFamily::Llama3Bpe);
        assert_eq!(spec.tokens.len(), GGUF_LLAMA3_BPE_EXPECTED_VOCAB_SIZE);
        assert_eq!(spec.merges, vec!["tok1 tok2", "tok3 tok4"]);
        assert_eq!(spec.special_token_ids.get("bos"), Some(&128000));
        assert_eq!(spec.special_token_ids.get("eos"), Some(&128001));
        assert!(metadata
            .tokenizer_summary
            .notes
            .iter()
            .any(|note| { note.contains("no public/runtime tokenizer execution") }));

        let public_json = serde_json::to_value(&metadata).unwrap();
        assert!(public_json.get("tokenizer_spec").is_none());
        assert!(public_json["metadata"]
            .as_array()
            .unwrap()
            .iter()
            .filter_map(|entry| entry.get("value"))
            .all(|value| value.get("full_strings").is_none()));
        fs::remove_file(path).unwrap();
    }

    #[test]
    fn rejects_malformed_llama3_bpe_metadata_before_internal_retention() {
        let unique = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_nanos();

        let short_path =
            std::env::temp_dir().join(format!("fathom-gguf-llama3-short-{unique}.gguf"));
        write_minimal_llama3_bpe_tokenizer_gguf(
            &short_path,
            &["tok0".into(), "tok1".into()],
            &["tok0 tok1"],
            &[],
        );
        let metadata = read_gguf_metadata_summary(&short_path).unwrap();
        assert!(metadata.tokenizer_spec.is_none());
        assert!(metadata
            .tokenizer_summary
            .notes
            .iter()
            .any(|note| { note.contains("does not match expected Llama 3 vocab size") }));
        fs::remove_file(short_path).unwrap();

        let malformed_merge_path =
            std::env::temp_dir().join(format!("fathom-gguf-llama3-merge-{unique}.gguf"));
        write_minimal_llama3_bpe_tokenizer_gguf(
            &malformed_merge_path,
            &exact_size_llama3_bpe_tokens_for_tests(),
            &["tok0"],
            &[],
        );
        let metadata = read_gguf_metadata_summary(&malformed_merge_path).unwrap();
        assert!(metadata.tokenizer_spec.is_none());
        assert!(metadata
            .tokenizer_summary
            .notes
            .iter()
            .any(|note| note.contains("not a two-token BPE merge")));
        fs::remove_file(malformed_merge_path).unwrap();

        let duplicate_path =
            std::env::temp_dir().join(format!("fathom-gguf-llama3-dupe-{unique}.gguf"));
        let mut duplicate_tokens = exact_size_llama3_bpe_tokens_for_tests();
        duplicate_tokens[42] = duplicate_tokens[41].clone();
        write_minimal_llama3_bpe_tokenizer_gguf(
            &duplicate_path,
            &duplicate_tokens,
            &["tok1 tok2"],
            &[],
        );
        let metadata = read_gguf_metadata_summary(&duplicate_path).unwrap();
        assert!(metadata.tokenizer_spec.is_none());
        assert!(metadata
            .tokenizer_summary
            .notes
            .iter()
            .any(|note| note.contains("duplicate token")));
        fs::remove_file(duplicate_path).unwrap();

        let special_path =
            std::env::temp_dir().join(format!("fathom-gguf-llama3-special-{unique}.gguf"));
        write_minimal_llama3_bpe_tokenizer_gguf(
            &special_path,
            &exact_size_llama3_bpe_tokens_for_tests(),
            &["tok1 tok2"],
            &[(
                "tokenizer.ggml.bos_token_id",
                GGUF_LLAMA3_BPE_EXPECTED_VOCAB_SIZE as u32,
            )],
        );
        let metadata = read_gguf_metadata_summary(&special_path).unwrap();
        assert!(metadata.tokenizer_spec.is_none());
        assert!(metadata
            .tokenizer_summary
            .notes
            .iter()
            .any(|note| note.contains("outside token array length")));
        fs::remove_file(special_path).unwrap();
    }

    #[test]
    fn gguf_bpe_registry_merges_with_std_binary_heap_priority() {
        let tokens = ["a", "b", "c", "ab", "bc", "abc"]
            .into_iter()
            .map(str::to_string)
            .collect::<Vec<_>>();
        let merges = ["a b", "b c", "ab c"]
            .into_iter()
            .map(str::to_string)
            .collect::<Vec<_>>();
        let registry = GgufBpeRegistry::from_tokens_and_merges(&tokens, &merges).unwrap();

        assert_eq!(registry.encode_piece("abc").unwrap(), vec![5]);
        assert_eq!(
            registry
                .encode_pretokenized(&["ab".to_string(), "c".to_string()])
                .unwrap(),
            vec![3, 2]
        );
    }

    #[test]
    fn gguf_bpe_registry_fails_closed_on_bad_metadata() {
        let duplicate_tokens = vec!["a".to_string(), "a".to_string()];
        let merges = vec!["a a".to_string()];
        assert!(matches!(
            GgufBpeRegistry::from_tokens_and_merges(&duplicate_tokens, &merges),
            Err(GgufBpeError::DuplicateToken { .. })
        ));

        let tokens = vec!["a".to_string(), "b".to_string()];
        let bad_merges = vec!["a".to_string()];
        assert!(matches!(
            GgufBpeRegistry::from_tokens_and_merges(&tokens, &bad_merges),
            Err(GgufBpeError::MalformedMerge { .. })
        ));
    }

    fn gguf_tokenizer_llama_tokenize_bin_for_tests() -> Option<PathBuf> {
        std::env::var("LLAMA_TOKENIZE_BIN").ok().map(PathBuf::from)
    }

    fn gguf_tokenizer_assert_pinned_fixture_for_tests(path: &Path) {
        let metadata = fs::metadata(path).unwrap_or_else(|error| {
            panic!("FATHOM_GGUF_LLAMA_TINY_FIXTURE must point at the pinned GGUF fixture: {error}")
        });
        assert_eq!(
            metadata.len(),
            LLAMA_TINY_GGUF_FIXTURE_SIZE,
            "pinned llama-2-tiny-random.gguf fixture size changed"
        );
        let bytes = fs::read(path).unwrap();
        let digest = Sha256::digest(&bytes);
        assert_eq!(
            format!("{digest:x}"),
            LLAMA_TINY_GGUF_FIXTURE_SHA256,
            "pinned llama-2-tiny-random.gguf fixture SHA256 changed"
        );
    }

    fn gguf_tokenizer_run_llama_tokenize_ids_for_tests(
        llama_tokenize_bin: &Path,
        model_path: &Path,
        prompt_path: &Path,
        no_bos_no_special: bool,
    ) -> String {
        let mut command = Command::new(llama_tokenize_bin);
        command
            .arg("-m")
            .arg(model_path)
            .arg("-f")
            .arg(prompt_path)
            .arg("--ids")
            .arg("--log-disable");
        if no_bos_no_special {
            command.arg("--no-bos").arg("--no-parse-special");
        }
        let output = command.output().unwrap_or_else(|error| {
            panic!(
                "failed to execute {}: {error}",
                llama_tokenize_bin.display()
            )
        });
        assert!(
            output.status.success(),
            "llama-tokenize failed with status {:?}\nstdout:\n{}\nstderr:\n{}",
            output.status.code(),
            String::from_utf8_lossy(&output.stdout),
            String::from_utf8_lossy(&output.stderr)
        );
        String::from_utf8(output.stdout)
            .expect("llama-tokenize --ids should emit UTF-8")
            .trim()
            .to_string()
    }

    fn gguf_tokenizer_format_ids_for_tests(ids: &[u32]) -> String {
        let body = ids
            .iter()
            .map(u32::to_string)
            .collect::<Vec<_>>()
            .join(", ");
        format!("[{body}]")
    }

    fn gguf_tokenizer_llama_fixture_for_internal_encode_tests() -> Option<PathBuf> {
        std::env::var("FATHOM_GGUF_LLAMA_TINY_FIXTURE")
            .ok()
            .map(PathBuf::from)
            .or_else(|| {
                let path = std::env::temp_dir().join("llama-2-tiny-random.gguf");
                path.exists().then_some(path)
            })
    }

    fn gguf_tokenizer_llama3_fixture_for_tests() -> Option<PathBuf> {
        std::env::var("FATHOM_GGUF_LLAMA3_FIXTURE")
            .ok()
            .map(PathBuf::from)
            .or_else(|| {
                let path = PathBuf::from(
                    "/Volumes/SSK Drive/Camelid/models/Meta-Llama-3-8B-Instruct-Q8_0.gguf",
                );
                path.exists().then_some(path)
            })
    }

    fn gguf_tokenizer_assert_llama3_fixture_for_tests(path: &Path) {
        let metadata = fs::metadata(path).unwrap_or_else(|error| {
            panic!("FATHOM_GGUF_LLAMA3_FIXTURE must point at the local Llama 3 GGUF: {error}")
        });
        assert_eq!(
            metadata.len(),
            LLAMA3_8B_INSTRUCT_Q8_0_GGUF_SIZE,
            "local Llama 3 8B Instruct Q8_0 GGUF fixture size changed"
        );
    }

    #[test]
    #[ignore = "requires FATHOM_GGUF_LLAMA3_FIXTURE or the local Camelid Llama 3 GGUF"]
    fn gguf_tokenizer_llama3_fixture_retains_exact_128k_bpe_metadata_before_runtime_use() {
        let Some(fixture_path) = gguf_tokenizer_llama3_fixture_for_tests() else {
            return;
        };
        gguf_tokenizer_assert_llama3_fixture_for_tests(&fixture_path);
        let metadata = read_gguf_metadata_summary(&fixture_path).unwrap();
        assert_eq!(
            metadata.tokenizer_summary.token_count,
            Some(GGUF_LLAMA3_BPE_EXPECTED_VOCAB_SIZE as u64)
        );
        assert_eq!(metadata.tokenizer_summary.model.as_deref(), Some("gpt2"));
        let spec = metadata
            .tokenizer_spec
            .as_ref()
            .expect("Llama 3 fixture should retain private BPE tokenizer metadata");
        assert_eq!(spec.family, GgufTokenizerSpecFamily::Llama3Bpe);
        assert_eq!(spec.tokens.len(), GGUF_LLAMA3_BPE_EXPECTED_VOCAB_SIZE);
        assert_eq!(
            spec.tokens.get(128000).map(String::as_str),
            Some("<|begin_of_text|>")
        );
        assert_eq!(
            spec.tokens.get(128001).map(String::as_str),
            Some("<|end_of_text|>")
        );
        assert_eq!(spec.special_token_ids.get("bos"), Some(&128000));
        assert_eq!(spec.special_token_ids.get("eos"), Some(&128001));
        let registry = GgufBpeRegistry::from_llama3_spec(spec).unwrap();
        assert_eq!(registry.token_to_id.get("<|begin_of_text|>"), Some(&128000));
        assert_eq!(registry.token_to_id.get("<|end_of_text|>"), Some(&128001));
    }

    fn llama_encode_test_spec(extra: &[(&str, f32)]) -> GgufTokenizerSpec {
        let mut tokens = vec!["<unk>".to_string(), "<s>".to_string(), "</s>".to_string()];
        let mut scores = vec![0.0f32, 0.0, 0.0];
        let mut token_types = vec![2i32, 3, 3];
        for byte in 0u16..=255 {
            tokens.push(format!("<0x{byte:02X}>"));
            scores.push(0.0);
            token_types.push(6);
        }
        for (token, score) in extra {
            tokens.push((*token).to_string());
            scores.push(*score);
            token_types.push(1);
        }
        GgufTokenizerSpec {
            family: GgufTokenizerSpecFamily::LlamaSentencePiece,
            model: "llama".to_string(),
            tokens,
            merges: Vec::new(),
            scores,
            token_types,
            special_token_ids: BTreeMap::from([
                ("unk".to_string(), 0),
                ("bos".to_string(), 1),
                ("eos".to_string(), 2),
            ]),
            has_byte_fallback: true,
        }
    }

    #[test]
    fn gguf_tokenizer_llama_internal_encode_goldens_match_retained_fixture_when_available() {
        let Some(fixture_path) = gguf_tokenizer_llama_fixture_for_internal_encode_tests() else {
            return;
        };
        gguf_tokenizer_assert_pinned_fixture_for_tests(&fixture_path);
        let metadata = read_gguf_metadata_summary(&fixture_path).unwrap();
        let spec = metadata
            .tokenizer_spec
            .as_ref()
            .expect("pinned fixture should retain private Llama tokenizer metadata");

        for golden in GGUF_TOKENIZER_LLAMA_TINY_ENCODE_GOLDENS {
            let actual_no_bos_no_special =
                gguf_llama_sentencepiece_encode(spec, golden.input, false, false).unwrap();
            assert_eq!(
                actual_no_bos_no_special,
                golden
                    .no_bos_no_special_ids
                    .iter()
                    .copied()
                    .map(|token_id| token_id as i32)
                    .collect::<Vec<_>>(),
                "private internal Llama encode mismatch for {} in no-BOS/no-special mode",
                golden.label
            );

            let actual_default =
                gguf_llama_sentencepiece_encode(spec, golden.input, true, true).unwrap();
            assert_eq!(
                actual_default,
                golden
                    .default_ids
                    .iter()
                    .copied()
                    .map(|token_id| token_id as i32)
                    .collect::<Vec<_>>(),
                "private internal Llama encode mismatch for {} in default mode",
                golden.label
            );
        }
    }

    #[test]
    fn gguf_tokenizer_llama_internal_encode_uses_lower_left_tie_break() {
        let spec =
            llama_encode_test_spec(&[("a", 0.0), ("b", 0.0), ("c", 0.0), ("ab", 1.0), ("bc", 1.0)]);
        let vocab = gguf_llama_sentencepiece_build_vocab(&spec).unwrap();
        let mut token_ids = Vec::new();
        gguf_llama_sentencepiece_encode_escaped(&vocab, "abc", &mut token_ids).unwrap();
        assert_eq!(token_ids, vec![262, 261]);
    }

    #[test]
    fn gguf_tokenizer_llama_internal_encode_parses_specials_and_dummy_prefix_context() {
        let spec = llama_encode_test_spec(&[("▁x", 0.5), ("▁y", 0.5)]);
        let token_ids = gguf_llama_sentencepiece_encode(&spec, "x<s>y", true, true).unwrap();
        assert_eq!(token_ids, vec![1, 259, 1, 260]);

        let literal_ids = gguf_llama_sentencepiece_encode(&spec, "x<s>y", false, false).unwrap();
        assert_eq!(literal_ids, vec![259, 63, 118, 65, 124]);
    }

    #[test]
    fn gguf_tokenizer_llama_internal_encode_fails_closed_for_malformed_specs() {
        let mut unsupported = llama_encode_test_spec(&[("▁x", 0.0)]);
        unsupported.family = GgufTokenizerSpecFamily::SyntheticGpt2ByteLevelBpe;
        assert!(matches!(
            gguf_llama_sentencepiece_encode(&unsupported, "x", false, false),
            Err(GgufLlamaTokenizerError::UnsupportedFamily { .. })
        ));

        let mut incomplete_byte_fallback = llama_encode_test_spec(&[("▁x", 0.0)]);
        incomplete_byte_fallback.tokens.remove(3);
        incomplete_byte_fallback.scores.remove(3);
        incomplete_byte_fallback.token_types.remove(3);
        assert_eq!(
            gguf_llama_sentencepiece_encode(&incomplete_byte_fallback, "x", false, false),
            Err(GgufLlamaTokenizerError::IncompleteByteFallback)
        );

        let mut duplicate = llama_encode_test_spec(&[("▁x", 0.0), ("▁x", 1.0)]);
        assert!(matches!(
            gguf_llama_sentencepiece_encode(&duplicate, "x", false, false),
            Err(GgufLlamaTokenizerError::DuplicateOrdinaryPiece { .. })
        ));

        duplicate.tokens[259].clear();
        assert!(matches!(
            gguf_llama_sentencepiece_encode(&duplicate, "x", false, false),
            Err(GgufLlamaTokenizerError::EmptyOrdinaryPiece { token_id: 259 })
        ));

        let mut bad_score = llama_encode_test_spec(&[("▁x", f32::NAN)]);
        assert!(matches!(
            gguf_llama_sentencepiece_encode(&bad_score, "x", false, false),
            Err(GgufLlamaTokenizerError::InvalidScore { token_id: 259 })
        ));

        bad_score.scores.pop();
        assert_eq!(
            gguf_llama_sentencepiece_encode(&bad_score, "x", false, false),
            Err(GgufLlamaTokenizerError::IncompleteTokenMetadata)
        );
    }

    #[test]
    fn gguf_tokenizer_llama_private_helpers_reject_unsupported_token_types() {
        for (token_type, label) in [
            (4, "user-defined partitioning"),
            (5, "unused token semantics"),
            (0, "undefined token semantics"),
            (99, "unknown future token type"),
        ] {
            let mut spec = llama_encode_test_spec(&[("▁x", 0.0)]);
            spec.token_types[259] = token_type;

            assert_eq!(
                gguf_llama_sentencepiece_encode(&spec, "x", false, false),
                Err(GgufLlamaTokenizerError::UnsupportedTokenType {
                    token_id: 259,
                    token_type,
                }),
                "private encode helper must not treat {label} as an ordinary SPM piece"
            );
            assert_eq!(
                gguf_llama_sentencepiece_build_vocab(&spec).unwrap_err(),
                GgufLlamaTokenizerError::UnsupportedTokenType {
                    token_id: 259,
                    token_type,
                },
                "private vocab construction must fail closed for {label}"
            );
            assert_eq!(
                gguf_llama_sentencepiece_token_to_piece_bytes(&spec, 259, true),
                Err(GgufLlamaTokenizerError::UnsupportedTokenType {
                    token_id: 259,
                    token_type,
                }),
                "private token-to-piece must not silently execute unsupported {label}"
            );
        }
    }

    #[test]
    fn gguf_tokenizer_llama_attribute_and_strip_semantics_remain_unsupported_without_goldens() {
        // Pinned llama.cpp source shows token_type=4 user-defined partitioning,
        // token_type=5 unused/0 undefined semantics, wider special ID classes,
        // LSTRIP/RSTRIP/SINGLE_WORD/NORMALIZED/strip behavior, add_eos,
        // remove_extra_whitespaces, precompiled charsmap, and tokenizer.ggml.pre
        // variations need separate oracle fixtures/goldens before execution.
        // This lane intentionally hardens the private helper to reject rather
        // than implement or guess those semantics.
        let unsupported_until_pinned = [
            "user-defined partitioning",
            "unused token_type",
            "undefined token_type",
            "LSTRIP",
            "RSTRIP",
            "SINGLE_WORD",
            "NORMALIZED",
            "wider special IDs",
            "add_eos",
            "remove_extra_whitespaces",
            "precompiled charsmap",
            "tokenizer.ggml.pre variations",
        ];
        assert!(unsupported_until_pinned.contains(&"user-defined partitioning"));
        assert!(unsupported_until_pinned.contains(&"LSTRIP"));
        assert!(unsupported_until_pinned.contains(&"add_eos"));
    }

    #[test]
    fn gguf_tokenizer_llama_internal_encode_then_detokenize_preserves_unambiguous_bytes() {
        let Some(fixture_path) = gguf_tokenizer_llama_fixture_for_internal_encode_tests() else {
            return;
        };
        gguf_tokenizer_assert_pinned_fixture_for_tests(&fixture_path);
        let metadata = read_gguf_metadata_summary(&fixture_path).unwrap();
        let spec = metadata
            .tokenizer_spec
            .as_ref()
            .expect("pinned fixture should retain private Llama tokenizer metadata");

        for golden in GGUF_TOKENIZER_LLAMA_TINY_ENCODE_GOLDENS {
            if matches!(
                golden.label,
                "literal <s>" | "literal </s>" | "literal <unk>"
            ) {
                continue;
            }
            let token_ids = gguf_llama_sentencepiece_encode(spec, golden.input, false, false)
                .unwrap_or_else(|error| panic!("encode failed for {}: {error:?}", golden.label));
            let bytes = gguf_llama_sentencepiece_detokenize_bytes(spec, &token_ids, false, false)
                .unwrap_or_else(|error| {
                    panic!("detokenize failed for {}: {error:?}", golden.label)
                });
            assert_eq!(
                bytes,
                golden.input.as_bytes(),
                "private encode→detokenize bytes changed for {}",
                golden.label
            );
        }
    }

    #[test]
    fn gguf_tokenizer_llama_internal_encode_detokenize_handles_mixed_unicode_byte_fallback() {
        let spec = llama_encode_test_spec(&[("▁", 0.1), ("▁h", 1.0), ("▁hi", 2.0), ("!", 0.5)]);
        let token_ids = gguf_llama_sentencepiece_encode(&spec, "hi 🦞!", false, true).unwrap();
        assert_eq!(token_ids, vec![261, 259, 243, 162, 169, 161, 262]);
        assert_eq!(
            gguf_llama_sentencepiece_detokenize_bytes(&spec, &token_ids, false, false).unwrap(),
            "hi 🦞!".as_bytes()
        );
    }

    #[test]
    fn gguf_tokenizer_llama_internal_special_ids_are_deterministic() {
        let mut only_bos = llama_encode_test_spec(&[("▁x", 0.0)]);
        only_bos.special_token_ids.remove("eos");
        only_bos.special_token_ids.remove("unk");
        assert_eq!(
            gguf_llama_sentencepiece_encode(&only_bos, "x</s><unk>", true, true).unwrap(),
            vec![1, 259, 63, 50, 118, 65, 63, 120, 113, 110, 65]
        );

        let mut missing_bos = llama_encode_test_spec(&[("▁x", 0.0)]);
        missing_bos.special_token_ids.remove("bos");
        assert_eq!(
            gguf_llama_sentencepiece_encode(&missing_bos, "x", false, false),
            Err(GgufLlamaTokenizerError::MissingSpecialToken { label: "bos" })
        );

        let mut out_of_range_bos = llama_encode_test_spec(&[("▁x", 0.0)]);
        out_of_range_bos
            .special_token_ids
            .insert("bos".to_string(), out_of_range_bos.tokens.len() as u64);
        assert_eq!(
            gguf_llama_sentencepiece_encode(&out_of_range_bos, "x", false, false),
            Err(GgufLlamaTokenizerError::TokenIdOutOfRange {
                token_id: out_of_range_bos.tokens.len() as i32,
                vocab_size: out_of_range_bos.tokens.len(),
            })
        );

        let mut malformed_bos = llama_encode_test_spec(&[("▁x", 0.0)]);
        malformed_bos
            .special_token_ids
            .insert("bos".to_string(), u64::MAX);
        assert_eq!(
            gguf_llama_sentencepiece_encode(&malformed_bos, "x", false, false),
            Err(GgufLlamaTokenizerError::TokenIdOutOfRange {
                token_id: i32::MAX,
                vocab_size: malformed_bos.tokens.len(),
            })
        );
    }

    #[test]
    fn gguf_tokenizer_llama_internal_duplicate_special_text_prefers_bos_fragment() {
        let mut spec = llama_encode_test_spec(&[("▁y", 0.0)]);
        spec.tokens[0] = "<s>".to_string();
        let token_ids = gguf_llama_sentencepiece_encode(&spec, "<s>y", true, true).unwrap();
        assert_eq!(token_ids, vec![1, 1, 259]);
    }

    #[test]
    fn gguf_tokenizer_llama_internal_byte_fallback_shapes_fail_deterministically() {
        let mut malformed_byte_token = llama_encode_test_spec(&[("▁x", 0.0)]);
        malformed_byte_token.tokens[3] = "<0xGG>".to_string();
        assert_eq!(
            gguf_llama_sentencepiece_encode(&malformed_byte_token, "x", false, false),
            Err(GgufLlamaTokenizerError::IncompleteByteFallback)
        );

        let mut duplicate_byte_token = llama_encode_test_spec(&[("▁x", 0.0)]);
        duplicate_byte_token.tokens[4] = "<0x00>".to_string();
        assert_eq!(
            gguf_llama_sentencepiece_encode(&duplicate_byte_token, "x", false, false),
            Err(GgufLlamaTokenizerError::IncompleteByteFallback)
        );
    }

    #[test]
    fn gguf_tokenizer_llama_private_detokenize_error_taxonomy_is_stable() {
        let mut unsupported = gguf_tokenizer_llama_tiny_spec_for_tests();
        unsupported.family = GgufTokenizerSpecFamily::SyntheticGpt2ByteLevelBpe;
        assert_eq!(
            gguf_llama_sentencepiece_detokenize_bytes(&unsupported, &[1], false, false),
            Err(GgufLlamaTokenizerError::UnsupportedFamily {
                family: GgufTokenizerSpecFamily::SyntheticGpt2ByteLevelBpe,
            })
        );

        let spec = gguf_tokenizer_llama_tiny_spec_for_tests();
        assert_eq!(
            gguf_llama_sentencepiece_token_to_piece_bytes(&spec, i32::MAX, false),
            Err(GgufLlamaTokenizerError::TokenIdOutOfRange {
                token_id: i32::MAX,
                vocab_size: LLAMA_TINY_GGUF_VOCAB_SIZE as usize,
            })
        );
        assert_eq!(
            gguf_llama_sentencepiece_detokenize_bytes(&spec, &[-1], false, false),
            Err(GgufLlamaTokenizerError::TokenIdOutOfRange {
                token_id: -1,
                vocab_size: LLAMA_TINY_GGUF_VOCAB_SIZE as usize,
            })
        );
    }

    #[test]
    #[ignore = "requires FATHOM_GGUF_LLAMA_TINY_FIXTURE plus LLAMA_TOKENIZE_BIN"]
    fn gguf_tokenizer_llama_tiny_encode_goldens_match_llama_cpp_when_configured() {
        let Ok(fixture_path) = std::env::var("FATHOM_GGUF_LLAMA_TINY_FIXTURE") else {
            return;
        };
        let Some(llama_tokenize_bin) = gguf_tokenizer_llama_tokenize_bin_for_tests() else {
            return;
        };
        let fixture_path = PathBuf::from(fixture_path);
        gguf_tokenizer_assert_pinned_fixture_for_tests(&fixture_path);

        let unique = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_nanos();
        let prompt_path = std::env::temp_dir().join(format!(
            "fathom-gguf-tokenizer-llama-tokenize-prompt-{unique}.txt"
        ));

        for golden in GGUF_TOKENIZER_LLAMA_TINY_ENCODE_GOLDENS {
            fs::write(&prompt_path, golden.input.as_bytes()).unwrap();
            let actual_no_bos_no_special = gguf_tokenizer_run_llama_tokenize_ids_for_tests(
                &llama_tokenize_bin,
                &fixture_path,
                &prompt_path,
                true,
            );
            assert_eq!(
                actual_no_bos_no_special,
                gguf_tokenizer_format_ids_for_tests(golden.no_bos_no_special_ids),
                "llama.cpp no-BOS/no-special encode IDs changed for {}",
                golden.label
            );

            let actual_default = gguf_tokenizer_run_llama_tokenize_ids_for_tests(
                &llama_tokenize_bin,
                &fixture_path,
                &prompt_path,
                false,
            );
            assert_eq!(
                actual_default,
                gguf_tokenizer_format_ids_for_tests(golden.default_ids),
                "llama.cpp default encode IDs changed for {}",
                golden.label
            );
        }

        fs::remove_file(prompt_path).unwrap();
    }

    #[test]
    #[ignore = "requires FATHOM_GGUF_LLAMA3_FIXTURE plus LLAMA_TOKENIZE_BIN"]
    fn gguf_tokenizer_llama3_encode_goldens_match_llama_cpp_when_configured() {
        let Some(fixture_path) = gguf_tokenizer_llama3_fixture_for_tests() else {
            return;
        };
        let Some(llama_tokenize_bin) = gguf_tokenizer_llama_tokenize_bin_for_tests() else {
            return;
        };
        gguf_tokenizer_assert_llama3_fixture_for_tests(&fixture_path);

        let unique = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_nanos();
        let prompt_path = std::env::temp_dir().join(format!(
            "fathom-gguf-tokenizer-llama3-tokenize-prompt-{unique}.txt"
        ));

        for golden in GGUF_TOKENIZER_LLAMA3_ENCODE_GOLDENS {
            fs::write(&prompt_path, golden.input.as_bytes()).unwrap();
            let actual_no_bos_no_special = gguf_tokenizer_run_llama_tokenize_ids_for_tests(
                &llama_tokenize_bin,
                &fixture_path,
                &prompt_path,
                true,
            );
            assert_eq!(
                actual_no_bos_no_special,
                gguf_tokenizer_format_ids_for_tests(golden.no_bos_no_special_ids),
                "llama.cpp Llama 3 no-BOS/no-special encode IDs changed for {}",
                golden.label
            );

            let actual_default = gguf_tokenizer_run_llama_tokenize_ids_for_tests(
                &llama_tokenize_bin,
                &fixture_path,
                &prompt_path,
                false,
            );
            assert_eq!(
                actual_default,
                gguf_tokenizer_format_ids_for_tests(golden.default_ids),
                "llama.cpp Llama 3 default encode IDs changed for {}",
                golden.label
            );
        }

        fs::remove_file(prompt_path).unwrap();
    }

    #[test]
    fn gguf_tokenizer_llama_tiny_decode_goldens_are_pinned_and_cover_reference_modes() {
        assert_eq!(LLAMA_CPP_DECODE_REFERENCE_REVISION, "15f786e65");
        assert_eq!(LLAMA_CPP_HOMEBREW_REFERENCE_VERSION, "8680");
        assert_eq!(LLAMA_TINY_GGUF_VOCAB_SIZE, 32_000);
        assert!(GGUF_TOKENIZER_LLAMA_TINY_PIECE_GOLDENS.len() >= 24);
        assert!(GGUF_TOKENIZER_LLAMA_TINY_DETOKENIZE_GOLDENS.len() >= 16);

        for golden in GGUF_TOKENIZER_LLAMA_TINY_PIECE_GOLDENS {
            assert!(
                (0..LLAMA_TINY_GGUF_VOCAB_SIZE).contains(&golden.token_id),
                "piece golden token id must be in bounds: {}",
                golden.label
            );
            assert_eq!(
                golden.special_false_hex.len() % 2,
                0,
                "piece hex must be byte-aligned: {}",
                golden.label
            );
            assert_eq!(
                golden.special_true_hex.len() % 2,
                0,
                "piece hex must be byte-aligned: {}",
                golden.label
            );
        }

        for golden in GGUF_TOKENIZER_LLAMA_TINY_DETOKENIZE_GOLDENS {
            assert!(
                golden
                    .token_ids
                    .iter()
                    .all(|token_id| (0..LLAMA_TINY_GGUF_VOCAB_SIZE).contains(token_id)),
                "detokenize golden token ids must be in bounds: {}",
                golden.label
            );
            assert_eq!(
                golden.expected_hex.len() % 2,
                0,
                "detokenize hex must be byte-aligned: {}",
                golden.label
            );
        }
    }

    #[test]
    fn gguf_tokenizer_llama_tiny_decode_out_of_range_ids_are_fathom_bounds_expectations() {
        for token_id in [
            -1,
            LLAMA_TINY_GGUF_VOCAB_SIZE,
            LLAMA_TINY_GGUF_VOCAB_SIZE + 1,
        ] {
            assert!(
                !(0..LLAMA_TINY_GGUF_VOCAB_SIZE).contains(&token_id),
                "invalid token id {token_id} should be rejected before any llama.cpp reference call"
            );
        }
    }

    fn gguf_tokenizer_llama_decode_probe_for_tests() -> Option<PathBuf> {
        if let Ok(path) = std::env::var("LLAMA_DECODE_PROBE_BIN") {
            return Some(PathBuf::from(path));
        }
        if std::env::var_os("FATHOM_GGUF_BUILD_LLAMA_DECODE_PROBE").is_none() {
            return None;
        }
        gguf_tokenizer_build_llama_decode_probe_for_tests()
    }

    fn gguf_tokenizer_build_llama_decode_probe_for_tests() -> Option<PathBuf> {
        let unique = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_nanos();
        let source_path =
            std::env::temp_dir().join(format!("fathom-llama-decode-probe-{unique}.cpp"));
        let binary_path = std::env::temp_dir().join(format!("fathom-llama-decode-probe-{unique}"));
        fs::write(&source_path, GGUF_TOKENIZER_LLAMA_DECODE_PROBE_CPP).unwrap();

        let Ok(llama_cpp_lib_dir) = std::env::var("LLAMA_CPP_LIB_DIR") else {
            eprintln!("skipping GGUF llama decode probe build; LLAMA_CPP_LIB_DIR is not set");
            let _ = fs::remove_file(source_path);
            return None;
        };
        let Ok(ggml_lib_dir) = std::env::var("GGML_LIB_DIR") else {
            eprintln!("skipping GGUF llama decode probe build; GGML_LIB_DIR is not set");
            let _ = fs::remove_file(source_path);
            return None;
        };
        let pkg_config = Command::new("pkg-config")
            .args(["--cflags", "llama", "ggml"])
            .output()
            .ok();
        let mut compile_command = Command::new("c++");
        compile_command
            .arg("-std=c++17")
            .arg(&source_path)
            .arg("-o")
            .arg(&binary_path);
        if let Some(pkg_config) = pkg_config.as_ref().filter(|output| output.status.success()) {
            for flag in String::from_utf8_lossy(&pkg_config.stdout).split_whitespace() {
                compile_command.arg(flag);
            }
        }
        let compile = compile_command
            .arg(format!("-L{llama_cpp_lib_dir}"))
            .arg(format!("-L{ggml_lib_dir}"))
            .arg("-lllama")
            .arg("-lggml")
            .arg("-lggml-base")
            .arg(format!("-Wl,-rpath,{llama_cpp_lib_dir}"))
            .arg(format!("-Wl,-rpath,{ggml_lib_dir}"))
            .output()
            .unwrap_or_else(|error| panic!("failed to invoke C++ decode probe build: {error}"));

        if !compile.status.success() {
            eprintln!(
                "skipping GGUF llama decode probe build; status {:?}\nstdout:\n{}\nstderr:\n{}",
                compile.status.code(),
                String::from_utf8_lossy(&compile.stdout),
                String::from_utf8_lossy(&compile.stderr)
            );
            let _ = fs::remove_file(source_path);
            return None;
        }

        let _ = fs::remove_file(source_path);
        Some(binary_path)
    }

    fn gguf_tokenizer_run_decode_probe_for_tests(
        probe_bin: &Path,
        model_path: &Path,
        args: &[String],
    ) -> String {
        let output = Command::new(probe_bin)
            .arg(model_path)
            .args(args)
            .output()
            .unwrap_or_else(|error| panic!("failed to execute {}: {error}", probe_bin.display()));
        assert!(
            output.status.success(),
            "decode probe failed with status {:?}\nstdout:\n{}\nstderr:\n{}",
            output.status.code(),
            String::from_utf8_lossy(&output.stdout),
            String::from_utf8_lossy(&output.stderr)
        );
        String::from_utf8(output.stdout)
            .expect("decode probe should emit UTF-8 metadata around hex bytes")
            .trim()
            .to_string()
    }

    fn gguf_tokenizer_hex_from_probe_output(output: &str) -> &str {
        output
            .split_whitespace()
            .find_map(|field| field.strip_prefix("hex="))
            .expect("decode probe output should include hex=...")
    }

    fn gguf_tokenizer_hex_to_bytes_for_tests(hex: &str) -> Vec<u8> {
        assert_eq!(hex.len() % 2, 0);
        (0..hex.len())
            .step_by(2)
            .map(|index| u8::from_str_radix(&hex[index..index + 2], 16).unwrap())
            .collect()
    }

    fn gguf_tokenizer_llama_tiny_spec_for_tests() -> GgufTokenizerSpec {
        let mut tokens = (0..LLAMA_TINY_GGUF_VOCAB_SIZE)
            .map(|index| format!("<unused-{index}>").to_string())
            .collect::<Vec<_>>();
        let mut token_types = vec![1; LLAMA_TINY_GGUF_VOCAB_SIZE as usize];
        tokens[0] = "<unk>".to_string();
        tokens[1] = "<s>".to_string();
        tokens[2] = "</s>".to_string();
        token_types[0] = 2;
        token_types[1] = 3;
        token_types[2] = 3;
        for byte in 0u16..=255 {
            let token_id = 3 + byte as usize;
            tokens[token_id] = format!("<0x{byte:02X}>");
            token_types[token_id] = 6;
        }
        for (token_id, piece) in [
            (12usize, "\t"),
            (13, "\n"),
            (259, "▁▁"),
            (263, "▁a"),
            (274, "▁c"),
            (321, "▁e"),
            (11526, "world"),
            (3186, "▁world"),
            (15043, "▁Hello"),
            (22172, "▁hello"),
            (28059, "afé"),
            (29871, "▁"),
            (29889, "."),
            (29890, "b"),
            (29892, ","),
            (29991, "!"),
            (30103, "\u{0301}"),
        ] {
            tokens[token_id] = piece.to_string();
            token_types[token_id] = 1;
        }
        let mut special_token_ids = BTreeMap::new();
        special_token_ids.insert("unk".to_string(), 0);
        special_token_ids.insert("bos".to_string(), 1);
        special_token_ids.insert("eos".to_string(), 2);
        GgufTokenizerSpec {
            family: GgufTokenizerSpecFamily::LlamaSentencePiece,
            model: "llama".to_string(),
            tokens,
            merges: Vec::new(),
            scores: vec![0.0; LLAMA_TINY_GGUF_VOCAB_SIZE as usize],
            token_types,
            special_token_ids,
            has_byte_fallback: true,
        }
    }

    #[test]
    fn gguf_tokenizer_llama_tiny_private_token_to_piece_matches_pinned_goldens() {
        let spec = gguf_tokenizer_llama_tiny_spec_for_tests();
        for golden in GGUF_TOKENIZER_LLAMA_TINY_PIECE_GOLDENS {
            assert_eq!(
                gguf_llama_sentencepiece_token_to_piece_bytes(&spec, golden.token_id, false)
                    .unwrap(),
                gguf_tokenizer_hex_to_bytes_for_tests(golden.special_false_hex),
                "special=false token-to-piece changed for {}",
                golden.label
            );
            assert_eq!(
                gguf_llama_sentencepiece_token_to_piece_bytes(&spec, golden.token_id, true)
                    .unwrap(),
                gguf_tokenizer_hex_to_bytes_for_tests(golden.special_true_hex),
                "special=true token-to-piece changed for {}",
                golden.label
            );
        }
    }

    #[test]
    fn gguf_tokenizer_llama_tiny_private_detokenize_matches_pinned_byte_goldens() {
        let spec = gguf_tokenizer_llama_tiny_spec_for_tests();
        for golden in GGUF_TOKENIZER_LLAMA_TINY_DETOKENIZE_GOLDENS {
            assert_eq!(
                gguf_llama_sentencepiece_detokenize_bytes(
                    &spec,
                    golden.token_ids,
                    golden.remove_special,
                    golden.unparse_special,
                )
                .unwrap(),
                gguf_tokenizer_hex_to_bytes_for_tests(golden.expected_hex),
                "private detokenize bytes changed for {}",
                golden.label
            );
        }
    }

    #[test]
    fn gguf_tokenizer_llama_private_helpers_fail_deterministically_for_out_of_range_ids() {
        let spec = gguf_tokenizer_llama_tiny_spec_for_tests();
        assert_eq!(
            gguf_llama_sentencepiece_token_to_piece_bytes(&spec, -1, false).unwrap_err(),
            GgufLlamaTokenizerError::TokenIdOutOfRange {
                token_id: -1,
                vocab_size: LLAMA_TINY_GGUF_VOCAB_SIZE as usize,
            }
        );
        assert_eq!(
            gguf_llama_sentencepiece_detokenize_bytes(
                &spec,
                &[LLAMA_TINY_GGUF_VOCAB_SIZE],
                false,
                false,
            )
            .unwrap_err(),
            GgufLlamaTokenizerError::TokenIdOutOfRange {
                token_id: LLAMA_TINY_GGUF_VOCAB_SIZE,
                vocab_size: LLAMA_TINY_GGUF_VOCAB_SIZE as usize,
            }
        );
    }

    #[test]
    #[ignore = "requires FATHOM_GGUF_LLAMA_TINY_FIXTURE plus LLAMA_DECODE_PROBE_BIN or FATHOM_GGUF_BUILD_LLAMA_DECODE_PROBE=1"]
    fn gguf_tokenizer_llama_tiny_decode_goldens_match_llama_cpp_when_configured() {
        let Ok(fixture_path) = std::env::var("FATHOM_GGUF_LLAMA_TINY_FIXTURE") else {
            return;
        };
        let Some(probe_bin) = gguf_tokenizer_llama_decode_probe_for_tests() else {
            return;
        };
        let fixture_path = PathBuf::from(fixture_path);
        gguf_tokenizer_assert_pinned_fixture_for_tests(&fixture_path);

        for golden in GGUF_TOKENIZER_LLAMA_TINY_PIECE_GOLDENS {
            for (special, expected_hex) in [
                (false, golden.special_false_hex),
                (true, golden.special_true_hex),
            ] {
                let output = gguf_tokenizer_run_decode_probe_for_tests(
                    &probe_bin,
                    &fixture_path,
                    &[
                        "piece".to_string(),
                        special.to_string(),
                        golden.token_id.to_string(),
                    ],
                );
                assert_eq!(
                    gguf_tokenizer_hex_from_probe_output(&output),
                    expected_hex,
                    "llama.cpp token-to-piece bytes changed for {} special={special}",
                    golden.label
                );
            }
        }

        for golden in GGUF_TOKENIZER_LLAMA_TINY_DETOKENIZE_GOLDENS {
            let mut args = vec![
                "detok".to_string(),
                golden.remove_special.to_string(),
                golden.unparse_special.to_string(),
            ];
            args.extend(golden.token_ids.iter().map(i32::to_string));
            let output =
                gguf_tokenizer_run_decode_probe_for_tests(&probe_bin, &fixture_path, &args);
            assert_eq!(
                gguf_tokenizer_hex_from_probe_output(&output),
                golden.expected_hex,
                "llama.cpp detokenize bytes changed for {} remove_special={} unparse_special={}",
                golden.label,
                golden.remove_special,
                golden.unparse_special
            );
        }
    }

    const GGUF_TOKENIZER_LLAMA_DECODE_PROBE_CPP: &str = r#"
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>
#include <llama.h>

static bool parse_bool(const char *value) {
    return std::strcmp(value, "true") == 0 || std::strcmp(value, "1") == 0;
}

static void print_hex(const char *bytes, int32_t len) {
    std::printf("hex=");
    for (int32_t i = 0; i < len; ++i) {
        std::printf("%02x", static_cast<unsigned char>(bytes[i]));
    }
}

int main(int argc, char **argv) {
    if (argc < 3) {
        std::fprintf(stderr, "usage: %s MODEL piece SPECIAL TOKEN | detok REMOVE_SPECIAL UNPARSE_SPECIAL TOKENS...\n", argv[0]);
        return 2;
    }

    llama_backend_init();
    llama_model_params params = llama_model_default_params();
    params.vocab_only = true;
    llama_model *model = llama_model_load_from_file(argv[1], params);
    if (model == nullptr) {
        std::fprintf(stderr, "failed to load model\n");
        llama_backend_free();
        return 3;
    }
    const llama_vocab *vocab = llama_model_get_vocab(model);
    const std::string mode(argv[2]);
    char buffer[4096];

    if (mode == "piece") {
        if (argc != 5) {
            std::fprintf(stderr, "piece requires SPECIAL TOKEN\n");
            llama_model_free(model);
            llama_backend_free();
            return 2;
        }
        bool special = parse_bool(argv[3]);
        llama_token token = static_cast<llama_token>(std::strtol(argv[4], nullptr, 10));
        int32_t rc = llama_token_to_piece(vocab, token, buffer, sizeof(buffer), 0, special);
        if (rc < 0) {
            std::fprintf(stderr, "token_to_piece failed rc=%d\n", rc);
            llama_model_free(model);
            llama_backend_free();
            return 4;
        }
        std::printf("rc=%d len=%d ", rc, rc);
        print_hex(buffer, rc);
        std::printf("\n");
    } else if (mode == "detok") {
        if (argc < 6) {
            std::fprintf(stderr, "detok requires REMOVE_SPECIAL UNPARSE_SPECIAL TOKENS...\n");
            llama_model_free(model);
            llama_backend_free();
            return 2;
        }
        bool remove_special = parse_bool(argv[3]);
        bool unparse_special = parse_bool(argv[4]);
        std::vector<llama_token> tokens;
        for (int i = 5; i < argc; ++i) {
            tokens.push_back(static_cast<llama_token>(std::strtol(argv[i], nullptr, 10)));
        }
        int32_t rc = llama_detokenize(vocab, tokens.data(), tokens.size(), buffer, sizeof(buffer), remove_special, unparse_special);
        if (rc < 0) {
            std::fprintf(stderr, "detokenize failed rc=%d\n", rc);
            llama_model_free(model);
            llama_backend_free();
            return 4;
        }
        std::printf("rc=%d len=%d ", rc, rc);
        print_hex(buffer, rc);
        std::printf("\n");
    } else {
        std::fprintf(stderr, "unknown mode %s\n", mode.c_str());
        llama_model_free(model);
        llama_backend_free();
        return 2;
    }

    llama_model_free(model);
    llama_backend_free();
    return 0;
}
"#;

    #[test]
    fn hf_safetensors_runtime_package_key_is_stable_and_invalidates() {
        let unique = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_nanos();
        let dir = std::env::temp_dir().join(format!("fathom-runtime-key-{unique}"));
        write_runtime_key_package(&dir);

        let first = RuntimePackageKey::for_hf_safetensors(&dir, "gpt2", "f32", "cpu").unwrap();
        let same = RuntimePackageKey::for_hf_safetensors(&dir, "gpt2", "f32", "cpu").unwrap();
        assert_eq!(first, same);

        let llama = RuntimePackageKey::for_hf_safetensors(&dir, "llama", "f32", "cpu").unwrap();
        let qwen2 = RuntimePackageKey::for_hf_safetensors(&dir, "qwen2", "f32", "cpu").unwrap();
        let phi = RuntimePackageKey::for_hf_safetensors(&dir, "phi", "f32", "cpu").unwrap();
        let mistral = RuntimePackageKey::for_hf_safetensors(&dir, "mistral", "f32", "cpu").unwrap();
        let gemma = RuntimePackageKey::for_hf_safetensors(&dir, "gemma", "f32", "cpu").unwrap();
        let f16 = RuntimePackageKey::for_hf_safetensors(&dir, "gpt2", "f16", "cpu").unwrap();
        let gpu = RuntimePackageKey::for_hf_safetensors(&dir, "gpt2", "f32", "metal").unwrap();
        assert_ne!(first, llama);
        assert_ne!(first, qwen2);
        assert_ne!(first, phi);
        assert_ne!(first, mistral);
        assert_ne!(first, gemma);
        assert_ne!(llama, qwen2);
        assert_ne!(llama, phi);
        assert_ne!(llama, mistral);
        assert_ne!(llama, gemma);
        assert_ne!(qwen2, phi);
        assert_ne!(qwen2, mistral);
        assert_ne!(qwen2, gemma);
        assert_ne!(phi, mistral);
        assert_ne!(phi, gemma);
        assert_ne!(mistral, gemma);
        assert_ne!(first, f16);
        assert_ne!(first, gpu);

        fs::write(
            dir.join("config.json"),
            br#"{"model_type":"gpt2","n_layer":1}"#,
        )
        .unwrap();
        let config_changed =
            RuntimePackageKey::for_hf_safetensors(&dir, "gpt2", "f32", "cpu").unwrap();
        assert_ne!(first, config_changed);

        fs::write(dir.join("tokenizer.json"), b"{\"changed\":true}").unwrap();
        let tokenizer_changed =
            RuntimePackageKey::for_hf_safetensors(&dir, "gpt2", "f32", "cpu").unwrap();
        assert_ne!(config_changed, tokenizer_changed);

        fs::write(dir.join("model.safetensors"), b"changed weights").unwrap();
        let weights_changed =
            RuntimePackageKey::for_hf_safetensors(&dir, "gpt2", "f32", "cpu").unwrap();
        assert_ne!(tokenizer_changed, weights_changed);

        fs::remove_dir_all(dir).unwrap();
    }

    #[test]
    fn unsupported_gpt2_package_does_not_populate_chat_runtime_cache() {
        clear_candle_chat_runtime_cache_for_tests().unwrap();
        let unique = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_nanos();
        let dir = std::env::temp_dir().join(format!("fathom-runtime-cache-reject-{unique}"));
        write_runtime_key_package(&dir);

        assert!(load_candle_gpt2_runtime_cached(&dir).is_err());
        assert_eq!(candle_chat_runtime_cache_len().unwrap(), 0);

        fs::remove_dir_all(dir).unwrap();
    }

    #[test]
    fn unsupported_llama_package_does_not_populate_chat_runtime_cache() {
        clear_candle_chat_runtime_cache_for_tests().unwrap();
        let unique = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_nanos();
        let dir = std::env::temp_dir().join(format!("fathom-llama-runtime-cache-reject-{unique}"));
        write_runtime_key_package(&dir);

        assert!(load_candle_llama_runtime_cached(&dir).is_err());
        assert_eq!(candle_chat_runtime_cache_len().unwrap(), 0);

        fs::remove_dir_all(dir).unwrap();
    }

    #[test]
    fn unsupported_qwen2_package_does_not_populate_chat_runtime_cache() {
        clear_candle_chat_runtime_cache_for_tests().unwrap();
        let unique = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_nanos();
        let dir = std::env::temp_dir().join(format!("fathom-qwen2-runtime-cache-reject-{unique}"));
        write_runtime_key_package(&dir);

        assert!(load_candle_qwen2_runtime_cached(&dir).is_err());
        assert_eq!(candle_chat_runtime_cache_len().unwrap(), 0);

        fs::remove_dir_all(dir).unwrap();
    }

    #[test]
    fn unsupported_phi_package_does_not_populate_chat_runtime_cache() {
        clear_candle_chat_runtime_cache_for_tests().unwrap();
        let unique = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_nanos();
        let dir = std::env::temp_dir().join(format!("fathom-phi-runtime-cache-reject-{unique}"));
        write_runtime_key_package(&dir);
        fs::write(
            dir.join("config.json"),
            r#"{"model_type":"phi","architectures":["PhiForCausalLM"],"vocab_size":3,"hidden_size":4,"intermediate_size":8,"num_hidden_layers":1,"num_attention_heads":1,"num_key_value_heads":1,"max_position_embeddings":8,"layer_norm_eps":1e-5,"tie_word_embeddings":false,"rope_theta":10000.0,"partial_rotary_factor":0.5,"qk_layernorm":false,"hidden_act":"gelu","eos_token_id":2}"#,
        )
        .unwrap();

        assert!(load_candle_phi_runtime_cached(&dir).is_err());
        assert_eq!(candle_chat_runtime_cache_len().unwrap(), 0);

        fs::remove_dir_all(dir).unwrap();
    }

    #[test]
    fn unsupported_mistral_package_does_not_populate_chat_runtime_cache() {
        clear_candle_chat_runtime_cache_for_tests().unwrap();
        let unique = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_nanos();
        let dir =
            std::env::temp_dir().join(format!("fathom-mistral-runtime-cache-reject-{unique}"));
        write_runtime_key_package(&dir);

        assert!(load_candle_mistral_runtime_cached(&dir).is_err());
        assert_eq!(candle_chat_runtime_cache_len().unwrap(), 0);

        fs::remove_dir_all(dir).unwrap();
    }

    #[test]
    fn unsupported_gemma_package_does_not_populate_chat_runtime_cache() {
        clear_candle_chat_runtime_cache_for_tests().unwrap();
        let unique = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_nanos();
        let dir = std::env::temp_dir().join(format!("fathom-gemma-runtime-cache-reject-{unique}"));
        write_runtime_key_package(&dir);

        assert!(load_candle_gemma_runtime_cached(&dir).is_err());
        assert_eq!(candle_chat_runtime_cache_len().unwrap(), 0);

        fs::remove_dir_all(dir).unwrap();
    }

    #[test]
    fn qwen2_runtime_cache_live_fixture_reports_cold_then_warm_when_configured() {
        let Ok(path) = std::env::var("FATHOM_QWEN2_FIXTURE") else {
            return;
        };
        clear_candle_chat_runtime_cache_for_tests().unwrap();
        let options = GenerationOptions {
            temperature: 0.0,
            top_k: None,
            top_p: None,
        };
        let cold = generate_with_candle_qwen2_options(&path, "Hello", 1, options).unwrap();
        let warm = generate_with_candle_qwen2_options(&path, "Goodbye", 1, options).unwrap();
        clear_candle_chat_runtime_cache_for_tests().unwrap();
        let fresh = generate_with_candle_qwen2_options(&path, "Goodbye", 1, options).unwrap();

        assert!(!cold.metrics.runtime_cache_hit);
        assert_eq!(
            cold.metrics.runtime_residency.as_deref(),
            Some("cold_loaded")
        );
        assert_eq!(cold.metrics.runtime_family.as_deref(), Some("qwen2"));
        assert!(warm.metrics.runtime_cache_hit);
        assert_eq!(
            warm.metrics.runtime_residency.as_deref(),
            Some("warm_reused")
        );
        assert_eq!(warm.metrics.runtime_family.as_deref(), Some("qwen2"));
        assert!(!fresh.metrics.runtime_cache_hit);
        assert_eq!(warm.text, fresh.text);
    }

    #[test]
    fn phi_runtime_cache_live_fixture_reports_cold_then_warm_and_resets_when_configured() {
        let Ok(path) = std::env::var("FATHOM_PHI_FIXTURE") else {
            return;
        };
        clear_candle_chat_runtime_cache_for_tests().unwrap();
        let options = GenerationOptions {
            temperature: 0.0,
            top_k: None,
            top_p: None,
        };
        let cold = generate_with_candle_phi_options(&path, "Hello", 1, options).unwrap();
        let warm = generate_with_candle_phi_options(&path, "Goodbye", 1, options).unwrap();

        let config = read_candle_phi_config(&Path::new(&path).join("config.json")).unwrap();
        let tokenizer = load_tokenizer_cached(&Path::new(&path).join("tokenizer.json")).unwrap();
        let mut too_long = String::from("Hello");
        while tokenizer
            .encode(too_long.as_str(), true)
            .unwrap()
            .get_ids()
            .len()
            < config.max_position_embeddings
        {
            too_long.push_str(" Hello");
            if too_long.len() > 100_000 {
                break;
            }
        }
        assert!(generate_with_candle_phi_options(&path, &too_long, 1, options).is_err());
        let after_error = generate_with_candle_phi_options(&path, "Goodbye", 1, options).unwrap();
        clear_candle_chat_runtime_cache_for_tests().unwrap();
        let fresh = generate_with_candle_phi_options(&path, "Goodbye", 1, options).unwrap();

        assert!(!cold.metrics.runtime_cache_hit);
        assert_eq!(
            cold.metrics.runtime_residency.as_deref(),
            Some("cold_loaded")
        );
        assert_eq!(cold.metrics.runtime_family.as_deref(), Some("phi"));
        assert!(warm.metrics.runtime_cache_hit);
        assert_eq!(
            warm.metrics.runtime_residency.as_deref(),
            Some("warm_reused")
        );
        assert_eq!(warm.metrics.runtime_family.as_deref(), Some("phi"));
        assert!(after_error.metrics.runtime_cache_hit);
        assert!(!fresh.metrics.runtime_cache_hit);
        assert_eq!(warm.text, fresh.text);
        assert_eq!(after_error.text, fresh.text);
    }

    #[test]
    fn mistral_runtime_cache_live_fixture_reports_cold_then_warm_when_configured() {
        let Ok(path) = std::env::var("FATHOM_MISTRAL_FIXTURE") else {
            return;
        };
        clear_candle_chat_runtime_cache_for_tests().unwrap();
        let options = GenerationOptions {
            temperature: 0.0,
            top_k: None,
            top_p: None,
        };
        let cold = generate_with_candle_mistral_options(&path, "Hello", 1, options).unwrap();
        let warm = generate_with_candle_mistral_options(&path, "Goodbye", 1, options).unwrap();
        clear_candle_chat_runtime_cache_for_tests().unwrap();
        let fresh = generate_with_candle_mistral_options(&path, "Goodbye", 1, options).unwrap();

        assert!(!cold.metrics.runtime_cache_hit);
        assert_eq!(
            cold.metrics.runtime_residency.as_deref(),
            Some("cold_loaded")
        );
        assert_eq!(cold.metrics.runtime_family.as_deref(), Some("mistral"));
        assert!(warm.metrics.runtime_cache_hit);
        assert_eq!(
            warm.metrics.runtime_residency.as_deref(),
            Some("warm_reused")
        );
        assert_eq!(warm.metrics.runtime_family.as_deref(), Some("mistral"));
        assert!(!fresh.metrics.runtime_cache_hit);
        assert_eq!(warm.text, fresh.text);
    }

    #[test]
    fn gemma_runtime_cache_live_fixture_reports_cold_then_warm_when_configured() {
        let Ok(path) = std::env::var("FATHOM_GEMMA_FIXTURE") else {
            return;
        };
        clear_candle_chat_runtime_cache_for_tests().unwrap();
        let options = GenerationOptions {
            temperature: 0.0,
            top_k: None,
            top_p: None,
        };
        let cold = generate_with_candle_gemma_options(&path, "Hello", 1, options).unwrap();
        let warm = generate_with_candle_gemma_options(&path, "Goodbye", 1, options).unwrap();

        let config = read_candle_gemma_config(&Path::new(&path).join("config.json")).unwrap();
        let tokenizer = load_tokenizer_cached(&Path::new(&path).join("tokenizer.json")).unwrap();
        let mut too_long = String::from("Hello");
        while tokenizer
            .encode(too_long.as_str(), true)
            .unwrap()
            .get_ids()
            .len()
            < config.max_position_embeddings
        {
            too_long.push_str(" Hello");
            if too_long.len() > 100_000 {
                break;
            }
        }
        assert!(generate_with_candle_gemma_options(&path, &too_long, 1, options).is_err());
        let after_error = generate_with_candle_gemma_options(&path, "Goodbye", 1, options).unwrap();
        clear_candle_chat_runtime_cache_for_tests().unwrap();
        let fresh = generate_with_candle_gemma_options(&path, "Goodbye", 1, options).unwrap();

        assert!(!cold.metrics.runtime_cache_hit);
        assert_eq!(
            cold.metrics.runtime_residency.as_deref(),
            Some("cold_loaded")
        );
        assert_eq!(cold.metrics.runtime_family.as_deref(), Some("gemma"));
        assert!(warm.metrics.runtime_cache_hit);
        assert_eq!(
            warm.metrics.runtime_residency.as_deref(),
            Some("warm_reused")
        );
        assert_eq!(warm.metrics.runtime_family.as_deref(), Some("gemma"));
        assert!(after_error.metrics.runtime_cache_hit);
        assert!(!fresh.metrics.runtime_cache_hit);
        assert_eq!(warm.text, fresh.text);
        assert_eq!(after_error.text, fresh.text);
    }

    #[test]
    fn candle_runtime_metrics_report_cold_and_warm_status() {
        let cold = CandleRuntimeCacheStatus::ColdLoaded;
        let warm = CandleRuntimeCacheStatus::WarmReused;

        assert!(!cold.cache_hit());
        assert_eq!(cold.residency(), "cold_loaded");
        assert!(warm.cache_hit());
        assert_eq!(warm.residency(), "warm_reused");
    }

    #[test]
    fn detects_common_model_formats() {
        let cases = [
            ("llama.gguf", ModelFormat::Gguf),
            ("model.safetensors", ModelFormat::SafeTensors),
            (
                "model.safetensors.index.json",
                ModelFormat::SafeTensorsIndex,
            ),
            ("pytorch_model.bin", ModelFormat::PyTorchBin),
            ("model.onnx", ModelFormat::Onnx),
            ("model.mlpackage", ModelFormat::CoreMl),
            ("model.engine", ModelFormat::TensorRtPlan),
            ("tokenizer.json", ModelFormat::TokenizerJson),
            ("tokenizer_config.json", ModelFormat::TokenizerConfigJson),
            ("chat_template.jinja", ModelFormat::ChatTemplate),
            ("spiece.model", ModelFormat::SentencePiece),
            ("config.json", ModelFormat::ConfigJson),
        ];
        for (path, expected) in cases {
            assert_eq!(detect_model_artifact(path).format, expected);
        }
    }

    #[test]
    fn reads_gguf_header_metadata_and_tensor_infos_without_claiming_runnable_support() {
        let unique = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_nanos();
        let path = std::env::temp_dir().join(format!("fathom-gguf-metadata-{unique}.gguf"));
        let mut bytes = Vec::new();
        bytes.extend_from_slice(b"GGUF");
        bytes.extend_from_slice(&3u32.to_le_bytes());
        bytes.extend_from_slice(&2u64.to_le_bytes()); // tensor_count
        bytes.extend_from_slice(&4u64.to_le_bytes()); // metadata_kv_count
        write_gguf_string(&mut bytes, "general.architecture");
        bytes.extend_from_slice(&8u32.to_le_bytes()); // string
        write_gguf_string(&mut bytes, "llama");
        write_gguf_string(&mut bytes, "llama.context_length");
        bytes.extend_from_slice(&4u32.to_le_bytes()); // u32
        bytes.extend_from_slice(&4096u32.to_le_bytes());
        write_gguf_string(&mut bytes, "general.alignment");
        bytes.extend_from_slice(&4u32.to_le_bytes()); // u32
        bytes.extend_from_slice(&32u32.to_le_bytes());
        write_gguf_string(&mut bytes, "tokenizer.ggml.tokens");
        bytes.extend_from_slice(&9u32.to_le_bytes()); // array
        bytes.extend_from_slice(&8u32.to_le_bytes()); // string array
        bytes.extend_from_slice(&10u64.to_le_bytes());
        for token in ["<s>", "</s>", "a", "b", "c", "d", "e", "f", "g", "h"] {
            write_gguf_string(&mut bytes, token);
        }
        write_gguf_tensor_info(&mut bytes, "blk.0.weight", &[2, 3], 0, 0);
        write_gguf_tensor_info(&mut bytes, "blk.0.qweight", &[32], 2, 32);
        while bytes.len() % 32 != 0 {
            bytes.push(0);
        }
        bytes.extend(vec![0u8; 24]); // F32 [2, 3]
        bytes.extend(vec![0u8; 8]); // aligned gap before offset 32
        bytes.extend(vec![0u8; 18]); // Q4_0 [32]
        fs::write(&path, bytes).unwrap();

        let artifact = detect_model_artifact(&path);
        assert_eq!(artifact.format, ModelFormat::Gguf);
        assert_eq!(artifact.support, SupportLevel::MetadataReadable);
        assert!(!artifact.runnable_today);
        let metadata = artifact.gguf_metadata.as_ref().unwrap();
        assert_eq!(metadata.version, 3);
        assert_eq!(metadata.tensor_count, 2);
        assert_eq!(metadata.metadata_kv_count, 4);
        assert_eq!(metadata.metadata[0].key, "general.architecture");
        assert_eq!(metadata.hints.architecture.as_deref(), Some("llama"));
        assert_eq!(metadata.hints.context_length, Some(4096));
        assert_eq!(metadata.hints.tokenizer_token_count, Some(10));
        match &metadata.metadata[3].value {
            GgufMetadataValueSummary::Array { len, preview, .. } => {
                assert_eq!(*len, 10);
                assert_eq!(preview.len(), 8);
            }
            other => panic!("expected GGUF token array, got {other:?}"),
        }
        assert_eq!(
            metadata.metadata[0].value,
            GgufMetadataValueSummary::String {
                value: "llama".into()
            }
        );
        assert_eq!(metadata.tensors.len(), 2);
        assert_eq!(metadata.tensors[0].name, "blk.0.weight");
        assert_eq!(metadata.tensors[0].shape, vec![2, 3]);
        assert_eq!(metadata.tensors[0].ggml_type_name, "F32");
        assert_eq!(metadata.tensors[0].element_count, Some(6));
        assert_eq!(metadata.tensors[0].estimated_bytes, Some(24));
        assert_eq!(metadata.tensors[1].name, "blk.0.qweight");
        assert_eq!(metadata.tensors[1].ggml_type_name, "Q4_0");
        assert_eq!(metadata.tensors[1].estimated_bytes, Some(18));
        let tensor_summary = metadata.tensor_summary.as_ref().unwrap();
        assert_eq!(tensor_summary.type_counts.get("F32"), Some(&1));
        assert_eq!(tensor_summary.type_counts.get("Q4_0"), Some(&1));
        assert_eq!(tensor_summary.total_estimated_tensor_bytes, Some(42));
        assert!(tensor_summary.unknown_type_tags.is_empty());
        assert_eq!(tensor_summary.largest_tensors[0].name, "blk.0.weight");
        assert!(artifact.notes[0].contains("metadata-only and not runnable"));

        let package = inspect_model_package(&path).unwrap();
        let report = capability_report_for_package(package, &current_machine_profile());
        assert_eq!(report.best_status, CapabilityStatus::MetadataOnly);
        assert!(!report.runnable);
        assert!(report.summary.contains("tensor descriptors"));
        assert!(report
            .summary
            .contains("GGUF remains metadata-only and not runnable"));

        fs::remove_file(path).unwrap();
    }

    #[test]
    fn represents_unknown_ggml_tensor_type_without_claiming_bytes() {
        let unique = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_nanos();
        let path = std::env::temp_dir().join(format!("fathom-gguf-unknown-type-{unique}.gguf"));
        let mut bytes = Vec::new();
        bytes.extend_from_slice(b"GGUF");
        bytes.extend_from_slice(&3u32.to_le_bytes());
        bytes.extend_from_slice(&1u64.to_le_bytes()); // tensor_count
        bytes.extend_from_slice(&0u64.to_le_bytes()); // metadata_kv_count
        write_gguf_tensor_info(&mut bytes, "mystery.weight", &[4], 1234, 0);
        while bytes.len() % 32 != 0 {
            bytes.push(0);
        }
        fs::write(&path, bytes).unwrap();

        let metadata = read_gguf_metadata_summary(&path).unwrap();
        assert_eq!(metadata.tensors[0].ggml_type, 1234);
        assert_eq!(metadata.tensors[0].ggml_type_name, "UNKNOWN_1234");
        assert_eq!(metadata.tensors[0].element_count, Some(4));
        assert_eq!(metadata.tensors[0].estimated_bytes, None);
        let tensor_summary = metadata.tensor_summary.as_ref().unwrap();
        assert_eq!(tensor_summary.total_estimated_tensor_bytes, None);
        assert_eq!(tensor_summary.unknown_type_tags, vec![1234]);

        fs::remove_file(path).unwrap();
    }

    #[test]
    fn rejects_duplicate_gguf_tensor_names_safely() {
        let unique = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_nanos();
        let path = std::env::temp_dir().join(format!("fathom-gguf-duplicate-tensor-{unique}.gguf"));
        let mut bytes = Vec::new();
        bytes.extend_from_slice(b"GGUF");
        bytes.extend_from_slice(&3u32.to_le_bytes());
        bytes.extend_from_slice(&2u64.to_le_bytes()); // tensor_count
        bytes.extend_from_slice(&0u64.to_le_bytes()); // metadata_kv_count
        write_gguf_tensor_info(&mut bytes, "dup.weight", &[1], 0, 0);
        write_gguf_tensor_info(&mut bytes, "dup.weight", &[1], 0, 32);
        while bytes.len() % 32 != 0 {
            bytes.push(0);
        }
        bytes.extend(vec![0u8; 36]);
        fs::write(&path, bytes).unwrap();

        let error = read_gguf_metadata_summary(&path).unwrap_err().to_string();
        assert!(error.contains("duplicate GGUF tensor name dup.weight"));

        fs::remove_file(path).unwrap();
    }

    #[test]
    fn derives_sorted_gguf_payload_ranges_for_known_f32_and_f16_tensors() {
        let unique = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_nanos();
        let path = std::env::temp_dir().join(format!("fathom-gguf-payload-ranges-{unique}.gguf"));
        let mut bytes = minimal_gguf_header(2, 32);
        write_gguf_tensor_info(&mut bytes, "late.f16", &[2, 2], 1, 64);
        write_gguf_tensor_info(&mut bytes, "early.f32", &[2], 0, 0);
        pad_to_alignment(&mut bytes, 32);
        bytes.extend(vec![0u8; 8]);
        bytes.extend(vec![0u8; 56]);
        bytes.extend(vec![0u8; 8]);
        fs::write(&path, bytes).unwrap();

        let metadata = read_gguf_metadata_summary(&path).unwrap();
        assert_eq!(
            metadata.payload_range_status,
            GgufTensorPayloadRangeStatus::Ready
        );
        assert_eq!(metadata.payload_ranges.len(), 2);
        assert_eq!(metadata.payload_ranges[0].name, "early.f32");
        assert_eq!(metadata.payload_ranges[0].shape, vec![2]);
        assert_eq!(metadata.payload_ranges[0].element_count, 2);
        assert_eq!(metadata.payload_ranges[0].relative_offset, 0);
        assert_eq!(metadata.payload_ranges[0].byte_len, 8);
        assert_eq!(metadata.payload_ranges[1].name, "late.f16");
        assert_eq!(metadata.payload_ranges[1].shape, vec![2, 2]);
        assert_eq!(metadata.payload_ranges[1].element_count, 4);
        assert_eq!(metadata.payload_ranges[1].relative_offset, 64);
        assert_eq!(metadata.payload_ranges[1].byte_len, 8);
        let tensor_data_start = metadata.tensor_summary.as_ref().unwrap().tensor_data_start;
        assert_eq!(metadata.payload_ranges[0].absolute_start, tensor_data_start);
        assert_eq!(
            metadata.payload_ranges[0].absolute_end,
            tensor_data_start + 8
        );
        assert_eq!(
            metadata.payload_ranges[1].absolute_start,
            tensor_data_start + 64
        );
        assert_eq!(
            metadata.payload_ranges[1].absolute_end,
            tensor_data_start + 72
        );
        let public_json = serde_json::to_value(&metadata).unwrap();
        assert!(public_json.get("payload_ranges").is_none());
        assert!(public_json.get("payload_range_status").is_none());

        fs::remove_file(path).unwrap();
    }

    #[test]
    fn accepts_gaps_and_rejects_overlapping_gguf_payload_ranges() {
        let unique = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_nanos();
        let gap_path = std::env::temp_dir().join(format!("fathom-gguf-payload-gap-{unique}.gguf"));
        let mut gap_bytes = minimal_gguf_header(2, 32);
        write_gguf_tensor_info(&mut gap_bytes, "a", &[1], 0, 0);
        write_gguf_tensor_info(&mut gap_bytes, "b", &[1], 0, 32);
        pad_to_alignment(&mut gap_bytes, 32);
        gap_bytes.extend(vec![0u8; 36]);
        fs::write(&gap_path, gap_bytes).unwrap();
        let metadata = read_gguf_metadata_summary(&gap_path).unwrap();
        assert_eq!(
            metadata.payload_range_status,
            GgufTensorPayloadRangeStatus::Ready
        );
        assert_eq!(metadata.payload_ranges.len(), 2);

        let overlap_path =
            std::env::temp_dir().join(format!("fathom-gguf-payload-overlap-{unique}.gguf"));
        let mut overlap_bytes = minimal_gguf_header(2, 4);
        write_gguf_tensor_info(&mut overlap_bytes, "a", &[2], 0, 0);
        write_gguf_tensor_info(&mut overlap_bytes, "b", &[2], 0, 4);
        pad_to_alignment(&mut overlap_bytes, 4);
        overlap_bytes.extend(vec![0u8; 12]);
        fs::write(&overlap_path, overlap_bytes).unwrap();
        let error = read_gguf_metadata_summary(&overlap_path)
            .unwrap_err()
            .to_string();
        assert!(error.contains("GGUF tensor payload ranges overlap"));

        fs::remove_file(gap_path).unwrap();
        fs::remove_file(overlap_path).unwrap();
    }

    #[test]
    fn rejects_malformed_gguf_payload_range_prerequisites() {
        let unique = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_nanos();
        let past_eof =
            std::env::temp_dir().join(format!("fathom-gguf-payload-past-eof-{unique}.gguf"));
        let mut bytes = minimal_gguf_header(1, 32);
        write_gguf_tensor_info(&mut bytes, "past", &[8], 0, 0);
        pad_to_alignment(&mut bytes, 32);
        bytes.extend(vec![0u8; 16]);
        fs::write(&past_eof, bytes).unwrap();
        let error = read_gguf_metadata_summary(&past_eof)
            .unwrap_err()
            .to_string();
        assert!(error.contains("estimated byte range ends"));

        let misaligned =
            std::env::temp_dir().join(format!("fathom-gguf-payload-misaligned-{unique}.gguf"));
        let mut bytes = minimal_gguf_header(1, 32);
        write_gguf_tensor_info(&mut bytes, "misaligned", &[1], 0, 4);
        pad_to_alignment(&mut bytes, 32);
        bytes.extend(vec![0u8; 8]);
        fs::write(&misaligned, bytes).unwrap();
        let error = read_gguf_metadata_summary(&misaligned)
            .unwrap_err()
            .to_string();
        assert!(error.contains("is not aligned to 32"));

        let zero_alignment =
            std::env::temp_dir().join(format!("fathom-gguf-payload-align-zero-{unique}.gguf"));
        let mut bytes = minimal_gguf_header(1, 0);
        write_gguf_tensor_info(&mut bytes, "zero", &[1], 0, 0);
        fs::write(&zero_alignment, bytes).unwrap();
        let error = read_gguf_metadata_summary(&zero_alignment)
            .unwrap_err()
            .to_string();
        assert!(error.contains("GGUF alignment cannot be zero"));

        let data_start_past_eof =
            std::env::temp_dir().join(format!("fathom-gguf-payload-start-eof-{unique}.gguf"));
        let mut bytes = minimal_gguf_header(1, 32);
        write_gguf_tensor_info(&mut bytes, "no_payload", &[1], 0, 0);
        fs::write(&data_start_past_eof, bytes).unwrap();
        let error = read_gguf_metadata_summary(&data_start_past_eof)
            .unwrap_err()
            .to_string();
        assert!(error.contains("GGUF tensor data starts"));

        let overflow =
            std::env::temp_dir().join(format!("fathom-gguf-payload-overflow-{unique}.gguf"));
        let mut bytes = minimal_gguf_header(1, 32);
        write_gguf_tensor_info(&mut bytes, "overflow", &[u64::MAX, 2], 0, 0);
        fs::write(&overflow, bytes).unwrap();
        let error = read_gguf_metadata_summary(&overflow)
            .unwrap_err()
            .to_string();
        assert!(error.contains("element count overflow"));

        fs::remove_file(past_eof).unwrap();
        fs::remove_file(misaligned).unwrap();
        fs::remove_file(zero_alignment).unwrap();
        fs::remove_file(data_start_past_eof).unwrap();
        fs::remove_file(overflow).unwrap();
    }

    #[test]
    fn keeps_unknown_ggml_types_metadata_only_without_payload_ranges() {
        let unique = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_nanos();
        let path = std::env::temp_dir().join(format!("fathom-gguf-payload-unknown-{unique}.gguf"));
        let mut bytes = minimal_gguf_header(1, 32);
        write_gguf_tensor_info(&mut bytes, "mystery", &[4], 1234, 0);
        pad_to_alignment(&mut bytes, 32);
        fs::write(&path, bytes).unwrap();

        let metadata = read_gguf_metadata_summary(&path).unwrap();
        assert_eq!(metadata.payload_ranges.len(), 0);
        assert_eq!(
            metadata.payload_range_status,
            GgufTensorPayloadRangeStatus::UnsupportedTypesPresent
        );
        assert_eq!(metadata.tensors[0].estimated_bytes, None);
        assert_eq!(
            metadata.tensor_summary.as_ref().unwrap().unknown_type_tags,
            vec![1234]
        );

        fs::remove_file(path).unwrap();
    }

    #[test]
    fn refuses_internal_payload_ranges_when_known_payload_budget_is_exceeded() {
        let unique = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_nanos();
        let path = std::env::temp_dir().join(format!("fathom-gguf-payload-budget-{unique}.gguf"));
        let mut bytes = minimal_gguf_header(1, 32);
        let elements = (512 * 1024 * 1024u64 / 4) + 1;
        write_gguf_tensor_info(&mut bytes, "huge.f32", &[elements], 0, 0);
        pad_to_alignment(&mut bytes, 32);
        let required_len = bytes.len() as u64 + elements * 4;
        fs::write(&path, bytes).unwrap();
        fs::OpenOptions::new()
            .write(true)
            .open(&path)
            .unwrap()
            .set_len(required_len)
            .unwrap();

        let metadata = read_gguf_metadata_summary(&path).unwrap();
        assert!(metadata.payload_ranges.is_empty());
        assert_eq!(
            metadata.payload_range_status,
            GgufTensorPayloadRangeStatus::PayloadBudgetExceeded {
                requested: elements * 4,
                limit: 512 * 1024 * 1024,
            }
        );
        assert_eq!(
            metadata
                .tensor_summary
                .as_ref()
                .unwrap()
                .total_estimated_tensor_bytes,
            Some(elements * 4)
        );

        fs::remove_file(path).unwrap();
    }

    #[test]
    fn gguf_payload_reads_validated_f32_and_f16_ranges_and_decodes_exactly() {
        let unique = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_nanos();
        let path = std::env::temp_dir().join(format!("fathom-gguf-payload-read-{unique}.gguf"));
        let mut bytes = minimal_gguf_header(2, 32);
        write_gguf_tensor_info(&mut bytes, "weights.f32", &[2], 0, 0);
        write_gguf_tensor_info(&mut bytes, "weights.f16", &[4], 1, 32);
        pad_to_alignment(&mut bytes, 32);
        bytes.extend_from_slice(&1.25f32.to_le_bytes());
        bytes.extend_from_slice(&(-2.5f32).to_le_bytes());
        bytes.extend(vec![0xA5; 24]);
        for half in [0x3c00u16, 0xc000, 0x4200, 0x3800] {
            bytes.extend_from_slice(&half.to_le_bytes());
        }
        fs::write(&path, bytes).unwrap();

        let metadata = read_gguf_metadata_summary(&path).unwrap();
        assert_eq!(
            metadata.payload_range_status,
            GgufTensorPayloadRangeStatus::Ready
        );
        let f32_range = metadata
            .payload_ranges
            .iter()
            .find(|range| range.name == "weights.f32")
            .unwrap();
        let f32_raw = read_validated_gguf_payload_range(&path, f32_range, 8).unwrap();
        assert_eq!(
            decode_gguf_payload_to_f32(&f32_raw, f32_range).unwrap(),
            vec![1.25, -2.5]
        );

        let f16_range = metadata
            .payload_ranges
            .iter()
            .find(|range| range.name == "weights.f16")
            .unwrap();
        let f16_raw = read_validated_gguf_payload_range(&path, f16_range, 8).unwrap();
        assert_eq!(f16_raw.len(), 8);
        let decoded = decode_gguf_payload_to_f32(&f16_raw, f16_range).unwrap();
        assert_eq!(decoded, vec![1.0, -2.0, 3.0, 0.5]);

        fs::remove_file(path).unwrap();
    }

    #[test]
    fn gguf_payload_read_fails_closed_for_wrong_lengths_budget_and_truncation() {
        let unique = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_nanos();
        let path = std::env::temp_dir().join(format!("fathom-gguf-payload-closed-{unique}.gguf"));
        let mut bytes = minimal_gguf_header(1, 32);
        write_gguf_tensor_info(&mut bytes, "weights.f32", &[2], 0, 0);
        pad_to_alignment(&mut bytes, 32);
        bytes.extend_from_slice(&1.0f32.to_le_bytes());
        bytes.extend_from_slice(&2.0f32.to_le_bytes());
        fs::write(&path, bytes).unwrap();

        let metadata = read_gguf_metadata_summary(&path).unwrap();
        let range = metadata.payload_ranges.first().unwrap();
        assert_eq!(
            read_validated_gguf_payload_range(&path, range, 7).unwrap_err(),
            GgufPayloadReadError::PerReadBudgetExceeded {
                name: "weights.f32".to_string(),
                requested: 8,
                limit: 7,
            }
        );
        assert_eq!(
            decode_gguf_payload_to_f32(&[0, 0, 0, 0], range).unwrap_err(),
            GgufPayloadDecodeError::WrongLength {
                name: "weights.f32".to_string(),
                expected: 8,
                actual: 4,
            }
        );

        fs::OpenOptions::new()
            .write(true)
            .open(&path)
            .unwrap()
            .set_len(range.absolute_end - 1)
            .unwrap();
        assert_eq!(
            read_validated_gguf_payload_range(&path, range, 8).unwrap_err(),
            GgufPayloadReadError::FileTruncatedOrMutated {
                name: "weights.f32".to_string(),
                required_end: range.absolute_end,
                file_len: range.absolute_end - 1,
            }
        );

        fs::remove_file(path).unwrap();
    }

    #[test]
    fn gguf_payload_reader_does_not_offer_arbitrary_offsets() {
        let unique = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_nanos();
        let path =
            std::env::temp_dir().join(format!("fathom-gguf-payload-no-offset-{unique}.gguf"));
        let mut bytes = minimal_gguf_header(2, 32);
        write_gguf_tensor_info(&mut bytes, "first.f32", &[1], 0, 0);
        write_gguf_tensor_info(&mut bytes, "second.f32", &[1], 0, 32);
        pad_to_alignment(&mut bytes, 32);
        bytes.extend_from_slice(&9.0f32.to_le_bytes());
        bytes.extend(vec![0xCC; 28]);
        bytes.extend_from_slice(&4.0f32.to_le_bytes());
        fs::write(&path, bytes).unwrap();

        let metadata = read_gguf_metadata_summary(&path).unwrap();
        let second = metadata
            .payload_ranges
            .iter()
            .find(|range| range.name == "second.f32")
            .unwrap();
        let raw = read_validated_gguf_payload_range(&path, second, 4).unwrap();
        assert_eq!(decode_gguf_payload_to_f32(&raw, second).unwrap(), vec![4.0]);
        assert!(!raw.contains(&0xCC));

        fs::remove_file(path).unwrap();
    }

    #[test]
    fn gguf_payload_decode_quantized_q8_0_and_q4_0_golden_vectors() {
        let q8_range = GgufTensorPayloadRange {
            name: "quantized.q8".to_string(),
            ggml_type: 8,
            ggml_type_name: "Q8_0".to_string(),
            shape: vec![32],
            element_count: 32,
            relative_offset: 0,
            absolute_start: 0,
            absolute_end: 34,
            byte_len: 34,
        };
        let mut q8_raw = Vec::new();
        q8_raw.extend_from_slice(&0x3800u16.to_le_bytes()); // F16 0.5 scale.
        q8_raw.extend((-16i8..16i8).map(|value| value as u8));
        let q8_decoded = decode_gguf_payload_to_f32(&q8_raw, &q8_range).unwrap();
        assert_eq!(q8_decoded.len(), 32);
        assert_eq!(q8_decoded[0], -8.0);
        assert_eq!(q8_decoded[15], -0.5);
        assert_eq!(q8_decoded[16], 0.0);
        assert_eq!(q8_decoded[31], 7.5);

        let q4_range = GgufTensorPayloadRange {
            name: "quantized.q4".to_string(),
            ggml_type: 2,
            ggml_type_name: "Q4_0".to_string(),
            shape: vec![32],
            element_count: 32,
            relative_offset: 0,
            absolute_start: 0,
            absolute_end: 18,
            byte_len: 18,
        };
        let mut q4_raw = Vec::new();
        q4_raw.extend_from_slice(&0x4000u16.to_le_bytes()); // F16 2.0 scale.
        for low in 0u8..16 {
            let high = 15 - low;
            q4_raw.push((high << 4) | low);
        }
        let q4_decoded = decode_gguf_payload_to_f32(&q4_raw, &q4_range).unwrap();
        assert_eq!(q4_decoded.len(), 32);
        assert_eq!(&q4_decoded[..4], &[-16.0, -14.0, -12.0, -10.0]);
        assert_eq!(&q4_decoded[14..18], &[12.0, 14.0, 14.0, 12.0]);
        assert_eq!(&q4_decoded[28..], &[-10.0, -12.0, -14.0, -16.0]);
    }

    #[test]
    fn gguf_payload_decode_quantized_wrong_block_lengths_fail_closed() {
        let q8_partial_range = GgufTensorPayloadRange {
            name: "quantized.q8.partial".to_string(),
            ggml_type: 8,
            ggml_type_name: "Q8_0".to_string(),
            shape: vec![3],
            element_count: 3,
            relative_offset: 0,
            absolute_start: 0,
            absolute_end: 34,
            byte_len: 34,
        };
        let mut q8_raw = Vec::new();
        q8_raw.extend_from_slice(&0x3c00u16.to_le_bytes()); // F16 1.0 scale.
        q8_raw.extend([3i8 as u8, -2i8 as u8, 7i8 as u8]);
        q8_raw.extend([99u8; 29]);
        assert_eq!(
            decode_gguf_payload_to_f32(&q8_raw, &q8_partial_range).unwrap(),
            vec![3.0, -2.0, 7.0]
        );
        assert_eq!(
            decode_gguf_payload_to_f32(&q8_raw[..33], &q8_partial_range).unwrap_err(),
            GgufPayloadDecodeError::WrongLength {
                name: "quantized.q8.partial".to_string(),
                expected: 34,
                actual: 33,
            }
        );

        let q4_range = GgufTensorPayloadRange {
            name: "quantized.q4".to_string(),
            ggml_type: 2,
            ggml_type_name: "Q4_0".to_string(),
            shape: vec![32],
            element_count: 32,
            relative_offset: 0,
            absolute_start: 0,
            absolute_end: 18,
            byte_len: 18,
        };
        assert_eq!(
            decode_gguf_payload_to_f32(&[0; 17], &q4_range).unwrap_err(),
            GgufPayloadDecodeError::WrongLength {
                name: "quantized.q4".to_string(),
                expected: 18,
                actual: 17,
            }
        );
    }

    #[test]
    fn gguf_payload_decode_refuses_unknown_and_unsupported_types() {
        let q4_1_range = GgufTensorPayloadRange {
            name: "quantized.q4_1".to_string(),
            ggml_type: 3,
            ggml_type_name: "Q4_1".to_string(),
            shape: vec![32],
            element_count: 32,
            relative_offset: 0,
            absolute_start: 0,
            absolute_end: 20,
            byte_len: 20,
        };
        assert_eq!(
            decode_gguf_payload_to_f32(&[0; 20], &q4_1_range).unwrap_err(),
            GgufPayloadDecodeError::UnsupportedGgmlType {
                name: "quantized.q4_1".to_string(),
                tag: 3,
                type_name: "Q4_1".to_string(),
            }
        );

        let unknown_range = GgufTensorPayloadRange {
            name: "mystery".to_string(),
            ggml_type: 1234,
            ggml_type_name: "UNKNOWN_1234".to_string(),
            shape: vec![1],
            element_count: 1,
            relative_offset: 0,
            absolute_start: 0,
            absolute_end: 1,
            byte_len: 1,
        };
        assert_eq!(
            decode_gguf_payload_to_f32(&[0], &unknown_range).unwrap_err(),
            GgufPayloadDecodeError::UnknownGgmlType {
                name: "mystery".to_string(),
                tag: 1234,
            }
        );
        assert_eq!(
            read_validated_gguf_payload_range("/does/not/matter.gguf", &unknown_range, 1)
                .unwrap_err(),
            GgufPayloadReadError::UnknownGgmlTypeForPayload {
                name: "mystery".to_string(),
                tag: 1234,
            }
        );
    }

    #[test]
    #[ignore]
    fn live_pinned_gguf_fixture_has_expected_metadata_only_payload_ranges() {
        let path = std::env::var("FATHOM_GGUF_LLAMA_TINY_FIXTURE")
            .expect("set FATHOM_GGUF_LLAMA_TINY_FIXTURE to llama-2-tiny-random.gguf");
        let metadata = read_gguf_metadata_summary(path).unwrap();
        assert_eq!(metadata.version, 3);
        assert_eq!(metadata.tensor_count, 12);
        assert_eq!(metadata.metadata_kv_count, 18);
        assert_eq!(metadata.hints.architecture.as_deref(), Some("llama"));
        assert_eq!(metadata.tokenizer_summary.model.as_deref(), Some("llama"));
        assert_eq!(metadata.tokenizer_summary.token_count, Some(32_000));
        let spec = metadata
            .tokenizer_spec
            .as_ref()
            .expect("llama tokenizer spec");
        assert_eq!(spec.family, GgufTokenizerSpecFamily::LlamaSentencePiece);
        assert_eq!(spec.tokens.len(), 32_000);
        assert_eq!(spec.scores.len(), 32_000);
        assert_eq!(spec.token_types.len(), 32_000);
        assert!(spec.has_byte_fallback);
        let summary = metadata.tensor_summary.as_ref().unwrap();
        assert_eq!(summary.tensor_data_start, 724_416);
        assert_eq!(summary.file_size, 1_750_560);
        assert_eq!(summary.total_estimated_tensor_bytes, Some(1_026_144));
        assert_eq!(summary.type_counts.get("F16"), Some(&9));
        assert_eq!(summary.type_counts.get("F32"), Some(&3));
        assert!(summary.unknown_type_tags.is_empty());
        assert_eq!(
            metadata.payload_range_status,
            GgufTensorPayloadRangeStatus::Ready
        );
        assert_eq!(metadata.payload_ranges.len(), 12);
        assert_eq!(
            metadata.payload_ranges.first().unwrap().name,
            "token_embd.weight"
        );
        assert_eq!(
            metadata.payload_ranges.first().unwrap().absolute_start,
            724_416
        );
        assert_eq!(metadata.payload_ranges.first().unwrap().byte_len, 512_000);
        assert_eq!(
            metadata.payload_ranges.last().unwrap().name,
            "output.weight"
        );
        assert_eq!(
            metadata.payload_ranges.last().unwrap().absolute_start,
            1_238_560
        );
        assert_eq!(
            metadata.payload_ranges.last().unwrap().absolute_end,
            1_750_560
        );
        assert!(metadata
            .compatibility
            .runtime_blockers
            .contains(&"gguf_generation_not_implemented".to_string()));
    }

    #[test]
    fn classifies_gguf_tokenizer_and_architecture_metadata_without_runtime_claims() {
        let unique = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_nanos();
        let path = std::env::temp_dir().join(format!("fathom-gguf-compat-{unique}.gguf"));
        let mut bytes = Vec::new();
        bytes.extend_from_slice(b"GGUF");
        bytes.extend_from_slice(&3u32.to_le_bytes());
        bytes.extend_from_slice(&1u64.to_le_bytes()); // tensor_count
        bytes.extend_from_slice(&15u64.to_le_bytes()); // metadata_kv_count
        write_gguf_string(&mut bytes, "general.architecture");
        bytes.extend_from_slice(&8u32.to_le_bytes()); // string
        write_gguf_string(&mut bytes, "llama");
        write_gguf_string(&mut bytes, "general.alignment");
        bytes.extend_from_slice(&4u32.to_le_bytes()); // u32
        bytes.extend_from_slice(&32u32.to_le_bytes());
        write_gguf_string(&mut bytes, "general.file_type");
        bytes.extend_from_slice(&4u32.to_le_bytes()); // u32
        bytes.extend_from_slice(&15u32.to_le_bytes());
        for (key, value) in [
            ("llama.context_length", 4096u32),
            ("llama.embedding_length", 128),
            ("llama.block_count", 2),
            ("llama.attention.head_count", 4),
            ("llama.attention.head_count_kv", 2),
            ("llama.feed_forward_length", 256),
            ("llama.rope.dimension_count", 64),
        ] {
            write_gguf_string(&mut bytes, key);
            bytes.extend_from_slice(&4u32.to_le_bytes()); // u32
            bytes.extend_from_slice(&value.to_le_bytes());
        }
        write_gguf_string(&mut bytes, "tokenizer.ggml.model");
        bytes.extend_from_slice(&8u32.to_le_bytes()); // string
        write_gguf_string(&mut bytes, "llama");
        write_gguf_string(&mut bytes, "tokenizer.ggml.tokens");
        bytes.extend_from_slice(&9u32.to_le_bytes()); // array
        bytes.extend_from_slice(&8u32.to_le_bytes()); // string array
        bytes.extend_from_slice(&3u64.to_le_bytes());
        for token in ["<s>", "</s>", "hello"] {
            write_gguf_string(&mut bytes, token);
        }
        write_gguf_string(&mut bytes, "tokenizer.ggml.merges");
        bytes.extend_from_slice(&9u32.to_le_bytes()); // array
        bytes.extend_from_slice(&8u32.to_le_bytes()); // string array
        bytes.extend_from_slice(&2u64.to_le_bytes());
        for merge in ["h e", "he llo"] {
            write_gguf_string(&mut bytes, merge);
        }
        write_gguf_string(&mut bytes, "tokenizer.ggml.bos_token_id");
        bytes.extend_from_slice(&4u32.to_le_bytes()); // u32
        bytes.extend_from_slice(&1u32.to_le_bytes());
        write_gguf_string(&mut bytes, "tokenizer.chat_template");
        bytes.extend_from_slice(&8u32.to_le_bytes()); // string
        write_gguf_string(
            &mut bytes,
            "{% for message in messages %}{{ message.content }}{% endfor %}",
        );
        write_gguf_tensor_info(&mut bytes, "blk.0.weight", &[1], 0, 0);
        while bytes.len() % 32 != 0 {
            bytes.push(0);
        }
        bytes.extend(vec![0u8; 4]);
        fs::write(&path, bytes).unwrap();

        let metadata = read_gguf_metadata_summary(&path).unwrap();
        assert_eq!(
            metadata.tokenizer_summary.status,
            GgufTokenizerMetadataStatus::Present
        );
        assert_eq!(metadata.tokenizer_summary.model.as_deref(), Some("llama"));
        assert_eq!(metadata.tokenizer_summary.token_count, Some(3));
        assert_eq!(
            metadata.tokenizer_summary.token_samples,
            vec!["<s>", "</s>", "hello"]
        );
        assert_eq!(metadata.tokenizer_summary.merge_count, Some(2));
        assert_eq!(
            metadata.tokenizer_summary.special_token_ids.get("bos"),
            Some(&1)
        );
        assert!(
            metadata
                .tokenizer_summary
                .chat_template
                .as_ref()
                .unwrap()
                .byte_len
                > 0
        );
        assert_eq!(
            metadata.architecture_summary.status,
            GgufArchitectureMetadataStatus::Recognized
        );
        assert_eq!(
            metadata.architecture_summary.family.as_deref(),
            Some("llama_like_metadata")
        );
        assert_eq!(metadata.architecture_summary.context_length, Some(4096));
        assert_eq!(metadata.architecture_summary.embedding_length, Some(128));
        assert_eq!(
            metadata.architecture_summary.quantization_hint.as_deref(),
            Some("mostly_q4_k_m")
        );
        assert_eq!(
            metadata.compatibility.tokenizer_metadata,
            GgufTokenizerMetadataStatus::Present
        );
        assert_eq!(
            metadata.compatibility.architecture_metadata,
            GgufArchitectureMetadataStatus::Recognized
        );
        assert!(metadata
            .compatibility
            .categories
            .contains(&"llama_like_metadata".to_string()));
        assert!(metadata
            .compatibility
            .runtime_blockers
            .contains(&"gguf_general_dequantization_not_implemented".to_string()));
        assert!(metadata
            .compatibility
            .runtime_blockers
            .contains(&"gguf_quantized_kernels_not_implemented".to_string()));

        let package = inspect_model_package(&path).unwrap();
        let report = capability_report_for_package(package, &current_machine_profile());
        assert_eq!(report.best_status, CapabilityStatus::MetadataOnly);
        assert!(!report.runnable);

        fs::remove_file(path).unwrap();
    }

    #[test]
    fn builds_internal_synthetic_gpt2_bpe_tokenizer_spec_without_public_array_dump() {
        let unique = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_nanos();
        let path = std::env::temp_dir().join(format!("fathom-gguf-tokenizer-spec-{unique}.gguf"));
        let mut bytes = Vec::new();
        bytes.extend_from_slice(b"GGUF");
        bytes.extend_from_slice(&3u32.to_le_bytes());
        bytes.extend_from_slice(&0u64.to_le_bytes());
        bytes.extend_from_slice(&5u64.to_le_bytes());
        write_gguf_string_kv(&mut bytes, "tokenizer.ggml.model", "gpt2");
        write_gguf_string_array_kv(
            &mut bytes,
            "tokenizer.ggml.tokens",
            &["<|endoftext|>", "H", "e", "l", "o", "He", "ll", "Hello"],
        );
        write_gguf_string_array_kv(
            &mut bytes,
            "tokenizer.ggml.merges",
            &["H e", "l l", "He ll", "Hell o"],
        );
        write_gguf_u32_kv(&mut bytes, "tokenizer.ggml.eos_token_id", 0);
        write_gguf_string_kv(
            &mut bytes,
            "tokenizer.chat_template",
            "{{ messages[0].content }}",
        );
        fs::write(&path, bytes).unwrap();

        let metadata = read_gguf_metadata_summary(&path).unwrap();
        assert_eq!(
            metadata.tokenizer_summary.status,
            GgufTokenizerMetadataStatus::Present
        );
        assert_eq!(metadata.tokenizer_summary.model.as_deref(), Some("gpt2"));
        assert_eq!(metadata.tokenizer_summary.token_count, Some(8));
        assert_eq!(metadata.tokenizer_summary.merge_count, Some(4));
        let spec = metadata.tokenizer_spec.as_ref().expect("internal spec");
        assert_eq!(
            spec.family,
            GgufTokenizerSpecFamily::SyntheticGpt2ByteLevelBpe
        );
        assert_eq!(spec.tokens.len(), 8);
        assert_eq!(spec.merges.len(), 4);
        assert_eq!(spec.special_token_ids.get("eos"), Some(&0));

        let public_json = serde_json::to_value(&metadata).unwrap();
        assert!(public_json.get("tokenizer_spec").is_none());
        let public_tokens = &public_json["metadata"][1]["value"];
        assert_eq!(public_tokens["len"], 8);
        assert!(public_tokens.get("full_strings").is_none());
        assert!(metadata.tokenizer_summary.notes.iter().any(|note| {
            note.contains("Internal tokenizer spec built for synthetic GPT-2/ByteLevel-BPE")
        }));

        fs::remove_file(path).unwrap();
    }

    #[test]
    fn builds_internal_llama_sentencepiece_tokenizer_spec_without_public_array_dump() {
        let unique = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_nanos();
        let path = std::env::temp_dir().join(format!("fathom-gguf-llama-spec-{unique}.gguf"));
        write_minimal_llama_tokenizer_gguf(
            &path,
            &["<unk>", "<s>", "</s>", "▁hello"],
            &[0.0, 0.0, 0.0, -1.25],
            &[2, 3, 3, 1],
            &[
                ("tokenizer.ggml.unknown_token_id", 0),
                ("tokenizer.ggml.bos_token_id", 1),
                ("tokenizer.ggml.eos_token_id", 2),
            ],
        );

        let metadata = read_gguf_metadata_summary(&path).unwrap();
        let spec = metadata.tokenizer_spec.as_ref().expect("llama spec");
        assert_eq!(spec.family, GgufTokenizerSpecFamily::LlamaSentencePiece);
        assert_eq!(spec.tokens, vec!["<unk>", "<s>", "</s>", "▁hello"]);
        assert_eq!(spec.scores, vec![0.0, 0.0, 0.0, -1.25]);
        assert_eq!(spec.token_types, vec![2, 3, 3, 1]);
        assert_eq!(spec.special_token_ids.get("bos"), Some(&1));
        assert!(!spec.has_byte_fallback);
        assert!(metadata.tokenizer_summary.notes.iter().any(|note| {
            note.contains("bounded Llama/SentencePiece-shaped GGUF metadata only")
        }));
        assert!(metadata
            .tokenizer_summary
            .notes
            .iter()
            .any(|note| { note.contains("does not include explicit add_bos_token") }));

        let public_json = serde_json::to_value(&metadata).unwrap();
        assert!(public_json.get("tokenizer_spec").is_none());
        assert!(public_json["metadata"]
            .as_array()
            .unwrap()
            .iter()
            .filter_map(|entry| entry.get("value"))
            .all(|value| value.get("full_strings").is_none()
                && value.get("full_floats").is_none()
                && value.get("full_i32s").is_none()));

        fs::remove_file(path).unwrap();
    }

    #[test]
    fn rejects_llama_sentencepiece_score_and_type_length_mismatches() {
        let unique = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_nanos();
        let score_path =
            std::env::temp_dir().join(format!("fathom-gguf-llama-score-{unique}.gguf"));
        write_minimal_llama_tokenizer_gguf(&score_path, &["<unk>", "<s>"], &[0.0], &[2, 3], &[]);
        let metadata = read_gguf_metadata_summary(&score_path).unwrap();
        assert!(metadata.tokenizer_spec.is_none());
        assert!(metadata
            .tokenizer_summary
            .notes
            .iter()
            .any(|note| note
                .contains("tokenizer.ggml.scores length 1 does not match token length 2")));
        fs::remove_file(score_path).unwrap();

        let type_path = std::env::temp_dir().join(format!("fathom-gguf-llama-type-{unique}.gguf"));
        write_minimal_llama_tokenizer_gguf(&type_path, &["<unk>", "<s>"], &[0.0, 0.0], &[2], &[]);
        let metadata = read_gguf_metadata_summary(&type_path).unwrap();
        assert!(metadata.tokenizer_spec.is_none());
        assert!(metadata.tokenizer_summary.notes.iter().any(|note| note
            .contains("tokenizer.ggml.token_type length 1 does not match token length 2")));
        fs::remove_file(type_path).unwrap();
    }

    #[test]
    fn rejects_llama_sentencepiece_special_id_out_of_range() {
        let unique = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_nanos();
        let path =
            std::env::temp_dir().join(format!("fathom-gguf-llama-bad-special-{unique}.gguf"));
        write_minimal_llama_tokenizer_gguf(
            &path,
            &["<unk>", "<s>"],
            &[0.0, 0.0],
            &[2, 3],
            &[("tokenizer.ggml.eos_token_id", 2)],
        );
        let metadata = read_gguf_metadata_summary(&path).unwrap();
        assert!(metadata.tokenizer_spec.is_none());
        assert!(metadata.tokenizer_summary.notes.iter().any(|note| note
            .contains("tokenizer.ggml.eos_token_id value 2 is outside token array length 2")));
        fs::remove_file(path).unwrap();
    }

    #[test]
    fn detects_llama_sentencepiece_byte_fallback_evidence() {
        let unique = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_nanos();
        let path = std::env::temp_dir().join(format!("fathom-gguf-llama-byte-{unique}.gguf"));
        let mut token_storage = vec!["<unk>".to_string(), "<s>".to_string(), "</s>".to_string()];
        for byte in 0u16..=255 {
            token_storage.push(format!("<0x{byte:02X}>"));
        }
        let token_refs: Vec<&str> = token_storage.iter().map(String::as_str).collect();
        let mut scores = vec![0.0f32; token_refs.len()];
        scores[258] = -1.0;
        let mut token_types = vec![6i32; token_refs.len()];
        token_types[0] = 2;
        token_types[1] = 3;
        token_types[2] = 3;
        write_minimal_llama_tokenizer_gguf(&path, &token_refs, &scores, &token_types, &[]);

        let metadata = read_gguf_metadata_summary(&path).unwrap();
        let spec = metadata.tokenizer_spec.as_ref().expect("llama byte spec");
        assert!(spec.has_byte_fallback);
        assert!(metadata
            .tokenizer_summary
            .notes
            .iter()
            .any(|note| note.contains("Detected GGUF Llama byte-fallback evidence")));
        fs::remove_file(path).unwrap();
    }

    #[test]
    fn leaves_incomplete_gguf_tokenizer_metadata_without_internal_spec() {
        let unique = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_nanos();
        let path =
            std::env::temp_dir().join(format!("fathom-gguf-tokenizer-incomplete-{unique}.gguf"));
        let mut bytes = Vec::new();
        bytes.extend_from_slice(b"GGUF");
        bytes.extend_from_slice(&3u32.to_le_bytes());
        bytes.extend_from_slice(&0u64.to_le_bytes());
        bytes.extend_from_slice(&2u64.to_le_bytes());
        write_gguf_string_kv(&mut bytes, "tokenizer.ggml.model", "gpt2");
        write_gguf_string_array_kv(&mut bytes, "tokenizer.ggml.tokens", &["<|endoftext|>", "H"]);
        fs::write(&path, bytes).unwrap();

        let metadata = read_gguf_metadata_summary(&path).unwrap();
        assert_eq!(
            metadata.tokenizer_summary.status,
            GgufTokenizerMetadataStatus::Partial
        );
        assert!(metadata.tokenizer_spec.is_none());
        assert!(metadata
            .tokenizer_summary
            .notes
            .iter()
            .any(|note| note.contains("tokenizer.ggml.merges is missing")));

        fs::remove_file(path).unwrap();
    }

    #[test]
    fn leaves_unsupported_gguf_tokenizer_family_without_internal_spec() {
        let unique = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_nanos();
        let path =
            std::env::temp_dir().join(format!("fathom-gguf-tokenizer-unsupported-{unique}.gguf"));
        let mut bytes = Vec::new();
        bytes.extend_from_slice(b"GGUF");
        bytes.extend_from_slice(&3u32.to_le_bytes());
        bytes.extend_from_slice(&0u64.to_le_bytes());
        bytes.extend_from_slice(&3u64.to_le_bytes());
        write_gguf_string_kv(&mut bytes, "tokenizer.ggml.model", "qwen2");
        write_gguf_string_array_kv(
            &mut bytes,
            "tokenizer.ggml.tokens",
            &["<s>", "</s>", "hello"],
        );
        write_gguf_string_array_kv(&mut bytes, "tokenizer.ggml.merges", &["h e"]);
        fs::write(&path, bytes).unwrap();

        let metadata = read_gguf_metadata_summary(&path).unwrap();
        assert!(metadata.tokenizer_spec.is_none());
        assert!(metadata.tokenizer_summary.notes.iter().any(|note| note
            .contains("outside the narrow synthetic GPT-2/ByteLevel-BPE, Llama 3 BPE, and Llama/SentencePiece")));

        fs::remove_file(path).unwrap();
    }

    #[test]
    fn reads_oversized_gguf_tokenizer_arrays_without_retaining_internal_spec() {
        let unique = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_nanos();
        let path =
            std::env::temp_dir().join(format!("fathom-gguf-tokenizer-oversize-{unique}.gguf"));
        let mut bytes = Vec::new();
        bytes.extend_from_slice(b"GGUF");
        bytes.extend_from_slice(&3u32.to_le_bytes());
        bytes.extend_from_slice(&0u64.to_le_bytes());
        bytes.extend_from_slice(&2u64.to_le_bytes());
        write_gguf_string_kv(&mut bytes, "tokenizer.ggml.model", "gpt2");
        write_gguf_string(&mut bytes, "tokenizer.ggml.tokens");
        bytes.extend_from_slice(&9u32.to_le_bytes());
        bytes.extend_from_slice(&8u32.to_le_bytes());
        bytes.extend_from_slice(&128_257u64.to_le_bytes());
        for index in 0..128_257 {
            write_gguf_string(&mut bytes, &format!("token-{index}"));
        }
        fs::write(&path, bytes).unwrap();

        let metadata = read_gguf_metadata_summary(&path).unwrap();
        assert_eq!(metadata.tokenizer_summary.token_count, Some(128_257));
        assert_eq!(metadata.tokenizer_summary.token_samples.len(), 8);
        assert!(metadata.tokenizer_spec.is_none());
        assert!(metadata
            .tokenizer_summary
            .notes
            .iter()
            .any(|note| note.contains(
                "tokenizer.ggml.tokens is missing, malformed, or over the internal retention limit"
            )));

        fs::remove_file(path).unwrap();
    }

    #[test]
    fn rejects_malformed_gguf_tokenizer_merges_without_runtime_claims() {
        let unique = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_nanos();
        let path =
            std::env::temp_dir().join(format!("fathom-gguf-tokenizer-bad-merge-{unique}.gguf"));
        let mut bytes = Vec::new();
        bytes.extend_from_slice(b"GGUF");
        bytes.extend_from_slice(&3u32.to_le_bytes());
        bytes.extend_from_slice(&0u64.to_le_bytes());
        bytes.extend_from_slice(&3u64.to_le_bytes());
        write_gguf_string_kv(&mut bytes, "tokenizer.ggml.model", "gpt2");
        write_gguf_string_array_kv(
            &mut bytes,
            "tokenizer.ggml.tokens",
            &["<|endoftext|>", "a", "b"],
        );
        write_gguf_string_array_kv(&mut bytes, "tokenizer.ggml.merges", &["a"]);
        fs::write(&path, bytes).unwrap();

        let metadata = read_gguf_metadata_summary(&path).unwrap();
        assert!(metadata.tokenizer_spec.is_none());
        assert!(metadata
            .tokenizer_summary
            .notes
            .iter()
            .any(|note| note.contains("not a two-token BPE merge")));

        fs::remove_file(path).unwrap();
    }

    #[test]
    fn marks_unsupported_gguf_tokenizer_metadata_shape_without_loading_runtime() {
        let unique = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_nanos();
        let path = std::env::temp_dir().join(format!("fathom-gguf-tokenizer-shape-{unique}.gguf"));
        let mut bytes = Vec::new();
        bytes.extend_from_slice(b"GGUF");
        bytes.extend_from_slice(&3u32.to_le_bytes());
        bytes.extend_from_slice(&0u64.to_le_bytes()); // tensor_count
        bytes.extend_from_slice(&1u64.to_le_bytes()); // metadata_kv_count
        write_gguf_string(&mut bytes, "tokenizer.ggml.tokens");
        bytes.extend_from_slice(&9u32.to_le_bytes()); // array
        bytes.extend_from_slice(&4u32.to_le_bytes()); // u32 array, not string
        bytes.extend_from_slice(&2u64.to_le_bytes());
        bytes.extend_from_slice(&1u32.to_le_bytes());
        bytes.extend_from_slice(&2u32.to_le_bytes());
        fs::write(&path, bytes).unwrap();

        let error = read_gguf_metadata_summary(&path).unwrap_err().to_string();
        assert!(error.contains("must be an array of strings"));

        fs::remove_file(path).unwrap();
    }

    #[test]
    fn rejects_malformed_gguf_metadata_safely() {
        let unique = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_nanos();
        let path = std::env::temp_dir().join(format!("fathom-gguf-bad-{unique}.gguf"));
        fs::write(&path, b"GGUF\x03\0\0").unwrap();

        let artifact = detect_model_artifact(&path);
        assert_eq!(artifact.format, ModelFormat::Gguf);
        assert_eq!(artifact.support, SupportLevel::LoadPlanned);
        assert!(artifact.gguf_metadata.is_none());
        assert!(artifact
            .notes
            .iter()
            .any(|note| note.contains("metadata could not be read safely")));

        fs::remove_file(path).unwrap();
    }

    fn write_gguf_string(bytes: &mut Vec<u8>, value: &str) {
        bytes.extend_from_slice(&(value.len() as u64).to_le_bytes());
        bytes.extend_from_slice(value.as_bytes());
    }

    fn write_gguf_string_kv(bytes: &mut Vec<u8>, key: &str, value: &str) {
        write_gguf_string(bytes, key);
        bytes.extend_from_slice(&8u32.to_le_bytes());
        write_gguf_string(bytes, value);
    }

    fn write_gguf_u32_kv(bytes: &mut Vec<u8>, key: &str, value: u32) {
        write_gguf_string(bytes, key);
        bytes.extend_from_slice(&4u32.to_le_bytes());
        bytes.extend_from_slice(&value.to_le_bytes());
    }

    fn write_gguf_string_array_kv(bytes: &mut Vec<u8>, key: &str, values: &[&str]) {
        write_gguf_string(bytes, key);
        bytes.extend_from_slice(&9u32.to_le_bytes());
        bytes.extend_from_slice(&8u32.to_le_bytes());
        bytes.extend_from_slice(&(values.len() as u64).to_le_bytes());
        for value in values {
            write_gguf_string(bytes, value);
        }
    }

    fn write_gguf_f32_array_kv(bytes: &mut Vec<u8>, key: &str, values: &[f32]) {
        write_gguf_string(bytes, key);
        bytes.extend_from_slice(&9u32.to_le_bytes());
        bytes.extend_from_slice(&6u32.to_le_bytes());
        bytes.extend_from_slice(&(values.len() as u64).to_le_bytes());
        for value in values {
            bytes.extend_from_slice(&value.to_le_bytes());
        }
    }

    fn write_gguf_i32_array_kv(bytes: &mut Vec<u8>, key: &str, values: &[i32]) {
        write_gguf_string(bytes, key);
        bytes.extend_from_slice(&9u32.to_le_bytes());
        bytes.extend_from_slice(&5u32.to_le_bytes());
        bytes.extend_from_slice(&(values.len() as u64).to_le_bytes());
        for value in values {
            bytes.extend_from_slice(&value.to_le_bytes());
        }
    }

    fn write_minimal_llama_tokenizer_gguf(
        path: &Path,
        tokens: &[&str],
        scores: &[f32],
        token_types: &[i32],
        special_ids: &[(&str, u32)],
    ) {
        let mut bytes = Vec::new();
        bytes.extend_from_slice(b"GGUF");
        bytes.extend_from_slice(&3u32.to_le_bytes());
        bytes.extend_from_slice(&0u64.to_le_bytes());
        bytes.extend_from_slice(&(4u64 + special_ids.len() as u64).to_le_bytes());
        write_gguf_string_kv(&mut bytes, "tokenizer.ggml.model", "llama");
        write_gguf_string_array_kv(&mut bytes, "tokenizer.ggml.tokens", tokens);
        write_gguf_f32_array_kv(&mut bytes, "tokenizer.ggml.scores", scores);
        write_gguf_i32_array_kv(&mut bytes, "tokenizer.ggml.token_type", token_types);
        for (key, value) in special_ids {
            write_gguf_u32_kv(&mut bytes, key, *value);
        }
        fs::write(path, bytes).unwrap();
    }

    fn write_gguf_tensor_info(
        bytes: &mut Vec<u8>,
        name: &str,
        shape: &[u64],
        ggml_type: u32,
        offset: u64,
    ) {
        write_gguf_string(bytes, name);
        bytes.extend_from_slice(&(shape.len() as u32).to_le_bytes());
        for dim in shape {
            bytes.extend_from_slice(&dim.to_le_bytes());
        }
        bytes.extend_from_slice(&ggml_type.to_le_bytes());
        bytes.extend_from_slice(&offset.to_le_bytes());
    }

    fn minimal_gguf_header(tensor_count: u64, alignment: u32) -> Vec<u8> {
        let mut bytes = Vec::new();
        bytes.extend_from_slice(b"GGUF");
        bytes.extend_from_slice(&3u32.to_le_bytes());
        bytes.extend_from_slice(&tensor_count.to_le_bytes());
        bytes.extend_from_slice(&1u64.to_le_bytes());
        write_gguf_u32_kv(&mut bytes, "general.alignment", alignment);
        bytes
    }

    fn pad_to_alignment(bytes: &mut Vec<u8>, alignment: usize) {
        if alignment == 0 {
            return;
        }
        while bytes.len() % alignment != 0 {
            bytes.push(0);
        }
    }

    #[test]
    fn generation_metrics_split_prefill_and_decode_rates() {
        let metrics = build_generation_metrics(
            25,
            std::time::Duration::from_millis(250),
            300,
            20,
            5,
            Some(std::time::Duration::from_millis(50)),
        );
        assert_eq!(metrics.model_load_ms, 25);
        assert_eq!(metrics.generation_ms, 250);
        assert_eq!(metrics.total_ms, 300);
        assert_eq!(metrics.ttft_ms, Some(50));
        assert_eq!(metrics.prefill_ms, Some(50));
        assert_eq!(metrics.decode_ms, Some(200));
        assert_eq!(metrics.prefill_tokens_per_second, Some(400.0));
        assert_eq!(metrics.decode_tokens_per_second, Some(20.0));
        assert_eq!(metrics.tokens_per_second, Some(20.0));
        assert!(!metrics.runtime_cache_hit);
        assert_eq!(metrics.runtime_cache_lookup_ms, 0);
        assert_eq!(metrics.runtime_residency.as_deref(), Some("not_cached"));
        assert_eq!(metrics.runtime_family, None);
    }

    #[test]
    fn generation_metrics_serialize_runtime_cache_fields() {
        let mut metrics = build_generation_metrics(
            0,
            std::time::Duration::from_millis(100),
            110,
            4,
            2,
            Some(std::time::Duration::from_millis(25)),
        );
        metrics.runtime_cache_hit = true;
        metrics.runtime_cache_lookup_ms = 3;
        metrics.runtime_residency = Some("warm_reused".to_string());
        metrics.runtime_family = Some("gpt2".to_string());

        let value = serde_json::to_value(metrics).unwrap();
        assert_eq!(value["runtime_cache_hit"], true);
        assert_eq!(value["runtime_cache_lookup_ms"], 3);
        assert_eq!(value["runtime_residency"], "warm_reused");
        assert_eq!(value["runtime_family"], "gpt2");
    }

    #[test]
    fn inspects_hugging_face_style_package() {
        let unique = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_nanos();
        let dir = std::env::temp_dir().join(format!("fathom-package-test-{unique}"));
        fs::create_dir_all(&dir).unwrap();
        fs::write(
            dir.join("config.json"),
            r#"{"model_type":"llama","architectures":["LlamaForCausalLM"]}"#,
        )
        .unwrap();
        fs::write(dir.join("tokenizer.json"), b"{}").unwrap();
        fs::write(
            dir.join("tokenizer_config.json"),
            r#"{"bos_token":"<s>","eos_token":{"content":"</s>"},"chat_template":"{{ messages }}"}"#,
        )
        .unwrap();
        fs::write(dir.join("model.safetensors"), b"").unwrap();

        let package = inspect_model_package(&dir).unwrap();
        assert_eq!(package.model_type.as_deref(), Some("llama"));
        assert_eq!(package.architectures, vec!["LlamaForCausalLM"]);
        assert_eq!(package.tokenizer_files.len(), 2);
        assert!(package.hf_validation.ready_for_loader_metadata);
        assert!(package.hf_validation.has_chat_template);
        let tokenizer = package.tokenizer.as_ref().unwrap();
        assert_eq!(tokenizer.kind, TokenizerKind::HuggingFaceTokenizerJson);
        assert_eq!(tokenizer.bos_token.as_deref(), Some("<s>"));
        assert_eq!(tokenizer.eos_token.as_deref(), Some("</s>"));
        assert_eq!(
            tokenizer.chat_template.as_ref().unwrap().format,
            ChatTemplateFormat::HuggingFaceJinja
        );
        assert!(
            tokenizer
                .chat_template
                .as_ref()
                .unwrap()
                .needs_template_engine
        );
        let tokenizer = package.tokenizer.as_ref().expect("tokenizer metadata");
        assert_eq!(tokenizer.kind, TokenizerKind::HuggingFaceTokenizerJson);
        assert_eq!(
            tokenizer.chat_template.as_ref().unwrap().template,
            "{{ messages }}"
        );
        assert!(package
            .artifacts
            .iter()
            .any(|artifact| artifact.format == ModelFormat::SafeTensors));

        fs::remove_dir_all(dir).unwrap();
    }

    #[test]
    fn maps_hf_safetensors_package_to_planned_backend_lane() {
        let unique = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_nanos();
        let dir = std::env::temp_dir().join(format!("fathom-capability-test-{unique}"));
        fs::create_dir_all(&dir).unwrap();
        fs::write(dir.join("config.json"), r#"{"model_type":"llama"}"#).unwrap();
        fs::write(dir.join("tokenizer.json"), b"{}").unwrap();
        fs::write(dir.join("model.safetensors"), b"").unwrap();

        let package = inspect_model_package(&dir).unwrap();
        let machine = current_machine_profile();
        let report = capability_report_for_package(package, &machine);

        assert!(!report.runnable);
        assert_eq!(report.best_status, CapabilityStatus::Planned);
        assert!(report
            .matching_lanes
            .iter()
            .any(|lane| lane.kind == BackendLaneKind::SafeTensorsHf));

        fs::remove_dir_all(dir).unwrap();
    }

    #[test]
    fn recommends_retrieval_for_small_context_hf_package() {
        let unique = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_nanos();
        let dir = std::env::temp_dir().join(format!("fathom-context-test-{unique}"));
        fs::create_dir_all(&dir).unwrap();
        fs::write(
            dir.join("config.json"),
            r#"{"model_type":"gpt2","architectures":["GPT2LMHeadModel"],"n_positions":1024}"#,
        )
        .unwrap();
        fs::write(dir.join("tokenizer.json"), b"{}").unwrap();
        fs::write(dir.join("model.safetensors"), b"").unwrap();

        let package = inspect_model_package(&dir).unwrap();
        let advice = recommend_context_strategy_for_package(&package);

        assert_eq!(advice.max_context_tokens, Some(1024));
        assert_eq!(advice.engine, ContextEngineRecommendation::RetrievalIndex);
        assert!(advice.needs_retrieval);
        assert!(advice.recommended_chunk_tokens <= 500);
        assert!(advice.reserve_output_tokens > 0);

        fs::remove_dir_all(dir).unwrap();
    }

    #[test]
    fn recommends_code_index_for_qwen_package() {
        let unique = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_nanos();
        let dir = std::env::temp_dir().join(format!("fathom-context-code-test-{unique}"));
        fs::create_dir_all(&dir).unwrap();
        fs::write(
            dir.join("config.json"),
            r#"{"model_type":"qwen2","architectures":["Qwen2ForCausalLM"],"max_position_embeddings":32768}"#,
        )
        .unwrap();
        fs::write(dir.join("tokenizer.json"), b"{}").unwrap();
        fs::write(dir.join("model.safetensors"), b"").unwrap();

        let package = inspect_model_package(&dir).unwrap();
        let advice = recommend_context_strategy_for_package(&package);

        assert_eq!(advice.max_context_tokens, Some(32768));
        assert_eq!(advice.engine, ContextEngineRecommendation::CodeIndex);
        assert!(advice.needs_retrieval);
        assert!(advice.top_k >= 5);

        fs::remove_dir_all(dir).unwrap();
    }

    fn write_minimal_wordlevel_tokenizer(path: &Path) {
        fs::write(
            path,
            r#"{"version":"1.0","truncation":null,"padding":null,"added_tokens":[],"normalizer":null,"pre_tokenizer":{"type":"Whitespace"},"post_processor":null,"decoder":null,"model":{"type":"WordLevel","vocab":{"<unk>":0,"hello":1,"assistant":2},"unk_token":"<unk>"}}"#,
        )
        .unwrap();
    }

    fn write_minimal_onnx_embedding_config(path: &Path) {
        fs::write(
            path,
            r#"{"model_type":"bert","architectures":["BertModel"],"hidden_size":384}"#,
        )
        .unwrap();
    }

    fn write_pinned_minilm_onnx_fixture_placeholder(path: &Path) {
        let file = fs::File::create(path).unwrap();
        file.set_len(PINNED_MINILM_ONNX_EMBEDDING_BYTES).unwrap();
    }

    fn write_minimal_llama_safetensors_header(path: &Path) {
        write_minimal_llama_safetensors_header_with_lm_head(path, true);
    }

    fn write_minimal_llama_safetensors_header_with_lm_head(path: &Path, include_lm_head: bool) {
        let mut specs = vec![
            ("model.embed_tokens.weight", vec![3, 4]),
            ("model.norm.weight", vec![4]),
            ("model.layers.0.input_layernorm.weight", vec![4]),
            ("model.layers.0.post_attention_layernorm.weight", vec![4]),
            ("model.layers.0.self_attn.q_proj.weight", vec![4, 4]),
            ("model.layers.0.self_attn.k_proj.weight", vec![4, 4]),
            ("model.layers.0.self_attn.v_proj.weight", vec![4, 4]),
            ("model.layers.0.self_attn.o_proj.weight", vec![4, 4]),
            ("model.layers.0.mlp.gate_proj.weight", vec![8, 4]),
            ("model.layers.0.mlp.up_proj.weight", vec![8, 4]),
            ("model.layers.0.mlp.down_proj.weight", vec![4, 8]),
        ];
        if include_lm_head {
            specs.insert(2, ("lm_head.weight", vec![3, 4]));
        }
        let mut tensors = serde_json::Map::new();
        let mut offset = 0usize;
        for (name, shape) in specs {
            let elements = shape.iter().product::<usize>();
            let byte_len = elements * 2;
            tensors.insert(
                name.to_string(),
                serde_json::json!({"dtype":"BF16","shape":shape,"data_offsets":[offset, offset + byte_len]}),
            );
            offset += byte_len;
        }
        let header = serde_json::Value::Object(tensors).to_string();
        let mut bytes = (header.len() as u64).to_le_bytes().to_vec();
        bytes.extend_from_slice(header.as_bytes());
        bytes.resize(bytes.len() + offset, 0);
        fs::write(path, bytes).unwrap();
    }

    fn write_minimal_mistral_safetensors_header(path: &Path) {
        let specs = [
            ("model.embed_tokens.weight", vec![3, 4]),
            ("model.norm.weight", vec![4]),
            ("lm_head.weight", vec![3, 4]),
            ("model.layers.0.input_layernorm.weight", vec![4]),
            ("model.layers.0.post_attention_layernorm.weight", vec![4]),
            ("model.layers.0.self_attn.q_proj.weight", vec![4, 4]),
            ("model.layers.0.self_attn.k_proj.weight", vec![4, 4]),
            ("model.layers.0.self_attn.v_proj.weight", vec![4, 4]),
            ("model.layers.0.self_attn.o_proj.weight", vec![4, 4]),
            ("model.layers.0.mlp.gate_proj.weight", vec![8, 4]),
            ("model.layers.0.mlp.up_proj.weight", vec![8, 4]),
            ("model.layers.0.mlp.down_proj.weight", vec![4, 8]),
        ];
        let mut tensors = serde_json::Map::new();
        let mut offset = 0usize;
        for (name, shape) in specs {
            let elements = shape.iter().product::<usize>();
            let byte_len = elements * 4;
            tensors.insert(
                name.to_string(),
                serde_json::json!({"dtype":"F32","shape":shape,"data_offsets":[offset, offset + byte_len]}),
            );
            offset += byte_len;
        }
        let header = serde_json::Value::Object(tensors).to_string();
        let mut bytes = (header.len() as u64).to_le_bytes().to_vec();
        bytes.extend_from_slice(header.as_bytes());
        bytes.resize(bytes.len() + offset, 0);
        fs::write(path, bytes).unwrap();
    }

    fn write_minimal_gemma_safetensors_header(path: &Path) {
        let specs = [
            ("model.embed_tokens.weight", vec![3, 4]),
            ("model.norm.weight", vec![4]),
            ("model.layers.0.input_layernorm.weight", vec![4]),
            ("model.layers.0.post_attention_layernorm.weight", vec![4]),
            ("model.layers.0.self_attn.q_proj.weight", vec![4, 4]),
            ("model.layers.0.self_attn.k_proj.weight", vec![4, 4]),
            ("model.layers.0.self_attn.v_proj.weight", vec![4, 4]),
            ("model.layers.0.self_attn.o_proj.weight", vec![4, 4]),
            ("model.layers.0.mlp.gate_proj.weight", vec![8, 4]),
            ("model.layers.0.mlp.up_proj.weight", vec![8, 4]),
            ("model.layers.0.mlp.down_proj.weight", vec![4, 8]),
        ];
        let mut tensors = serde_json::Map::new();
        let mut offset = 0usize;
        for (name, shape) in specs {
            let elements = shape.iter().product::<usize>();
            let byte_len = elements * 2;
            tensors.insert(
                name.to_string(),
                serde_json::json!({"dtype":"BF16","shape":shape,"data_offsets":[offset, offset + byte_len]}),
            );
            offset += byte_len;
        }
        let header = serde_json::Value::Object(tensors).to_string();
        let mut bytes = (header.len() as u64).to_le_bytes().to_vec();
        bytes.extend_from_slice(header.as_bytes());
        bytes.resize(bytes.len() + offset, 0);
        fs::write(path, bytes).unwrap();
    }

    fn write_minimal_qwen2_safetensors_header(path: &Path) {
        write_minimal_qwen2_safetensors_header_with_lm_head(path, true);
    }

    fn write_minimal_qwen2_safetensors_header_with_lm_head(path: &Path, include_lm_head: bool) {
        let mut specs = vec![
            ("model.embed_tokens.weight", vec![3, 4]),
            ("model.norm.weight", vec![4]),
            ("model.layers.0.input_layernorm.weight", vec![4]),
            ("model.layers.0.post_attention_layernorm.weight", vec![4]),
            ("model.layers.0.self_attn.q_proj.weight", vec![4, 4]),
            ("model.layers.0.self_attn.q_proj.bias", vec![4]),
            ("model.layers.0.self_attn.k_proj.weight", vec![4, 4]),
            ("model.layers.0.self_attn.k_proj.bias", vec![4]),
            ("model.layers.0.self_attn.v_proj.weight", vec![4, 4]),
            ("model.layers.0.self_attn.v_proj.bias", vec![4]),
            ("model.layers.0.self_attn.o_proj.weight", vec![4, 4]),
            ("model.layers.0.mlp.gate_proj.weight", vec![8, 4]),
            ("model.layers.0.mlp.up_proj.weight", vec![8, 4]),
            ("model.layers.0.mlp.down_proj.weight", vec![4, 8]),
        ];
        if include_lm_head {
            specs.insert(2, ("lm_head.weight", vec![3, 4]));
        }
        let mut tensors = serde_json::Map::new();
        let mut offset = 0usize;
        for (name, shape) in specs {
            let elements = shape.iter().product::<usize>();
            let byte_len = elements * 2;
            tensors.insert(
                name.to_string(),
                serde_json::json!({"dtype":"BF16","shape":shape,"data_offsets":[offset, offset + byte_len]}),
            );
            offset += byte_len;
        }
        let header = serde_json::Value::Object(tensors).to_string();
        let mut bytes = (header.len() as u64).to_le_bytes().to_vec();
        bytes.extend_from_slice(header.as_bytes());
        bytes.resize(bytes.len() + offset, 0);
        fs::write(path, bytes).unwrap();
    }

    fn write_minimal_phi_safetensors_header(path: &Path) {
        let specs = [
            ("model.embed_tokens.weight", vec![3, 4]),
            ("model.final_layernorm.weight", vec![4]),
            ("model.final_layernorm.bias", vec![4]),
            ("lm_head.weight", vec![3, 4]),
            ("lm_head.bias", vec![3]),
            ("model.layers.0.input_layernorm.weight", vec![4]),
            ("model.layers.0.input_layernorm.bias", vec![4]),
            ("model.layers.0.self_attn.q_proj.weight", vec![4, 4]),
            ("model.layers.0.self_attn.q_proj.bias", vec![4]),
            ("model.layers.0.self_attn.k_proj.weight", vec![4, 4]),
            ("model.layers.0.self_attn.k_proj.bias", vec![4]),
            ("model.layers.0.self_attn.v_proj.weight", vec![4, 4]),
            ("model.layers.0.self_attn.v_proj.bias", vec![4]),
            ("model.layers.0.self_attn.dense.weight", vec![4, 4]),
            ("model.layers.0.self_attn.dense.bias", vec![4]),
            ("model.layers.0.mlp.fc1.weight", vec![8, 4]),
            ("model.layers.0.mlp.fc1.bias", vec![8]),
            ("model.layers.0.mlp.fc2.weight", vec![4, 8]),
            ("model.layers.0.mlp.fc2.bias", vec![4]),
        ];
        let mut tensors = serde_json::Map::new();
        let mut offset = 0usize;
        for (name, shape) in specs {
            let elements = shape.iter().product::<usize>();
            let byte_len = elements * 4;
            tensors.insert(
                name.to_string(),
                serde_json::json!({"dtype":"F32","shape":shape,"data_offsets":[offset, offset + byte_len]}),
            );
            offset += byte_len;
        }
        let header = serde_json::Value::Object(tensors).to_string();
        let mut bytes = (header.len() as u64).to_le_bytes().to_vec();
        bytes.extend_from_slice(header.as_bytes());
        bytes.resize(bytes.len() + offset, 0);
        fs::write(path, bytes).unwrap();
    }

    fn write_minimal_gpt2_safetensors_header(path: &Path) {
        let specs = [
            ("transformer.wte.weight", vec![3, 2]),
            ("transformer.wpe.weight", vec![4, 2]),
            ("transformer.ln_f.weight", vec![2]),
            ("transformer.ln_f.bias", vec![2]),
            ("transformer.h.0.ln_1.weight", vec![2]),
            ("transformer.h.0.ln_1.bias", vec![2]),
            ("transformer.h.0.attn.c_attn.weight", vec![2, 6]),
            ("transformer.h.0.attn.c_attn.bias", vec![6]),
            ("transformer.h.0.attn.c_proj.weight", vec![2, 2]),
            ("transformer.h.0.attn.c_proj.bias", vec![2]),
            ("transformer.h.0.ln_2.weight", vec![2]),
            ("transformer.h.0.ln_2.bias", vec![2]),
            ("transformer.h.0.mlp.c_fc.weight", vec![2, 8]),
            ("transformer.h.0.mlp.c_fc.bias", vec![8]),
            ("transformer.h.0.mlp.c_proj.weight", vec![8, 2]),
            ("transformer.h.0.mlp.c_proj.bias", vec![2]),
        ];
        write_minimal_safetensors_header(path, &specs, 4, "F32");
    }

    fn write_minimal_bert_safetensors_header(path: &Path) {
        let specs = [
            ("embeddings.word_embeddings.weight", vec![3, 4]),
            ("embeddings.position_embeddings.weight", vec![8, 4]),
            ("embeddings.token_type_embeddings.weight", vec![2, 4]),
            ("embeddings.LayerNorm.weight", vec![4]),
            ("embeddings.LayerNorm.bias", vec![4]),
            ("encoder.layer.0.attention.self.query.weight", vec![4, 4]),
            ("encoder.layer.0.attention.self.query.bias", vec![4]),
            ("encoder.layer.0.attention.self.key.weight", vec![4, 4]),
            ("encoder.layer.0.attention.self.key.bias", vec![4]),
            ("encoder.layer.0.attention.self.value.weight", vec![4, 4]),
            ("encoder.layer.0.attention.self.value.bias", vec![4]),
            ("encoder.layer.0.attention.output.dense.weight", vec![4, 4]),
            ("encoder.layer.0.attention.output.dense.bias", vec![4]),
            ("encoder.layer.0.attention.output.LayerNorm.weight", vec![4]),
            ("encoder.layer.0.attention.output.LayerNorm.bias", vec![4]),
            ("encoder.layer.0.intermediate.dense.weight", vec![8, 4]),
            ("encoder.layer.0.intermediate.dense.bias", vec![8]),
            ("encoder.layer.0.output.dense.weight", vec![4, 8]),
            ("encoder.layer.0.output.dense.bias", vec![4]),
            ("encoder.layer.0.output.LayerNorm.weight", vec![4]),
            ("encoder.layer.0.output.LayerNorm.bias", vec![4]),
        ];
        write_minimal_safetensors_header(path, &specs, 4, "F32");
    }

    fn write_minimal_safetensors_header(
        path: &Path,
        specs: &[(&str, Vec<usize>)],
        bytes_per_element: usize,
        dtype: &str,
    ) {
        let mut tensors = serde_json::Map::new();
        let mut offset = 0usize;
        for (name, shape) in specs {
            let elements = shape.iter().product::<usize>();
            let byte_len = elements * bytes_per_element;
            tensors.insert(
                (*name).to_string(),
                serde_json::json!({"dtype":dtype,"shape":shape,"data_offsets":[offset, offset + byte_len]}),
            );
            offset += byte_len;
        }
        let header = serde_json::Value::Object(tensors).to_string();
        let mut bytes = (header.len() as u64).to_le_bytes().to_vec();
        bytes.extend_from_slice(header.as_bytes());
        bytes.resize(bytes.len() + offset, 0);
        fs::write(path, bytes).unwrap();
    }

    #[test]
    fn refuses_fake_gpt2_metadata_without_loadable_artifacts() {
        let unique = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_nanos();
        let dir = std::env::temp_dir().join(format!("fathom-fake-gpt2-capability-test-{unique}"));
        fs::create_dir_all(&dir).unwrap();
        fs::write(
            dir.join("config.json"),
            r#"{"model_type":"gpt2","architectures":["GPT2LMHeadModel"]}"#,
        )
        .unwrap();
        fs::write(dir.join("tokenizer.json"), b"{}").unwrap();
        fs::write(dir.join("model.safetensors"), b"").unwrap();

        let package = inspect_model_package(&dir).unwrap();
        assert!(!is_candle_gpt2_supported_package(&package));
        let report = capability_report_for_package(package, &current_machine_profile());
        assert!(!report.runnable);
        assert_eq!(report.best_status, CapabilityStatus::Planned);

        fs::remove_dir_all(dir).unwrap();
    }

    #[test]
    fn maps_bert_safetensors_package_to_default_candle_embedding_lane_when_artifacts_validate() {
        let unique = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_nanos();
        let dir = std::env::temp_dir().join(format!("fathom-bert-embed-capability-test-{unique}"));
        fs::create_dir_all(&dir).unwrap();
        fs::write(
            dir.join("config.json"),
            r#"{"model_type":"bert","architectures":["BertModel"],"vocab_size":3,"hidden_size":4,"intermediate_size":8,"num_hidden_layers":1,"num_attention_heads":1,"max_position_embeddings":8,"type_vocab_size":2,"layer_norm_eps":1e-12}"#,
        )
        .unwrap();
        write_minimal_wordlevel_tokenizer(&dir.join("tokenizer.json"));
        write_minimal_bert_safetensors_header(&dir.join("model.safetensors"));

        let package = inspect_model_package(&dir).unwrap();
        validate_candle_bert_embedding_supported_package(&package).unwrap();
        let status = embedding_model_status_for_package(&package).unwrap();
        assert_eq!(status.task, ModelTaskKind::TextEmbedding);
        assert_eq!(status.runtime_lane, "candle-bert-embeddings");
        assert!(status.runnable);
        assert_eq!(status.embedding_dimension, Some(4));
        let report = capability_report_for_package(package, &current_machine_profile());
        assert!(report.runnable);
        assert_eq!(report.best_status, CapabilityStatus::Runnable);
        assert!(report
            .matching_lanes
            .iter()
            .any(|lane| lane.kind == BackendLaneKind::LocalEmbeddingsRetrieval));
        assert!(report.summary.contains("embedding-only"));

        fs::remove_dir_all(dir).unwrap();
    }

    #[test]
    fn refuses_fake_bert_embedding_metadata_without_loadable_artifacts() {
        let unique = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_nanos();
        let dir = std::env::temp_dir().join(format!("fathom-fake-bert-embed-test-{unique}"));
        fs::create_dir_all(&dir).unwrap();
        fs::write(
            dir.join("config.json"),
            r#"{"model_type":"bert","architectures":["BertModel"],"vocab_size":3,"hidden_size":4,"intermediate_size":8,"num_hidden_layers":1,"num_attention_heads":1,"max_position_embeddings":8,"type_vocab_size":2}"#,
        )
        .unwrap();
        write_minimal_wordlevel_tokenizer(&dir.join("tokenizer.json"));
        fs::write(dir.join("model.safetensors"), b"").unwrap();

        let package = inspect_model_package(&dir).unwrap();
        assert!(!is_candle_bert_embedding_supported_package(&package));
        let status = embedding_model_status_for_package(&package).unwrap();
        assert!(!status.runnable);
        assert!(!status.blockers.is_empty());
        let report = capability_report_for_package(package, &current_machine_profile());
        assert!(!report.runnable);

        fs::remove_dir_all(dir).unwrap();
    }

    #[cfg(unix)]
    #[test]
    fn refuses_runnable_safetensors_package_when_weights_symlink_escape_root() {
        let unique = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_nanos();
        let dir = std::env::temp_dir().join(format!("fathom-gpt2-weight-escape-test-{unique}"));
        let outside = std::env::temp_dir().join(format!(
            "fathom-gpt2-weight-escape-outside-{unique}.safetensors"
        ));
        fs::create_dir_all(&dir).unwrap();
        fs::write(
            dir.join("config.json"),
            r#"{"model_type":"gpt2","architectures":["GPT2LMHeadModel"],"vocab_size":3,"n_positions":4,"n_embd":2,"n_layer":1,"n_head":1}"#,
        )
        .unwrap();
        write_minimal_wordlevel_tokenizer(&dir.join("tokenizer.json"));
        write_minimal_gpt2_safetensors_header(&outside);
        std::os::unix::fs::symlink(&outside, dir.join("model.safetensors")).unwrap();

        let package = inspect_model_package(&dir).unwrap();
        let error = validate_candle_gpt2_supported_package(&package)
            .unwrap_err()
            .to_string();
        assert!(error.contains(ARTIFACT_PATH_ESCAPE_BLOCKER));
        let report = capability_report_for_package(package, &current_machine_profile());
        assert!(!report.runnable);
        assert_ne!(report.best_status, CapabilityStatus::Runnable);

        fs::remove_dir_all(dir).unwrap();
        fs::remove_file(outside).unwrap();
    }

    #[cfg(unix)]
    #[test]
    fn refuses_runnable_safetensors_package_when_config_or_tokenizer_symlink_escape_root() {
        for escaped_filename in ["config.json", "tokenizer.json"] {
            let unique = SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .unwrap()
                .as_nanos();
            let dir = std::env::temp_dir().join(format!(
                "fathom-gpt2-metadata-escape-test-{escaped_filename}-{unique}"
            ));
            let outside = std::env::temp_dir().join(format!(
                "fathom-gpt2-metadata-escape-outside-{escaped_filename}-{unique}"
            ));
            fs::create_dir_all(&dir).unwrap();
            fs::write(
                dir.join("config.json"),
                r#"{"model_type":"gpt2","architectures":["GPT2LMHeadModel"],"vocab_size":3,"n_positions":4,"n_embd":2,"n_layer":1,"n_head":1}"#,
            )
            .unwrap();
            write_minimal_wordlevel_tokenizer(&dir.join("tokenizer.json"));
            write_minimal_gpt2_safetensors_header(&dir.join("model.safetensors"));
            if escaped_filename == "config.json" {
                fs::write(
                    &outside,
                    r#"{"model_type":"gpt2","architectures":["GPT2LMHeadModel"],"vocab_size":3,"n_positions":4,"n_embd":2,"n_layer":1,"n_head":1}"#,
                )
                .unwrap();
            } else {
                write_minimal_wordlevel_tokenizer(&outside);
            }
            fs::remove_file(dir.join(escaped_filename)).unwrap();
            std::os::unix::fs::symlink(&outside, dir.join(escaped_filename)).unwrap();

            let package = inspect_model_package(&dir).unwrap();
            let error = validate_candle_gpt2_supported_package(&package)
                .unwrap_err()
                .to_string();
            assert!(error.contains(ARTIFACT_PATH_ESCAPE_BLOCKER));
            let report = capability_report_for_package(package, &current_machine_profile());
            assert!(!report.runnable);
            assert_ne!(report.best_status, CapabilityStatus::Runnable);

            fs::remove_dir_all(dir).unwrap();
            fs::remove_file(outside).unwrap();
        }
    }

    #[test]
    fn maps_gpt2_safetensors_package_to_narrow_candle_runnable_when_artifacts_validate() {
        let unique = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_nanos();
        let dir = std::env::temp_dir().join(format!("fathom-gpt2-capability-test-{unique}"));
        fs::create_dir_all(&dir).unwrap();
        fs::write(
            dir.join("config.json"),
            r#"{"model_type":"gpt2","architectures":["GPT2LMHeadModel"],"vocab_size":3,"n_positions":4,"n_embd":2,"n_layer":1,"n_head":1}"#,
        )
        .unwrap();
        write_minimal_wordlevel_tokenizer(&dir.join("tokenizer.json"));
        write_minimal_gpt2_safetensors_header(&dir.join("model.safetensors"));

        let package = inspect_model_package(&dir).unwrap();
        validate_candle_gpt2_supported_package(&package).unwrap();
        let report = capability_report_for_package(package, &current_machine_profile());
        assert!(report.runnable);
        assert_eq!(report.best_status, CapabilityStatus::Runnable);
        assert!(report.summary.contains("custom Rust GPT-2"));

        fs::remove_dir_all(dir).unwrap();
    }

    #[test]
    fn maps_llama_safetensors_package_to_narrow_candle_runnable_when_artifacts_validate() {
        let unique = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_nanos();
        let dir = std::env::temp_dir().join(format!("fathom-llama-capability-test-{unique}"));
        fs::create_dir_all(&dir).unwrap();
        fs::write(
            dir.join("config.json"),
            r#"{"model_type":"llama","architectures":["LlamaForCausalLM"],"vocab_size":3,"hidden_size":4,"intermediate_size":8,"num_hidden_layers":1,"num_attention_heads":1,"num_key_value_heads":1,"max_position_embeddings":8,"rms_norm_eps":1e-5}"#,
        )
        .unwrap();
        write_minimal_wordlevel_tokenizer(&dir.join("tokenizer.json"));
        write_minimal_llama_safetensors_header(&dir.join("model.safetensors"));

        let package = inspect_model_package(&dir).unwrap();
        validate_candle_llama_supported_package(&package).unwrap();
        let report = capability_report_for_package(package, &current_machine_profile());
        assert!(report.runnable);
        assert_eq!(report.best_status, CapabilityStatus::Runnable);
        assert!(report.summary.contains("custom Rust Llama"));

        fs::remove_dir_all(dir).unwrap();
    }

    #[test]
    fn maps_tied_llama_safetensors_without_lm_head_when_config_allows() {
        let unique = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_nanos();
        let dir = std::env::temp_dir().join(format!("fathom-llama-tied-capability-test-{unique}"));
        fs::create_dir_all(&dir).unwrap();
        fs::write(
            dir.join("config.json"),
            r#"{"model_type":"llama","architectures":["LlamaForCausalLM"],"vocab_size":3,"hidden_size":4,"intermediate_size":8,"num_hidden_layers":1,"num_attention_heads":1,"num_key_value_heads":1,"max_position_embeddings":8,"rms_norm_eps":1e-5,"tie_word_embeddings":true}"#,
        )
        .unwrap();
        write_minimal_wordlevel_tokenizer(&dir.join("tokenizer.json"));
        write_minimal_llama_safetensors_header_with_lm_head(&dir.join("model.safetensors"), false);

        let package = inspect_model_package(&dir).unwrap();
        validate_candle_llama_supported_package(&package).unwrap();
        let report = capability_report_for_package(package, &current_machine_profile());
        assert!(report.runnable);
        assert_eq!(report.best_status, CapabilityStatus::Runnable);
        assert!(report.summary.contains("custom Rust Llama"));

        fs::remove_dir_all(dir).unwrap();
    }

    #[test]
    fn rejects_untied_llama_safetensors_without_lm_head() {
        let unique = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_nanos();
        let dir = std::env::temp_dir().join(format!(
            "fathom-llama-untied-no-head-capability-test-{unique}"
        ));
        fs::create_dir_all(&dir).unwrap();
        fs::write(
            dir.join("config.json"),
            r#"{"model_type":"llama","architectures":["LlamaForCausalLM"],"vocab_size":3,"hidden_size":4,"intermediate_size":8,"num_hidden_layers":1,"num_attention_heads":1,"num_key_value_heads":1,"max_position_embeddings":8,"rms_norm_eps":1e-5,"tie_word_embeddings":false}"#,
        )
        .unwrap();
        write_minimal_wordlevel_tokenizer(&dir.join("tokenizer.json"));
        write_minimal_llama_safetensors_header_with_lm_head(&dir.join("model.safetensors"), false);

        let package = inspect_model_package(&dir).unwrap();
        let error = validate_candle_llama_supported_package(&package).unwrap_err();
        assert!(error.to_string().contains("missing lm_head.weight"));

        fs::remove_dir_all(dir).unwrap();
    }

    #[test]
    fn refuses_fake_llama_metadata_without_loadable_artifacts() {
        let unique = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_nanos();
        let dir = std::env::temp_dir().join(format!("fathom-fake-llama-capability-test-{unique}"));
        fs::create_dir_all(&dir).unwrap();
        fs::write(
            dir.join("config.json"),
            r#"{"model_type":"llama","architectures":["LlamaForCausalLM"],"vocab_size":3,"hidden_size":4,"intermediate_size":8,"num_hidden_layers":1,"num_attention_heads":1,"max_position_embeddings":8}"#,
        )
        .unwrap();
        write_minimal_wordlevel_tokenizer(&dir.join("tokenizer.json"));
        fs::write(dir.join("model.safetensors"), b"").unwrap();

        let package = inspect_model_package(&dir).unwrap();
        assert!(!is_candle_llama_supported_package(&package));
        let report = capability_report_for_package(package, &current_machine_profile());
        assert!(!report.runnable);
        assert_eq!(report.best_status, CapabilityStatus::Planned);

        fs::remove_dir_all(dir).unwrap();
    }

    #[test]
    fn maps_qwen2_safetensors_package_to_narrow_candle_runnable_when_artifacts_validate() {
        let unique = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_nanos();
        let dir = std::env::temp_dir().join(format!("fathom-qwen2-capability-test-{unique}"));
        fs::create_dir_all(&dir).unwrap();
        fs::write(
            dir.join("config.json"),
            r#"{"model_type":"qwen2","architectures":["Qwen2ForCausalLM"],"vocab_size":3,"hidden_size":4,"intermediate_size":8,"num_hidden_layers":1,"num_attention_heads":1,"num_key_value_heads":1,"max_position_embeddings":8,"sliding_window":8,"max_window_layers":1,"tie_word_embeddings":false,"rope_theta":10000.0,"rms_norm_eps":1e-6,"use_sliding_window":false,"hidden_act":"silu","eos_token_id":2}"#,
        )
        .unwrap();
        write_minimal_wordlevel_tokenizer(&dir.join("tokenizer.json"));
        write_minimal_qwen2_safetensors_header(&dir.join("model.safetensors"));

        let package = inspect_model_package(&dir).unwrap();
        validate_candle_qwen2_supported_package(&package).unwrap();
        let report = capability_report_for_package(package, &current_machine_profile());
        assert!(report.runnable);
        assert_eq!(report.best_status, CapabilityStatus::Runnable);
        assert!(report.summary.contains("custom Rust Qwen2"));

        fs::remove_dir_all(dir).unwrap();
    }

    #[test]
    fn maps_tied_qwen2_safetensors_without_lm_head_when_config_allows() {
        let unique = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_nanos();
        let dir = std::env::temp_dir().join(format!("fathom-qwen2-tied-capability-test-{unique}"));
        fs::create_dir_all(&dir).unwrap();
        fs::write(
            dir.join("config.json"),
            r#"{"model_type":"qwen2","architectures":["Qwen2ForCausalLM"],"vocab_size":3,"hidden_size":4,"intermediate_size":8,"num_hidden_layers":1,"num_attention_heads":1,"num_key_value_heads":1,"max_position_embeddings":8,"sliding_window":8,"max_window_layers":1,"tie_word_embeddings":true,"rope_theta":10000.0,"rms_norm_eps":1e-6,"use_sliding_window":false,"hidden_act":"silu","eos_token_id":2}"#,
        )
        .unwrap();
        write_minimal_wordlevel_tokenizer(&dir.join("tokenizer.json"));
        write_minimal_qwen2_safetensors_header_with_lm_head(&dir.join("model.safetensors"), false);

        let package = inspect_model_package(&dir).unwrap();
        validate_candle_qwen2_supported_package(&package).unwrap();
        let report = capability_report_for_package(package, &current_machine_profile());
        assert!(report.runnable);
        assert_eq!(report.best_status, CapabilityStatus::Runnable);
        assert!(report.summary.contains("custom Rust Qwen2"));

        fs::remove_dir_all(dir).unwrap();
    }

    #[test]
    fn rejects_untied_qwen2_safetensors_without_lm_head() {
        let unique = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_nanos();
        let dir = std::env::temp_dir().join(format!(
            "fathom-qwen2-untied-no-head-capability-test-{unique}"
        ));
        fs::create_dir_all(&dir).unwrap();
        fs::write(
            dir.join("config.json"),
            r#"{"model_type":"qwen2","architectures":["Qwen2ForCausalLM"],"vocab_size":3,"hidden_size":4,"intermediate_size":8,"num_hidden_layers":1,"num_attention_heads":1,"num_key_value_heads":1,"max_position_embeddings":8,"sliding_window":8,"max_window_layers":1,"tie_word_embeddings":false,"rope_theta":10000.0,"rms_norm_eps":1e-6,"use_sliding_window":false,"hidden_act":"silu","eos_token_id":2}"#,
        )
        .unwrap();
        write_minimal_wordlevel_tokenizer(&dir.join("tokenizer.json"));
        write_minimal_qwen2_safetensors_header_with_lm_head(&dir.join("model.safetensors"), false);

        let package = inspect_model_package(&dir).unwrap();
        let error = validate_candle_qwen2_supported_package(&package).unwrap_err();
        assert!(error.to_string().contains("missing lm_head.weight"));
        assert!(!is_candle_qwen2_supported_package(&package));

        fs::remove_dir_all(dir).unwrap();
    }

    #[test]
    fn refuses_fake_qwen2_metadata_without_loadable_artifacts() {
        let unique = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_nanos();
        let dir = std::env::temp_dir().join(format!("fathom-fake-qwen2-capability-test-{unique}"));
        fs::create_dir_all(&dir).unwrap();
        fs::write(
            dir.join("config.json"),
            r#"{"model_type":"qwen2","architectures":["Qwen2ForCausalLM"],"vocab_size":3,"hidden_size":4,"intermediate_size":8,"num_hidden_layers":1,"num_attention_heads":1,"num_key_value_heads":1,"max_position_embeddings":8,"sliding_window":8,"max_window_layers":1,"tie_word_embeddings":false,"rope_theta":10000.0,"rms_norm_eps":1e-6,"use_sliding_window":false,"hidden_act":"silu"}"#,
        )
        .unwrap();
        write_minimal_wordlevel_tokenizer(&dir.join("tokenizer.json"));
        fs::write(dir.join("model.safetensors"), b"").unwrap();

        let package = inspect_model_package(&dir).unwrap();
        assert!(!is_candle_qwen2_supported_package(&package));
        let report = capability_report_for_package(package, &current_machine_profile());
        assert!(!report.runnable);
        assert_eq!(report.best_status, CapabilityStatus::Planned);

        fs::remove_dir_all(dir).unwrap();
    }

    #[test]
    fn maps_phi_safetensors_package_to_narrow_candle_runnable_when_artifacts_validate() {
        let unique = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_nanos();
        let dir = std::env::temp_dir().join(format!("fathom-phi-capability-test-{unique}"));
        fs::create_dir_all(&dir).unwrap();
        fs::write(
            dir.join("config.json"),
            r#"{"model_type":"phi","architectures":["PhiForCausalLM"],"vocab_size":3,"hidden_size":4,"intermediate_size":8,"num_hidden_layers":1,"num_attention_heads":1,"num_key_value_heads":1,"max_position_embeddings":8,"layer_norm_eps":1e-5,"tie_word_embeddings":false,"rope_theta":10000.0,"partial_rotary_factor":0.5,"qk_layernorm":false,"hidden_act":"gelu","eos_token_id":2}"#,
        )
        .unwrap();
        write_minimal_wordlevel_tokenizer(&dir.join("tokenizer.json"));
        write_minimal_phi_safetensors_header(&dir.join("model.safetensors"));

        let package = inspect_model_package(&dir).unwrap();
        validate_candle_phi_supported_package(&package).unwrap();
        let report = capability_report_for_package(package, &current_machine_profile());
        assert!(report.runnable);
        assert_eq!(report.best_status, CapabilityStatus::Runnable);
        assert!(report.summary.contains("custom Rust Phi"));

        fs::remove_dir_all(dir).unwrap();
    }

    #[test]
    fn refuses_fake_phi_metadata_without_loadable_artifacts() {
        let unique = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_nanos();
        let dir = std::env::temp_dir().join(format!("fathom-fake-phi-capability-test-{unique}"));
        fs::create_dir_all(&dir).unwrap();
        fs::write(
            dir.join("config.json"),
            r#"{"model_type":"phi","architectures":["PhiForCausalLM"],"vocab_size":3,"hidden_size":4,"intermediate_size":8,"num_hidden_layers":1,"num_attention_heads":1,"num_key_value_heads":1,"max_position_embeddings":8,"layer_norm_eps":1e-5,"tie_word_embeddings":false,"rope_theta":10000.0,"partial_rotary_factor":0.5,"qk_layernorm":false,"hidden_act":"gelu"}"#,
        )
        .unwrap();
        write_minimal_wordlevel_tokenizer(&dir.join("tokenizer.json"));
        fs::write(dir.join("model.safetensors"), b"").unwrap();

        let package = inspect_model_package(&dir).unwrap();
        assert!(!is_candle_phi_supported_package(&package));
        let report = capability_report_for_package(package, &current_machine_profile());
        assert!(!report.runnable);
        assert_eq!(report.best_status, CapabilityStatus::Planned);

        fs::remove_dir_all(dir).unwrap();
    }

    #[test]
    fn maps_mistral_safetensors_package_to_narrow_candle_runnable_when_artifacts_validate() {
        let unique = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_nanos();
        let dir = std::env::temp_dir().join(format!("fathom-mistral-capability-test-{unique}"));
        fs::create_dir_all(&dir).unwrap();
        fs::write(
            dir.join("config.json"),
            r#"{"model_type":"mistral","architectures":["MistralForCausalLM"],"vocab_size":3,"hidden_size":4,"intermediate_size":8,"num_hidden_layers":1,"num_attention_heads":1,"num_key_value_heads":1,"max_position_embeddings":8,"rms_norm_eps":1e-6,"rope_theta":10000.0,"sliding_window":8,"hidden_act":"silu","eos_token_id":2}"#,
        )
        .unwrap();
        write_minimal_wordlevel_tokenizer(&dir.join("tokenizer.json"));
        write_minimal_mistral_safetensors_header(&dir.join("model.safetensors"));

        let package = inspect_model_package(&dir).unwrap();
        validate_candle_mistral_supported_package(&package).unwrap();
        let report = capability_report_for_package(package, &current_machine_profile());
        assert!(report.runnable);
        assert_eq!(report.best_status, CapabilityStatus::Runnable);
        assert!(report.summary.contains("custom Rust Mistral"));

        fs::remove_dir_all(dir).unwrap();
    }

    #[test]
    fn refuses_fake_mistral_metadata_without_loadable_artifacts() {
        let unique = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_nanos();
        let dir =
            std::env::temp_dir().join(format!("fathom-fake-mistral-capability-test-{unique}"));
        fs::create_dir_all(&dir).unwrap();
        fs::write(
            dir.join("config.json"),
            r#"{"model_type":"mistral","architectures":["MistralForCausalLM"],"vocab_size":3,"hidden_size":4,"intermediate_size":8,"num_hidden_layers":1,"num_attention_heads":1,"num_key_value_heads":1,"max_position_embeddings":8,"rms_norm_eps":1e-6,"rope_theta":10000.0,"sliding_window":8,"hidden_act":"silu"}"#,
        )
        .unwrap();
        write_minimal_wordlevel_tokenizer(&dir.join("tokenizer.json"));
        fs::write(dir.join("model.safetensors"), b"").unwrap();

        let package = inspect_model_package(&dir).unwrap();
        assert!(!is_candle_mistral_supported_package(&package));
        let report = capability_report_for_package(package, &current_machine_profile());
        assert!(!report.runnable);
        assert_eq!(report.best_status, CapabilityStatus::Planned);

        fs::remove_dir_all(dir).unwrap();
    }

    #[test]
    fn maps_gemma_safetensors_package_to_narrow_candle_runnable_when_artifacts_validate() {
        let unique = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_nanos();
        let dir = std::env::temp_dir().join(format!("fathom-gemma-capability-test-{unique}"));
        fs::create_dir_all(&dir).unwrap();
        fs::write(
            dir.join("config.json"),
            r#"{"model_type":"gemma","architectures":["GemmaForCausalLM"],"vocab_size":3,"hidden_size":4,"intermediate_size":8,"num_hidden_layers":1,"num_attention_heads":1,"num_key_value_heads":1,"head_dim":4,"max_position_embeddings":8,"rms_norm_eps":1e-6,"rope_theta":10000.0,"attention_bias":false,"hidden_act":"gelu","eos_token_id":2}"#,
        )
        .unwrap();
        write_minimal_wordlevel_tokenizer(&dir.join("tokenizer.json"));
        write_minimal_gemma_safetensors_header(&dir.join("model.safetensors"));

        let package = inspect_model_package(&dir).unwrap();
        validate_candle_gemma_supported_package(&package).unwrap();
        let report = capability_report_for_package(package, &current_machine_profile());
        assert!(report.runnable);
        assert_eq!(report.best_status, CapabilityStatus::Runnable);
        assert!(report.summary.contains("custom Rust Gemma"));

        fs::remove_dir_all(dir).unwrap();
    }

    #[test]
    fn refuses_fake_gemma_metadata_without_loadable_artifacts() {
        let unique = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_nanos();
        let dir = std::env::temp_dir().join(format!("fathom-fake-gemma-capability-test-{unique}"));
        fs::create_dir_all(&dir).unwrap();
        fs::write(
            dir.join("config.json"),
            r#"{"model_type":"gemma","architectures":["GemmaForCausalLM"],"vocab_size":3,"hidden_size":4,"intermediate_size":8,"num_hidden_layers":1,"num_attention_heads":1,"num_key_value_heads":1,"head_dim":4,"max_position_embeddings":8,"rms_norm_eps":1e-6,"rope_theta":10000.0,"attention_bias":false,"hidden_act":"gelu"}"#,
        )
        .unwrap();
        write_minimal_wordlevel_tokenizer(&dir.join("tokenizer.json"));
        fs::write(dir.join("model.safetensors"), b"").unwrap();

        let package = inspect_model_package(&dir).unwrap();
        assert!(!is_candle_gemma_supported_package(&package));
        let report = capability_report_for_package(package, &current_machine_profile());
        assert!(!report.runnable);
        assert_eq!(report.best_status, CapabilityStatus::Planned);

        fs::remove_dir_all(dir).unwrap();
    }

    #[test]
    fn mean_pools_last_hidden_state_over_attention_mask_and_normalizes() {
        let hidden_states = vec![1.0, 0.0, 3.0, 0.0, 99.0, 99.0, 0.0, 2.0, 0.0, 4.0, 0.0, 6.0];
        let attention_mask = vec![1, 1, 0, 1, 1, 1];
        let vectors =
            mean_pool_last_hidden_state(&hidden_states, &[2, 3, 2], &attention_mask, 2, 3, 2, true)
                .unwrap();

        assert_eq!(vectors.len(), 2);
        assert!((vectors[0].values[0] - 1.0).abs() < 1e-6);
        assert!(vectors[0].values[1].abs() < 1e-6);
        assert!(vectors[1].values[0].abs() < 1e-6);
        assert!((vectors[1].values[1] - 1.0).abs() < 1e-6);
    }

    #[test]
    fn mean_pool_rejects_dimension_mismatch() {
        let error = mean_pool_last_hidden_state(&[1.0, 2.0], &[1, 1, 2], &[1], 1, 1, 384, false)
            .unwrap_err()
            .to_string();
        assert!(error.contains("did not match expected"));
    }

    fn vector_test_chunk(id: &str, document_id: &str, text: &str) -> VectorIndexChunk {
        VectorIndexChunk {
            id: id.to_string(),
            document_id: document_id.to_string(),
            text: text.to_string(),
            byte_start: 0,
            byte_end: text.len(),
        }
    }

    #[test]
    fn vector_index_returns_cosine_top_k_for_hard_coded_vectors() {
        let mut index = VectorIndex::new("notes", "fixture-embedder", 3).unwrap();
        index
            .add_chunk(
                vector_test_chunk("rust", "doc-a", "Rust ownership and lifetimes"),
                EmbeddingVector::new(vec![1.0, 0.0, 0.0]).unwrap(),
            )
            .unwrap();
        index
            .add_chunk(
                vector_test_chunk("python", "doc-b", "Python scripting notes"),
                EmbeddingVector::new(vec![0.0, 1.0, 0.0]).unwrap(),
            )
            .unwrap();
        index
            .add_chunk(
                vector_test_chunk("rust-async", "doc-a", "Async Rust runtime notes"),
                EmbeddingVector::new(vec![0.8, 0.2, 0.0]).unwrap(),
            )
            .unwrap();

        let hits = index
            .search(
                &EmbeddingVector::new(vec![1.0, 0.1, 0.0]).unwrap(),
                2,
                VectorSearchMetric::Cosine,
            )
            .unwrap();

        assert_eq!(hits.len(), 2);
        assert_eq!(hits[0].chunk.id, "rust");
        assert_eq!(hits[1].chunk.id, "rust-async");
        assert!(hits[0].score > hits[1].score);
        assert_eq!(index.summary().document_count, 2);
        assert_eq!(index.summary().chunk_count, 3);
        assert_eq!(index.summary().schema_version, VECTOR_INDEX_SCHEMA_VERSION);
    }

    #[test]
    fn vector_index_dot_product_top_k_uses_raw_magnitude() {
        let mut index = VectorIndex::new("scores", "fixture-embedder", 2).unwrap();
        index
            .add_chunk(
                vector_test_chunk("small-same-direction", "doc-a", "small"),
                EmbeddingVector::new(vec![1.0, 0.0]).unwrap(),
            )
            .unwrap();
        index
            .add_chunk(
                vector_test_chunk("large-same-direction", "doc-b", "large"),
                EmbeddingVector::new(vec![4.0, 0.0]).unwrap(),
            )
            .unwrap();
        index
            .add_chunk(
                vector_test_chunk("orthogonal", "doc-c", "other"),
                EmbeddingVector::new(vec![0.0, 5.0]).unwrap(),
            )
            .unwrap();

        let hits = index
            .search(
                &EmbeddingVector::new(vec![1.0, 0.0]).unwrap(),
                2,
                VectorSearchMetric::DotProduct,
            )
            .unwrap();

        assert_eq!(hits[0].chunk.id, "large-same-direction");
        assert_eq!(hits[0].score, 4.0);
        assert_eq!(hits[1].chunk.id, "small-same-direction");
    }

    #[test]
    fn vector_index_rejects_dimension_mismatches_and_bad_schema() {
        let mut index = VectorIndex::new("bad_dims", "fixture-embedder", 2).unwrap();
        let err = index
            .add_chunk(
                vector_test_chunk("wrong", "doc-a", "bad"),
                EmbeddingVector::new(vec![1.0, 0.0, 0.0]).unwrap(),
            )
            .unwrap_err();
        assert!(err.to_string().contains("does not match index dimension"));

        index
            .add_chunk(
                vector_test_chunk("ok", "doc-a", "good"),
                EmbeddingVector::new(vec![1.0, 0.0]).unwrap(),
            )
            .unwrap();
        let err = index
            .search(
                &EmbeddingVector::new(vec![1.0, 0.0, 0.0]).unwrap(),
                1,
                VectorSearchMetric::Cosine,
            )
            .unwrap_err();
        assert!(err.to_string().contains("query dimension"));

        index.schema_version = 999;
        let err = index
            .search(
                &EmbeddingVector::new(vec![1.0, 0.0]).unwrap(),
                1,
                VectorSearchMetric::Cosine,
            )
            .unwrap_err();
        assert!(err
            .to_string()
            .contains("unsupported vector index schema version"));
    }

    #[test]
    fn vector_index_enforces_small_developer_index_limits() {
        let err = VectorIndex::new(
            "too_wide",
            "fixture-embedder",
            MAX_VECTOR_INDEX_DIMENSION + 1,
        )
        .unwrap_err();
        assert!(err.to_string().contains("exceeds maximum"));

        let mut index = VectorIndex::new("notes", "fixture-embedder", 2).unwrap();
        let oversized_text = "x".repeat(MAX_VECTOR_CHUNK_TEXT_CHARS + 1);
        let err = index
            .add_chunk(
                vector_test_chunk("huge", "doc-a", &oversized_text),
                EmbeddingVector::new(vec![1.0, 0.0]).unwrap(),
            )
            .unwrap_err();
        assert!(err.to_string().contains("chunk text"));
        assert_eq!(index.summary().chunk_count, 0);
    }

    #[test]
    fn vector_index_persists_under_state_dir_and_loads_back() {
        let unique = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_nanos();
        let state_dir = std::env::temp_dir().join(format!("fathom-vector-index-state-{unique}"));
        fs::create_dir_all(&state_dir).unwrap();

        let mut index = VectorIndex::new("local_notes", "fixture-embedder", 2).unwrap();
        index
            .add_chunk(
                vector_test_chunk("a", "doc-a", "first chunk"),
                EmbeddingVector::new(vec![0.9, 0.1]).unwrap(),
            )
            .unwrap();
        let path = index.save_to_state_dir(&state_dir).unwrap();
        assert!(path.starts_with(&state_dir));
        assert_eq!(path.file_name().unwrap(), "local_notes.json");

        let loaded = VectorIndex::load_from_state_dir(&state_dir, "local_notes").unwrap();
        assert_eq!(loaded, index);
        let hits = loaded
            .search(
                &EmbeddingVector::new(vec![1.0, 0.0]).unwrap(),
                1,
                VectorSearchMetric::Cosine,
            )
            .unwrap();
        assert_eq!(hits[0].chunk.id, "a");

        assert!(VectorIndex::new("../escape", "fixture-embedder", 2).is_err());
        assert!(VectorIndex::load_from_state_dir(&state_dir, "../escape").is_err());

        fs::remove_dir_all(state_dir).unwrap();
    }

    #[test]
    fn detects_onnx_embedding_package_as_metadata_only_not_runnable() {
        let unique = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_nanos();
        let dir = std::env::temp_dir().join(format!("fathom-onnx-embedding-test-{unique}"));
        fs::create_dir_all(&dir).unwrap();
        fs::write(dir.join("model.onnx"), b"fake onnx fixture bytes").unwrap();
        fs::write(
            dir.join("config.json"),
            r#"{"model_type":"bert","architectures":["BertModel"],"hidden_size":384}"#,
        )
        .unwrap();
        write_minimal_wordlevel_tokenizer(&dir.join("tokenizer.json"));

        let package = inspect_model_package(&dir).unwrap();
        let embedding = embedding_model_status_for_package(&package).unwrap();
        assert_eq!(embedding.task, ModelTaskKind::TextEmbedding);
        assert_eq!(embedding.status, CapabilityStatus::Planned);
        assert_eq!(embedding.embedding_dimension, Some(384));
        assert_eq!(
            embedding.runtime_installed,
            cfg!(feature = "onnx-embeddings-ort")
        );
        assert!(!embedding.runnable);
        assert!(embedding.summary.contains("not ONNX chat/LLM support"));

        let report = capability_report_for_package(package, &current_machine_profile());
        assert!(!report.runnable);
        assert_eq!(report.best_status, CapabilityStatus::Planned);
        assert!(report
            .matching_lanes
            .iter()
            .any(|lane| lane.kind == BackendLaneKind::LocalEmbeddingsRetrieval));
        assert!(
            !report
                .matching_lanes
                .iter()
                .any(|lane| lane.kind == BackendLaneKind::SafeTensorsHf),
            "ONNX embedding metadata files must not map the package to the SafeTensors/HF lane"
        );

        fs::remove_dir_all(dir).unwrap();
    }

    #[test]
    fn pinned_onnx_embedding_fixture_policy_preserves_feature_gated_status() {
        let unique = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_nanos();
        let dir =
            std::env::temp_dir().join(format!("fathom-pinned-onnx-embedding-policy-test-{unique}"));
        fs::create_dir_all(&dir).unwrap();
        write_minimal_onnx_embedding_config(&dir.join("config.json"));
        write_minimal_wordlevel_tokenizer(&dir.join("tokenizer.json"));
        write_pinned_minilm_onnx_fixture_placeholder(&dir.join("model_quantized.onnx"));

        let package = inspect_model_package(&dir).unwrap();
        let embedding = embedding_model_status_for_package(&package).unwrap();
        assert_eq!(embedding.task, ModelTaskKind::TextEmbedding);
        assert_eq!(embedding.runtime_lane, "onnx-embeddings");
        assert_eq!(embedding.embedding_dimension, Some(384));
        assert_eq!(
            embedding.runtime_installed,
            cfg!(feature = "onnx-embeddings-ort")
        );
        assert_eq!(embedding.runnable, cfg!(feature = "onnx-embeddings-ort"));
        if cfg!(feature = "onnx-embeddings-ort") {
            assert_eq!(embedding.status, CapabilityStatus::Runnable);
            assert!(embedding.blockers.is_empty());
        } else {
            assert_eq!(embedding.status, CapabilityStatus::MetadataOnly);
            assert!(embedding
                .blockers
                .iter()
                .any(|blocker| blocker.contains("runtime adapter is not installed")));
        }

        fs::remove_dir_all(dir).unwrap();
    }

    #[test]
    fn refuses_wrong_size_or_non_pinned_onnx_embedding_artifact() {
        for (filename, bytes) in [
            ("model_quantized.onnx", 128_u64),
            ("model.onnx", PINNED_MINILM_ONNX_EMBEDDING_BYTES),
        ] {
            let unique = SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .unwrap()
                .as_nanos();
            let dir = std::env::temp_dir()
                .join(format!("fathom-wrong-onnx-embedding-policy-test-{unique}"));
            fs::create_dir_all(&dir).unwrap();
            write_minimal_onnx_embedding_config(&dir.join("config.json"));
            write_minimal_wordlevel_tokenizer(&dir.join("tokenizer.json"));
            let file = fs::File::create(dir.join(filename)).unwrap();
            file.set_len(bytes).unwrap();

            let package = inspect_model_package(&dir).unwrap();
            let embedding = embedding_model_status_for_package(&package).unwrap();
            assert!(!embedding.runnable);
            assert_ne!(embedding.status, CapabilityStatus::Runnable);
            assert!(embedding
                .blockers
                .iter()
                .any(|blocker| blocker.contains(ONNX_EMBEDDING_PINNED_FIXTURE_BLOCKER)));

            fs::remove_dir_all(dir).unwrap();
        }
    }

    #[test]
    fn refuses_onnx_embedding_external_data_sidecar() {
        let unique = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_nanos();
        let dir =
            std::env::temp_dir().join(format!("fathom-onnx-external-data-policy-test-{unique}"));
        fs::create_dir_all(&dir).unwrap();
        write_minimal_onnx_embedding_config(&dir.join("config.json"));
        write_minimal_wordlevel_tokenizer(&dir.join("tokenizer.json"));
        write_pinned_minilm_onnx_fixture_placeholder(&dir.join("model_quantized.onnx"));
        fs::write(
            dir.join("model_quantized.onnx.data"),
            b"external tensor bytes",
        )
        .unwrap();

        let package = inspect_model_package(&dir).unwrap();
        let embedding = embedding_model_status_for_package(&package).unwrap();
        assert!(!embedding.runnable);
        assert_ne!(embedding.status, CapabilityStatus::Runnable);
        assert!(embedding
            .blockers
            .iter()
            .any(|blocker| blocker.contains(ONNX_EMBEDDING_EXTERNAL_DATA_BLOCKER)));

        fs::remove_dir_all(dir).unwrap();
    }

    #[test]
    fn refuses_onnx_embedding_custom_op_or_shared_library_configuration() {
        let unique = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_nanos();
        let dir = std::env::temp_dir().join(format!("fathom-onnx-custom-op-policy-test-{unique}"));
        fs::create_dir_all(&dir).unwrap();
        fs::write(
            dir.join("config.json"),
            r#"{"model_type":"bert","architectures":["BertModel"],"hidden_size":384,"custom_op_library":"libcustom_ops.so"}"#,
        )
        .unwrap();
        write_minimal_wordlevel_tokenizer(&dir.join("tokenizer.json"));
        write_pinned_minilm_onnx_fixture_placeholder(&dir.join("model_quantized.onnx"));

        let package = inspect_model_package(&dir).unwrap();
        let embedding = embedding_model_status_for_package(&package).unwrap();
        assert!(!embedding.runnable);
        assert_ne!(embedding.status, CapabilityStatus::Runnable);
        assert!(embedding
            .blockers
            .iter()
            .any(|blocker| blocker.contains(ONNX_EMBEDDING_CUSTOM_OP_BLOCKER)));

        fs::remove_dir_all(dir).unwrap();
    }

    #[cfg(unix)]
    #[test]
    fn refuses_onnx_embedding_package_when_config_or_tokenizer_symlink_escapes_root() {
        for escaped_filename in ["config.json", "tokenizer.json"] {
            let unique = SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .unwrap()
                .as_nanos();
            let dir = std::env::temp_dir().join(format!(
                "fathom-onnx-metadata-escape-test-{escaped_filename}-{unique}"
            ));
            let outside = std::env::temp_dir().join(format!(
                "fathom-onnx-metadata-escape-outside-{escaped_filename}-{unique}"
            ));
            fs::create_dir_all(&dir).unwrap();
            if escaped_filename == "config.json" {
                write_minimal_onnx_embedding_config(&outside);
                write_minimal_wordlevel_tokenizer(&dir.join("tokenizer.json"));
            } else {
                write_minimal_onnx_embedding_config(&dir.join("config.json"));
                write_minimal_wordlevel_tokenizer(&outside);
            }
            std::os::unix::fs::symlink(&outside, dir.join(escaped_filename)).unwrap();
            write_pinned_minilm_onnx_fixture_placeholder(&dir.join("model_quantized.onnx"));

            let package = inspect_model_package(&dir).unwrap();
            let embedding = embedding_model_status_for_package(&package).unwrap();
            assert!(!embedding.runnable);
            assert_ne!(embedding.status, CapabilityStatus::Runnable);
            assert!(embedding
                .blockers
                .iter()
                .any(|blocker| blocker.contains(ARTIFACT_PATH_ESCAPE_BLOCKER)));

            fs::remove_dir_all(dir).unwrap();
            fs::remove_file(outside).unwrap();
        }
    }

    #[cfg(all(unix, feature = "onnx-embeddings-ort"))]
    #[test]
    fn refuses_runnable_onnx_embedding_package_when_selected_artifact_symlink_escapes_root() {
        let unique = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_nanos();
        let dir = std::env::temp_dir().join(format!("fathom-onnx-embedding-escape-test-{unique}"));
        let outside = std::env::temp_dir().join(format!(
            "fathom-onnx-embedding-escape-outside-{unique}.onnx"
        ));
        fs::create_dir_all(&dir).unwrap();
        fs::write(
            dir.join("config.json"),
            r#"{"model_type":"bert","architectures":["BertModel"],"hidden_size":384}"#,
        )
        .unwrap();
        write_minimal_wordlevel_tokenizer(&dir.join("tokenizer.json"));
        let outside_file = fs::File::create(&outside).unwrap();
        outside_file.set_len(22_972_869).unwrap();
        std::os::unix::fs::symlink(&outside, dir.join("model_quantized.onnx")).unwrap();

        let package = inspect_model_package(&dir).unwrap();
        let embedding = embedding_model_status_for_package(&package).unwrap();
        assert!(!embedding.runnable);
        assert_ne!(embedding.status, CapabilityStatus::Runnable);
        assert!(embedding
            .blockers
            .iter()
            .any(|blocker| blocker.contains(ARTIFACT_PATH_ESCAPE_BLOCKER)));

        fs::remove_dir_all(dir).unwrap();
        fs::remove_file(outside).unwrap();
    }

    #[test]
    fn detects_onnx_without_tokenizer_without_claiming_embedding_readiness() {
        let unique = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_nanos();
        let dir = std::env::temp_dir().join(format!("fathom-onnx-unknown-test-{unique}"));
        fs::create_dir_all(&dir).unwrap();
        fs::write(dir.join("model.onnx"), b"fake onnx fixture bytes").unwrap();

        let package = inspect_model_package(&dir).unwrap();
        let embedding = embedding_model_status_for_package(&package).unwrap();
        assert_eq!(embedding.task, ModelTaskKind::Unknown);
        assert_eq!(embedding.status, CapabilityStatus::Planned);
        assert!(!embedding.runnable);
        assert!(embedding
            .blockers
            .iter()
            .any(|blocker| blocker.contains("missing tokenizer")));

        fs::remove_dir_all(dir).unwrap();
    }

    #[test]
    fn marks_pytorch_bin_as_blocked_not_planned() {
        let unique = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_nanos();
        let path = std::env::temp_dir().join(format!("pytorch_model-{unique}.bin"));
        fs::write(&path, b"").unwrap();

        let package = inspect_model_package(&path).unwrap();
        let machine = current_machine_profile();
        let report = capability_report_for_package(package, &machine);

        assert!(!report.runnable);
        assert_eq!(report.best_status, CapabilityStatus::Blocked);
        assert!(report
            .matching_lanes
            .iter()
            .any(|lane| lane.kind == BackendLaneKind::PyTorchTrustedImport));

        fs::remove_file(path).unwrap();
    }

    #[test]
    fn validates_missing_hf_loader_metadata_without_claiming_runnable() {
        let unique = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_nanos();
        let dir = std::env::temp_dir().join(format!("fathom-validation-test-{unique}"));
        fs::create_dir_all(&dir).unwrap();
        fs::write(dir.join("model.safetensors"), b"").unwrap();

        let package = inspect_model_package(&dir).unwrap();
        assert!(package.hf_validation.has_safetensors_weights);
        assert!(!package.hf_validation.ready_for_loader_metadata);
        assert_eq!(
            package.hf_validation.missing_required,
            vec![
                "config.json".to_string(),
                "tokenizer.json or SentencePiece model".to_string()
            ]
        );

        let report = capability_report_for_package(package, &current_machine_profile());
        assert!(!report.runnable);
        assert_eq!(report.best_status, CapabilityStatus::MetadataOnly);
        assert!(report.summary.contains("missing config.json"));

        fs::remove_dir_all(dir).unwrap();
    }

    #[test]
    fn generation_options_validate_openai_style_sampling_bounds() {
        GenerationOptions {
            temperature: 0.0,
            top_k: Some(1),
            top_p: Some(1.0),
        }
        .validate()
        .unwrap();

        assert!(GenerationOptions {
            temperature: -0.1,
            top_k: None,
            top_p: None,
        }
        .validate()
        .is_err());
        assert!(GenerationOptions {
            temperature: 1.0,
            top_k: Some(0),
            top_p: None,
        }
        .validate()
        .is_err());
        assert!(GenerationOptions {
            temperature: 1.0,
            top_k: None,
            top_p: Some(1.1),
        }
        .validate()
        .is_err());
    }

    #[test]
    fn cached_decode_step_prefills_prompt_then_decodes_one_token() {
        let prompt_tokens = [10, 11, 12];
        let (prefill, prefill_pos) = cached_decode_step(&prompt_tokens, true);
        assert_eq!(prefill, &[10, 11, 12]);
        assert_eq!(prefill_pos, 0);

        let generated_tokens = [10, 11, 12, 42];
        let (decode, decode_pos) = cached_decode_step(&generated_tokens, false);
        assert_eq!(decode, &[42]);
        assert_eq!(decode_pos, 3);
    }

    #[test]
    fn deterministic_sampler_respects_greedy_and_top_k_restriction() {
        let logits = [0.0, 2.0, 1.0];
        let greedy = select_next_token_from_logits(
            &logits,
            GenerationOptions {
                temperature: 0.0,
                top_k: None,
                top_p: None,
            },
        )
        .unwrap();
        assert_eq!(greedy, 1);

        let restricted = select_next_token_from_logits(
            &logits,
            GenerationOptions {
                temperature: 1.0,
                top_k: Some(2),
                top_p: Some(0.99),
            },
        )
        .unwrap();
        assert_eq!(restricted, 1);
    }

    #[test]
    fn renders_plain_role_prompt_for_debug_paths_only() {
        let prompt = render_plain_role_prompt(
            &[
                ChatMessage {
                    role: "system".into(),
                    content: "Be direct.".into(),
                },
                ChatMessage {
                    role: "user".into(),
                    content: "Hi".into(),
                },
            ],
            &PromptRenderOptions {
                add_generation_prompt: true,
            },
        )
        .unwrap();

        assert_eq!(prompt, "system: Be direct.\nuser: Hi\nassistant: ");
    }

    #[test]
    fn renders_common_chatml_hf_chat_template() {
        let template = ChatTemplateMetadata {
            source: PathBuf::from("tokenizer_config.json"),
            template: "{% for message in messages %}{{'<|im_start|>' + message['role'] + '\\n' + message['content'] + '<|im_end|>' + '\\n'}}{% endfor %}{% if add_generation_prompt %}{{ '<|im_start|>assistant\\n' }}{% endif %}".into(),
            format: ChatTemplateFormat::HuggingFaceJinja,
            needs_template_engine: true,
        };

        let prompt = render_hf_chat_template_prompt(
            &template,
            &[
                ChatMessage {
                    role: "system".into(),
                    content: "Be direct.".into(),
                },
                ChatMessage {
                    role: "user".into(),
                    content: "Hi".into(),
                },
            ],
            &PromptRenderOptions {
                add_generation_prompt: true,
            },
        )
        .unwrap();

        assert_eq!(
            prompt,
            "<|im_start|>system\nBe direct.<|im_end|>\n<|im_start|>user\nHi<|im_end|>\n<|im_start|>assistant\n"
        );
    }

    #[test]
    fn renders_inst_hf_chat_template_with_system_and_history() {
        let template = ChatTemplateMetadata {
            source: PathBuf::from("tokenizer_config.json"),
            template: "{% for message in messages %}{% if message['role'] == 'user' %}{{ '<s>[INST] ' + message['content'] + ' [/INST]' }}{% elif message['role'] == 'assistant' %}{{ ' ' + message['content'] + ' </s>' }}{% endif %}{% endfor %}".into(),
            format: ChatTemplateFormat::HuggingFaceJinja,
            needs_template_engine: true,
        };

        let prompt = render_hf_chat_template_prompt(
            &template,
            &[
                ChatMessage {
                    role: "system".into(),
                    content: "Be direct.".into(),
                },
                ChatMessage {
                    role: "user".into(),
                    content: "Hi".into(),
                },
                ChatMessage {
                    role: "assistant".into(),
                    content: "Hello.".into(),
                },
                ChatMessage {
                    role: "user".into(),
                    content: "Again".into(),
                },
            ],
            &PromptRenderOptions {
                add_generation_prompt: true,
            },
        )
        .unwrap();

        assert_eq!(
            prompt,
            "<s>[INST] <<SYS>>\nBe direct.\n<</SYS>>\n\nHi [/INST] Hello. </s><s>[INST] Again [/INST]"
        );
    }

    #[test]
    fn renders_llama3_header_hf_chat_template_with_generation_prompt() {
        let template = ChatTemplateMetadata {
            source: PathBuf::from("tokenizer_config.json"),
            template: "{% for message in messages %}{{ '<|start_header_id|>' + message['role'] + '<|end_header_id|>\\n\\n' + message['content'] | trim + '<|eot_id|>' }}{% endfor %}{% if add_generation_prompt %}{{ '<|start_header_id|>assistant<|end_header_id|>\\n\\n' }}{% endif %}".into(),
            format: ChatTemplateFormat::HuggingFaceJinja,
            needs_template_engine: true,
        };

        let prompt = render_hf_chat_template_prompt(
            &template,
            &[
                ChatMessage {
                    role: "system".into(),
                    content: "Be direct.".into(),
                },
                ChatMessage {
                    role: "user".into(),
                    content: "Hi".into(),
                },
                ChatMessage {
                    role: "assistant".into(),
                    content: "Hello.".into(),
                },
                ChatMessage {
                    role: "user".into(),
                    content: "Again".into(),
                },
            ],
            &PromptRenderOptions {
                add_generation_prompt: true,
            },
        )
        .unwrap();

        assert_eq!(
            prompt,
            "<|start_header_id|>system<|end_header_id|>\n\nBe direct.<|eot_id|><|start_header_id|>user<|end_header_id|>\n\nHi<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\nHello.<|eot_id|><|start_header_id|>user<|end_header_id|>\n\nAgain<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n"
        );
    }

    #[test]
    fn refuses_llama3_like_template_without_generation_prompt_marker() {
        let template = ChatTemplateMetadata {
            source: PathBuf::from("tokenizer_config.json"),
            template: "{% for message in messages %}{{ '<|start_header_id|>' + message['role'] + '<|end_header_id|>' + message['content'] + '<|eot_id|>' }}{% endfor %}".into(),
            format: ChatTemplateFormat::HuggingFaceJinja,
            needs_template_engine: true,
        };

        let error = render_hf_chat_template_prompt(
            &template,
            &[ChatMessage {
                role: "user".into(),
                content: "Hi".into(),
            }],
            &PromptRenderOptions {
                add_generation_prompt: true,
            },
        )
        .unwrap_err();

        assert!(error.to_string().contains("chat_template_not_supported"));
    }

    #[test]
    fn renders_gemma_hf_chat_template_with_model_generation_prompt() {
        let template = ChatTemplateMetadata {
            source: PathBuf::from("tokenizer_config.json"),
            template: "{% for message in messages %}{% if message['role'] == 'user' %}{{'<start_of_turn>user\n' + message['content'] + '<end_of_turn>\n'}}{% elif message['role'] == 'assistant' %}{{'<start_of_turn>model\n' + message['content'] + '<end_of_turn>\n'}}{% endif %}{% endfor %}{% if add_generation_prompt %}{{'<start_of_turn>model\n'}}{% endif %}".into(),
            format: ChatTemplateFormat::HuggingFaceJinja,
            needs_template_engine: true,
        };

        let prompt = render_hf_chat_template_prompt(
            &template,
            &[
                ChatMessage {
                    role: "user".into(),
                    content: "Hi".into(),
                },
                ChatMessage {
                    role: "assistant".into(),
                    content: "Hello.".into(),
                },
            ],
            &PromptRenderOptions {
                add_generation_prompt: true,
            },
        )
        .unwrap();

        assert_eq!(
            prompt,
            "<start_of_turn>user\nHi<end_of_turn>\n<start_of_turn>model\nHello.<end_of_turn>\n<start_of_turn>model\n"
        );
    }

    #[test]
    fn renders_dynamic_gemma_hf_chat_template_with_model_generation_prompt() {
        let template = ChatTemplateMetadata {
            source: PathBuf::from("tokenizer_config.json"),
            template: "{% for message in messages %}{{'<start_of_turn>' + (message['role'] == 'assistant' and 'model' or message['role']) + '\\n' + message['content'] + '<end_of_turn>\\n'}}{% endfor %}{% if add_generation_prompt %}{{'<start_of_turn>model\\n'}}{% endif %}".into(),
            format: ChatTemplateFormat::HuggingFaceJinja,
            needs_template_engine: true,
        };

        let prompt = render_hf_chat_template_prompt(
            &template,
            &[
                ChatMessage {
                    role: "user".into(),
                    content: "Hi".into(),
                },
                ChatMessage {
                    role: "assistant".into(),
                    content: "Hello.".into(),
                },
            ],
            &PromptRenderOptions {
                add_generation_prompt: true,
            },
        )
        .unwrap();

        assert_eq!(
            prompt,
            "<start_of_turn>user\nHi<end_of_turn>\n<start_of_turn>model\nHello.<end_of_turn>\n<start_of_turn>model\n"
        );
    }

    #[test]
    fn refuses_unknown_role_for_supported_hf_chat_templates() {
        let template = ChatTemplateMetadata {
            source: PathBuf::from("tokenizer_config.json"),
            template: "{% for message in messages %}{{'<|im_start|>' + message['role'] + '\n' + message['content'] + '<|im_end|>' + '\n'}}{% endfor %}".into(),
            format: ChatTemplateFormat::HuggingFaceJinja,
            needs_template_engine: true,
        };

        let error = render_hf_chat_template_prompt(
            &template,
            &[ChatMessage {
                role: "tool".into(),
                content: "Nope".into(),
            }],
            &PromptRenderOptions {
                add_generation_prompt: true,
            },
        )
        .unwrap_err();

        assert!(error.to_string().contains("chat_template_not_supported"));
        assert!(error.to_string().contains("unsupported chat message role"));
    }

    #[test]
    fn refuses_unsupported_hf_jinja_chat_template_until_engine_exists() {
        let template = ChatTemplateMetadata {
            source: PathBuf::from("tokenizer_config.json"),
            template: "{{ messages }}".into(),
            format: ChatTemplateFormat::HuggingFaceJinja,
            needs_template_engine: true,
        };

        let error = render_hf_chat_template_prompt(
            &template,
            &[ChatMessage {
                role: "user".into(),
                content: "Hi".into(),
            }],
            &PromptRenderOptions {
                add_generation_prompt: true,
            },
        )
        .unwrap_err();

        assert!(error.to_string().contains("chat_template_not_supported"));
    }

    #[test]
    fn tokenizer_config_alone_does_not_satisfy_tokenizer_requirement() {
        let unique = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_nanos();
        let dir = std::env::temp_dir().join(format!("fathom-tokenizer-config-test-{unique}"));
        fs::create_dir_all(&dir).unwrap();
        fs::write(dir.join("config.json"), r#"{"model_type":"llama"}"#).unwrap();
        fs::write(
            dir.join("tokenizer_config.json"),
            r#"{"chat_template":"{{ messages }}"}"#,
        )
        .unwrap();
        fs::write(dir.join("model.safetensors"), b"").unwrap();

        let package = inspect_model_package(&dir).unwrap();
        assert!(package.hf_validation.has_chat_template);
        assert!(!package.hf_validation.has_tokenizer);
        assert_eq!(
            package.hf_validation.missing_required,
            vec!["tokenizer.json or SentencePiece model".to_string()]
        );

        fs::remove_dir_all(dir).unwrap();
    }
}
