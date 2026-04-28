use axum::{
    extract::{Path, State},
    http::StatusCode,
    routing::{get, patch, post},
    Json, Router,
};
use fathom_core::{
    backend_lanes_for_machine, capability_report_for_package, current_machine_profile,
    embedding_model_status_for_package, external_context_strategy_advice,
    generate_candle_bert_embeddings, generate_onnx_embeddings, generate_with_candle_hf_options,
    inspect_model_package, recommend_context_strategy_for_package, render_hf_chat_template_prompt,
    CapabilityStatus, ChatMessage, ChatPromptRenderer, ContextStrategyAdvice, EmbeddingOutput,
    EmbeddingRequest, EmbeddingVector, GenerationMetrics, GenerationOptions,
    PlainRolePromptRenderer, PromptRenderOptions, RetrievalIndexSummary, VectorIndex,
    VectorIndexChunk, VectorSearchHit, VectorSearchMetric,
};
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::{
    collections::BTreeMap,
    net::SocketAddr,
    path::{Path as FsPath, PathBuf},
    sync::Arc,
};
use tokio::sync::Mutex;
use tokio::{fs::OpenOptions, io::AsyncWriteExt};
use tower_http::cors::{Any, CorsLayer};
use uuid::Uuid;

#[derive(Clone)]
struct AppState {
    inner: Arc<Mutex<Store>>,
    model_state_path: PathBuf,
}

#[derive(Default)]
struct Store {
    models: Vec<ModelRecord>,
    conversations: Vec<ConversationRecord>,
    memories: Vec<MemoryRecord>,
    active_model_id: Option<String>,
    startup_warnings: Vec<RuntimeWarning>,
}

type ApiError = (StatusCode, Json<serde_json::Value>);

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
struct PersistedModelState {
    #[serde(default)]
    models: Vec<ModelRecord>,
    #[serde(default)]
    active_model_id: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
struct RuntimeWarning {
    #[serde(rename = "type")]
    warning_type: String,
    message: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    preserved_path: Option<String>,
}

#[derive(Debug, Clone)]
struct ModelStateLoadResult {
    state: PersistedModelState,
    warnings: Vec<RuntimeWarning>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct ModelRecord {
    id: String,
    name: String,
    status: String,
    provider_kind: String,
    model_path: Option<String>,
    runtime_model_name: Option<String>,
    format: Option<String>,
    source: Option<String>,
    engine: Option<String>,
    quant: Option<String>,
    hf_repo: Option<String>,
    hf_filename: Option<String>,
    bytes_downloaded: Option<i64>,
    total_bytes: Option<i64>,
    progress: Option<f64>,
    install_error: Option<String>,
    api_base: Option<String>,
    api_key_configured: Option<bool>,
    capability_status: String,
    capability_summary: String,
    backend_lanes: Vec<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    task: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    download_manifest: Option<DownloadManifest>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct DownloadManifest {
    schema_version: u32,
    repo_id: String,
    revision: String,
    source_url: String,
    license: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    license_status: Option<String>,
    #[serde(default)]
    license_acknowledgement_required: bool,
    #[serde(default)]
    license_acknowledged: bool,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    license_policy_note: Option<String>,
    installed_at: String,
    verification_status: String,
    files: Vec<DownloadManifestFile>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct DownloadManifestFile {
    filename: String,
    size_bytes: i64,
    sha256: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct ConversationRecord {
    id: String,
    title: String,
    model_id: Option<String>,
    created_at: String,
    updated_at: String,
    messages: Vec<MessageRecord>,
    #[serde(skip_serializing_if = "Option::is_none")]
    memory_context: Option<serde_json::Value>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct MessageRecord {
    id: String,
    role: String,
    content: String,
    created_at: String,
    model_id: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    tokens_in_per_sec: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    tokens_out_per_sec: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    runtime_cache_hit: Option<bool>,
    #[serde(skip_serializing_if = "Option::is_none")]
    runtime_residency: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    runtime_family: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    runtime_cache_lookup_ms: Option<u128>,
    #[serde(skip_serializing_if = "Option::is_none")]
    model_load_ms: Option<u128>,
    #[serde(skip_serializing_if = "Option::is_none")]
    generation_ms: Option<u128>,
    #[serde(skip_serializing_if = "Option::is_none")]
    total_ms: Option<u128>,
    #[serde(skip_serializing_if = "Option::is_none")]
    ttft_ms: Option<u128>,
    #[serde(skip_serializing_if = "Option::is_none")]
    prefill_ms: Option<u128>,
    #[serde(skip_serializing_if = "Option::is_none")]
    decode_ms: Option<u128>,
    #[serde(skip_serializing_if = "Option::is_none")]
    prefill_tokens_per_second: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    decode_tokens_per_second: Option<f64>,
}

impl MessageRecord {
    fn user(id: String, content: String, created_at: String, model_id: Option<String>) -> Self {
        Self {
            id,
            role: "user".into(),
            content,
            created_at,
            model_id,
            tokens_in_per_sec: None,
            tokens_out_per_sec: None,
            runtime_cache_hit: None,
            runtime_residency: None,
            runtime_family: None,
            runtime_cache_lookup_ms: None,
            model_load_ms: None,
            generation_ms: None,
            total_ms: None,
            ttft_ms: None,
            prefill_ms: None,
            decode_ms: None,
            prefill_tokens_per_second: None,
            decode_tokens_per_second: None,
        }
    }

    fn assistant_from_generation(
        id: String,
        content: String,
        created_at: String,
        model_id: Option<String>,
        metrics: &GenerationMetrics,
    ) -> Self {
        Self {
            id,
            role: "assistant".into(),
            content,
            created_at,
            model_id,
            tokens_in_per_sec: None,
            tokens_out_per_sec: metrics.tokens_per_second,
            runtime_cache_hit: Some(metrics.runtime_cache_hit),
            runtime_residency: metrics.runtime_residency.clone(),
            runtime_family: metrics.runtime_family.clone(),
            runtime_cache_lookup_ms: Some(metrics.runtime_cache_lookup_ms),
            model_load_ms: Some(metrics.model_load_ms),
            generation_ms: Some(metrics.generation_ms),
            total_ms: Some(metrics.total_ms),
            ttft_ms: metrics.ttft_ms,
            prefill_ms: metrics.prefill_ms,
            decode_ms: metrics.decode_ms,
            prefill_tokens_per_second: metrics.prefill_tokens_per_second,
            decode_tokens_per_second: metrics.decode_tokens_per_second,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct MemoryRecord {
    id: String,
    title: String,
    body: String,
    scope: String,
    created_at: String,
    updated_at: String,
}

#[derive(Debug, Deserialize)]
struct RegisterModelRequest {
    id: Option<String>,
    name: String,
    model_path: String,
    runtime_model_name: Option<String>,
}

#[derive(Debug, Deserialize)]
struct ExternalModelRequest {
    id: String,
    name: String,
    source: String,
    api_base: String,
    api_key: String,
    model_name: String,
}

#[derive(Debug, Deserialize)]
struct CreateConversationRequest {
    title: Option<String>,
    model_id: Option<String>,
}

#[derive(Debug, Deserialize)]
struct PatchConversationRequest {
    title: Option<String>,
}

#[derive(Debug, Deserialize)]
struct ChatRequest {
    content: String,
    model_id: Option<String>,
}

#[derive(Debug, Deserialize)]
struct V1EmbeddingsRequest {
    model: Option<String>,
    input: V1EmbeddingInput,
    encoding_format: Option<String>,
}

#[derive(Debug, Deserialize)]
#[serde(untagged)]
enum V1EmbeddingInput {
    Single(String),
    Batch(Vec<String>),
}

impl V1EmbeddingInput {
    fn into_vec(self) -> Vec<String> {
        match self {
            Self::Single(input) => vec![input],
            Self::Batch(inputs) => inputs,
        }
    }
}

#[derive(Debug, Deserialize)]
struct V1ChatCompletionRequest {
    model: Option<String>,
    messages: Vec<ChatMessage>,
    max_tokens: Option<usize>,
    temperature: Option<f32>,
    top_p: Option<f32>,
    top_k: Option<usize>,
    stream: Option<bool>,
    #[serde(default, rename = "fathom.retrieval")]
    retrieval: Option<FathomRetrievalRequest>,
}

#[derive(Debug, Clone, Deserialize)]
struct FathomRetrievalRequest {
    index_id: String,
    query_vector: Vec<f32>,
    top_k: Option<usize>,
    metric: Option<VectorSearchMetric>,
    max_context_chars: Option<usize>,
}

#[derive(Debug, Clone, Serialize)]
struct FathomRetrievalMetadata {
    index: RetrievalIndexSummary,
    metric: VectorSearchMetric,
    requested_top_k: usize,
    hit_count: usize,
    inserted_chunk_count: usize,
    context_chars: usize,
    max_context_chars: usize,
    hits: Vec<FathomRetrievalHitMetadata>,
}

#[derive(Debug, Clone, Serialize)]
struct FathomRetrievalHitMetadata {
    chunk_id: String,
    document_id: String,
    score: f32,
    inserted: bool,
    text_chars: usize,
}

#[derive(Debug, Deserialize)]
struct MemoryRequest {
    title: Option<String>,
    body: Option<String>,
    scope: Option<String>,
}

#[derive(Debug, Deserialize)]
struct CatalogInstallRequest {
    repo_id: String,
    filename: String,
    #[serde(default)]
    accept_license: Option<bool>,
    #[serde(default)]
    accepted_license: Option<bool>,
}

#[derive(Debug, Deserialize)]
struct CreateRetrievalIndexRequest {
    id: String,
    embedding_model_id: String,
    embedding_dimension: usize,
}

#[derive(Debug, Deserialize)]
struct AddRetrievalChunkRequest {
    chunk: VectorIndexChunk,
    vector: Vec<f32>,
}

#[derive(Debug, Deserialize)]
struct SearchRetrievalIndexRequest {
    vector: Vec<f32>,
    top_k: Option<usize>,
    metric: Option<VectorSearchMetric>,
}

#[derive(Debug, Deserialize)]
struct EmbedRequest {
    input: Vec<String>,
    #[serde(default = "default_embed_normalize")]
    normalize: bool,
}

fn default_embed_normalize() -> bool {
    true
}

#[derive(Debug, Serialize)]
struct RetrievalIndexListResponse {
    items: Vec<RetrievalIndexSummary>,
    summary: String,
}

#[derive(Debug, Serialize)]
struct RetrievalSearchResponse {
    index: RetrievalIndexSummary,
    metric: VectorSearchMetric,
    hits: Vec<VectorSearchHit>,
}

#[derive(Debug, Clone)]
struct CatalogModelSpec {
    catalog_id: &'static str,
    title: &'static str,
    repo_id: &'static str,
    revision: Option<&'static str>,
    primary_filename: &'static str,
    files: Vec<CatalogFileSpec>,
    size_bytes: Option<i64>,
    license: &'static str,
    description: &'static str,
}

#[derive(Debug, Clone, Copy)]
struct CatalogFileSpec {
    filename: &'static str,
    size_bytes: Option<i64>,
    sha256: Option<&'static str>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum CatalogLicenseStatus {
    Permissive,
    Unknown,
    Restrictive,
}

impl CatalogLicenseStatus {
    const fn as_str(self) -> &'static str {
        match self {
            Self::Permissive => "permissive",
            Self::Unknown => "unknown",
            Self::Restrictive => "restrictive",
        }
    }
}

const fn catalog_file(
    filename: &'static str,
    size_bytes: i64,
    sha256: &'static str,
) -> CatalogFileSpec {
    CatalogFileSpec {
        filename,
        size_bytes: Some(size_bytes),
        sha256: Some(sha256),
    }
}

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    tracing_subscriber::fmt().with_env_filter("info").init();
    let model_state_path = fathom_state_file();
    let loaded_state = load_model_state(&model_state_path).await?;
    let state = AppState {
        inner: Arc::new(Mutex::new(Store {
            models: loaded_state.state.models,
            active_model_id: loaded_state.state.active_model_id,
            startup_warnings: loaded_state.warnings,
            ..Store::default()
        })),
        model_state_path,
    };
    let app = Router::new()
        .route("/api/runtime", get(runtime_state))
        .route("/api/dashboard", get(dashboard))
        .route("/api/capabilities", get(capabilities))
        .route("/api/conversations", post(create_conversation))
        .route(
            "/api/conversations/:id",
            patch(rename_conversation).delete(delete_conversation),
        )
        .route("/api/conversations/:id/chat", post(chat_conversation))
        .route("/api/embedding-models", get(embedding_models))
        .route(
            "/api/embedding-models/:id/embed",
            post(embed_with_embedding_model),
        )
        .route(
            "/api/retrieval-indexes",
            get(list_retrieval_indexes).post(create_retrieval_index),
        )
        .route(
            "/api/retrieval-indexes/:id/chunks",
            post(add_retrieval_chunk),
        )
        .route(
            "/api/retrieval-indexes/:id/search",
            post(search_retrieval_index),
        )
        .route("/api/memories", post(create_memory))
        .route(
            "/api/memories/:id",
            patch(update_memory).delete(delete_memory),
        )
        .route("/api/models/catalog", get(model_catalog))
        .route("/api/models/catalog/install", post(install_catalog_model))
        .route("/api/models/install", post(install_model))
        .route("/api/models/register", post(register_model))
        .route("/api/models/external", post(connect_external_model))
        .route(
            "/api/models/:id/context-strategy",
            get(model_context_strategy),
        )
        .route("/api/models/:id/activate", post(activate_model))
        .route("/api/models/:id/cancel", post(cancel_model_download))
        .route("/v1/models", get(v1_models))
        .route("/v1/health", get(v1_health))
        .route("/v1/chat/completions", post(v1_chat_completions))
        .route("/v1/embeddings", post(v1_embeddings))
        .layer(
            CorsLayer::new()
                .allow_origin(Any)
                .allow_methods(Any)
                .allow_headers(Any),
        )
        .with_state(state);

    let port = std::env::var("FATHOM_PORT")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(8180);
    let addr = SocketAddr::from(([127, 0, 0, 1], port));
    tracing::info!(%addr, "Fathom server listening");
    let listener = tokio::net::TcpListener::bind(addr).await?;
    axum::serve(listener, app).await?;
    Ok(())
}

async fn runtime_state(State(state): State<AppState>) -> Json<serde_json::Value> {
    let store = state.inner.lock().await;
    Json(runtime_json(&store))
}

async fn dashboard(State(state): State<AppState>) -> Json<serde_json::Value> {
    let store = state.inner.lock().await;
    let machine = current_machine_profile();
    let context_strategies = context_strategies_for_models(&store.models);
    Json(serde_json::json!({
        "runtime": runtime_json(&store),
        "models": store.models,
        "conversations": store.conversations,
        "memories": store.memories,
        "context_strategies": context_strategies,
        "capabilities": {
            "machine": machine,
            "backend_lanes": backend_lanes_for_machine(&machine)
        }
    }))
}

async fn capabilities() -> Json<serde_json::Value> {
    let machine = current_machine_profile();
    Json(serde_json::json!({
        "machine": machine,
        "backend_lanes": backend_lanes_for_machine(&machine)
    }))
}

async fn embedding_models(State(state): State<AppState>) -> Json<serde_json::Value> {
    let store = state.inner.lock().await;
    let items = store
        .models
        .iter()
        .filter_map(|model| {
            let model_path = model.model_path.as_ref()?;
            let package = inspect_model_package(model_path).ok()?;
            let embedding_status = embedding_model_status_for_package(&package)?;
            Some(serde_json::json!({
                "id": model.id,
                "name": model.name,
                "model_path": model_path,
                "provider_kind": model.provider_kind,
                "status": embedding_status,
            }))
        })
        .collect::<Vec<_>>();

    Json(serde_json::json!({
        "items": items,
        "retrieval": {
            "status": if fathom_core::onnx_embeddings_ort_compiled() { "vector_index_ready_default_minilm_and_onnx_embedding_inference_available" } else { "vector_index_ready_default_minilm_embedding_inference_available" },
            "runtime_lane": "local-embeddings-retrieval",
            "summary": if fathom_core::onnx_embeddings_ort_compiled() { "Developer vector indexes are available for caller-supplied embeddings. Default-build Candle BERT/MiniLM embeddings are available for verified SafeTensors fixtures, and the non-default ONNX embeddings feature can generate vectors for the pinned ONNX MiniLM fixture. Automatic document embedding/context assembly is still explicit and opt-in; Fathom does not claim ONNX chat/LLM support." } else { "Developer vector indexes are available for caller-supplied embeddings. Default-build Candle BERT/MiniLM embeddings are available for verified SafeTensors fixtures. ONNX embedding inference remains feature-gated behind onnx-embeddings-ort, and Fathom does not claim ONNX chat/LLM support." }
        }
    }))
}

async fn embed_with_embedding_model(
    State(state): State<AppState>,
    Path(id): Path<String>,
    Json(request): Json<EmbedRequest>,
) -> Result<Json<serde_json::Value>, ApiError> {
    let EmbeddingRun {
        model,
        runtime_label,
        output,
        normalize,
    } = run_embedding_model(&state, &id, request.input, request.normalize).await?;

    Ok(Json(serde_json::json!({
        "model": model.id,
        "runtime": runtime_label,
        "embedding_dimension": output.dimension,
        "data": embedding_data_json(&output, false),
        "fathom": {
            "runtime": runtime_label,
            "metrics": output.metrics,
            "normalize": normalize
        }
    })))
}

struct EmbeddingRun {
    model: ModelRecord,
    runtime_label: &'static str,
    output: EmbeddingOutput,
    normalize: bool,
}

async fn run_embedding_model(
    state: &AppState,
    id: &str,
    input: Vec<String>,
    normalize: bool,
) -> Result<EmbeddingRun, ApiError> {
    let model = {
        let store = state.inner.lock().await;
        store.models.iter().find(|model| model.id == id).cloned()
    }
    .ok_or_else(|| {
        error_json(
            StatusCode::NOT_FOUND,
            &format!("Embedding model '{id}' was not found."),
            "embedding_model_not_found",
        )
    })?;

    if input.is_empty() {
        return Err(error_json(
            StatusCode::BAD_REQUEST,
            "Embedding input cannot be empty.",
            "invalid_embedding_input",
        ));
    }
    if input.iter().any(|input| input.trim().is_empty()) {
        return Err(error_json(
            StatusCode::BAD_REQUEST,
            "Embedding input strings cannot be empty.",
            "invalid_embedding_input",
        ));
    }

    if model.task.as_deref() == Some("text_generation")
        && model
            .format
            .as_deref()
            .is_some_and(|format| format.eq_ignore_ascii_case("SafeTensors"))
        && model
            .backend_lanes
            .iter()
            .any(|lane| lane == "safetensors-hf")
    {
        return Err(error_json(
            StatusCode::BAD_REQUEST,
            "Model is a chat/generation model, not a text-embedding model.",
            "not_embedding_model",
        ));
    }
    if model
        .format
        .as_deref()
        .is_some_and(|format| format.eq_ignore_ascii_case("GGUF"))
        || model
            .backend_lanes
            .iter()
            .any(|lane| lane == "gguf-native" || lane == "gguf")
    {
        return Err(error_json(
            StatusCode::NOT_IMPLEMENTED,
            "GGUF models are metadata/readiness-only in Fathom and cannot be used for embeddings. No fake vectors were produced.",
            "embedding_runtime_unavailable",
        ));
    }
    if model.capability_status == "blocked"
        || model
            .format
            .as_deref()
            .is_some_and(|format| format.eq_ignore_ascii_case("PyTorchBin"))
        || model
            .backend_lanes
            .iter()
            .any(|lane| lane == "pytorch-trusted-import")
    {
        return Err(error_json(
            StatusCode::NOT_IMPLEMENTED,
            "PyTorch .bin models are blocked because legacy pickle artifacts can execute code. No embedding runtime was used and no fake vectors were produced.",
            "embedding_runtime_unavailable",
        ));
    }

    let model_path = model.model_path.as_ref().ok_or_else(|| {
        error_json(
            StatusCode::BAD_REQUEST,
            "Embedding model does not have a local package path.",
            "embedding_model_not_local",
        )
    })?;
    let package = inspect_model_package(model_path).map_err(|error| {
        error_json(
            StatusCode::BAD_REQUEST,
            &format!("Could not inspect embedding model package: {error}"),
            "embedding_model_inspection_failed",
        )
    })?;
    let status = embedding_model_status_for_package(&package).ok_or_else(|| {
        error_json(
            StatusCode::BAD_REQUEST,
            "Model package is not a known embedding package.",
            "not_embedding_model",
        )
    })?;
    if status.task != fathom_core::ModelTaskKind::TextEmbedding {
        return Err(error_json(
            StatusCode::BAD_REQUEST,
            "Package is not classified as a text-embedding model.",
            "not_text_embedding_model",
        ));
    }
    let embedding_request = EmbeddingRequest {
        inputs: &input,
        normalize,
    };
    let (runtime_label, output) = match status.runtime_lane {
        "candle-bert-embeddings" => (
            "candle-bert-embeddings",
            generate_candle_bert_embeddings(model_path, &embedding_request).map_err(|error| {
                error_json(
                    StatusCode::INTERNAL_SERVER_ERROR,
                    &error.to_string(),
                    "embedding_runtime_error",
                )
            })?,
        ),
        "onnx-embeddings" => (
            "onnx-embeddings-ort",
            generate_onnx_embeddings(model_path, &embedding_request).map_err(|error| {
                error_json(
                    StatusCode::NOT_IMPLEMENTED,
                    &error.to_string(),
                    "embedding_runtime_unavailable",
                )
            })?,
        ),
        other => {
            return Err(error_json(
                StatusCode::NOT_IMPLEMENTED,
                &format!("Embedding runtime lane '{other}' is not implemented."),
                "embedding_runtime_unavailable",
            ))
        }
    };

    Ok(EmbeddingRun {
        model,
        runtime_label,
        output,
        normalize,
    })
}

fn embedding_data_json(output: &EmbeddingOutput, openai_object: bool) -> Vec<serde_json::Value> {
    output
        .vectors
        .iter()
        .enumerate()
        .map(|(index, vector)| {
            let mut item = serde_json::json!({ "index": index, "embedding": vector.values });
            if openai_object {
                item["object"] = serde_json::json!("embedding");
            }
            item
        })
        .collect::<Vec<_>>()
}

async fn create_retrieval_index(
    State(state): State<AppState>,
    Json(request): Json<CreateRetrievalIndexRequest>,
) -> Result<Json<RetrievalIndexSummary>, ApiError> {
    let index = VectorIndex::new(
        request.id,
        request.embedding_model_id,
        request.embedding_dimension,
    )
    .map_err(|error| api_error(StatusCode::BAD_REQUEST, error, "invalid_retrieval_index"))?;
    let state_dir = retrieval_state_dir(&state)
        .map_err(|error| raw_api_error(error, "retrieval_state_error"))?;
    let path = state_dir
        .join("retrieval-indexes")
        .join(format!("{}.json", index.id));
    if path.exists() {
        return Err(api_error(
            StatusCode::CONFLICT,
            format!("Retrieval index '{}' already exists.", index.id),
            "retrieval_index_exists",
        ));
    }
    index.save_to_state_dir(&state_dir).map_err(|error| {
        api_error(
            StatusCode::INTERNAL_SERVER_ERROR,
            error,
            "retrieval_state_error",
        )
    })?;
    Ok(Json(index.summary()))
}

async fn list_retrieval_indexes(
    State(state): State<AppState>,
) -> Result<Json<RetrievalIndexListResponse>, ApiError> {
    let state_dir = retrieval_state_dir(&state)
        .map_err(|error| raw_api_error(error, "retrieval_state_error"))?;
    let dir = state_dir.join("retrieval-indexes");
    let mut items = Vec::new();
    if !dir.exists() {
        return Ok(Json(RetrievalIndexListResponse {
            items,
            summary: "No retrieval indexes have been created yet. Create one with caller-supplied vector dimensions; Fathom does not infer embeddings here.".into(),
        }));
    }

    let entries = std::fs::read_dir(&dir).map_err(|error| {
        api_error(
            StatusCode::INTERNAL_SERVER_ERROR,
            error,
            "retrieval_state_error",
        )
    })?;
    for entry in entries {
        let entry = entry.map_err(|error| {
            api_error(
                StatusCode::INTERNAL_SERVER_ERROR,
                error,
                "retrieval_state_error",
            )
        })?;
        let path = entry.path();
        if path.extension().and_then(|value| value.to_str()) != Some("json") {
            continue;
        }
        let Some(id) = path.file_stem().and_then(|value| value.to_str()) else {
            continue;
        };
        let index = VectorIndex::load_from_state_dir(&state_dir, id).map_err(|error| {
            api_error(
                StatusCode::INTERNAL_SERVER_ERROR,
                error,
                "retrieval_state_error",
            )
        })?;
        items.push(index.summary());
    }
    items.sort_by(|left, right| left.id.cmp(&right.id));
    Ok(Json(RetrievalIndexListResponse {
        items,
        summary: "Retrieval indexes use explicit caller-supplied vectors only; no embedding inference is performed by these APIs.".into(),
    }))
}

async fn add_retrieval_chunk(
    State(state): State<AppState>,
    Path(id): Path<String>,
    Json(request): Json<AddRetrievalChunkRequest>,
) -> Result<Json<RetrievalIndexSummary>, ApiError> {
    let state_dir = retrieval_state_dir(&state)
        .map_err(|error| raw_api_error(error, "retrieval_state_error"))?;
    let mut index = load_retrieval_index(&state_dir, &id).map_err(retrieval_load_error_json)?;
    let vector = EmbeddingVector::new(request.vector)
        .map_err(|error| api_error(StatusCode::BAD_REQUEST, error, "invalid_retrieval_vector"))?;
    index
        .add_chunk(request.chunk, vector)
        .map_err(|error| api_error(StatusCode::BAD_REQUEST, error, "invalid_retrieval_chunk"))?;
    index.save_to_state_dir(&state_dir).map_err(|error| {
        api_error(
            StatusCode::INTERNAL_SERVER_ERROR,
            error,
            "retrieval_state_error",
        )
    })?;
    Ok(Json(index.summary()))
}

async fn search_retrieval_index(
    State(state): State<AppState>,
    Path(id): Path<String>,
    Json(request): Json<SearchRetrievalIndexRequest>,
) -> Result<Json<RetrievalSearchResponse>, ApiError> {
    let state_dir = retrieval_state_dir(&state)
        .map_err(|error| raw_api_error(error, "retrieval_state_error"))?;
    let index = load_retrieval_index(&state_dir, &id).map_err(retrieval_load_error_json)?;
    let query = EmbeddingVector::new(request.vector)
        .map_err(|error| api_error(StatusCode::BAD_REQUEST, error, "invalid_retrieval_vector"))?;
    let top_k = request.top_k.unwrap_or(5).min(100);
    let metric = request.metric.unwrap_or(VectorSearchMetric::Cosine);
    let hits = index
        .search(&query, top_k, metric)
        .map_err(|error| api_error(StatusCode::BAD_REQUEST, error, "invalid_retrieval_search"))?;
    Ok(Json(RetrievalSearchResponse {
        index: index.summary(),
        metric,
        hits,
    }))
}

fn retrieval_state_dir(state: &AppState) -> Result<PathBuf, (StatusCode, String)> {
    state
        .model_state_path
        .parent()
        .map(FsPath::to_path_buf)
        .ok_or_else(|| internal_error("Fathom model state path has no parent"))
}

fn load_retrieval_index(state_dir: &FsPath, id: &str) -> Result<VectorIndex, (StatusCode, String)> {
    VectorIndex::load_from_state_dir(state_dir, id).map_err(|error| {
        if error
            .downcast_ref::<std::io::Error>()
            .is_some_and(|io_error| io_error.kind() == std::io::ErrorKind::NotFound)
        {
            (
                StatusCode::NOT_FOUND,
                format!("Retrieval index '{id}' was not found."),
            )
        } else {
            (StatusCode::BAD_REQUEST, error.to_string())
        }
    })
}

fn internal_error(error: impl std::fmt::Display) -> (StatusCode, String) {
    (StatusCode::INTERNAL_SERVER_ERROR, error.to_string())
}

async fn model_context_strategy(
    State(state): State<AppState>,
    Path(id): Path<String>,
) -> Result<Json<ContextStrategyAdvice>, ApiError> {
    let store = state.inner.lock().await;
    let model = store
        .models
        .iter()
        .find(|model| model.id == id)
        .ok_or_else(|| api_error(StatusCode::NOT_FOUND, "Model not found.", "model_not_found"))?;
    Ok(Json(context_strategy_for_model(model)))
}

fn context_strategies_for_models(
    models: &[ModelRecord],
) -> BTreeMap<String, ContextStrategyAdvice> {
    models
        .iter()
        .map(|model| (model.id.clone(), context_strategy_for_model(model)))
        .collect()
}

fn context_strategy_for_model(model: &ModelRecord) -> ContextStrategyAdvice {
    if model.provider_kind == "external" {
        return external_context_strategy_advice();
    }

    model
        .model_path
        .as_ref()
        .and_then(|path| inspect_model_package(path).ok())
        .map(|package| recommend_context_strategy_for_package(&package))
        .unwrap_or_else(|| ContextStrategyAdvice {
            label: "Metadata needed".into(),
            engine: fathom_core::ContextEngineRecommendation::RetrievalIndex,
            summary: "Fathom cannot inspect this model path right now, so it cannot make a model-specific context recommendation yet.".into(),
            max_context_tokens: None,
            reserve_output_tokens: 512,
            recommended_chunk_tokens: 500,
            recommended_overlap_tokens: 100,
            top_k: 4,
            needs_retrieval: true,
            caveats: vec!["Register or install a readable local model package to make this recommendation specific.".into()],
            suggested_workflow: vec![
                "Keep memories short and explicit until model metadata is readable.".into(),
                "Prefer retrieval snippets over large pasted documents.".into(),
            ],
        })
}

async fn create_conversation(
    State(state): State<AppState>,
    Json(req): Json<CreateConversationRequest>,
) -> Json<ConversationRecord> {
    let now = now();
    let mut conversation = ConversationRecord {
        id: Uuid::new_v4().to_string(),
        title: req.title.unwrap_or_else(|| "New Fathom chat".into()),
        model_id: req.model_id,
        created_at: now.clone(),
        updated_at: now.clone(),
        messages: Vec::new(),
        memory_context: None,
    };
    if conversation.title.trim().is_empty() {
        conversation.title = "New Fathom chat".into();
    }
    let mut store = state.inner.lock().await;
    store.conversations.insert(0, conversation.clone());
    Json(conversation)
}

async fn rename_conversation(
    State(state): State<AppState>,
    Path(id): Path<String>,
    Json(req): Json<PatchConversationRequest>,
) -> Result<Json<ConversationRecord>, ApiError> {
    let mut store = state.inner.lock().await;
    let conversation = store
        .conversations
        .iter_mut()
        .find(|item| item.id == id)
        .ok_or_else(|| {
            api_error(
                StatusCode::NOT_FOUND,
                "Conversation not found.",
                "conversation_not_found",
            )
        })?;
    if let Some(title) = req.title {
        conversation.title = title;
        conversation.updated_at = now();
    }
    Ok(Json(conversation.clone()))
}

async fn delete_conversation(
    State(state): State<AppState>,
    Path(id): Path<String>,
) -> Result<StatusCode, ApiError> {
    let mut store = state.inner.lock().await;
    let before = store.conversations.len();
    store.conversations.retain(|item| item.id != id);
    if store.conversations.len() == before {
        Err(api_error(
            StatusCode::NOT_FOUND,
            "Conversation not found.",
            "conversation_not_found",
        ))
    } else {
        Ok(StatusCode::NO_CONTENT)
    }
}

async fn chat_conversation(
    State(state): State<AppState>,
    Path(id): Path<String>,
    Json(req): Json<ChatRequest>,
) -> Result<Json<ConversationRecord>, ApiError> {
    if req.content.trim().is_empty() {
        return Err(api_error(
            StatusCode::BAD_REQUEST,
            "Message content is required.",
            "invalid_chat_message",
        ));
    }
    let model_id = req.model_id.clone();
    let prompt = req.content.clone();
    let model = {
        let store = state.inner.lock().await;
        if !store.conversations.iter().any(|item| item.id == id) {
            return Err(api_error(
                StatusCode::NOT_FOUND,
                "Conversation not found.",
                "conversation_not_found",
            ));
        }
        model_id
            .as_ref()
            .and_then(|model_id| store.models.iter().find(|model| &model.id == model_id))
            .cloned()
    };
    let model = model.ok_or_else(|| {
        api_error(
            StatusCode::BAD_REQUEST,
            "Select a runnable model before chatting.",
            "model_not_runnable",
        )
    })?;
    if model.provider_kind == "external" {
        return Err(api_error(
            StatusCode::NOT_IMPLEMENTED,
            "External OpenAI-compatible API entries are connected metadata only. Fathom does not proxy chat to external endpoints yet. No fake answer was produced.",
            "external_proxy_not_implemented",
        ));
    }
    if !is_runnable_model(&model)
        || !model
            .backend_lanes
            .iter()
            .any(|lane| lane == "safetensors-hf")
    {
        return Err(api_error(
            StatusCode::BAD_REQUEST,
            "Fathom detected the model, but no real generation backend is implemented for it yet. No fake answer was produced.",
            "model_not_runnable",
        ));
    }
    let model_path = model.model_path.as_ref().ok_or_else(|| {
        api_error(
            StatusCode::BAD_REQUEST,
            "Runnable local model is missing a model_path.",
            "model_not_runnable",
        )
    })?;
    let generation = generate_with_candle_hf_options(
        model_path,
        &prompt,
        default_chat_max_tokens(),
        demo_generation_options(),
    )
    .map_err(|error| api_error(StatusCode::INTERNAL_SERVER_ERROR, error, "generation_error"))?;
    if generation.text.trim().is_empty() {
        return Err(api_error(
            StatusCode::INTERNAL_SERVER_ERROR,
            "The model produced only whitespace or special tokens. Try a more specific completion-style prompt.",
            "generation_error",
        ));
    }

    let mut store = state.inner.lock().await;
    let conversation = store
        .conversations
        .iter_mut()
        .find(|item| item.id == id)
        .ok_or_else(|| {
            api_error(
                StatusCode::NOT_FOUND,
                "Conversation not found.",
                "conversation_not_found",
            )
        })?;
    let now = now();
    conversation.messages.push(MessageRecord::user(
        Uuid::new_v4().to_string(),
        req.content,
        now.clone(),
        req.model_id.clone(),
    ));
    conversation
        .messages
        .push(MessageRecord::assistant_from_generation(
            Uuid::new_v4().to_string(),
            generation.text,
            now.clone(),
            req.model_id,
            &generation.metrics,
        ));
    conversation.updated_at = now;
    Ok(Json(conversation.clone()))
}

async fn create_memory(
    State(state): State<AppState>,
    Json(req): Json<MemoryRequest>,
) -> Result<Json<MemoryRecord>, ApiError> {
    let title = req.title.unwrap_or_default();
    let body = req.body.unwrap_or_default();
    if title.trim().is_empty() || body.trim().is_empty() {
        return Err(api_error(
            StatusCode::BAD_REQUEST,
            "Memory title and body are required.",
            "invalid_memory",
        ));
    }
    let now = now();
    let memory = MemoryRecord {
        id: Uuid::new_v4().to_string(),
        title,
        body,
        scope: req.scope.unwrap_or_else(|| "General".into()),
        created_at: now.clone(),
        updated_at: now,
    };
    let mut store = state.inner.lock().await;
    store.memories.insert(0, memory.clone());
    Ok(Json(memory))
}

async fn update_memory(
    State(state): State<AppState>,
    Path(id): Path<String>,
    Json(req): Json<MemoryRequest>,
) -> Result<Json<MemoryRecord>, ApiError> {
    if req
        .title
        .as_ref()
        .is_some_and(|title| title.trim().is_empty())
        || req.body.as_ref().is_some_and(|body| body.trim().is_empty())
    {
        return Err(api_error(
            StatusCode::BAD_REQUEST,
            "Memory title and body cannot be empty.",
            "invalid_memory",
        ));
    }
    let mut store = state.inner.lock().await;
    let memory = store
        .memories
        .iter_mut()
        .find(|item| item.id == id)
        .ok_or_else(|| {
            api_error(
                StatusCode::NOT_FOUND,
                "Memory not found.",
                "memory_not_found",
            )
        })?;
    if let Some(title) = req.title {
        memory.title = title;
    }
    if let Some(body) = req.body {
        memory.body = body;
    }
    if let Some(scope) = req.scope {
        memory.scope = scope;
    }
    memory.updated_at = now();
    Ok(Json(memory.clone()))
}

async fn delete_memory(
    State(state): State<AppState>,
    Path(id): Path<String>,
) -> Result<StatusCode, ApiError> {
    let mut store = state.inner.lock().await;
    let before = store.memories.len();
    store.memories.retain(|item| item.id != id);
    if store.memories.len() == before {
        Err(api_error(
            StatusCode::NOT_FOUND,
            "Memory not found.",
            "memory_not_found",
        ))
    } else {
        Ok(StatusCode::NO_CONTENT)
    }
}

async fn register_model(
    State(state): State<AppState>,
    Json(req): Json<RegisterModelRequest>,
) -> Result<Json<ModelRecord>, ApiError> {
    let path = PathBuf::from(req.model_path.trim());
    let package = inspect_model_package(&path)
        .map_err(|error| api_error(StatusCode::BAD_REQUEST, error, "invalid_model_package"))?;
    let primary_artifact = package.artifacts.iter().find(|artifact| {
        !matches!(
            artifact.format,
            fathom_core::ModelFormat::TokenizerJson
                | fathom_core::ModelFormat::TokenizerConfigJson
                | fathom_core::ModelFormat::SentencePiece
                | fathom_core::ModelFormat::ConfigJson
                | fathom_core::ModelFormat::ChatTemplate
        )
    });
    let capability_report =
        capability_report_for_package(package.clone(), &current_machine_profile());
    let id = req.id.unwrap_or_else(|| slugify(&req.name));
    if id.trim().is_empty() || req.name.trim().is_empty() {
        return Err(api_error(
            StatusCode::BAD_REQUEST,
            "Model id and name are required.",
            "invalid_model_request",
        ));
    }
    let model = ModelRecord {
        id: id.clone(),
        name: req.name,
        status: if capability_report.runnable {
            "ready"
        } else {
            "registered"
        }
        .into(),
        provider_kind: "local".into(),
        model_path: Some(path.display().to_string()),
        runtime_model_name: req.runtime_model_name.or(Some(id.clone())),
        format: Some(
            primary_artifact
                .map(|artifact| format!("{:?}", artifact.format))
                .unwrap_or_else(|| "Package".into()),
        ),
        source: Some(
            package
                .model_type
                .as_deref()
                .map(|model_type| format!("Local artifact · {model_type}"))
                .unwrap_or_else(|| "Local artifact".into()),
        ),
        engine: Some("Fathom".into()),
        quant: None,
        hf_repo: None,
        hf_filename: None,
        bytes_downloaded: None,
        total_bytes: None,
        progress: None,
        install_error: if capability_report.runnable {
            None
        } else {
            Some(
                primary_artifact
                    .map(|artifact| artifact.notes.join(" "))
                    .unwrap_or_else(|| package.notes.join(" ")),
            )
        },
        api_base: None,
        api_key_configured: None,
        capability_status: capability_status_label(&capability_report.best_status).into(),
        capability_summary: if primary_artifact
            .is_some_and(|artifact| matches!(artifact.format, fathom_core::ModelFormat::PyTorchBin))
        {
            "Blocked PyTorch .bin/pickle artifact: Fathom requires an explicit trusted-import safety policy because pickle can execute code; this is not runnable.".into()
        } else {
            capability_report.summary.clone()
        },
        backend_lanes: capability_report
            .matching_lanes
            .iter()
            .map(|lane| lane.id.to_string())
            .collect(),
        task: model_task_label_for_package(&package),
        download_manifest: None,
    };
    let model = mutate_model_state_and_persist(&state, |store| {
        upsert_model(&mut store.models, model.clone());
        Ok(model)
    })
    .await?;
    Ok(Json(model))
}

async fn connect_external_model(
    State(state): State<AppState>,
    Json(req): Json<ExternalModelRequest>,
) -> Result<Json<ModelRecord>, ApiError> {
    if req.api_key.trim().is_empty() {
        return Err(api_error(
            StatusCode::BAD_REQUEST,
            "API key is required.",
            "invalid_external_model",
        ));
    }
    let model = ModelRecord {
        id: req.id.clone(),
        name: req.name,
        status: "ready".into(),
        provider_kind: "external".into(),
        model_path: None,
        runtime_model_name: Some(req.model_name),
        format: Some("OpenAI-compatible API".into()),
        source: Some(req.source),
        engine: Some("External API".into()),
        quant: None,
        hf_repo: None,
        hf_filename: None,
        bytes_downloaded: None,
        total_bytes: None,
        progress: None,
        install_error: None,
        api_base: Some(req.api_base),
        api_key_configured: Some(true),
        capability_status: "planned".into(),
        capability_summary:
            "External OpenAI-compatible API details are connected as metadata only; Fathom does not proxy chat to external endpoints yet.".into(),
        backend_lanes: vec!["external-openai".into()],
        task: None,
        download_manifest: None,
    };
    let model = mutate_model_state_and_persist(&state, |store| {
        upsert_model(&mut store.models, model.clone());
        Ok(model)
    })
    .await?;
    Ok(Json(model))
}

async fn activate_model(
    State(state): State<AppState>,
    Path(id): Path<String>,
) -> Result<Json<ModelRecord>, ApiError> {
    let model = mutate_model_state_and_persist(&state, |store| {
        let model = store
            .models
            .iter()
            .find(|item| item.id == id)
            .cloned()
            .ok_or_else(|| {
                api_error(StatusCode::NOT_FOUND, "Model not found.", "model_not_found")
            })?;
        if model.provider_kind == "external" {
            return Err(api_error(
                StatusCode::NOT_IMPLEMENTED,
                "External OpenAI-compatible API entries are connected metadata only. Fathom does not proxy chat to external endpoints yet, so they cannot be activated for local chat readiness.",
                "external_proxy_not_implemented",
            ));
        }
        if !is_runnable_model(&model) {
            return Err(api_error(
                StatusCode::NOT_IMPLEMENTED,
                format!(
                    "{} was detected as {}, but Fathom does not have a real loader/runtime for it yet.",
                    model.name,
                    model.format.clone().unwrap_or_else(|| "Unknown".into())
                ),
                "not_implemented",
            ));
        }
        store.active_model_id = Some(model.id.clone());
        Ok(model)
    })
    .await?;
    Ok(Json(model))
}

async fn cancel_model_download(
    State(state): State<AppState>,
    Path(id): Path<String>,
) -> Result<StatusCode, ApiError> {
    let store = state.inner.lock().await;
    let model = store
        .models
        .iter()
        .find(|item| item.id == id)
        .ok_or_else(|| api_error(StatusCode::NOT_FOUND, "Model not found.", "model_not_found"))?;

    if model.status == "downloading" || model.status == "canceling" {
        return Err(api_error(
            StatusCode::NOT_IMPLEMENTED,
            "Fathom does not have a cancellable background download task for this model yet.",
            "model_download_cancellation_not_supported",
        ));
    }

    Err(api_error(
        StatusCode::CONFLICT,
        "This model is not actively downloading, so there is no download to cancel.",
        "model_download_not_active",
    ))
}

async fn install_model() -> ApiError {
    api_error(
        StatusCode::NOT_IMPLEMENTED,
        "Bundled model downloads are not implemented in Fathom yet. Register a local artifact to test detection.",
        "model_install_not_implemented",
    )
}

async fn install_catalog_model(
    State(state): State<AppState>,
    Json(req): Json<CatalogInstallRequest>,
) -> Result<Json<ModelRecord>, ApiError> {
    let spec = catalog_specs()
        .into_iter()
        .find(|spec| spec.repo_id == req.repo_id && spec.primary_filename == req.filename)
        .ok_or_else(|| {
            api_error(
                StatusCode::BAD_REQUEST,
                "Unknown catalog model.",
                "catalog_model_not_found",
            )
        })?;

    let license_acknowledged = catalog_install_license_acknowledged(&req);

    validate_catalog_license_ack(&spec, license_acknowledged)?;

    let final_model_dir = fathom_model_dir(&spec.repo_id);
    let staging_dir = create_catalog_install_staging_dir(&spec)
        .await
        .map_err(|error| raw_api_error(error, "catalog_staging_failed"))?;
    let install_result = async {
        let mut downloaded_bytes = 0_i64;
        for file in &spec.files {
            let bytes = download_hf_file(&spec, file)
                .await
                .map_err(catalog_error_json)?;
            downloaded_bytes += bytes.len() as i64;
            write_verified_catalog_file(&spec, file, &staging_dir, &bytes)
                .await
                .map_err(catalog_error_json)?;
        }

        verify_catalog_download_total(&spec, downloaded_bytes).map_err(catalog_error_json)?;
        let manifest =
            build_download_manifest(&spec, license_acknowledged).map_err(catalog_error_json)?;
        write_download_manifest(&staging_dir, &manifest)
            .await
            .map_err(|error| raw_api_error(error, "catalog_manifest_write_failed"))?;

        let package = inspect_model_package(&staging_dir).map_err(|error| {
            api_error(
                StatusCode::INTERNAL_SERVER_ERROR,
                error,
                "catalog_inspect_failed",
            )
        })?;
        let capability_report =
            capability_report_for_package(package.clone(), &current_machine_profile());
        promote_staged_catalog_install(&staging_dir, &final_model_dir)
            .await
            .map_err(|error| raw_api_error(error, "catalog_promotion_failed"))?;
        let model = build_catalog_model_record(
            &spec,
            &final_model_dir,
            downloaded_bytes,
            manifest,
            package,
            capability_report,
        );
        mutate_model_state_and_persist(&state, |store| {
            upsert_model(&mut store.models, model.clone());
            Ok(())
        })
        .await?;
        Ok(model)
    }
    .await;

    if install_result.is_err() {
        let _ = tokio::fs::remove_dir_all(&staging_dir).await;
    }

    install_result.map(Json)
}

async fn model_catalog() -> Json<serde_json::Value> {
    let items: Vec<_> = catalog_specs()
        .into_iter()
        .map(|spec| {
            serde_json::json!({
                "catalog_id": spec.catalog_id,
                "title": spec.title,
                "repo_id": spec.repo_id,
                "filename": spec.primary_filename,
                "revision": spec.revision,
                "source": "Hugging Face",
                "source_url": format!("https://huggingface.co/{}", spec.repo_id),
                "license": spec.license,
                "license_status": catalog_license_status(spec.license).as_str(),
                "license_acknowledgement_required": catalog_license_acknowledgement_required(&spec),
                "license_warning": catalog_license_warning(&spec),
                "quant": format_from_filename(spec.primary_filename),
                "size_bytes": spec.size_bytes,
                "files": spec.files.iter().map(|file| serde_json::json!({"filename": file.filename, "size_bytes": file.size_bytes, "sha256": file.sha256})).collect::<Vec<_>>(),
                "description": spec.description,
            })
        })
        .collect();

    Json(serde_json::json!({
        "items": items,
        "next_cursor": null
    }))
}

async fn v1_health(State(state): State<AppState>) -> Json<serde_json::Value> {
    let store = state.inner.lock().await;
    Json(serde_json::json!({
        "ok": true,
        "engine": "fathom",
        "generation_ready": store.models.iter().any(is_v1_chat_runnable_model)
    }))
}

async fn v1_models(State(state): State<AppState>) -> Json<serde_json::Value> {
    let store = state.inner.lock().await;
    Json(serde_json::json!({
        "object": "list",
        "data": store.models.iter().filter(|m| is_v1_chat_runnable_model(m)).map(|m| serde_json::json!({
            "id": m.id,
            "object": "model",
            "created": 0,
            "owned_by": "fathom",
            "fathom": {
                "provider_kind": m.provider_kind,
                "status": m.status,
                "capability_status": m.capability_status,
                "capability_summary": m.capability_summary,
                "backend_lanes": m.backend_lanes,
            }
        })).collect::<Vec<_>>()
    }))
}

async fn v1_embeddings(
    State(state): State<AppState>,
    Json(req): Json<V1EmbeddingsRequest>,
) -> (StatusCode, Json<serde_json::Value>) {
    if req
        .encoding_format
        .as_deref()
        .is_some_and(|format| format != "float")
    {
        return error_json(
            StatusCode::BAD_REQUEST,
            "Only encoding_format='float' is supported for /v1/embeddings. Base64 embeddings are not implemented.",
            "invalid_request",
        );
    }
    let Some(model_id) = req
        .model
        .as_deref()
        .filter(|model| !model.trim().is_empty())
    else {
        return error_json(
            StatusCode::BAD_REQUEST,
            "model is required for /v1/embeddings.",
            "model_not_found",
        );
    };

    match run_embedding_model(&state, model_id, req.input.into_vec(), false).await {
        Ok(run) => (
            StatusCode::OK,
            Json(serde_json::json!({
                "object": "list",
                "data": embedding_data_json(&run.output, true),
                "model": run.model.id,
                "usage": {
                    "prompt_tokens": 0,
                    "total_tokens": 0
                },
                "fathom": {
                    "runtime": run.runtime_label,
                    "embedding_dimension": run.output.dimension,
                    "metrics": run.output.metrics,
                    "normalize": run.normalize,
                    "scope": "verified local embedding runtime only"
                }
            })),
        ),
        Err(error) => error,
    }
}

fn render_chat_prompt_for_model(
    model_path: &str,
    messages: &[ChatMessage],
    options: &PromptRenderOptions,
) -> anyhow::Result<String> {
    let package = inspect_model_package(FsPath::new(model_path))?;
    if let Some(template) = package
        .tokenizer
        .as_ref()
        .and_then(|tokenizer| tokenizer.chat_template.as_ref())
    {
        return render_hf_chat_template_prompt(template, messages, options);
    }

    PlainRolePromptRenderer.render_chat_prompt(messages, options)
}

const DEFAULT_RETRIEVAL_CONTEXT_CHARS: usize = 4_000;
const MAX_RETRIEVAL_CONTEXT_CHARS: usize = 16_000;

fn normalize_retrieval_context_chars(requested: Option<usize>) -> usize {
    requested
        .unwrap_or(DEFAULT_RETRIEVAL_CONTEXT_CHARS)
        .clamp(1, MAX_RETRIEVAL_CONTEXT_CHARS)
}

fn truncate_to_char_boundary(value: &str, max_chars: usize) -> &str {
    if value.chars().count() <= max_chars {
        return value;
    }

    let byte_end = value
        .char_indices()
        .nth(max_chars)
        .map(|(index, _)| index)
        .unwrap_or(value.len());
    &value[..byte_end]
}

fn retrieval_context_from_hits(
    index: RetrievalIndexSummary,
    metric: VectorSearchMetric,
    requested_top_k: usize,
    hits: Vec<VectorSearchHit>,
    max_context_chars: usize,
) -> (Option<ChatMessage>, FathomRetrievalMetadata) {
    let mut context = String::from(
        "Fathom retrieval context (caller-supplied explicit vectors; no embedding inference was performed). Use these snippets only if relevant.\n",
    );
    let mut inserted_chunk_count = 0;
    let mut hit_metadata = Vec::with_capacity(hits.len());

    for (position, hit) in hits.iter().enumerate() {
        let header = format!(
            "\n[{}] document_id={} chunk_id={} score={:.6}\n",
            position + 1,
            hit.chunk.document_id,
            hit.chunk.id,
            hit.score
        );
        let remaining = max_context_chars.saturating_sub(context.chars().count());
        let text_chars = hit.chunk.text.chars().count();
        let needed = header.chars().count() + text_chars;
        let mut inserted = false;

        if remaining > header.chars().count() {
            context.push_str(&header);
            let remaining_text_chars = max_context_chars.saturating_sub(context.chars().count());
            let snippet = truncate_to_char_boundary(&hit.chunk.text, remaining_text_chars);
            context.push_str(snippet);
            inserted = true;
            inserted_chunk_count += 1;
        } else if needed == 0 {
            inserted = true;
        }

        hit_metadata.push(FathomRetrievalHitMetadata {
            chunk_id: hit.chunk.id.clone(),
            document_id: hit.chunk.document_id.clone(),
            score: hit.score,
            inserted,
            text_chars,
        });

        if context.chars().count() >= max_context_chars {
            break;
        }
    }

    let context_chars = context.chars().count().min(max_context_chars);
    let message = if inserted_chunk_count > 0 {
        Some(ChatMessage {
            role: "system".into(),
            content: context,
        })
    } else {
        None
    };

    (
        message,
        FathomRetrievalMetadata {
            index,
            metric,
            requested_top_k,
            hit_count: hit_metadata.len(),
            inserted_chunk_count,
            context_chars,
            max_context_chars,
            hits: hit_metadata,
        },
    )
}

fn resolve_retrieval_for_chat(
    state: &AppState,
    request: &FathomRetrievalRequest,
) -> Result<(Option<ChatMessage>, FathomRetrievalMetadata), (StatusCode, Json<serde_json::Value>)> {
    if request.index_id.trim().is_empty() {
        return Err(error_json(
            StatusCode::BAD_REQUEST,
            "fathom.retrieval.index_id must not be empty",
            "invalid_request",
        ));
    }

    let state_dir = retrieval_state_dir(state)
        .map_err(|(status, message)| error_json(status, &message, "retrieval_error"))?;
    let index = load_retrieval_index(&state_dir, &request.index_id)
        .map_err(|(status, message)| error_json(status, &message, "retrieval_error"))?;
    let query = EmbeddingVector::new(request.query_vector.clone()).map_err(|error| {
        error_json(
            StatusCode::BAD_REQUEST,
            &error.to_string(),
            "invalid_request",
        )
    })?;
    let top_k = request.top_k.unwrap_or(5).min(20);
    let metric = request.metric.unwrap_or(VectorSearchMetric::Cosine);
    let max_context_chars = normalize_retrieval_context_chars(request.max_context_chars);
    let hits = index.search(&query, top_k, metric).map_err(|error| {
        error_json(
            StatusCode::BAD_REQUEST,
            &error.to_string(),
            "invalid_request",
        )
    })?;

    Ok(retrieval_context_from_hits(
        index.summary(),
        metric,
        top_k,
        hits,
        max_context_chars,
    ))
}

async fn v1_chat_completions(
    State(state): State<AppState>,
    Json(req): Json<V1ChatCompletionRequest>,
) -> (StatusCode, Json<serde_json::Value>) {
    if req.stream.unwrap_or(false) {
        return error_json(
            StatusCode::NOT_IMPLEMENTED,
            "Streaming chat completions are not implemented by Fathom yet. Send stream=false or omit stream for a regular JSON response.",
            "not_implemented",
        );
    }
    if req.messages.is_empty() {
        return error_json(
            StatusCode::BAD_REQUEST,
            "messages must not be empty",
            "invalid_request",
        );
    }
    let selected_model = {
        let store = state.inner.lock().await;
        let requested = req.model.as_ref().or(store.active_model_id.as_ref());
        requested
            .and_then(|id| store.models.iter().find(|model| &model.id == id))
            .cloned()
    };
    let Some(model) = selected_model else {
        return error_json(
            StatusCode::BAD_REQUEST,
            "No runnable model is active or requested.",
            "model_not_found",
        );
    };
    if model.provider_kind == "external" {
        return error_json(
            StatusCode::NOT_IMPLEMENTED,
            "External OpenAI-compatible proxying is not implemented by Fathom yet. No fake inference was produced.",
            "not_implemented",
        );
    }
    if !is_runnable_model(&model)
        || !model
            .backend_lanes
            .iter()
            .any(|lane| lane == "safetensors-hf")
    {
        let is_gguf = model
            .format
            .as_deref()
            .is_some_and(|format| format.eq_ignore_ascii_case("GGUF"))
            || model
                .backend_lanes
                .iter()
                .any(|lane| lane == "gguf-native" || lane == "gguf");
        if is_gguf {
            return error_json(
                StatusCode::NOT_IMPLEMENTED,
                "This GGUF model is metadata-only and not runnable in Fathom. Fathom may inspect and privately retain bounded tokenizer metadata for narrow synthetic GPT-2/BPE and Llama/SentencePiece GGUF shapes, with private fixture-scoped Llama/SentencePiece encode/decode parity helpers, but public/runtime tokenizer execution, runtime weight loading, general dequantization, quantized kernels, an architecture runtime, and generation are still missing. No fake inference was produced.",
                "not_implemented",
            );
        }
        let is_blocked_pytorch_bin = model.capability_status == "blocked"
            || model
                .format
                .as_deref()
                .is_some_and(|format| format.eq_ignore_ascii_case("PyTorchBin"))
            || model
                .backend_lanes
                .iter()
                .any(|lane| lane == "pytorch-trusted-import");
        if is_blocked_pytorch_bin {
            return error_json(
                StatusCode::NOT_IMPLEMENTED,
                "This PyTorch .bin model is blocked because legacy pickle artifacts can execute code. Fathom requires an explicit trusted-import safety policy before any PyTorch/pickle/bin artifact can be loaded, routed, or used for generation. No fake inference was produced.",
                "not_implemented",
            );
        }
        return error_json(
            StatusCode::NOT_IMPLEMENTED,
            "This model is known to Fathom, but no real local generation backend is implemented for its format yet. No fake inference was produced.",
            "not_implemented",
        );
    }
    let max_tokens = match normalize_max_tokens(req.max_tokens) {
        Ok(max_tokens) => max_tokens,
        Err(error) => {
            return error_json(
                StatusCode::BAD_REQUEST,
                &error.to_string(),
                "invalid_request",
            )
        }
    };
    let options = match (GenerationOptions {
        temperature: req.temperature.unwrap_or_else(default_chat_temperature),
        top_k: req.top_k.or_else(default_chat_top_k),
        top_p: req.top_p.or_else(default_chat_top_p),
    })
    .validate()
    {
        Ok(options) => options,
        Err(error) => {
            return error_json(
                StatusCode::BAD_REQUEST,
                &error.to_string(),
                "invalid_request",
            )
        }
    };
    let Some(model_path) = model.model_path.clone() else {
        return error_json(
            StatusCode::BAD_REQUEST,
            "Runnable local model is missing a model_path.",
            "invalid_model",
        );
    };
    let (retrieval_message, retrieval_metadata) = match req.retrieval.as_ref() {
        Some(retrieval_request) => match resolve_retrieval_for_chat(&state, retrieval_request) {
            Ok((message, metadata)) => (message, Some(metadata)),
            Err(error) => return error,
        },
        None => (None, None),
    };
    let messages = if let Some(message) = retrieval_message {
        let mut augmented = Vec::with_capacity(req.messages.len() + 1);
        augmented.push(message);
        augmented.extend(req.messages.clone());
        augmented
    } else {
        req.messages.clone()
    };
    let render_options = PromptRenderOptions {
        add_generation_prompt: true,
    };
    let prompt = match render_chat_prompt_for_model(&model_path, &messages, &render_options) {
        Ok(prompt) => prompt,
        Err(error) => {
            let message = error.to_string();
            let (status, code) = if message.contains("chat_template_not_supported")
                || message.contains("only the common ChatML/Qwen-style")
            {
                (StatusCode::NOT_IMPLEMENTED, "chat_template_not_supported")
            } else {
                (StatusCode::BAD_REQUEST, "invalid_request")
            };
            return error_json(status, &message, code);
        }
    };
    let generation =
        match generate_with_candle_hf_options(&model_path, &prompt, max_tokens, options) {
            Ok(generation) => generation,
            Err(error) => {
                return error_json(
                    StatusCode::INTERNAL_SERVER_ERROR,
                    &error.to_string(),
                    "generation_error",
                )
            }
        };
    if generation.text.trim().is_empty() {
        return error_json(
            StatusCode::INTERNAL_SERVER_ERROR,
            "The model produced only whitespace or special tokens. Try a more specific completion-style prompt.",
            "empty_completion",
        );
    }
    let completion_id = format!("chatcmpl-{}", Uuid::new_v4());
    (
        StatusCode::OK,
        Json(serde_json::json!({
            "id": completion_id,
            "object": "chat.completion",
            "created": chrono::Utc::now().timestamp(),
            "model": model.id,
            "choices": [{
                "index": 0,
                "message": {"role": "assistant", "content": generation.text},
                "finish_reason": generation.finish_reason
            }],
            "usage": {
                "prompt_tokens": generation.prompt_tokens,
                "completion_tokens": generation.completion_tokens,
                "total_tokens": generation.prompt_tokens + generation.completion_tokens
            },
            "fathom": {
                "runtime": "candle-safetensors-hf",
                "metrics": generation.metrics,
                "retrieval": retrieval_metadata
            }
        })),
    )
}

fn error_json(
    status: StatusCode,
    message: &str,
    error_type: &str,
) -> (StatusCode, Json<serde_json::Value>) {
    (
        status,
        Json(serde_json::json!({
            "error": {
                "message": message,
                "type": error_type,
                "code": error_type,
                "param": null
            }
        })),
    )
}

fn api_error(
    status: StatusCode,
    message: impl std::fmt::Display,
    error_type: &'static str,
) -> ApiError {
    error_json(status, &message.to_string(), error_type)
}

fn raw_api_error(error: (StatusCode, String), error_type: &'static str) -> ApiError {
    api_error(error.0, error.1, error_type)
}

fn retrieval_load_error_json(error: (StatusCode, String)) -> ApiError {
    let code = if error.0 == StatusCode::NOT_FOUND {
        "retrieval_index_not_found"
    } else {
        "retrieval_state_error"
    };
    raw_api_error(error, code)
}

fn catalog_install_license_acknowledged(req: &CatalogInstallRequest) -> bool {
    req.accept_license.unwrap_or(false) || req.accepted_license.unwrap_or(false)
}

fn catalog_license_status(license: &str) -> CatalogLicenseStatus {
    let normalized = license.trim().to_ascii_lowercase();
    if normalized.is_empty() || normalized == "unknown" || normalized == "other" {
        return CatalogLicenseStatus::Unknown;
    }

    if normalized.contains("noncommercial")
        || normalized.contains("non-commercial")
        || normalized.contains("cc-by-nc")
        || normalized.contains("cc-by-sa-nc")
        || normalized.ends_with("-nc")
        || normalized.contains("llama")
    {
        return CatalogLicenseStatus::Restrictive;
    }

    match normalized.as_str() {
        "apache-2.0"
        | "mit"
        | "bsd"
        | "bsd-2-clause"
        | "bsd-3-clause"
        | "isc"
        | "cc0-1.0"
        | "openrail"
        | "bigscience-openrail-m" => CatalogLicenseStatus::Permissive,
        _ => CatalogLicenseStatus::Unknown,
    }
}

fn catalog_license_acknowledgement_required(spec: &CatalogModelSpec) -> bool {
    catalog_license_status(spec.license) != CatalogLicenseStatus::Permissive
}

fn catalog_license_warning(spec: &CatalogModelSpec) -> Option<&'static str> {
    match catalog_license_status(spec.license) {
        CatalogLicenseStatus::Permissive => None,
        CatalogLicenseStatus::Unknown => Some(
            "Fathom could not classify this catalog license as permissive. Review the model card/license before downloading.",
        ),
        CatalogLicenseStatus::Restrictive => Some(
            "This catalog license may include use restrictions. Review the model card/license before downloading.",
        ),
    }
}

fn catalog_license_policy_note() -> &'static str {
    "Recorded from Fathom catalog metadata at install time; review the upstream model card and license before use."
}

fn validate_catalog_license_ack(
    spec: &CatalogModelSpec,
    acknowledged: bool,
) -> Result<(), ApiError> {
    if !catalog_license_acknowledgement_required(spec) || acknowledged {
        return Ok(());
    }

    Err(api_error(
        StatusCode::BAD_REQUEST,
        format!(
            "Review the catalog license before installing {}. Fathom lists this entry as {} and requires explicit acknowledgement for unknown or restrictive licenses.",
            spec.title,
            catalog_license_status(spec.license).as_str()
        ),
        "catalog_license_ack_required",
    ))
}

fn catalog_error_json(error: (StatusCode, String)) -> ApiError {
    let message = error.1.as_str();
    let code = match error.0 {
        StatusCode::BAD_REQUEST if message.contains("pinned Hugging Face revision") => {
            "catalog_revision_missing"
        }
        StatusCode::BAD_REQUEST if message.contains("missing size metadata") => {
            "catalog_size_metadata_missing"
        }
        StatusCode::BAD_REQUEST if message.contains("missing SHA256 metadata") => {
            "catalog_sha256_metadata_missing"
        }
        StatusCode::BAD_REQUEST if message.contains("safe relative path") => "catalog_path_unsafe",
        StatusCode::BAD_REQUEST => "invalid_catalog_request",
        StatusCode::BAD_GATEWAY if message.contains("server reported") => {
            "catalog_content_length_mismatch"
        }
        StatusCode::BAD_GATEWAY if message.contains("SHA256") => "catalog_sha256_mismatch",
        StatusCode::BAD_GATEWAY if message.contains("expected") => "catalog_size_mismatch",
        StatusCode::BAD_GATEWAY if message.starts_with("Could not read") => {
            "catalog_download_read_failed"
        }
        StatusCode::BAD_GATEWAY if message.starts_with("Could not download") => {
            "catalog_download_failed"
        }
        StatusCode::BAD_GATEWAY => "catalog_integrity_failed",
        StatusCode::INTERNAL_SERVER_ERROR => "catalog_filesystem_error",
        _ => "catalog_install_failed",
    };
    raw_api_error(error, code)
}

fn default_chat_max_tokens() -> usize {
    48
}

fn default_chat_temperature() -> f32 {
    0.8
}

fn default_chat_top_k() -> Option<usize> {
    Some(40)
}

fn default_chat_top_p() -> Option<f32> {
    Some(0.95)
}

fn demo_generation_options() -> GenerationOptions {
    GenerationOptions {
        temperature: default_chat_temperature(),
        top_k: default_chat_top_k(),
        top_p: default_chat_top_p(),
    }
}

fn normalize_max_tokens(max_tokens: Option<usize>) -> anyhow::Result<usize> {
    let requested = max_tokens.unwrap_or_else(default_chat_max_tokens);
    anyhow::ensure!(requested > 0, "max_tokens must be greater than 0");
    Ok(requested.min(128))
}

fn catalog_specs() -> Vec<CatalogModelSpec> {
    vec![
        CatalogModelSpec {
            catalog_id: "hf-vijaymohan-gpt2-tinystories-10m",
            title: "TinyStories GPT-2 10M",
            repo_id: "vijaymohan/gpt2-tinystories-from-scratch-10m",
            revision: Some("05adc1f2172f56876246fa05eb1ddb1c3bb7b6e8"),
            primary_filename: "model.safetensors",
            files: vec![
                catalog_file("config.json", 756, "437a502c1cc8cca97a3d7a0a9dc5a2877a86640564b9a232b363eab18bcd1325"),
                catalog_file("merges.txt", 144_367, "c31b046c27836a02b0f52993c454dee31d0d66f08ecf8f6d3cf1771f5bbc9ce0"),
                catalog_file("model.safetensors", 42_586_776, "ec681106644d9eb8ee57ff38d527684bb65d3ac010661e0371b995127b1c3aec"),
                catalog_file("tokenizer.json", 1_138_770, "e279fe0460f78586cd73d36302df2453e112e36734d274bba5012e59f60e5190"),
                catalog_file("tokenizer_config.json", 1_213, "25016919119007ef159e45aeda58e95815fbaf9ab81738e8cbdfde9b0efcbf88"),
                catalog_file("vocab.json", 248_934, "9fb09c105621875dec32450aa3b01c3cc5741b080297b575f536ce13b8fe0b8a"),
            ],
            size_bytes: Some(44_120_816),
            license: "mit",
            description: "Tiny trained GPT-2-compatible SafeTensors/HF causal-LM trained on TinyStories. Good for quick human-visible local generation demos on tester machines; style is story-like rather than general chat.",
        },
        CatalogModelSpec {
            catalog_id: "hf-distilgpt2",
            title: "DistilGPT-2",
            repo_id: "distilbert/distilgpt2",
            revision: Some("2290a62682d06624634c1f46a6ad5be0f47f38aa"),
            primary_filename: "model.safetensors",
            files: vec![
                catalog_file("config.json", 762, "4ec5947c1d59fee6212cdf3b0ec1a53eac02092554c5ff0a733488cbd2c64f3a"),
                catalog_file("generation_config.json", 124, "b90eadacf585a743a30ea51e8b5c88b8d282a2a34dc0c7e556d0987cdbd68805"),
                catalog_file("merges.txt", 456_318, "1ce1664773c50f3e0cc8842619a93edc4624525b728b188a9e0be33b7726adc5"),
                catalog_file("model.safetensors", 352_824_413, "e1ff18884359fe8beb795a5f414feb85a6ce3d929ad019c0d958c039d2b94a1b"),
                catalog_file("tokenizer.json", 1_355_256, "8414cab924d8b9b33013f0d221c5862f365ee9be39c5c2bfae8a5a9e970478a6"),
                catalog_file("tokenizer_config.json", 26, "5e04eb606e3a1583530a42e36c2a6b6615c86f34fe77e44d9ddeb43ff940931f"),
                catalog_file("vocab.json", 1_042_301, "196139668be63f3b5d6574427317ae82f612a97c5d1cdaf36ed2256dbf636783"),
            ],
            size_bytes: Some(355_679_200),
            license: "apache-2.0",
            description: "Small trained GPT-2-compatible SafeTensors/HF causal-LM for testing real custom Rust generation with visible text.",
        },
        CatalogModelSpec {
            catalog_id: "hf-intel-tiny-random-gpt2",
            title: "Intel Tiny Random GPT-2 fixture",
            repo_id: "Intel/tiny-random-gpt2",
            revision: Some("6538c1ee0523bdc149e4ad9f8b7a0258ead8781a"),
            primary_filename: "model.safetensors",
            files: vec![
                catalog_file("config.json", 955, "487248d6a21904c59392c001303040073203d40c5a258bf5084ed0bcdafda3e5"),
                catalog_file("generation_config.json", 119, "95274ad3324d4d34da19992fc297540a148638989733af6c07998f262f898a73"),
                catalog_file("model.safetensors", 27_845_776, "3225fe8f7d54b4bb2b5486aaaf5fa3afd921a3188396fe0f0a2e4b1c3d3727c2"),
                catalog_file("special_tokens_map.json", 90, "c0b3c279b6ecdb71996a86ffb4d4ab94dfdb5df95f00bac9515688faef2ff5dd"),
                catalog_file("tokenizer.json", 17_454, "e62fc5f642751cf5d0bf542adb6aba88624b696f26f5f40aff7e4a53788eb85c"),
                catalog_file("tokenizer_config.json", 236, "1e664fe1a5bc404dcb204a76c76bc4100c91bd12c66098d8a8090b11b6ee6f0d"),
                catalog_file("vocab.json", 10_419, "eda8bd11b5763cec815aa93b5524cf7aeed42b49c772134355451460f46c299c"),
            ],
            size_bytes: Some(27_875_049),
            license: "apache-2.0",
            description: "Very small public SafeTensors/HF causal-LM fixture for backend smoke tests. It is random, so output may be whitespace or gibberish; use DistilGPT-2 for a human-visible local test.",
        },
        CatalogModelSpec {
            catalog_id: "hf-stas-tiny-random-llama-2",
            title: "Stas Tiny Random Llama 2 fixture",
            repo_id: "stas/tiny-random-llama-2",
            revision: Some("3579d71fd57e04f5a364d824d3a2ec3e913dbb67"),
            primary_filename: "model.safetensors",
            files: vec![
                catalog_file("config.json", 680, "9197475bfcc987a4f9361dbc22b33397b101372c137c228b6a6fd7e4adf21622"),
                catalog_file("generation_config.json", 116, "40e6ecbcedfc2b67b7fa8ba37216c9546c18c00242020b2b34f0b58c3558f680"),
                catalog_file("model.safetensors", 210_712, "852db1b39acb2336abc997440c6f6d6e4ab640f91e5e2aa9e2488d5794159d30"),
                catalog_file("special_tokens_map.json", 414, "6fa06efa2785e450051989a6f8fb4416b10149ded485ddd3f127a40734f5cfd0"),
                catalog_file("tokenizer.json", 64_223, "0afe36ee1358ce1fa277f4eac935250bb90253ed5c27867eb6ff376ded7d1980"),
                catalog_file("tokenizer.model", 499_723, "9e556afd44213b6bd1be2b850ebbbd98f5481437a8021afaf58ee7fb1818d347"),
                catalog_file("tokenizer_config.json", 918, "e3dd4025f0dc9f23a8bea840afceb093dc4c3f250f6555ec0c536cc0615e0695"),
            ],
            size_bytes: Some(776_786),
            license: "apache-2.0",
            description: "Tiny BF16 LlamaForCausalLM SafeTensors/HF fixture for proving Fathom's custom Rust Llama lane, including tied-embedding handling. Output is random and only validates the runtime path.",
        },
        CatalogModelSpec {
            catalog_id: "hf-huggingfacetb-smollm2-135m-instruct",
            title: "SmolLM2 135M Instruct",
            repo_id: "HuggingFaceTB/SmolLM2-135M-Instruct",
            revision: Some("12fd25f77366fa6b3b4b768ec3050bf629380bac"),
            primary_filename: "model.safetensors",
            files: vec![
                catalog_file("config.json", 861, "8eb740e8bbe4cff95ea7b4588d17a2432deb16e8075bc5828ff7ba9be94d982a"),
                catalog_file("generation_config.json", 132, "87b916edaaab66b3899b9d0dd0752727dff6666686da0504d89ae0a6e055a013"),
                catalog_file("model.safetensors", 269_060_552, "5af571cbf074e6d21a03528d2330792e532ca608f24ac70a143f6b369968ab8c"),
                catalog_file("special_tokens_map.json", 655, "2b7379f3ae813529281a5c602bc5a11c1d4e0a99107aaa597fe936c1e813ca52"),
                catalog_file("tokenizer.json", 2_104_556, "9ca9acddb6525a194ec8ac7a87f24fbba7232a9a15ffa1af0c1224fcd888e47c"),
                catalog_file("tokenizer_config.json", 3_764, "4ec77d44f62efeb38d7e044a1db318f6a939438425312dfa333b8382dbad98df"),
            ],
            size_bytes: Some(271_170_520),
            license: "apache-2.0",
            description: "Larger Llama-family tied-embedding instruct demo using SmolLM2 135M SafeTensors/HF. Useful for showing Fathom can run a real chat-tuned Llama-style package; it is a larger demo, not the fastest first-run option.",
        },
        CatalogModelSpec {
            catalog_id: "hf-yujiepan-qwen2-tiny-random",
            title: "Yujiepan Tiny Random Qwen2 fixture",
            repo_id: "yujiepan/qwen2-tiny-random",
            revision: Some("b01b9c82aaf1efb4c26c94b6342d611b397245ff"),
            primary_filename: "model.safetensors",
            files: vec![
                catalog_file("config.json", 700, "0410c6d74d121e9bd6260c5cbf850249171eb958ce6e89fa048a578be244d012"),
                catalog_file("generation_config.json", 243, "9fe552f08e1c6dcb1822662c9a102bacb03028664c1bad12dad359691a3a79e6"),
                catalog_file("merges.txt", 1_671_853, "8831e4f1a044471340f7c0a83d7bd71306a5b867e95fd870f74d0c5308a904d5"),
                catalog_file("model.safetensors", 4_871_288, "5fe8ad6fa1496ccf5a51f6f7b9db94c3b1e0587c1c97ff16a326d7c501450ce9"),
                catalog_file("special_tokens_map.json", 367, "f4f79e08d97f4d1c87f8d89264f525c8789da3b73b3bb55d1e12f692f41a7b1b"),
                catalog_file("tokenizer.json", 7_028_015, "f7c9b2dba4a296b1aa76c16a34b8225c0c118978400d4bb66bff0902d702f5b8"),
                catalog_file("tokenizer_config.json", 1_300, "5603a6b602c42df7196593907e785c3a272afd0a1880c63f6c37dacf505f2dce"),
                catalog_file("vocab.json", 2_776_833, "ca10d7e9fb3ed18575dd1e277a2579c16d108e32f27439684afa0e10b1440910"),
            ],
            size_bytes: Some(16_350_599),
            license: "unknown",
            description: "Tiny BF16 Qwen2ForCausalLM SafeTensors/HF fixture. It is random/gibberish, but it verifies Fathom's custom Rust Qwen2 SafeTensors lane with tokenizer and generation wiring.",
        },
        CatalogModelSpec {
            catalog_id: "hf-echarlaix-tiny-random-phi",
            title: "Echarlaix Tiny Random Phi fixture",
            repo_id: "echarlaix/tiny-random-PhiForCausalLM",
            revision: Some("acb0ac41d787dd9c9bbed0a70a16ed05c9e2e8db"),
            primary_filename: "model.safetensors",
            files: vec![
                catalog_file("config.json", 881, "10ccbd955d9f5fb9a5d112c3f3e66dfbe6fd6a56280aab6de6534680a9b96ec8"),
                catalog_file("generation_config.json", 132, "7f7a6ba749294f1994299afc989d8f4d7d84707b3a60f5e91fedcbb09350291a"),
                catalog_file("model.safetensors", 323_520, "6fbbc177683bcd0c8d694d552461d9dba3cd6e7f5a883cb8c6c6cce36ce6882e"),
                catalog_file("special_tokens_map.json", 99, "6f50ab5a5a509a1c309d6171f339b196a900dc9c99ad0408ff23bb615fdae7ad"),
                catalog_file("tokenizer.json", 31_312, "7d146281cf62cafc0f3357090c8b7902c7e918a1fb10916996aea3ac7eb7d2ad"),
                catalog_file("tokenizer_config.json", 237, "7671fbb5b3d610e6e11d4f5fc78d3a7716e8846112ac7e0f72124caedf887570"),
            ],
            size_bytes: Some(356_181),
            license: "apache-2.0",
            description: "Tiny F32 PhiForCausalLM SafeTensors/HF fixture. It is random/gibberish, but it verifies Fathom's custom Rust Phi runtime path with tokenizer and generation wiring.",
        },
        CatalogModelSpec {
            catalog_id: "hf-fxmarty-tiny-random-gemma",
            title: "Fxmarty Tiny Random Gemma fixture",
            repo_id: "fxmarty/tiny-random-GemmaForCausalLM",
            revision: Some("ca53c1ebb8b142110b71662d702e4923e5426cb4"),
            primary_filename: "model.safetensors",
            files: vec![
                catalog_file("config.json", 654, "600dcb12c2b361f6d1773df058a716bfc791eb89a6b5f44c8cd2742d4c4fcd04"),
                catalog_file("generation_config.json", 137, "af2d463e4dc9c84e9eb54a0cc2734ec1b4b12215a17c533d0360147c33a06982"),
                catalog_file("model.safetensors", 32_776_472, "049f839dfbaaf81a9bf876984bc72d7e3261c3b6b7dd092246fe1d15c3037a8c"),
                catalog_file("special_tokens_map.json", 555, "db82f8bd9b25d14f9c788e6bde64de84d42f1c2538f1c245ba6cb3e872d14b18"),
                catalog_file("tokenizer.json", 17_477_553, "d0d908b4f9326e0998815690e325b6abbd378978553e10627924dd825db7e243"),
                catalog_file("tokenizer_config.json", 1_108, "acb0c767881929483d11af25020fc4744c474eed09101a5752d4d443a13d3e55"),
            ],
            size_bytes: Some(50_256_479),
            license: "mit",
            description: "Tiny random GemmaForCausalLM SafeTensors/HF fixture. It is random/gibberish, but it verifies Fathom's custom Rust Gemma runtime path with tied output projection, tokenizer, and generation wiring.",
        },
        CatalogModelSpec {
            catalog_id: "hf-sanchit-gandhi-tiny-random-mistral-1-layer",
            title: "Sanchit Gandhi Tiny Random Mistral 1-layer fixture",
            repo_id: "sanchit-gandhi/tiny-random-MistralForCausalLM-1-layer",
            revision: Some("53f378eafc3840b0a33e33ea5137e5978ffe4434"),
            primary_filename: "model.safetensors",
            files: vec![
                catalog_file("config.json", 659, "dcf08503948a50d00812c45024a265553a563d81ddc08cef1989989e8614e4df"),
                catalog_file("generation_config.json", 116, "c52d1b1cdfbf6d8e93153226de203a5c84b8c1528b85058824880f36f7cec6a8"),
                catalog_file("model.safetensors", 4_112_280, "f23831f620e562a02fdd1727ba451d408838bcf0bd91290ed40a4df70e90ec80"),
                catalog_file("special_tokens_map.json", 552, "4859e5dbde90e059988a0a2136d8df3f2773d4d2fc4c4543690028f0b2166e7f"),
                catalog_file("tokenizer.json", 1_795_303, "fc4f0bd70b3709312d9d1d9e5ba674794b6bc5abc17429897a540f93882f25fc"),
                catalog_file("tokenizer_config.json", 969, "166ae68a8db095da6eee2606a9d76fcbcc663dd44a9149070ed3a36796f9f17e"),
            ],
            size_bytes: Some(5_909_879),
            license: "unknown",
            description: "Tiny F32 MistralForCausalLM SafeTensors/HF fixture. It is random/gibberish, but it verifies Fathom's custom Rust Mistral runtime path with tokenizer and generation wiring.",
        },
        CatalogModelSpec {
            catalog_id: "hf-sentence-transformers-all-minilm-l6-v2-safetensors",
            title: "SentenceTransformers all-MiniLM-L6-v2 SafeTensors embeddings",
            repo_id: "sentence-transformers/all-MiniLM-L6-v2",
            revision: Some("c9745ed1d9f207416be6d2e6f8de32d1f16199bf"),
            primary_filename: "model.safetensors",
            files: vec![
                catalog_file("config.json", 612, "953f9c0d463486b10a6871cc2fd59f223b2c70184f49815e7efbcab5d8908b41"),
                catalog_file("model.safetensors", 90_868_376, "53aa51172d142c89d9012cce15ae4d6cc0ca6895895114379cacb4fab128d9db"),
                catalog_file("modules.json", 349, "84e40c8e006c9b1d6c122e02cba9b02458120b5fb0c87b746c41e0207cf642cf"),
                catalog_file("special_tokens_map.json", 112, "303df45a03609e4ead04bc3dc1536d0ab19b5358db685b6f3da123d05ec200e3"),
                catalog_file("tokenizer.json", 466_247, "be50c3628f2bf5bb5e3a7f17b1f74611b2561a3a27eeab05e5aa30f411572037"),
                catalog_file("tokenizer_config.json", 350, "acb92769e8195aabd29b7b2137a9e6d6e25c476a4f15aa4355c233426c61576b"),
                catalog_file("1_Pooling/config.json", 190, "4be450dde3b0273bb9787637cfbd28fe04a7ba6ab9d36ac48e92b11e350ffc23"),
            ],
            size_bytes: Some(91_336_236),
            license: "apache-2.0",
            description: "Default-build Candle BertModel/SentenceTransformers MiniLM text-embedding fixture. Produces local 384-dimensional pooled embeddings for retrieval; it is not a chat/generation model and is excluded from /v1/models.",
        },
        CatalogModelSpec {
            catalog_id: "gguf-llama-tiny",
            title: "Aladar Llama 2 tiny random GGUF metadata fixture",
            repo_id: "aladar/llama-2-tiny-random-GGUF",
            revision: Some("8d5321916486e1d33c46b16990e8da6567785769"),
            primary_filename: "llama-2-tiny-random.gguf",
            files: vec![catalog_file(
                "llama-2-tiny-random.gguf",
                1_750_560,
                "81f226c62d28ed4a1a9b9fa080fcd9f0cc40e0f9d5680036583ff98fbcd035cb",
            )],
            size_bytes: Some(1_750_560),
            license: "mit",
            description: "Pinned tiny random GGUF fixture for metadata/provenance inspection only. Fathom verifies the exact revision, byte size, SHA256, and MIT license, then registers it as metadata-only. Internal synthetic tokenizer-retention checks cover narrow GPT-2/BPE and Llama/SentencePiece-shaped metadata, private fixture-scoped Llama/SentencePiece encode/decode parity helpers exist as readiness groundwork, and internal synthetic payload decode tests cover F32/F16/Q8_0/Q4_0 readiness; these are not public/runtime tokenizer execution, runtime weight loading, general dequantization, quantized kernels, or generation.",
        },
        CatalogModelSpec {
            catalog_id: "onnx-nixiesearch-all-minilm-l6-v2-quantized",
            title: "Nixiesearch all-MiniLM-L6-v2 ONNX embedding fixture",
            repo_id: "nixiesearch/all-MiniLM-L6-v2-onnx",
            revision: Some("1e6ba950da2d9627f0e297996bd2bdb5fdb521cc"),
            primary_filename: "model_quantized.onnx",
            files: vec![
                catalog_file("config.json", 650, "93e8a996c74771a028349d19822798295f31c2a462687dcb133f319f6ca7db50"),
                catalog_file("model_quantized.onnx", 22_972_869, "740b0e562df8a054267cf1ff250f70c307959030880eaa556f5e1b38a71d5c1f"),
                catalog_file("tokenizer.json", 711_661, "da0e79933b9ed51798a3ae27893d3c5fa4a201126cef75586296df9b4d2c62a0"),
            ],
            size_bytes: Some(23_685_180),
            license: "apache-2.0",
            description: "Pinned ONNX text-embedding fixture for the future narrow onnx-embeddings-ort adapter. It is cataloged for metadata/preflight only today; Fathom does not claim ONNX chat or general ONNX LLM support.",
        },
    ]
}

const DOWNLOAD_MANIFEST_FILENAME: &str = "fathom-download-manifest.json";

const CATALOG_INSTALL_TMP_PREFIX: &str = ".fathom-install";

async fn create_catalog_install_staging_dir(
    spec: &CatalogModelSpec,
) -> Result<PathBuf, (StatusCode, String)> {
    let root = fathom_models_root();
    tokio::fs::create_dir_all(&root)
        .await
        .map_err(|error| (StatusCode::INTERNAL_SERVER_ERROR, error.to_string()))?;
    let staging_dir = root.join(format!(
        "{}-{}-{}.tmp",
        CATALOG_INSTALL_TMP_PREFIX,
        slugify(spec.repo_id),
        Uuid::new_v4()
    ));
    tokio::fs::create_dir(&staging_dir)
        .await
        .map_err(|error| (StatusCode::INTERNAL_SERVER_ERROR, error.to_string()))?;
    Ok(staging_dir)
}

fn catalog_file_path(
    root: &FsPath,
    file: &CatalogFileSpec,
) -> Result<PathBuf, (StatusCode, String)> {
    let relative = FsPath::new(file.filename);
    if relative.is_absolute()
        || relative
            .components()
            .any(|component| !matches!(component, std::path::Component::Normal(_)))
    {
        return Err((
            StatusCode::BAD_REQUEST,
            format!(
                "Catalog file {} from Fathom's catalog is not a safe relative path.",
                file.filename
            ),
        ));
    }
    Ok(root.join(relative))
}

async fn write_verified_catalog_file(
    spec: &CatalogModelSpec,
    file: &CatalogFileSpec,
    root: &FsPath,
    bytes: &[u8],
) -> Result<PathBuf, (StatusCode, String)> {
    if let Some(expected_size) = file.size_bytes {
        let actual_size = bytes.len() as i64;
        if actual_size != expected_size {
            return Err((
                StatusCode::BAD_GATEWAY,
                format!(
                    "Refusing {} from {}: downloaded {} bytes, expected {} bytes from Fathom's catalog metadata.",
                    file.filename, spec.repo_id, actual_size, expected_size
                ),
            ));
        }
    }
    verify_catalog_file_sha256(
        spec.repo_id,
        spec.revision.unwrap_or("unknown"),
        file,
        bytes,
    )?;

    let path = catalog_file_path(root, file)?;
    if let Some(parent) = path.parent() {
        tokio::fs::create_dir_all(parent)
            .await
            .map_err(|error| (StatusCode::INTERNAL_SERVER_ERROR, error.to_string()))?;
    }
    tokio::fs::write(&path, bytes)
        .await
        .map_err(|error| (StatusCode::INTERNAL_SERVER_ERROR, error.to_string()))?;
    verify_written_file_size(spec.repo_id, file, &path).await?;
    Ok(path)
}

async fn promote_staged_catalog_install(
    staging_dir: &FsPath,
    final_model_dir: &FsPath,
) -> Result<(), (StatusCode, String)> {
    let Some(root) = final_model_dir.parent() else {
        return Err((
            StatusCode::INTERNAL_SERVER_ERROR,
            "Catalog install target has no parent directory.".into(),
        ));
    };
    tokio::fs::create_dir_all(root)
        .await
        .map_err(|error| (StatusCode::INTERNAL_SERVER_ERROR, error.to_string()))?;

    match tokio::fs::rename(staging_dir, final_model_dir).await {
        Ok(()) => return Ok(()),
        Err(_error) if final_model_dir.exists() => {}
        Err(error) => return Err((StatusCode::INTERNAL_SERVER_ERROR, error.to_string())),
    }

    let backup_dir = root.join(format!(
        "{}-{}-{}.backup",
        CATALOG_INSTALL_TMP_PREFIX,
        final_model_dir
            .file_name()
            .and_then(|name| name.to_str())
            .unwrap_or("model"),
        Uuid::new_v4()
    ));

    tokio::fs::rename(final_model_dir, &backup_dir)
        .await
        .map_err(|error| (StatusCode::INTERNAL_SERVER_ERROR, error.to_string()))?;

    match tokio::fs::rename(staging_dir, final_model_dir).await {
        Ok(()) => {
            let _ = tokio::fs::remove_dir_all(&backup_dir).await;
            Ok(())
        }
        Err(error) => {
            let promote_error = error.to_string();
            if final_model_dir.exists() {
                let _ = tokio::fs::remove_dir_all(final_model_dir).await;
            }
            let _ = tokio::fs::rename(&backup_dir, final_model_dir).await;
            Err((StatusCode::INTERNAL_SERVER_ERROR, promote_error))
        }
    }
}

#[cfg(test)]
fn catalog_install_temp_dir_name(path: &FsPath) -> bool {
    path.file_name()
        .and_then(|name| name.to_str())
        .map(|name| {
            name.starts_with(CATALOG_INSTALL_TMP_PREFIX)
                && (name.ends_with(".tmp") || name.ends_with(".backup"))
        })
        .unwrap_or(false)
}

#[cfg(test)]
async fn cleanup_catalog_install_temp_dirs(root: &FsPath) -> Result<(), (StatusCode, String)> {
    let mut entries = match tokio::fs::read_dir(root).await {
        Ok(entries) => entries,
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => return Ok(()),
        Err(error) => return Err((StatusCode::INTERNAL_SERVER_ERROR, error.to_string())),
    };
    while let Some(entry) = entries
        .next_entry()
        .await
        .map_err(|error| (StatusCode::INTERNAL_SERVER_ERROR, error.to_string()))?
    {
        let path = entry.path();
        if catalog_install_temp_dir_name(&path) {
            let _ = tokio::fs::remove_dir_all(path).await;
        }
    }
    Ok(())
}

fn build_catalog_model_record(
    spec: &CatalogModelSpec,
    model_dir: &FsPath,
    downloaded_bytes: i64,
    manifest: DownloadManifest,
    package: fathom_core::ModelPackage,
    capability_report: fathom_core::PackageCapabilityReport,
) -> ModelRecord {
    let primary_artifact = package.artifacts.iter().find(|artifact| {
        !matches!(
            artifact.format,
            fathom_core::ModelFormat::TokenizerJson
                | fathom_core::ModelFormat::TokenizerConfigJson
                | fathom_core::ModelFormat::SentencePiece
                | fathom_core::ModelFormat::ConfigJson
                | fathom_core::ModelFormat::ChatTemplate
        )
    });
    let id = slugify(&format!("{}-{}", spec.repo_id, spec.primary_filename));
    ModelRecord {
        id: id.clone(),
        name: spec.title.into(),
        status: if capability_report.runnable {
            "ready"
        } else {
            "registered"
        }
        .into(),
        provider_kind: "local".into(),
        model_path: Some(model_dir.display().to_string()),
        runtime_model_name: Some(id),
        format: Some(
            primary_artifact
                .map(|artifact| format!("{:?}", artifact.format))
                .unwrap_or_else(|| "Package".into()),
        ),
        source: Some(
            package
                .model_type
                .as_deref()
                .map(|model_type| format!("Hugging Face · {model_type}"))
                .unwrap_or_else(|| "Hugging Face".into()),
        ),
        engine: Some("Fathom".into()),
        quant: Some(format_from_filename(spec.primary_filename).into()),
        hf_repo: Some(spec.repo_id.into()),
        hf_filename: Some(spec.primary_filename.into()),
        bytes_downloaded: Some(downloaded_bytes),
        total_bytes: Some(
            spec.size_bytes
                .unwrap_or(downloaded_bytes)
                .max(downloaded_bytes),
        ),
        progress: Some(100.0),
        install_error: if capability_report.runnable {
            None
        } else {
            primary_artifact.map(|artifact| artifact.notes.join(" "))
        },
        api_base: None,
        api_key_configured: None,
        capability_status: capability_status_label(&capability_report.best_status).into(),
        capability_summary: capability_report.summary.clone(),
        backend_lanes: capability_report
            .matching_lanes
            .iter()
            .map(|lane| lane.id.to_string())
            .collect(),
        task: model_task_label_for_package(&package),
        download_manifest: Some(manifest),
    }
}

fn build_download_manifest(
    spec: &CatalogModelSpec,
    license_acknowledged: bool,
) -> Result<DownloadManifest, (StatusCode, String)> {
    let Some(revision) = spec.revision else {
        return Err((
            StatusCode::BAD_REQUEST,
            format!(
                "Catalog model {} cannot be installed because it does not have a pinned Hugging Face revision yet.",
                spec.repo_id
            ),
        ));
    };

    let mut files = Vec::with_capacity(spec.files.len());
    for file in &spec.files {
        let Some(size_bytes) = file.size_bytes else {
            return Err((
                StatusCode::BAD_REQUEST,
                format!(
                    "Catalog file {} from {} cannot be installed because it is missing size metadata.",
                    file.filename, spec.repo_id
                ),
            ));
        };
        let Some(sha256) = file.sha256 else {
            return Err((
                StatusCode::BAD_REQUEST,
                format!(
                    "Catalog file {} from {} cannot be installed because it is missing SHA256 metadata.",
                    file.filename, spec.repo_id
                ),
            ));
        };
        files.push(DownloadManifestFile {
            filename: file.filename.into(),
            size_bytes,
            sha256: sha256.into(),
        });
    }

    let license_acknowledgement_required = catalog_license_acknowledgement_required(spec);

    Ok(DownloadManifest {
        schema_version: 1,
        repo_id: spec.repo_id.into(),
        revision: revision.into(),
        source_url: format!("https://huggingface.co/{}", spec.repo_id),
        license: spec.license.into(),
        license_status: Some(catalog_license_status(spec.license).as_str().into()),
        license_acknowledgement_required,
        license_acknowledged: license_acknowledgement_required && license_acknowledged,
        license_policy_note: Some(catalog_license_policy_note().into()),
        installed_at: now(),
        verification_status: "verified".into(),
        files,
    })
}

async fn write_download_manifest(
    model_dir: &FsPath,
    manifest: &DownloadManifest,
) -> Result<(), (StatusCode, String)> {
    let bytes = serde_json::to_vec_pretty(manifest)
        .map_err(|error| (StatusCode::INTERNAL_SERVER_ERROR, error.to_string()))?;
    let manifest_path = model_dir.join(DOWNLOAD_MANIFEST_FILENAME);
    let tmp_path = model_dir.join(format!(
        ".{DOWNLOAD_MANIFEST_FILENAME}.{}.tmp",
        Uuid::new_v4()
    ));
    tokio::fs::write(&tmp_path, bytes)
        .await
        .map_err(|error| (StatusCode::INTERNAL_SERVER_ERROR, error.to_string()))?;
    tokio::fs::rename(&tmp_path, &manifest_path)
        .await
        .map_err(|error| {
            let _ = std::fs::remove_file(&tmp_path);
            (StatusCode::INTERNAL_SERVER_ERROR, error.to_string())
        })
}

async fn verify_written_file_size(
    repo_id: &str,
    file: &CatalogFileSpec,
    path: &std::path::Path,
) -> Result<(), (StatusCode, String)> {
    let Some(expected_size) = file.size_bytes else {
        return Ok(());
    };
    let actual_size = tokio::fs::metadata(path)
        .await
        .map_err(|error| (StatusCode::INTERNAL_SERVER_ERROR, error.to_string()))?
        .len() as i64;
    if actual_size != expected_size {
        return Err((
            StatusCode::BAD_GATEWAY,
            format!(
                "Downloaded {} from {} was {} bytes, expected {} bytes.",
                file.filename, repo_id, actual_size, expected_size
            ),
        ));
    }
    Ok(())
}

fn verify_catalog_download_total(
    spec: &CatalogModelSpec,
    downloaded_bytes: i64,
) -> Result<(), (StatusCode, String)> {
    if let Some(expected_bytes) = spec.size_bytes {
        if downloaded_bytes != expected_bytes {
            return Err((
                StatusCode::BAD_GATEWAY,
                format!(
                    "Downloaded package for {} was {} bytes, expected {} bytes from Fathom's catalog metadata.",
                    spec.repo_id, downloaded_bytes, expected_bytes
                ),
            ));
        }
    }
    Ok(())
}

fn sha256_hex(bytes: &[u8]) -> String {
    let digest = Sha256::digest(bytes);
    let mut out = String::with_capacity(digest.len() * 2);
    for byte in digest {
        out.push_str(&format!("{byte:02x}"));
    }
    out
}

fn verify_catalog_file_sha256(
    repo_id: &str,
    revision: &str,
    file: &CatalogFileSpec,
    bytes: &[u8],
) -> Result<(), (StatusCode, String)> {
    let Some(expected_sha256) = file.sha256 else {
        return Ok(());
    };
    let actual_sha256 = sha256_hex(bytes);
    if actual_sha256 != expected_sha256 {
        return Err((
            StatusCode::BAD_GATEWAY,
            format!(
                "Refusing {} from {} at revision {revision}: SHA256 {} did not match Fathom's catalog metadata {}.",
                file.filename, repo_id, actual_sha256, expected_sha256
            ),
        ));
    }
    Ok(())
}

async fn download_hf_file(
    spec: &CatalogModelSpec,
    file: &CatalogFileSpec,
) -> Result<Vec<u8>, (StatusCode, String)> {
    let Some(revision) = spec.revision else {
        return Err((
            StatusCode::BAD_REQUEST,
            format!(
                "Catalog model {} cannot be installed because it does not have a pinned Hugging Face revision yet.",
                spec.repo_id
            ),
        ));
    };
    let url = hf_resolve_url(spec.repo_id, revision, file.filename);
    let response = reqwest::get(&url).await.map_err(|error| {
        (
            StatusCode::BAD_GATEWAY,
            format!(
                "Could not download {} from {} at revision {revision}: {error}",
                file.filename, spec.repo_id
            ),
        )
    })?;
    if !response.status().is_success() {
        return Err((
            StatusCode::BAD_GATEWAY,
            format!(
                "Could not download {} from {} at revision {revision}: HTTP {}",
                file.filename,
                spec.repo_id,
                response.status()
            ),
        ));
    }
    if let (Some(expected_size), Some(content_length)) =
        (file.size_bytes, response.content_length())
    {
        if content_length != expected_size as u64 {
            return Err((
                StatusCode::BAD_GATEWAY,
                format!(
                    "Refusing {} from {} at revision {revision}: server reported {} bytes, expected {} bytes from Fathom's catalog metadata.",
                    file.filename, spec.repo_id, content_length, expected_size
                ),
            ));
        }
    }
    let bytes = response.bytes().await.map_err(|error| {
        (
            StatusCode::BAD_GATEWAY,
            format!(
                "Could not read {} from {} at revision {revision}: {error}",
                file.filename, spec.repo_id
            ),
        )
    })?;
    if let Some(expected_size) = file.size_bytes {
        let actual_size = bytes.len() as i64;
        if actual_size != expected_size {
            return Err((
                StatusCode::BAD_GATEWAY,
                format!(
                    "Refusing {} from {} at revision {revision}: downloaded {} bytes, expected {} bytes from Fathom's catalog metadata.",
                    file.filename, spec.repo_id, actual_size, expected_size
                ),
            ));
        }
    }
    verify_catalog_file_sha256(spec.repo_id, revision, file, bytes.as_ref())?;
    Ok(bytes.to_vec())
}

fn hf_resolve_url(repo_id: &str, revision: &str, filename: &str) -> String {
    format!("https://huggingface.co/{repo_id}/resolve/{revision}/{filename}")
}

fn fathom_model_dir(repo_id: &str) -> PathBuf {
    fathom_models_root().join(slugify(repo_id))
}

fn fathom_models_root() -> PathBuf {
    std::env::var_os("FATHOM_MODELS_DIR")
        .map(PathBuf::from)
        .or_else(|| std::env::var_os("HOME").map(|home| PathBuf::from(home).join(".fathom/models")))
        .unwrap_or_else(|| PathBuf::from(".fathom/models"))
}

fn fathom_state_dir() -> PathBuf {
    std::env::var_os("FATHOM_STATE_DIR")
        .map(PathBuf::from)
        .or_else(|| std::env::var_os("HOME").map(|home| PathBuf::from(home).join(".fathom/state")))
        .unwrap_or_else(|| PathBuf::from(".fathom/state"))
}

fn fathom_state_file() -> PathBuf {
    fathom_state_dir().join("models.json")
}

async fn load_model_state(path: &FsPath) -> anyhow::Result<ModelStateLoadResult> {
    if let Some(parent) = path.parent() {
        cleanup_stale_model_state_temp_files(parent).await;
    }

    match tokio::fs::read(path).await {
        Ok(bytes) => match serde_json::from_slice::<PersistedModelState>(&bytes) {
            Ok(mut state) => {
                drop_missing_active_model(&mut state);
                Ok(ModelStateLoadResult {
                    state,
                    warnings: vec![],
                })
            }
            Err(error) => {
                let preserved_path = preserve_corrupt_model_state(path).await?;
                let preserved_path_string = preserved_path.display().to_string();
                let message = format!(
                    "Recovered from unreadable model registry; corrupt file preserved at {preserved_path_string}."
                );
                eprintln!(
                    "warning: recovered from unreadable model state at {}; preserved corrupt file at {}: {error}",
                    path.display(),
                    preserved_path.display()
                );
                Ok(ModelStateLoadResult {
                    state: PersistedModelState::default(),
                    warnings: vec![RuntimeWarning {
                        warning_type: "model_state_recovered".into(),
                        message,
                        preserved_path: Some(preserved_path_string),
                    }],
                })
            }
        },
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => Ok(ModelStateLoadResult {
            state: PersistedModelState::default(),
            warnings: vec![],
        }),
        Err(error) => Err(error.into()),
    }
}

fn drop_missing_active_model(state: &mut PersistedModelState) {
    if let Some(active_model_id) = &state.active_model_id {
        if !state
            .models
            .iter()
            .any(|model| &model.id == active_model_id)
        {
            state.active_model_id = None;
        }
    }
}

async fn preserve_corrupt_model_state(path: &FsPath) -> anyhow::Result<PathBuf> {
    let parent = path.parent().unwrap_or_else(|| FsPath::new("."));
    let file_name = path
        .file_name()
        .and_then(|name| name.to_str())
        .unwrap_or("models.json");
    let preserved_path = parent.join(format!("{file_name}.corrupt-{}", Uuid::new_v4()));
    tokio::fs::rename(path, &preserved_path).await?;
    sync_parent_dir(parent).await;
    Ok(preserved_path)
}

fn model_state_temp_path(path: &FsPath) -> PathBuf {
    let parent = path.parent().unwrap_or_else(|| FsPath::new("."));
    parent.join(format!(".models.json.{}.tmp", Uuid::new_v4()))
}

fn is_model_state_temp_file(path: &FsPath) -> bool {
    path.file_name()
        .and_then(|name| name.to_str())
        .map(|name| name.starts_with(".models.json.") && name.ends_with(".tmp"))
        .unwrap_or(false)
}

async fn cleanup_stale_model_state_temp_files(parent: &FsPath) {
    let Ok(mut entries) = tokio::fs::read_dir(parent).await else {
        return;
    };

    while let Ok(Some(entry)) = entries.next_entry().await {
        let path = entry.path();
        if is_model_state_temp_file(&path) {
            let _ = tokio::fs::remove_file(path).await;
        }
    }
}

async fn sync_parent_dir(path: &FsPath) {
    #[cfg(unix)]
    {
        if let Ok(dir) = tokio::fs::File::open(path).await {
            let _ = dir.sync_all().await;
        }
    }
    #[cfg(not(unix))]
    {
        let _ = path;
    }
}

async fn mutate_model_state_and_persist<R>(
    state: &AppState,
    mutation: impl FnOnce(&mut Store) -> Result<R, ApiError>,
) -> Result<R, ApiError> {
    let mut store = state.inner.lock().await;
    let original_models = store.models.clone();
    let original_active_model_id = store.active_model_id.clone();
    let result = mutation(&mut store)?;
    let persisted = PersistedModelState {
        models: store.models.clone(),
        active_model_id: store.active_model_id.clone(),
    };

    if let Err(error) = write_model_state(&state.model_state_path, &persisted).await {
        store.models = original_models;
        store.active_model_id = original_active_model_id;
        return Err(raw_api_error(error, "model_state_persist_failed"));
    }

    Ok(result)
}

async fn write_model_state(
    path: &FsPath,
    state: &PersistedModelState,
) -> Result<(), (StatusCode, String)> {
    let parent = path.parent().unwrap_or_else(|| FsPath::new("."));
    tokio::fs::create_dir_all(parent)
        .await
        .map_err(|error| (StatusCode::INTERNAL_SERVER_ERROR, error.to_string()))?;

    let bytes = serde_json::to_vec_pretty(state)
        .map_err(|error| (StatusCode::INTERNAL_SERVER_ERROR, error.to_string()))?;
    let temp_path = model_state_temp_path(path);
    let result = async {
        let mut file = OpenOptions::new()
            .write(true)
            .create_new(true)
            .open(&temp_path)
            .await?;
        file.write_all(&bytes).await?;
        file.flush().await?;
        file.sync_all().await?;
        drop(file);

        tokio::fs::rename(&temp_path, path).await?;
        sync_parent_dir(parent).await;
        Ok::<(), std::io::Error>(())
    }
    .await;

    if let Err(error) = result {
        let _ = tokio::fs::remove_file(&temp_path).await;
        return Err((StatusCode::INTERNAL_SERVER_ERROR, error.to_string()));
    }

    Ok(())
}

fn runtime_json(store: &Store) -> serde_json::Value {
    let machine = current_machine_profile();
    let backend_lanes = backend_lanes_for_machine(&machine);
    let port = std::env::var("FATHOM_PORT").unwrap_or_else(|_| "8180".into());
    let active_chat_model_id = store
        .active_model_id
        .as_ref()
        .filter(|active_id| {
            store
                .models
                .iter()
                .any(|model| &model.id == *active_id && is_runnable_model(model))
        })
        .cloned();
    serde_json::json!({
        "ready": true,
        "loaded_now": active_chat_model_id.is_some(),
        "active_model_id": active_chat_model_id,
        "engine": "Fathom Rust Runtime",
        "api_base": format!("http://127.0.0.1:{port}/v1"),
        "storage_root": "~/.fathom/models",
        "ready_model_count": store.models.iter().filter(|model| is_runnable_model(model)).count(),
        "machine": machine,
        "backend_lanes": backend_lanes,
        "warnings": store.startup_warnings.clone(),
        "llama_server_installed": false
    })
}

fn upsert_model(models: &mut Vec<ModelRecord>, model: ModelRecord) {
    models.retain(|existing| existing.id != model.id);
    models.insert(0, model);
}

fn model_task_label_for_package(package: &fathom_core::ModelPackage) -> Option<String> {
    embedding_model_status_for_package(package).map(|status| match status.task {
        fathom_core::ModelTaskKind::TextGeneration => "text_generation".to_string(),
        fathom_core::ModelTaskKind::TextEmbedding => "text_embedding".to_string(),
        fathom_core::ModelTaskKind::RetrievalContext => "retrieval_context".to_string(),
        fathom_core::ModelTaskKind::Unknown => "unknown".to_string(),
    })
}

fn is_runnable_model(model: &ModelRecord) -> bool {
    if model.status != "ready" {
        return false;
    }
    if model.provider_kind == "external" {
        return false;
    }
    if model.capability_status != "runnable" {
        return false;
    }
    let Some(model_path) = model.model_path.as_ref() else {
        return false;
    };
    if inspect_model_package(model_path)
        .ok()
        .and_then(|package| embedding_model_status_for_package(&package))
        .is_some()
    {
        return false;
    }
    true
}

fn is_v1_chat_runnable_model(model: &ModelRecord) -> bool {
    model.provider_kind != "external" && is_runnable_model(model)
}

fn now() -> String {
    chrono::Utc::now().to_rfc3339()
}

fn slugify(value: &str) -> String {
    value
        .to_ascii_lowercase()
        .chars()
        .map(|ch| if ch.is_ascii_alphanumeric() { ch } else { '-' })
        .collect::<String>()
        .trim_matches('-')
        .to_string()
}

fn format_from_filename(filename: &str) -> &'static str {
    let lowered = filename.to_ascii_lowercase();
    if lowered.ends_with(".gguf") {
        "GGUF"
    } else if lowered.ends_with(".safetensors") {
        "SafeTensors"
    } else if lowered.ends_with(".onnx") {
        "ONNX"
    } else if lowered.ends_with(".bin") {
        "PyTorchBin"
    } else {
        "Unknown"
    }
}

fn capability_status_label(status: &CapabilityStatus) -> &'static str {
    match status {
        CapabilityStatus::Runnable => "runnable",
        CapabilityStatus::Planned => "planned",
        CapabilityStatus::MetadataOnly => "metadata_only",
        CapabilityStatus::Blocked => "blocked",
        CapabilityStatus::Unsupported => "unsupported",
    }
}

#[cfg(test)]
mod catalog_tests {
    use super::*;

    #[test]
    fn catalog_expected_package_sizes_match_file_sizes() {
        for spec in catalog_specs()
            .into_iter()
            .filter(|spec| spec.size_bytes.is_some())
        {
            let file_total: i64 = spec
                .files
                .iter()
                .map(|file| {
                    file.size_bytes
                        .expect("catalog file should have expected size")
                })
                .sum();
            assert_eq!(Some(file_total), spec.size_bytes, "{}", spec.catalog_id);
            verify_catalog_download_total(&spec, file_total).unwrap();
        }
    }

    #[test]
    fn catalog_size_mismatch_is_rejected() {
        let spec = catalog_specs()
            .into_iter()
            .find(|spec| spec.catalog_id == "hf-intel-tiny-random-gpt2")
            .unwrap();
        let expected = spec.size_bytes.unwrap();
        let error = verify_catalog_download_total(&spec, expected - 1).unwrap_err();
        assert_eq!(error.0, StatusCode::BAD_GATEWAY);
        assert!(error.1.contains("expected"));

        let (status, Json(body)) = catalog_error_json(error);
        assert_eq!(status, StatusCode::BAD_GATEWAY);
        assert_eq!(body["error"]["code"], "catalog_size_mismatch");
        assert!(body["error"].get("param").is_some());
    }

    #[test]
    fn downloadable_catalog_files_have_sha256_metadata() {
        for spec in catalog_specs()
            .into_iter()
            .filter(|spec| spec.size_bytes.is_some())
        {
            for file in spec.files {
                let sha256 = file.sha256.unwrap_or_else(|| {
                    panic!(
                        "{}:{} must include SHA256 catalog metadata",
                        spec.catalog_id, file.filename
                    )
                });
                assert_eq!(sha256.len(), 64, "{}:{}", spec.catalog_id, file.filename);
                assert!(
                    sha256.chars().all(|ch| ch.is_ascii_hexdigit()),
                    "{}:{}",
                    spec.catalog_id,
                    file.filename
                );
            }
        }
    }

    #[test]
    fn catalog_sha256_mismatch_is_rejected() {
        let spec = catalog_specs()
            .into_iter()
            .find(|spec| spec.catalog_id == "hf-intel-tiny-random-gpt2")
            .unwrap();
        let file = spec
            .files
            .iter()
            .find(|file| file.filename == "config.json")
            .unwrap();
        let revision = spec.revision.unwrap();
        let error =
            verify_catalog_file_sha256(spec.repo_id, revision, file, b"wrong bytes").unwrap_err();

        assert_eq!(error.0, StatusCode::BAD_GATEWAY);
        assert!(error.1.contains("SHA256"));
        assert!(error.1.contains("did not match"));

        let (status, Json(body)) = catalog_error_json(error);
        assert_eq!(status, StatusCode::BAD_GATEWAY);
        assert_eq!(body["error"]["code"], "catalog_sha256_mismatch");
        assert_eq!(body["error"]["type"], "catalog_sha256_mismatch");
    }

    #[test]
    fn catalog_sha256_match_is_accepted() {
        let file = catalog_file(
            "fixture.txt",
            5,
            "2cf24dba5fb0a30e26e83b2ac5b9e29e1b161e5c1fa7425e73043362938b9824",
        );

        verify_catalog_file_sha256(
            "example/repo",
            "0123456789012345678901234567890123456789",
            &file,
            b"hello",
        )
        .unwrap();
    }

    #[test]
    fn downloadable_catalog_entries_have_pinned_hf_revisions() {
        for spec in catalog_specs()
            .into_iter()
            .filter(|spec| spec.size_bytes.is_some())
        {
            let revision = spec
                .revision
                .unwrap_or_else(|| panic!("{} must pin a Hugging Face revision", spec.catalog_id));
            assert_eq!(revision.len(), 40, "{}", spec.catalog_id);
            assert!(
                revision.chars().all(|ch| ch.is_ascii_hexdigit()),
                "{}",
                spec.catalog_id
            );
        }
    }

    #[test]
    fn hf_download_urls_use_pinned_revision_not_main() {
        let spec = catalog_specs()
            .into_iter()
            .find(|spec| spec.catalog_id == "hf-intel-tiny-random-gpt2")
            .unwrap();
        let revision = spec.revision.unwrap();

        assert_eq!(
            hf_resolve_url(spec.repo_id, revision, "model.safetensors"),
            format!(
                "https://huggingface.co/Intel/tiny-random-gpt2/resolve/{revision}/model.safetensors"
            )
        );
        assert_ne!(revision, "main");
    }

    #[test]
    fn catalog_license_policy_allows_permissive_without_ack() {
        let spec = catalog_specs()
            .into_iter()
            .find(|spec| spec.catalog_id == "hf-intel-tiny-random-gpt2")
            .unwrap();

        assert_eq!(
            catalog_license_status(spec.license),
            CatalogLicenseStatus::Permissive
        );
        assert!(!catalog_license_acknowledgement_required(&spec));
        validate_catalog_license_ack(&spec, false).unwrap();
    }

    #[test]
    fn unknown_catalog_license_requires_ack_and_has_stable_error_envelope() {
        let spec = catalog_specs()
            .into_iter()
            .find(|spec| spec.catalog_id == "hf-yujiepan-qwen2-tiny-random")
            .unwrap();

        assert_eq!(
            catalog_license_status(spec.license),
            CatalogLicenseStatus::Unknown
        );
        assert!(catalog_license_acknowledgement_required(&spec));
        validate_catalog_license_ack(&spec, true).unwrap();

        let (status, Json(body)) = validate_catalog_license_ack(&spec, false).unwrap_err();
        assert_eq!(status, StatusCode::BAD_REQUEST);
        assert_eq!(body["error"]["code"], "catalog_license_ack_required");
        assert_eq!(body["error"]["type"], "catalog_license_ack_required");
        assert!(body["error"]["message"]
            .as_str()
            .unwrap()
            .contains("requires explicit acknowledgement"));
        assert!(body["error"].get("param").is_some());
    }

    #[test]
    fn catalog_install_request_accepts_both_license_ack_field_names() {
        let accept_license: CatalogInstallRequest = serde_json::from_value(serde_json::json!({
            "repo_id": "example/repo",
            "filename": "model.safetensors",
            "accept_license": true
        }))
        .unwrap();
        let accepted_license: CatalogInstallRequest = serde_json::from_value(serde_json::json!({
            "repo_id": "example/repo",
            "filename": "model.safetensors",
            "accepted_license": true
        }))
        .unwrap();

        assert!(catalog_install_license_acknowledged(&accept_license));
        assert!(catalog_install_license_acknowledged(&accepted_license));
    }

    #[tokio::test]
    async fn catalog_listing_includes_license_policy_metadata() {
        let Json(body) = model_catalog().await;
        let items = body["items"].as_array().unwrap();
        let permissive = items
            .iter()
            .find(|item| item["catalog_id"] == "hf-intel-tiny-random-gpt2")
            .unwrap();
        let unknown = items
            .iter()
            .find(|item| item["catalog_id"] == "hf-yujiepan-qwen2-tiny-random")
            .unwrap();

        assert_eq!(permissive["license_status"], "permissive");
        assert_eq!(permissive["license_acknowledgement_required"], false);
        assert!(permissive["license_warning"].is_null());
        assert_eq!(unknown["license_status"], "unknown");
        assert_eq!(unknown["license_acknowledgement_required"], true);
        assert!(unknown["license_warning"]
            .as_str()
            .unwrap()
            .contains("Review the model card/license"));
    }

    #[test]
    fn download_manifest_captures_pinned_verified_catalog_provenance() {
        let spec = catalog_specs()
            .into_iter()
            .find(|spec| spec.catalog_id == "hf-intel-tiny-random-gpt2")
            .unwrap();

        let manifest = build_download_manifest(&spec, false).unwrap();

        assert_eq!(manifest.schema_version, 1);
        assert_eq!(manifest.repo_id, spec.repo_id);
        assert_eq!(manifest.revision, spec.revision.unwrap());
        assert_eq!(
            manifest.source_url,
            "https://huggingface.co/Intel/tiny-random-gpt2"
        );
        assert_eq!(manifest.license, spec.license);
        assert_eq!(manifest.license_status.as_deref(), Some("permissive"));
        assert!(!manifest.license_acknowledgement_required);
        assert!(!manifest.license_acknowledged);
        assert_eq!(
            manifest.license_policy_note.as_deref(),
            Some(catalog_license_policy_note())
        );
        assert_eq!(manifest.verification_status, "verified");
        assert_eq!(manifest.files.len(), spec.files.len());
        assert!(manifest.files.iter().all(|file| file.size_bytes > 0));
        assert!(manifest.files.iter().all(|file| file.sha256.len() == 64));
    }

    #[test]
    fn download_manifest_captures_license_acknowledgement_policy_state() {
        let spec = CatalogModelSpec {
            catalog_id: "test-unknown-license",
            title: "Unknown License Test",
            repo_id: "example/unknown-license",
            revision: Some("0123456789012345678901234567890123456789"),
            primary_filename: "config.json",
            files: vec![catalog_file(
                "config.json",
                32,
                "476771b86370aebb00db5d51d24d1427e319bef3aef67b0275f38ea0bcd69778",
            )],
            size_bytes: Some(32),
            license: "unknown",
            description: "No-network fixture.",
        };

        let manifest = build_download_manifest(&spec, true).unwrap();

        assert_eq!(manifest.license_status.as_deref(), Some("unknown"));
        assert!(manifest.license_acknowledgement_required);
        assert!(manifest.license_acknowledged);
        assert_eq!(
            manifest.license_policy_note.as_deref(),
            Some(catalog_license_policy_note())
        );
    }

    #[test]
    fn old_download_manifests_deserialize_without_license_audit_fields() {
        let manifest: DownloadManifest = serde_json::from_value(serde_json::json!({
            "schema_version": 1,
            "repo_id": "example/old",
            "revision": "0123456789012345678901234567890123456789",
            "source_url": "https://huggingface.co/example/old",
            "license": "mit",
            "installed_at": "2026-04-27T00:00:00Z",
            "verification_status": "verified",
            "files": []
        }))
        .unwrap();

        assert_eq!(manifest.license_status, None);
        assert!(!manifest.license_acknowledgement_required);
        assert!(!manifest.license_acknowledged);
        assert_eq!(manifest.license_policy_note, None);
    }

    #[test]
    fn gguf_catalog_fixture_has_exact_pinned_metadata_only_provenance() {
        let spec = catalog_specs()
            .into_iter()
            .find(|spec| spec.catalog_id == "gguf-llama-tiny")
            .unwrap();

        assert_eq!(spec.repo_id, "aladar/llama-2-tiny-random-GGUF");
        assert_eq!(
            spec.revision,
            Some("8d5321916486e1d33c46b16990e8da6567785769")
        );
        assert_eq!(spec.primary_filename, "llama-2-tiny-random.gguf");
        assert_eq!(spec.size_bytes, Some(1_750_560));
        assert_eq!(spec.license, "mit");
        assert!(spec
            .description
            .contains("metadata/provenance inspection only"));
        assert!(spec.description.contains("metadata-only"));
        assert!(spec
            .description
            .contains("Internal synthetic tokenizer-retention checks"));
        assert!(spec.description.contains("Llama/SentencePiece-shaped"));
        assert!(spec
            .description
            .contains("internal synthetic payload decode tests"));
        assert!(spec.description.contains("general dequantization"));
        assert!(spec
            .description
            .contains("private fixture-scoped Llama/SentencePiece encode/decode parity helpers"));
        assert!(spec
            .description
            .contains("not public/runtime tokenizer execution"));

        assert_eq!(spec.files.len(), 1);
        assert_eq!(spec.files[0].filename, "llama-2-tiny-random.gguf");
        assert_eq!(spec.files[0].size_bytes, Some(1_750_560));
        assert_eq!(
            spec.files[0].sha256,
            Some("81f226c62d28ed4a1a9b9fa080fcd9f0cc40e0f9d5680036583ff98fbcd035cb")
        );

        let manifest = build_download_manifest(&spec, false).unwrap();
        assert_eq!(manifest.repo_id, spec.repo_id);
        assert_eq!(manifest.revision, spec.revision.unwrap());
        assert_eq!(manifest.license, "mit");
        assert_eq!(manifest.license_status.as_deref(), Some("permissive"));
        assert!(!manifest.license_acknowledgement_required);
        assert!(!manifest.license_acknowledged);
        assert_eq!(manifest.verification_status, "verified");
        assert_eq!(manifest.files[0].filename, "llama-2-tiny-random.gguf");
        assert_eq!(manifest.files[0].size_bytes, 1_750_560);
        assert_eq!(
            manifest.files[0].sha256,
            "81f226c62d28ed4a1a9b9fa080fcd9f0cc40e0f9d5680036583ff98fbcd035cb"
        );
    }

    #[test]
    fn missing_catalog_provenance_is_rejected_before_download_manifest() {
        let spec = CatalogModelSpec {
            catalog_id: "placeholder",
            title: "Placeholder",
            repo_id: "example/placeholder",
            revision: None,
            primary_filename: "model.gguf",
            files: vec![CatalogFileSpec {
                filename: "model.gguf",
                size_bytes: None,
                sha256: None,
            }],
            size_bytes: None,
            license: "unknown",
            description: "Intentionally incomplete fixture.",
        };

        let error = build_download_manifest(&spec, false).unwrap_err();
        assert_eq!(error.0, StatusCode::BAD_REQUEST);
        assert!(error.1.contains("pinned Hugging Face revision"));
    }

    #[tokio::test]
    async fn cancel_unknown_model_returns_json_not_found() {
        let state = test_state(vec![], None);

        let (status, Json(body)) =
            cancel_model_download(State(state), Path("missing-model".into()))
                .await
                .unwrap_err();

        assert_eq!(status, StatusCode::NOT_FOUND);
        assert_eq!(body["error"]["code"], "model_not_found");
        assert_eq!(body["error"]["type"], "model_not_found");
        assert!(body["error"].get("param").is_some());
    }

    #[tokio::test]
    async fn cancel_known_non_active_model_returns_json_conflict() {
        let state = test_state(vec![test_model("local-gpt2")], None);

        let (status, Json(body)) = cancel_model_download(State(state), Path("local-gpt2".into()))
            .await
            .unwrap_err();

        assert_eq!(status, StatusCode::CONFLICT);
        assert_eq!(body["error"]["code"], "model_download_not_active");
        assert_eq!(body["error"]["type"], "model_download_not_active");
        assert!(body["error"]["message"]
            .as_str()
            .unwrap()
            .contains("not actively downloading"));
    }

    #[tokio::test]
    async fn cancel_downloading_model_returns_json_not_supported() {
        let mut model = test_model("downloading-gpt2");
        model.status = "downloading".into();
        model.bytes_downloaded = Some(128);
        model.total_bytes = Some(256);
        model.progress = Some(50.0);
        let state = test_state(vec![model], None);

        let (status, Json(body)) =
            cancel_model_download(State(state), Path("downloading-gpt2".into()))
                .await
                .unwrap_err();

        assert_eq!(status, StatusCode::NOT_IMPLEMENTED);
        assert_eq!(
            body["error"]["code"],
            "model_download_cancellation_not_supported"
        );
        assert_eq!(
            body["error"]["type"],
            "model_download_cancellation_not_supported"
        );
        assert!(body["error"]["message"]
            .as_str()
            .unwrap()
            .contains("cancellable background download task"));
    }

    #[tokio::test]
    async fn unknown_license_catalog_install_refuses_without_ack_before_download() {
        let root = std::env::temp_dir().join(format!("fathom-license-catalog-{}", Uuid::new_v4()));
        let state = test_state_at(root.join("models.json"), vec![], None);

        let (status, Json(body)) = install_catalog_model(
            State(state),
            Json(CatalogInstallRequest {
                repo_id: "yujiepan/qwen2-tiny-random".into(),
                filename: "model.safetensors".into(),
                accept_license: None,
                accepted_license: None,
            }),
        )
        .await
        .unwrap_err();

        assert_eq!(status, StatusCode::BAD_REQUEST);
        assert_eq!(body["error"]["code"], "catalog_license_ack_required");
        assert_eq!(body["error"]["type"], "catalog_license_ack_required");
        assert!(body["error"]["message"]
            .as_str()
            .unwrap()
            .contains("requires explicit acknowledgement"));
        let _ = tokio::fs::remove_dir_all(root).await;
    }

    #[tokio::test]
    async fn unknown_catalog_install_returns_json_error_shape() {
        let root = std::env::temp_dir().join(format!("fathom-unknown-catalog-{}", Uuid::new_v4()));
        let state = test_state_at(root.join("models.json"), vec![], None);

        let (status, Json(body)) = install_catalog_model(
            State(state),
            Json(CatalogInstallRequest {
                repo_id: "missing/repo".into(),
                filename: "missing.safetensors".into(),
                accept_license: None,
                accepted_license: None,
            }),
        )
        .await
        .unwrap_err();

        assert_eq!(status, StatusCode::BAD_REQUEST);
        assert_eq!(body["error"]["code"], "catalog_model_not_found");
        assert_eq!(body["error"]["type"], "catalog_model_not_found");
        assert!(body["error"].get("param").is_some());
        let _ = tokio::fs::remove_dir_all(root).await;
    }

    #[tokio::test]
    #[ignore = "optional live smoke: downloads the 1.75 MB pinned GGUF fixture and verifies metadata-only inspection"]
    async fn live_pinned_gguf_fixture_download_inspects_metadata_only() {
        let spec = catalog_specs()
            .into_iter()
            .find(|spec| spec.catalog_id == "gguf-llama-tiny")
            .unwrap();
        let file = &spec.files[0];
        let bytes = download_hf_file(&spec, file).await.unwrap();
        assert_eq!(bytes.len() as i64, file.size_bytes.unwrap());
        assert_eq!(sha256_hex(&bytes), file.sha256.unwrap());

        let root = std::env::temp_dir().join(format!("fathom-live-gguf-{}", Uuid::new_v4()));
        tokio::fs::create_dir_all(&root).await.unwrap();
        let path = root.join(file.filename);
        tokio::fs::write(&path, &bytes).await.unwrap();

        let package = inspect_model_package(&root).unwrap();
        let artifact = package
            .artifacts
            .iter()
            .find(|artifact| artifact.format == fathom_core::ModelFormat::Gguf)
            .unwrap();
        assert!(!artifact.runnable_today);
        let metadata = artifact
            .gguf_metadata
            .as_ref()
            .unwrap_or_else(|| panic!("expected live GGUF metadata, notes: {:?}", artifact.notes));
        assert_eq!(metadata.hints.architecture.as_deref(), Some("llama"));
        assert!(metadata.tensor_count > 0);
        assert!(metadata.metadata_kv_count > 0);
        assert!(metadata.tokenizer_summary.token_count.unwrap_or(0) > 0);
        assert!(metadata
            .compatibility
            .runtime_blockers
            .contains(&"gguf_generation_not_implemented".to_string()));

        let report = capability_report_for_package(package, &current_machine_profile());
        assert_eq!(report.best_status, CapabilityStatus::MetadataOnly);
        assert!(!report.runnable);
        let _ = tokio::fs::remove_dir_all(root).await;
    }

    fn transactional_test_spec() -> CatalogModelSpec {
        CatalogModelSpec {
            catalog_id: "test-transactional",
            title: "Transactional Test",
            repo_id: "example/transactional",
            revision: Some("0123456789012345678901234567890123456789"),
            primary_filename: "config.json",
            files: vec![catalog_file(
                "config.json",
                57,
                "6293a635cf91e50623224519aa1adaa93afcf5dc11ad038872a0064ec8977150",
            )],
            size_bytes: Some(57),
            license: "test",
            description: "Local no-network transactional install fixture.",
        }
    }

    fn transactional_config_bytes() -> &'static [u8] {
        br#"{"model_type":"gpt2","architectures":["GPT2LMHeadModel"]}"#
    }

    #[tokio::test]
    async fn catalog_size_or_sha_mismatch_leaves_no_registration_or_manifest() {
        let spec = transactional_test_spec();
        let root = std::env::temp_dir().join(format!("fathom-catalog-txn-{}", Uuid::new_v4()));
        let staging = root.join(".fathom-install-transactional.tmp");
        tokio::fs::create_dir_all(&staging).await.unwrap();
        let state = Store::default();

        let error = write_verified_catalog_file(&spec, &spec.files[0], &staging, b"wrong bytes")
            .await
            .unwrap_err();

        assert_eq!(error.0, StatusCode::BAD_GATEWAY);
        assert!(state.models.is_empty());
        assert!(!staging.join(DOWNLOAD_MANIFEST_FILENAME).exists());
        assert!(!staging.join("config.json").exists());
        let _ = tokio::fs::remove_dir_all(root).await;
    }

    #[tokio::test]
    async fn catalog_staging_failure_leaves_no_partial_final_package() {
        let spec = transactional_test_spec();
        let root = std::env::temp_dir().join(format!("fathom-catalog-txn-{}", Uuid::new_v4()));
        let final_dir = root.join("example-transactional");
        let staging = root.join(".fathom-install-transactional.tmp");
        tokio::fs::create_dir_all(&staging).await.unwrap();

        let error = write_verified_catalog_file(&spec, &spec.files[0], &staging, b"bad")
            .await
            .unwrap_err();

        assert_eq!(error.0, StatusCode::BAD_GATEWAY);
        assert!(!final_dir.exists());
        let _ = tokio::fs::remove_dir_all(root).await;
    }

    #[tokio::test]
    async fn catalog_manifest_is_written_only_after_verified_package() {
        let spec = transactional_test_spec();
        let root = std::env::temp_dir().join(format!("fathom-catalog-txn-{}", Uuid::new_v4()));
        let staging = root.join(".fathom-install-transactional.tmp");
        tokio::fs::create_dir_all(&staging).await.unwrap();

        assert!(
            write_verified_catalog_file(&spec, &spec.files[0], &staging, b"bad")
                .await
                .is_err()
        );
        assert!(!staging.join(DOWNLOAD_MANIFEST_FILENAME).exists());

        write_verified_catalog_file(
            &spec,
            &spec.files[0],
            &staging,
            transactional_config_bytes(),
        )
        .await
        .unwrap();
        verify_catalog_download_total(&spec, transactional_config_bytes().len() as i64).unwrap();
        let manifest = build_download_manifest(&spec, false).unwrap();
        write_download_manifest(&staging, &manifest).await.unwrap();

        assert!(staging.join(DOWNLOAD_MANIFEST_FILENAME).exists());
        assert!(tokio::fs::read_dir(&staging)
            .await
            .unwrap()
            .next_entry()
            .await
            .unwrap()
            .is_some());
        let _ = tokio::fs::remove_dir_all(root).await;
    }

    #[tokio::test]
    async fn catalog_nested_paths_are_preserved_after_promotion() {
        let spec = CatalogModelSpec {
            catalog_id: "test-nested",
            title: "Nested Test",
            repo_id: "example/nested",
            revision: Some("0123456789012345678901234567890123456789"),
            primary_filename: "1_Pooling/config.json",
            files: vec![catalog_file(
                "1_Pooling/config.json",
                32,
                "476771b86370aebb00db5d51d24d1427e319bef3aef67b0275f38ea0bcd69778",
            )],
            size_bytes: Some(32),
            license: "test",
            description: "Nested path no-network fixture.",
        };
        let bytes = br#"{"word_embedding_dimension":384}"#;
        let root = std::env::temp_dir().join(format!("fathom-catalog-txn-{}", Uuid::new_v4()));
        let staging = root.join(".fathom-install-nested.tmp");
        let final_dir = root.join("example-nested");
        tokio::fs::create_dir_all(&staging).await.unwrap();

        write_verified_catalog_file(&spec, &spec.files[0], &staging, bytes)
            .await
            .unwrap();
        promote_staged_catalog_install(&staging, &final_dir)
            .await
            .unwrap();

        assert!(final_dir.join("1_Pooling/config.json").exists());
        assert!(!staging.exists());
        let _ = tokio::fs::remove_dir_all(root).await;
    }

    #[tokio::test]
    async fn failed_reinstall_preserves_previous_final_package() {
        let root = std::env::temp_dir().join(format!("fathom-catalog-txn-{}", Uuid::new_v4()));
        let final_dir = root.join("example-transactional");
        let missing_staging = root.join(".fathom-install-missing.tmp");
        tokio::fs::create_dir_all(&final_dir).await.unwrap();
        tokio::fs::write(final_dir.join("sentinel.txt"), b"old-good")
            .await
            .unwrap();

        let error = promote_staged_catalog_install(&missing_staging, &final_dir)
            .await
            .unwrap_err();

        assert_eq!(error.0, StatusCode::INTERNAL_SERVER_ERROR);
        assert_eq!(
            tokio::fs::read(final_dir.join("sentinel.txt"))
                .await
                .unwrap(),
            b"old-good"
        );
        let mut entries = tokio::fs::read_dir(&root).await.unwrap();
        while let Some(entry) = entries.next_entry().await.unwrap() {
            assert!(!catalog_install_temp_dir_name(&entry.path()));
        }
        let _ = tokio::fs::remove_dir_all(root).await;
    }

    #[tokio::test]
    async fn catalog_temp_cleanup_removes_staging_and_backup_dirs() {
        let root = std::env::temp_dir().join(format!("fathom-catalog-txn-{}", Uuid::new_v4()));
        tokio::fs::create_dir_all(root.join(".fathom-install-a.tmp"))
            .await
            .unwrap();
        tokio::fs::create_dir_all(root.join(".fathom-install-b.backup"))
            .await
            .unwrap();
        tokio::fs::create_dir_all(root.join("ordinary-model"))
            .await
            .unwrap();

        cleanup_catalog_install_temp_dirs(&root).await.unwrap();

        assert!(!root.join(".fathom-install-a.tmp").exists());
        assert!(!root.join(".fathom-install-b.backup").exists());
        assert!(root.join("ordinary-model").exists());
        let _ = tokio::fs::remove_dir_all(root).await;
    }

    #[tokio::test]
    async fn catalog_model_record_points_to_final_dir_not_staging() {
        let spec = transactional_test_spec();
        let root = std::env::temp_dir().join(format!("fathom-catalog-txn-{}", Uuid::new_v4()));
        let staging = root.join(".fathom-install-transactional.tmp");
        let final_dir = root.join("example-transactional");
        tokio::fs::create_dir_all(&staging).await.unwrap();
        write_verified_catalog_file(
            &spec,
            &spec.files[0],
            &staging,
            transactional_config_bytes(),
        )
        .await
        .unwrap();
        let package = inspect_model_package(&staging).unwrap();
        let manifest = build_download_manifest(&spec, false).unwrap();

        let model = build_catalog_model_record(
            &spec,
            &final_dir,
            transactional_config_bytes().len() as i64,
            manifest,
            package.clone(),
            capability_report_for_package(package, &current_machine_profile()),
        );

        assert_eq!(
            model.model_path.as_deref(),
            Some(final_dir.to_str().unwrap())
        );
        assert!(!model
            .model_path
            .as_deref()
            .unwrap()
            .contains(".fathom-install"));
        let _ = tokio::fs::remove_dir_all(root).await;
    }

    #[tokio::test]
    async fn download_manifest_writes_json_next_to_installed_model() {
        let root = std::env::temp_dir().join(format!("fathom-manifest-test-{}", Uuid::new_v4()));
        tokio::fs::create_dir_all(&root).await.unwrap();
        let spec = catalog_specs()
            .into_iter()
            .find(|spec| spec.catalog_id == "hf-stas-tiny-random-llama-2")
            .unwrap();
        let manifest = build_download_manifest(&spec, false).unwrap();

        write_download_manifest(&root, &manifest).await.unwrap();
        let bytes = tokio::fs::read(root.join(DOWNLOAD_MANIFEST_FILENAME))
            .await
            .unwrap();
        let loaded: DownloadManifest = serde_json::from_slice(&bytes).unwrap();

        assert_eq!(loaded.repo_id, spec.repo_id);
        assert_eq!(loaded.revision, spec.revision.unwrap());
        assert_eq!(loaded.license_status.as_deref(), Some("permissive"));
        assert!(!loaded.license_acknowledgement_required);
        assert!(!loaded.license_acknowledged);
        assert_eq!(loaded.verification_status, "verified");
        let _ = tokio::fs::remove_dir_all(root).await;
    }

    async fn model_state_temp_files(root: &FsPath) -> Vec<PathBuf> {
        let mut temp_files = Vec::new();
        let Ok(mut entries) = tokio::fs::read_dir(root).await else {
            return temp_files;
        };
        while let Ok(Some(entry)) = entries.next_entry().await {
            let path = entry.path();
            if is_model_state_temp_file(&path) {
                temp_files.push(path);
            }
        }
        temp_files
    }

    async fn model_state_corrupt_files(root: &FsPath) -> Vec<PathBuf> {
        let mut corrupt_files = Vec::new();
        let Ok(mut entries) = tokio::fs::read_dir(root).await else {
            return corrupt_files;
        };
        while let Ok(Some(entry)) = entries.next_entry().await {
            let path = entry.path();
            if path
                .file_name()
                .and_then(|name| name.to_str())
                .map(|name| name.starts_with("models.json.corrupt-"))
                .unwrap_or(false)
            {
                corrupt_files.push(path);
            }
        }
        corrupt_files
    }

    #[tokio::test]
    async fn model_state_round_trips_models_and_active_model() {
        let root = std::env::temp_dir().join(format!("fathom-state-test-{}", Uuid::new_v4()));
        let path = root.join("models.json");
        let model = test_model("distilgpt2-fixture");
        let state = PersistedModelState {
            models: vec![model.clone()],
            active_model_id: Some(model.id.clone()),
        };

        write_model_state(&path, &state).await.unwrap();
        let bytes = tokio::fs::read(&path).await.unwrap();
        serde_json::from_slice::<PersistedModelState>(&bytes).unwrap();
        assert!(model_state_temp_files(&root).await.is_empty());

        let loaded = load_model_state(&path).await.unwrap();

        assert_eq!(loaded.state.models.len(), 1);
        assert_eq!(loaded.state.models[0].id, model.id);
        assert_eq!(loaded.state.active_model_id, Some(model.id));
        let _ = tokio::fs::remove_dir_all(root).await;
    }

    #[tokio::test]
    async fn missing_model_state_loads_as_empty() {
        let path = std::env::temp_dir()
            .join(format!("fathom-missing-state-{}", Uuid::new_v4()))
            .join("models.json");

        let loaded = load_model_state(&path).await.unwrap();

        assert!(loaded.state.models.is_empty());
        assert_eq!(loaded.state.active_model_id, None);
        assert!(loaded.warnings.is_empty());
    }

    #[tokio::test]
    async fn corrupt_model_state_is_preserved_and_loads_safe_empty_state() {
        let root = std::env::temp_dir().join(format!("fathom-corrupt-state-{}", Uuid::new_v4()));
        tokio::fs::create_dir_all(&root).await.unwrap();
        let path = root.join("models.json");
        tokio::fs::write(&path, b"{ truncated").await.unwrap();

        let loaded = load_model_state(&path).await.unwrap();

        assert!(loaded.state.models.is_empty());
        assert_eq!(loaded.state.active_model_id, None);
        assert_eq!(loaded.warnings.len(), 1);
        assert_eq!(loaded.warnings[0].warning_type, "model_state_recovered");
        assert!(loaded.warnings[0]
            .message
            .contains("Recovered from unreadable model registry"));
        let preserved_path = loaded.warnings[0].preserved_path.as_ref().unwrap();
        assert!(preserved_path.contains("models.json.corrupt-"));
        assert!(!path.exists());
        let corrupt_files = model_state_corrupt_files(&root).await;
        assert_eq!(corrupt_files.len(), 1);
        assert_eq!(preserved_path, &corrupt_files[0].display().to_string());
        assert_eq!(
            tokio::fs::read(&corrupt_files[0]).await.unwrap(),
            b"{ truncated"
        );
        let _ = tokio::fs::remove_dir_all(root).await;
    }

    #[test]
    fn runtime_json_includes_stable_startup_warning_shape() {
        let store = Store {
            startup_warnings: vec![RuntimeWarning {
                warning_type: "model_state_recovered".into(),
                message: "Recovered from unreadable model registry; corrupt file preserved at /tmp/models.json.corrupt-test.".into(),
                preserved_path: Some("/tmp/models.json.corrupt-test".into()),
            }],
            ..Store::default()
        };

        let body = runtime_json(&store);

        assert_eq!(body["warnings"][0]["type"], "model_state_recovered");
        assert_eq!(
            body["warnings"][0]["preserved_path"],
            "/tmp/models.json.corrupt-test"
        );
        assert!(body["warnings"][0]["message"]
            .as_str()
            .unwrap()
            .contains("Recovered from unreadable model registry"));
    }

    #[tokio::test]
    async fn dashboard_embeds_runtime_startup_warnings() {
        let state = AppState {
            inner: Arc::new(Mutex::new(Store {
                startup_warnings: vec![RuntimeWarning {
                    warning_type: "model_state_recovered".into(),
                    message: "Recovered from unreadable model registry; corrupt file preserved at /tmp/models.json.corrupt-test.".into(),
                    preserved_path: Some("/tmp/models.json.corrupt-test".into()),
                }],
                ..Store::default()
            })),
            model_state_path: std::env::temp_dir().join(format!(
                "fathom-dashboard-warning-{}.json",
                Uuid::new_v4()
            )),
        };

        let Json(body) = dashboard(State(state)).await;

        assert_eq!(
            body["runtime"]["warnings"][0]["type"],
            "model_state_recovered"
        );
    }

    #[tokio::test]
    async fn stale_model_state_temp_files_are_ignored_and_cleaned() {
        let root = std::env::temp_dir().join(format!("fathom-stale-state-{}", Uuid::new_v4()));
        tokio::fs::create_dir_all(&root).await.unwrap();
        let path = root.join("models.json");
        let model = test_model("present-model");
        let state = PersistedModelState {
            models: vec![model.clone()],
            active_model_id: Some("missing-model".into()),
        };
        write_model_state(&path, &state).await.unwrap();
        let stale_temp = root.join(format!(".models.json.{}.tmp", Uuid::new_v4()));
        tokio::fs::write(&stale_temp, b"not state").await.unwrap();

        let loaded = load_model_state(&path).await.unwrap();

        assert_eq!(loaded.state.models.len(), 1);
        assert_eq!(loaded.state.models[0].id, model.id);
        assert_eq!(loaded.state.active_model_id, None);
        assert!(!stale_temp.exists());
        assert!(model_state_temp_files(&root).await.is_empty());
        let _ = tokio::fs::remove_dir_all(root).await;
    }

    #[tokio::test]
    async fn failed_model_state_write_cleans_temp_file() {
        let root = std::env::temp_dir().join(format!("fathom-failed-state-{}", Uuid::new_v4()));
        let path = root.join("models.json");
        tokio::fs::create_dir_all(path.join("child")).await.unwrap();
        let state = PersistedModelState {
            models: vec![test_model("present-model")],
            active_model_id: None,
        };

        let result = write_model_state(&path, &state).await;

        assert!(result.is_err());
        assert!(model_state_temp_files(&root).await.is_empty());
        assert!(path.is_dir());
        let _ = tokio::fs::remove_dir_all(root).await;
    }

    #[tokio::test]
    async fn model_state_drops_missing_active_model() {
        let root = std::env::temp_dir().join(format!("fathom-state-test-{}", Uuid::new_v4()));
        let path = root.join("models.json");
        let state = PersistedModelState {
            models: vec![test_model("present-model")],
            active_model_id: Some("missing-model".into()),
        };

        write_model_state(&path, &state).await.unwrap();
        let loaded = load_model_state(&path).await.unwrap();

        assert_eq!(loaded.state.models.len(), 1);
        assert_eq!(loaded.state.active_model_id, None);
        let _ = tokio::fs::remove_dir_all(root).await;
    }

    async fn test_state_with_unwritable_model_file(
        models: Vec<ModelRecord>,
        active_model_id: Option<String>,
    ) -> (AppState, PathBuf) {
        let root = std::env::temp_dir().join(format!("fathom-unwritable-state-{}", Uuid::new_v4()));
        let path = root.join("models.json");
        tokio::fs::create_dir_all(path.join("child")).await.unwrap();
        (test_state_at(path, models, active_model_id), root)
    }

    #[tokio::test]
    async fn failed_register_model_persistence_does_not_publish_unsaved_model() {
        let package_root =
            std::env::temp_dir().join(format!("fathom-register-txn-{}", Uuid::new_v4()));
        tokio::fs::create_dir_all(&package_root).await.unwrap();
        let artifact_path = package_root.join("pytorch_model.bin");
        tokio::fs::write(&artifact_path, b"pickle-like fixture bytes")
            .await
            .unwrap();
        let (state, state_root) = test_state_with_unwritable_model_file(vec![], None).await;

        let error = register_model(
            State(state.clone()),
            Json(RegisterModelRequest {
                id: Some("unsaved-local".into()),
                name: "Unsaved local fixture".into(),
                model_path: artifact_path.to_string_lossy().to_string(),
                runtime_model_name: None,
            }),
        )
        .await
        .unwrap_err();

        assert_api_error(
            error,
            StatusCode::INTERNAL_SERVER_ERROR,
            "model_state_persist_failed",
        );
        let Json(body) = dashboard(State(state.clone())).await;
        assert!(body["models"].as_array().unwrap().is_empty());
        assert_eq!(body["runtime"]["active_model_id"], serde_json::Value::Null);
        let Json(models_body) = v1_models(State(state)).await;
        assert!(models_body["data"].as_array().unwrap().is_empty());
        let _ = tokio::fs::remove_dir_all(package_root).await;
        let _ = tokio::fs::remove_dir_all(state_root).await;
    }

    #[tokio::test]
    async fn failed_external_model_persistence_does_not_publish_unsaved_model() {
        let (state, state_root) = test_state_with_unwritable_model_file(vec![], None).await;

        let error = connect_external_model(
            State(state.clone()),
            Json(ExternalModelRequest {
                id: "unsaved-external".into(),
                name: "Unsaved external fixture".into(),
                model_name: "external-model".into(),
                api_base: "https://api.example.test/v1".into(),
                api_key: "test-key".into(),
                source: "OpenAI-compatible".into(),
            }),
        )
        .await
        .unwrap_err();

        assert_api_error(
            error,
            StatusCode::INTERNAL_SERVER_ERROR,
            "model_state_persist_failed",
        );
        let Json(body) = dashboard(State(state.clone())).await;
        assert!(body["models"].as_array().unwrap().is_empty());
        assert_eq!(body["runtime"]["loaded_now"], false);
        let Json(models_body) = v1_models(State(state)).await;
        assert!(models_body["data"].as_array().unwrap().is_empty());
        let _ = tokio::fs::remove_dir_all(state_root).await;
    }

    #[tokio::test]
    async fn failed_activate_model_persistence_restores_previous_active_model() {
        let previous = test_model("previous-local");
        let target = test_model("target-local");
        let (state, state_root) = test_state_with_unwritable_model_file(
            vec![previous.clone(), target.clone()],
            Some(previous.id.clone()),
        )
        .await;

        let error = activate_model(State(state.clone()), Path(target.id.clone()))
            .await
            .unwrap_err();

        assert_api_error(
            error,
            StatusCode::INTERNAL_SERVER_ERROR,
            "model_state_persist_failed",
        );
        let Json(runtime_body) = runtime_state(State(state.clone())).await;
        assert_eq!(runtime_body["active_model_id"], previous.id);
        let Json(dashboard_body) = dashboard(State(state)).await;
        assert_eq!(dashboard_body["runtime"]["active_model_id"], previous.id);
        let _ = tokio::fs::remove_dir_all(state_root).await;
    }

    #[tokio::test]
    async fn failed_catalog_final_registration_persistence_does_not_publish_unsaved_model() {
        let mut model = test_model("unsaved-catalog");
        model.source = Some("Catalog fixture".into());
        model.hf_repo = Some("example/catalog-fixture".into());
        let (state, state_root) = test_state_with_unwritable_model_file(vec![], None).await;

        let error = mutate_model_state_and_persist(&state, |store| {
            upsert_model(&mut store.models, model.clone());
            Ok(())
        })
        .await
        .unwrap_err();

        assert_api_error(
            error,
            StatusCode::INTERNAL_SERVER_ERROR,
            "model_state_persist_failed",
        );
        let Json(body) = dashboard(State(state.clone())).await;
        assert!(body["models"].as_array().unwrap().is_empty());
        let Json(models_body) = v1_models(State(state)).await;
        assert!(models_body["data"].as_array().unwrap().is_empty());
        let _ = tokio::fs::remove_dir_all(state_root).await;
    }

    #[test]
    fn max_tokens_default_cap_and_zero_validation_are_explicit() {
        assert_eq!(
            normalize_max_tokens(None).unwrap(),
            default_chat_max_tokens()
        );
        assert_eq!(normalize_max_tokens(Some(1)).unwrap(), 1);
        assert_eq!(normalize_max_tokens(Some(999)).unwrap(), 128);
        assert!(normalize_max_tokens(Some(0))
            .unwrap_err()
            .to_string()
            .contains("greater than 0"));
    }

    #[test]
    fn demo_generation_defaults_are_valid_and_bounded() {
        let options = demo_generation_options().validate().unwrap();
        assert_eq!(options.temperature, default_chat_temperature());
        assert_eq!(options.top_k, default_chat_top_k());
        assert_eq!(options.top_p, default_chat_top_p());
    }

    #[test]
    fn assistant_message_records_include_optional_runtime_generation_metrics() {
        let metrics = GenerationMetrics {
            model_load_ms: 63,
            generation_ms: 44,
            total_ms: 107,
            tokens_per_second: Some(8.125),
            ttft_ms: Some(12),
            prefill_ms: Some(12),
            decode_ms: Some(32),
            prefill_tokens_per_second: Some(20.0),
            decode_tokens_per_second: Some(7.5),
            runtime_cache_hit: true,
            runtime_cache_lookup_ms: 2,
            runtime_residency: Some("warm_reused".into()),
            runtime_family: Some("phi".into()),
        };

        let message = MessageRecord::assistant_from_generation(
            "msg-1".into(),
            "hello".into(),
            "2026-04-26T20:00:00Z".into(),
            Some("local-phi".into()),
            &metrics,
        );
        let value = serde_json::to_value(message).unwrap();

        assert_eq!(value["role"], "assistant");
        assert_eq!(value["tokens_out_per_sec"], 8.125);
        assert_eq!(value["runtime_cache_hit"], true);
        assert_eq!(value["runtime_residency"], "warm_reused");
        assert_eq!(value["runtime_family"], "phi");
        assert_eq!(value["runtime_cache_lookup_ms"], 2);
        assert_eq!(value["model_load_ms"], 63);
        assert_eq!(value["generation_ms"], 44);
        assert_eq!(value["total_ms"], 107);
        assert_eq!(value["ttft_ms"], 12);
        assert_eq!(value["prefill_ms"], 12);
        assert_eq!(value["decode_ms"], 32);
        assert_eq!(value["prefill_tokens_per_second"], 20.0);
        assert_eq!(value["decode_tokens_per_second"], 7.5);
    }

    #[test]
    fn older_message_records_without_runtime_metrics_still_deserialize() {
        let json = serde_json::json!({
            "id": "old-msg",
            "role": "assistant",
            "content": "older reply",
            "created_at": "2026-04-25T12:00:00Z",
            "model_id": "legacy-model",
            "tokens_out_per_sec": 11.0
        });

        let message: MessageRecord = serde_json::from_value(json).unwrap();

        assert_eq!(message.id, "old-msg");
        assert_eq!(message.tokens_out_per_sec, Some(11.0));
        assert_eq!(message.runtime_cache_hit, None);
        assert_eq!(message.runtime_residency, None);
        assert_eq!(message.runtime_family, None);
        assert_eq!(message.model_load_ms, None);
        assert_eq!(message.total_ms, None);
    }

    #[tokio::test]
    async fn dashboard_includes_context_strategy_advice_by_model_id() {
        let state = test_state(
            vec![test_external_model("external-gpt")],
            Some("external-gpt".into()),
        );

        let Json(body) = dashboard(State(state)).await;

        assert_eq!(
            body["context_strategies"]["external-gpt"]["engine"],
            "external_managed"
        );
        assert!(
            body["context_strategies"]["external-gpt"]["top_k"]
                .as_u64()
                .unwrap()
                > 0
        );
    }

    #[tokio::test]
    async fn embedding_models_endpoint_reports_onnx_embedding_metadata_without_runnable_claim() {
        let root = std::env::temp_dir().join(format!("fathom-server-onnx-emb-{}", Uuid::new_v4()));
        tokio::fs::create_dir_all(&root).await.unwrap();
        tokio::fs::write(root.join("model.onnx"), b"fake onnx fixture bytes")
            .await
            .unwrap();
        tokio::fs::write(
            root.join("config.json"),
            r#"{"model_type":"bert","architectures":["BertModel"],"hidden_size":384}"#,
        )
        .await
        .unwrap();
        tokio::fs::write(root.join("tokenizer.json"), b"{}")
            .await
            .unwrap();

        let mut model = test_model("local-embedder");
        model.name = "Local ONNX embedder".into();
        model.model_path = Some(root.to_string_lossy().to_string());
        model.format = Some("ONNX".into());
        model.status = "registered".into();
        model.capability_status = "metadata_only".into();
        model.backend_lanes = vec!["local-embeddings-retrieval".into()];
        let state = test_state(vec![model], None);

        let Json(body) = embedding_models(State(state)).await;
        let items = body["items"].as_array().unwrap();
        assert_eq!(items.len(), 1);
        assert_eq!(items[0]["id"], "local-embedder");
        assert_eq!(items[0]["status"]["task"], "text_embedding");
        assert_eq!(items[0]["status"]["status"], "Planned");
        assert_eq!(items[0]["status"]["runnable"], false);
        assert!(body["retrieval"]["summary"]
            .as_str()
            .unwrap()
            .contains("does not claim ONNX chat/LLM support"));

        let mut mislabeled_chat = test_model("mislabeled-embedder");
        mislabeled_chat.model_path = Some(root.to_string_lossy().to_string());
        mislabeled_chat.format = Some("ONNX".into());
        mislabeled_chat.status = "ready".into();
        mislabeled_chat.capability_status = "runnable".into();
        let Json(models_body) = v1_models(State(test_state(vec![mislabeled_chat], None))).await;
        assert_eq!(
            models_body["data"].as_array().unwrap().len(),
            0,
            "ONNX embedding packages must not be exposed as chat models"
        );

        let _ = tokio::fs::remove_dir_all(root).await;
    }

    #[tokio::test]
    async fn v1_models_excludes_ready_bert_embedding_package_from_chat_list() {
        let root =
            std::env::temp_dir().join(format!("fathom-server-bert-embed-{}", Uuid::new_v4()));
        tokio::fs::create_dir_all(&root).await.unwrap();
        tokio::fs::write(
            root.join("config.json"),
            r#"{"model_type":"bert","architectures":["BertModel"],"vocab_size":3,"hidden_size":4,"intermediate_size":8,"num_hidden_layers":1,"num_attention_heads":1,"max_position_embeddings":8,"type_vocab_size":2,"layer_norm_eps":1e-12,"hidden_act":"gelu","hidden_dropout_prob":0.0,"attention_probs_dropout_prob":0.0,"initializer_range":0.02,"pad_token_id":0}"#,
        )
        .await
        .unwrap();
        tokio::fs::write(
            root.join("tokenizer.json"),
            r#"{"version":"1.0","truncation":null,"padding":null,"added_tokens":[],"normalizer":null,"pre_tokenizer":{"type":"Whitespace"},"post_processor":null,"decoder":null,"model":{"type":"WordLevel","vocab":{"<unk>":0,"hello":1,"world":2},"unk_token":"<unk>"}}"#,
        )
        .await
        .unwrap();
        write_test_bert_safetensors_header(&root.join("model.safetensors"));

        let mut model = test_model("bert-embedder");
        model.name = "BERT embedder".into();
        model.model_path = Some(root.to_string_lossy().to_string());
        model.status = "ready".into();
        model.capability_status = "runnable".into();
        model.backend_lanes = vec!["local-embeddings-retrieval".into()];

        let Json(embedding_body) =
            embedding_models(State(test_state(vec![model.clone()], None))).await;
        assert_eq!(embedding_body["items"].as_array().unwrap().len(), 1);
        assert_eq!(
            embedding_body["items"][0]["status"]["runtime_lane"],
            "candle-bert-embeddings"
        );
        assert_eq!(embedding_body["items"][0]["status"]["runnable"], true);

        let Json(models_body) = v1_models(State(test_state(vec![model], None))).await;
        assert_eq!(
            models_body["data"].as_array().unwrap().len(),
            0,
            "BERT embedding packages must not be exposed as chat models"
        );

        let _ = tokio::fs::remove_dir_all(root).await;
    }

    #[tokio::test]
    async fn v1_embeddings_accepts_string_and_array_for_verified_embedding_model() {
        let root =
            std::env::temp_dir().join(format!("fathom-server-v1-bert-embed-{}", Uuid::new_v4()));
        tokio::fs::create_dir_all(&root).await.unwrap();
        tokio::fs::write(
            root.join("config.json"),
            r#"{"model_type":"bert","architectures":["BertModel"],"vocab_size":3,"hidden_size":4,"intermediate_size":8,"num_hidden_layers":1,"num_attention_heads":1,"max_position_embeddings":8,"type_vocab_size":2,"layer_norm_eps":1e-12,"hidden_act":"gelu","hidden_dropout_prob":0.0,"attention_probs_dropout_prob":0.0,"initializer_range":0.02,"pad_token_id":0}"#,
        )
        .await
        .unwrap();
        tokio::fs::write(
            root.join("tokenizer.json"),
            r#"{"version":"1.0","truncation":null,"padding":null,"added_tokens":[],"normalizer":null,"pre_tokenizer":{"type":"Whitespace"},"post_processor":null,"decoder":null,"model":{"type":"WordLevel","vocab":{"<unk>":0,"hello":1,"world":2},"unk_token":"<unk>"}}"#,
        )
        .await
        .unwrap();
        write_test_bert_safetensors_header(&root.join("model.safetensors"));

        let mut model = test_model("bert-embedder");
        model.name = "BERT embedder".into();
        model.model_path = Some(root.to_string_lossy().to_string());
        model.status = "ready".into();
        model.capability_status = "runnable".into();
        model.backend_lanes = vec!["local-embeddings-retrieval".into()];
        model.task = Some("text_embedding".into());
        let state = test_state(vec![model], None);

        let (single_status, Json(single_body)) = v1_embeddings(
            State(state.clone()),
            Json(V1EmbeddingsRequest {
                model: Some("bert-embedder".into()),
                input: V1EmbeddingInput::Single("hello world".into()),
                encoding_format: Some("float".into()),
            }),
        )
        .await;
        assert_eq!(single_status, StatusCode::OK, "{single_body}");
        assert_eq!(single_body["object"], "list");
        assert!(single_body.get("error").is_none());
        assert!(single_body.get("choices").is_none());
        assert_eq!(single_body["data"].as_array().unwrap().len(), 1);
        assert_eq!(single_body["data"][0]["object"], "embedding");
        assert_eq!(
            single_body["data"][0]["embedding"]
                .as_array()
                .unwrap()
                .len(),
            4
        );
        assert_eq!(single_body["fathom"]["runtime"], "candle-bert-embeddings");
        assert_eq!(single_body["fathom"]["embedding_dimension"], 4);

        let (array_status, Json(array_body)) = v1_embeddings(
            State(state),
            Json(V1EmbeddingsRequest {
                model: Some("bert-embedder".into()),
                input: V1EmbeddingInput::Batch(vec!["hello".into(), "world".into()]),
                encoding_format: None,
            }),
        )
        .await;
        assert_eq!(array_status, StatusCode::OK);
        assert_eq!(array_body["data"].as_array().unwrap().len(), 2);
        assert_eq!(array_body["data"][1]["index"], 1);
        assert_eq!(array_body["usage"]["prompt_tokens"], 0);

        let _ = tokio::fs::remove_dir_all(root).await;
    }

    #[tokio::test]
    async fn v1_embeddings_rejects_unsupported_encoding_format() {
        let state = test_state(vec![], None);

        let (status, Json(body)) = v1_embeddings(
            State(state),
            Json(V1EmbeddingsRequest {
                model: Some("embedder".into()),
                input: V1EmbeddingInput::Single("hello".into()),
                encoding_format: Some("base64".into()),
            }),
        )
        .await;

        assert_eq!(status, StatusCode::BAD_REQUEST);
        assert_v1_error_envelope(&body, "invalid_request");
        assert_no_fake_embedding_success(&body);
        assert!(body["error"]["message"]
            .as_str()
            .unwrap()
            .contains("encoding_format='float'"));
    }

    #[tokio::test]
    async fn v1_embeddings_rejects_chat_model_without_inspecting_runtime() {
        let state = test_state(vec![test_model("local-gpt2")], None);

        let (status, Json(body)) = v1_embeddings(
            State(state),
            Json(V1EmbeddingsRequest {
                model: Some("local-gpt2".into()),
                input: V1EmbeddingInput::Single("hello".into()),
                encoding_format: Some("float".into()),
            }),
        )
        .await;

        assert_eq!(status, StatusCode::BAD_REQUEST);
        assert_v1_error_envelope(&body, "not_embedding_model");
        assert_no_fake_embedding_success(&body);
        assert!(body["error"]["message"]
            .as_str()
            .unwrap()
            .contains("chat/generation model"));
    }

    #[tokio::test]
    async fn v1_embeddings_missing_or_unknown_model_returns_json_error() {
        let state = test_state(vec![], None);

        let (missing_status, Json(missing_body)) = v1_embeddings(
            State(state.clone()),
            Json(V1EmbeddingsRequest {
                model: None,
                input: V1EmbeddingInput::Batch(vec!["hello".into()]),
                encoding_format: Some("float".into()),
            }),
        )
        .await;
        assert_eq!(missing_status, StatusCode::BAD_REQUEST);
        assert_v1_error_envelope(&missing_body, "model_not_found");
        assert_no_fake_embedding_success(&missing_body);

        let (unknown_status, Json(unknown_body)) = v1_embeddings(
            State(state),
            Json(V1EmbeddingsRequest {
                model: Some("missing".into()),
                input: V1EmbeddingInput::Batch(vec!["hello".into()]),
                encoding_format: Some("float".into()),
            }),
        )
        .await;

        assert_eq!(unknown_status, StatusCode::NOT_FOUND);
        assert_v1_error_envelope(&unknown_body, "embedding_model_not_found");
        assert_no_fake_embedding_success(&unknown_body);
    }

    #[tokio::test]
    async fn v1_embeddings_refuses_gguf_and_pytorch_without_fake_vectors() {
        let mut gguf = test_model("gguf-metadata-only");
        gguf.format = Some("GGUF".into());
        gguf.backend_lanes = vec!["gguf-native".into()];
        gguf.capability_status = "metadata_only".into();
        gguf.task = None;

        let mut pytorch = test_model("blocked-bin");
        pytorch.format = Some("PyTorchBin".into());
        pytorch.backend_lanes = vec!["pytorch-trusted-import".into()];
        pytorch.capability_status = "blocked".into();
        pytorch.task = None;
        let state = test_state(vec![gguf, pytorch], None);

        for model_id in ["gguf-metadata-only", "blocked-bin"] {
            let (status, Json(body)) = v1_embeddings(
                State(state.clone()),
                Json(V1EmbeddingsRequest {
                    model: Some(model_id.into()),
                    input: V1EmbeddingInput::Single("hello".into()),
                    encoding_format: Some("float".into()),
                }),
            )
            .await;

            assert_eq!(status, StatusCode::NOT_IMPLEMENTED);
            assert_v1_error_envelope(&body, "embedding_runtime_unavailable");
            assert_no_fake_embedding_success(&body);
            assert!(body["error"]["message"]
                .as_str()
                .unwrap()
                .contains("fake vectors"));
        }
    }

    #[tokio::test]
    async fn embedding_endpoint_reports_runtime_unavailable_without_fake_vectors() {
        let root =
            std::env::temp_dir().join(format!("fathom-server-onnx-embed-api-{}", Uuid::new_v4()));
        tokio::fs::create_dir_all(&root).await.unwrap();
        tokio::fs::write(root.join("model.onnx"), b"fake onnx fixture bytes")
            .await
            .unwrap();
        tokio::fs::write(
            root.join("config.json"),
            r#"{"model_type":"bert","architectures":["BertModel"],"hidden_size":384}"#,
        )
        .await
        .unwrap();
        tokio::fs::write(root.join("tokenizer.json"), b"{}")
            .await
            .unwrap();

        let mut model = test_model("local-embedder");
        model.model_path = Some(root.to_string_lossy().to_string());
        model.format = Some("ONNX".into());
        model.status = "registered".into();
        model.capability_status = "metadata_only".into();
        let state = test_state(vec![model], None);

        let error = embed_with_embedding_model(
            State(state),
            Path("local-embedder".into()),
            Json(EmbedRequest {
                input: vec!["hello world".into()],
                normalize: true,
            }),
        )
        .await
        .unwrap_err();

        assert_eq!(error.0, StatusCode::NOT_IMPLEMENTED);
        assert_eq!(error.1["error"]["type"], "embedding_runtime_unavailable");
        let _ = tokio::fs::remove_dir_all(root).await;
    }

    #[tokio::test]
    async fn embedding_endpoint_rejects_empty_input_before_runtime_preflight() {
        let root = std::env::temp_dir().join(format!(
            "fathom-server-onnx-embed-invalid-input-{}",
            Uuid::new_v4()
        ));
        tokio::fs::create_dir_all(&root).await.unwrap();
        tokio::fs::write(root.join("model.onnx"), b"fake onnx fixture bytes")
            .await
            .unwrap();
        tokio::fs::write(
            root.join("config.json"),
            r#"{"model_type":"bert","architectures":["BertModel"],"hidden_size":384}"#,
        )
        .await
        .unwrap();
        tokio::fs::write(root.join("tokenizer.json"), b"{}")
            .await
            .unwrap();

        let mut model = test_model("local-embedder");
        model.model_path = Some(root.to_string_lossy().to_string());
        model.format = Some("ONNX".into());
        let state = test_state(vec![model], None);

        let empty_error = embed_with_embedding_model(
            State(state.clone()),
            Path("local-embedder".into()),
            Json(EmbedRequest {
                input: vec![],
                normalize: true,
            }),
        )
        .await
        .unwrap_err();
        assert_eq!(empty_error.0, StatusCode::BAD_REQUEST);
        assert_eq!(empty_error.1["error"]["type"], "invalid_embedding_input");

        let blank_error = embed_with_embedding_model(
            State(state),
            Path("local-embedder".into()),
            Json(EmbedRequest {
                input: vec!["  ".into()],
                normalize: true,
            }),
        )
        .await
        .unwrap_err();
        assert_eq!(blank_error.0, StatusCode::BAD_REQUEST);
        assert_eq!(blank_error.1["error"]["type"], "invalid_embedding_input");
        let _ = tokio::fs::remove_dir_all(root).await;
    }

    #[tokio::test]
    async fn retrieval_index_api_creates_persists_lists_adds_and_searches_supplied_vectors() {
        let root = std::env::temp_dir().join(format!("fathom-retrieval-api-{}", Uuid::new_v4()));
        let state = test_state_at(root.join("models.json"), vec![], None);

        let Json(created) = create_retrieval_index(
            State(state.clone()),
            Json(CreateRetrievalIndexRequest {
                id: "notes".into(),
                embedding_model_id: "explicit-fixture-vectors".into(),
                embedding_dimension: 3,
            }),
        )
        .await
        .unwrap();
        assert_eq!(created.id, "notes");
        assert_eq!(created.chunk_count, 0);

        let Json(after_rust) = add_retrieval_chunk(
            State(state.clone()),
            Path("notes".into()),
            Json(AddRetrievalChunkRequest {
                chunk: test_chunk("rust", "doc-a", "Rust ownership notes"),
                vector: vec![1.0, 0.0, 0.0],
            }),
        )
        .await
        .unwrap();
        assert_eq!(after_rust.chunk_count, 1);

        let _ = add_retrieval_chunk(
            State(state.clone()),
            Path("notes".into()),
            Json(AddRetrievalChunkRequest {
                chunk: test_chunk("python", "doc-b", "Python scripting notes"),
                vector: vec![0.0, 1.0, 0.0],
            }),
        )
        .await
        .unwrap();

        let Json(search) = search_retrieval_index(
            State(state.clone()),
            Path("notes".into()),
            Json(SearchRetrievalIndexRequest {
                vector: vec![0.9, 0.1, 0.0],
                top_k: Some(1),
                metric: Some(VectorSearchMetric::Cosine),
            }),
        )
        .await
        .unwrap();
        assert_eq!(search.index.chunk_count, 2);
        assert_eq!(search.hits.len(), 1);
        assert_eq!(search.hits[0].chunk.id, "rust");

        let Json(listed) = list_retrieval_indexes(State(state.clone())).await.unwrap();
        assert_eq!(listed.items.len(), 1);
        assert_eq!(listed.items[0].id, "notes");
        assert_eq!(listed.items[0].document_count, 2);
        assert!(listed.summary.contains("caller-supplied vectors"));

        let persisted = VectorIndex::load_from_state_dir(&root, "notes").unwrap();
        assert_eq!(persisted.entries.len(), 2);
        let _ = tokio::fs::remove_dir_all(root).await;
    }

    #[tokio::test]
    async fn chat_retrieval_extension_resolves_explicit_vector_context() {
        let root = std::env::temp_dir().join(format!("fathom-chat-retrieval-{}", Uuid::new_v4()));
        let state = test_state_at(root.join("models.json"), vec![], None);

        let _ = create_retrieval_index(
            State(state.clone()),
            Json(CreateRetrievalIndexRequest {
                id: "notes".into(),
                embedding_model_id: "explicit-fixture-vectors".into(),
                embedding_dimension: 3,
            }),
        )
        .await
        .unwrap();
        let _ = add_retrieval_chunk(
            State(state.clone()),
            Path("notes".into()),
            Json(AddRetrievalChunkRequest {
                chunk: test_chunk(
                    "rust",
                    "doc-a",
                    "Rust ownership lets one owner manage a value.",
                ),
                vector: vec![1.0, 0.0, 0.0],
            }),
        )
        .await
        .unwrap();
        let _ = add_retrieval_chunk(
            State(state.clone()),
            Path("notes".into()),
            Json(AddRetrievalChunkRequest {
                chunk: test_chunk("python", "doc-b", "Python scripting note"),
                vector: vec![0.0, 1.0, 0.0],
            }),
        )
        .await
        .unwrap();

        let (message, metadata) = resolve_retrieval_for_chat(
            &state,
            &FathomRetrievalRequest {
                index_id: "notes".into(),
                query_vector: vec![0.9, 0.1, 0.0],
                top_k: Some(1),
                metric: Some(VectorSearchMetric::Cosine),
                max_context_chars: Some(260),
            },
        )
        .unwrap();

        let message = message.expect("matching chunk should be inserted");
        assert_eq!(message.role, "system");
        assert!(message.content.contains("caller-supplied explicit vectors"));
        assert!(message.content.contains("chunk_id=rust"));
        assert!(message.content.contains("Rust ownership"));
        assert!(!message.content.contains("chunk_id=python"));
        assert_eq!(metadata.index.id, "notes");
        assert_eq!(metadata.hit_count, 1);
        assert_eq!(metadata.inserted_chunk_count, 1);
        assert_eq!(metadata.hits[0].chunk_id, "rust");
        assert!(metadata.hits[0].inserted);

        let _ = tokio::fs::remove_dir_all(root).await;
    }

    #[test]
    fn retrieval_context_respects_requested_character_bound() {
        let hit = VectorSearchHit {
            chunk: test_chunk("long", "doc-a", "abcdefghijklmnopqrstuvwxyz"),
            score: 0.5,
        };
        let (message, metadata) = retrieval_context_from_hits(
            RetrievalIndexSummary {
                schema_version: 1,
                id: "notes".into(),
                embedding_model_id: "fixture".into(),
                embedding_dimension: 3,
                document_count: 1,
                chunk_count: 1,
                status: CapabilityStatus::Runnable,
            },
            VectorSearchMetric::Cosine,
            1,
            vec![hit],
            220,
        );

        let message = message.expect("bounded snippet should fit");
        assert!(message.content.chars().count() <= 220);
        assert_eq!(metadata.max_context_chars, 220);
        assert_eq!(metadata.inserted_chunk_count, 1);
        assert_eq!(metadata.hits[0].chunk_id, "long");
    }

    #[tokio::test]
    async fn retrieval_index_api_rejects_dimension_mismatch_without_persisting_chunk() {
        let root = std::env::temp_dir().join(format!("fathom-retrieval-api-{}", Uuid::new_v4()));
        let state = test_state_at(root.join("models.json"), vec![], None);

        let _ = create_retrieval_index(
            State(state.clone()),
            Json(CreateRetrievalIndexRequest {
                id: "notes".into(),
                embedding_model_id: "explicit-fixture-vectors".into(),
                embedding_dimension: 3,
            }),
        )
        .await
        .unwrap();

        let (status, Json(body)) = add_retrieval_chunk(
            State(state.clone()),
            Path("notes".into()),
            Json(AddRetrievalChunkRequest {
                chunk: test_chunk("bad", "doc-a", "bad vector"),
                vector: vec![1.0, 0.0],
            }),
        )
        .await
        .unwrap_err();
        assert_eq!(status, StatusCode::BAD_REQUEST);
        assert_eq!(body["error"]["code"], "invalid_retrieval_chunk");
        assert!(body["error"]["message"]
            .as_str()
            .unwrap()
            .contains("dimension"));

        let persisted = VectorIndex::load_from_state_dir(&root, "notes").unwrap();
        assert_eq!(persisted.entries.len(), 0);
        let _ = tokio::fs::remove_dir_all(root).await;
    }

    #[tokio::test]
    async fn retrieval_index_api_returns_json_errors_for_duplicate_and_missing_index() {
        let root = std::env::temp_dir().join(format!("fathom-retrieval-api-{}", Uuid::new_v4()));
        let state = test_state_at(root.join("models.json"), vec![], None);
        let _ = create_retrieval_index(
            State(state.clone()),
            Json(CreateRetrievalIndexRequest {
                id: "notes".into(),
                embedding_model_id: "explicit-fixture-vectors".into(),
                embedding_dimension: 3,
            }),
        )
        .await
        .unwrap();
        let (duplicate_status, Json(duplicate_body)) = create_retrieval_index(
            State(state.clone()),
            Json(CreateRetrievalIndexRequest {
                id: "notes".into(),
                embedding_model_id: "explicit-fixture-vectors".into(),
                embedding_dimension: 3,
            }),
        )
        .await
        .unwrap_err();
        assert_eq!(duplicate_status, StatusCode::CONFLICT);
        assert_eq!(duplicate_body["error"]["code"], "retrieval_index_exists");
        assert!(duplicate_body["error"].get("param").is_some());

        let (missing_status, Json(missing_body)) = search_retrieval_index(
            State(state.clone()),
            Path("missing".into()),
            Json(SearchRetrievalIndexRequest {
                vector: vec![1.0, 0.0, 0.0],
                top_k: Some(1),
                metric: Some(VectorSearchMetric::Cosine),
            }),
        )
        .await
        .unwrap_err();
        assert_eq!(missing_status, StatusCode::NOT_FOUND);
        assert_eq!(missing_body["error"]["code"], "retrieval_index_not_found");
        let _ = tokio::fs::remove_dir_all(root).await;
    }

    #[tokio::test]
    async fn context_strategy_endpoint_returns_404_for_missing_model() {
        let state = test_state(vec![], None);

        let error = model_context_strategy(State(state), Path("missing".into()))
            .await
            .unwrap_err();

        assert_eq!(error.0, StatusCode::NOT_FOUND);
    }

    #[tokio::test]
    async fn v1_health_returns_small_success_shape() {
        let state = test_state(vec![test_model("local-gpt2")], None);

        let Json(body) = v1_health(State(state)).await;

        assert_eq!(body["ok"], true);
        assert_eq!(body["engine"], "fathom");
        assert_eq!(body["generation_ready"], true);
        assert!(body.get("data").is_none());
        assert!(body.get("choices").is_none());
        assert!(body.get("error").is_none());
    }

    #[tokio::test]
    async fn v1_health_excludes_external_proxy_models_from_generation_ready() {
        let state = test_state(vec![test_external_model("external-gpt")], None);

        let Json(body) = v1_health(State(state)).await;

        assert_eq!(body["ok"], true);
        assert_eq!(body["generation_ready"], false);
    }

    #[tokio::test]
    async fn v1_models_returns_openai_list_shape_with_fathom_metadata_nested() {
        let mut unsupported = test_model("known-unsupported");
        unsupported.status = "registered".into();
        unsupported.capability_status = "planned".into();
        let external = test_external_model("external-gpt");
        let state = test_state(vec![test_model("local-gpt2"), unsupported, external], None);

        let Json(body) = v1_models(State(state)).await;
        assert_eq!(body["object"], "list");
        let data = body["data"].as_array().unwrap();
        assert_eq!(
            data.len(),
            1,
            "only local models that /v1/chat/completions can run are advertised"
        );
        assert!(data.iter().any(|model| model["id"] == "local-gpt2"));
        assert!(!data.iter().any(|model| model["id"] == "external-gpt"));
        assert!(body.get("choices").is_none());
        assert!(body.get("error").is_none());

        let local = data
            .iter()
            .find(|model| model["id"] == "local-gpt2")
            .unwrap();
        assert_eq!(local["object"], "model");
        assert_eq!(local["owned_by"], "fathom");
        assert!(local.get("capability_status").is_none());
        assert_eq!(local["fathom"]["capability_status"], "runnable");
        assert_eq!(local["fathom"]["backend_lanes"][0], "safetensors-hf");
    }

    #[tokio::test]
    async fn v1_chat_missing_or_unknown_model_returns_openai_error_shape() {
        let state = test_state(vec![], None);

        for model in [None, Some("missing")] {
            let (status, Json(body)) =
                v1_chat_completions(State(state.clone()), Json(test_chat_request(model))).await;

            assert_eq!(status, StatusCode::BAD_REQUEST);
            assert_v1_error_envelope(&body, "model_not_found");
            assert_no_fake_chat_success(&body);
        }
    }

    #[tokio::test]
    async fn blocked_pytorch_bin_registration_stays_out_of_models_and_chat() {
        let root =
            std::env::temp_dir().join(format!("fathom-server-pytorch-bin-{}", Uuid::new_v4()));
        tokio::fs::create_dir_all(&root).await.unwrap();
        let artifact_path = root.join("pytorch_model.bin");
        tokio::fs::write(&artifact_path, b"pickle-like fixture bytes")
            .await
            .unwrap();
        let state = test_state(vec![], None);

        let Json(model) = register_model(
            State(state.clone()),
            Json(RegisterModelRequest {
                id: Some("unsafe-pytorch-bin".into()),
                name: "Unsafe PyTorch bin fixture".into(),
                model_path: artifact_path.to_string_lossy().to_string(),
                runtime_model_name: None,
            }),
        )
        .await
        .unwrap();

        assert_eq!(model.status, "registered");
        assert_eq!(model.capability_status, "blocked");
        assert_eq!(model.format.as_deref(), Some("PyTorchBin"));
        assert!(model
            .backend_lanes
            .iter()
            .any(|lane| lane == "pytorch-trusted-import"));
        let install_error = model.install_error.as_deref().unwrap_or_default();
        assert!(install_error.contains("PyTorch"));
        assert!(install_error.contains("pickle"));
        assert!(install_error.contains("bin"));
        assert!(install_error.contains("execute code"));
        assert!(!install_error.to_ascii_lowercase().contains("runnable"));
        let capability = model.capability_summary.to_ascii_lowercase();
        assert!(capability.contains("pytorch"));
        assert!(capability.contains("pickle"));
        assert!(capability.contains("blocked"));
        assert!(capability.contains("not runnable"));
        assert!(!capability.contains("chat-ready"));

        let Json(models_body) = v1_models(State(state.clone())).await;
        assert_eq!(models_body["data"].as_array().unwrap().len(), 0);

        let (status, Json(body)) = v1_chat_completions(
            State(state),
            Json(test_chat_request(Some("unsafe-pytorch-bin"))),
        )
        .await;
        assert_eq!(status, StatusCode::NOT_IMPLEMENTED);
        assert_v1_error_envelope(&body, "not_implemented");
        assert_no_fake_chat_success(&body);
        let message = body["error"]["message"].as_str().unwrap();
        assert!(message.contains("PyTorch"));
        assert!(message.contains("pickle"));
        assert!(message.contains(".bin"));
        assert!(message.contains("blocked"));
        assert!(message.contains("trusted-import"));
        assert!(message.contains("No fake inference"));

        let _ = tokio::fs::remove_dir_all(root).await;
    }

    #[tokio::test]
    async fn v1_models_excludes_metadata_only_gguf_and_chat_refuses_it() {
        let mut gguf = test_model("synthetic-gguf-tokenizer-spec");
        gguf.status = "ready".into();
        gguf.capability_status = "metadata_only".into();
        gguf.capability_summary =
            "GGUF metadata/tokenizer spec only; native inference is not runnable".into();
        gguf.format = Some("GGUF".into());
        gguf.model_path = Some("/tmp/synthetic-tokenizer-spec.gguf".into());
        gguf.backend_lanes = vec!["gguf-native".into()];
        gguf.task = None;
        let state = test_state(vec![gguf], None);

        let Json(models_body) = v1_models(State(state.clone())).await;
        assert_eq!(models_body["data"].as_array().unwrap().len(), 0);

        let (status, Json(body)) = v1_chat_completions(
            State(state),
            Json(test_chat_request(Some("synthetic-gguf-tokenizer-spec"))),
        )
        .await;
        assert_eq!(status, StatusCode::NOT_IMPLEMENTED);
        assert_v1_error_envelope(&body, "not_implemented");
        assert_no_fake_chat_success(&body);
        let message = body["error"]["message"].as_str().unwrap();
        assert!(message.contains("metadata-only"));
        assert!(message.contains("not runnable"));
        assert!(message.contains("No fake inference"));
        assert!(message
            .contains("private fixture-scoped Llama/SentencePiece encode/decode parity helpers"));
        assert!(message.contains("public/runtime tokenizer execution"));
        assert!(message.contains("runtime weight loading"));
    }

    #[tokio::test]
    async fn v1_chat_known_unsupported_local_model_does_not_fake_generation() {
        let mut model = test_model("known-unsupported");
        model.status = "registered".into();
        model.model_path = Some("/tmp/not-runnable".into());
        model.backend_lanes = vec!["gguf".into()];
        let state = test_state(vec![model], None);

        let (status, Json(body)) = v1_chat_completions(
            State(state),
            Json(test_chat_request(Some("known-unsupported"))),
        )
        .await;

        assert_eq!(status, StatusCode::NOT_IMPLEMENTED);
        assert_v1_error_envelope(&body, "not_implemented");
        assert_no_fake_chat_success(&body);
        assert!(body["error"]["message"]
            .as_str()
            .unwrap()
            .contains("No fake inference"));
    }

    #[tokio::test]
    async fn v1_chat_external_model_proxy_shape_is_truthful_not_implemented() {
        let state = test_state(vec![test_external_model("external-gpt")], None);

        let (status, Json(body)) =
            v1_chat_completions(State(state), Json(test_chat_request(Some("external-gpt")))).await;

        assert_eq!(status, StatusCode::NOT_IMPLEMENTED);
        assert_v1_error_envelope(&body, "not_implemented");
        assert_no_fake_chat_success(&body);
        assert!(body["error"]["message"]
            .as_str()
            .unwrap()
            .contains("External OpenAI-compatible proxying"));
    }

    #[tokio::test]
    async fn external_model_connection_is_metadata_only_and_activation_refuses() {
        let state = test_state(vec![], None);

        let Json(model) = connect_external_model(
            State(state.clone()),
            Json(ExternalModelRequest {
                id: "connected-external".into(),
                name: "Connected external placeholder".into(),
                model_name: "gpt-placeholder".into(),
                api_base: "https://api.example.test/v1".into(),
                api_key: "test-key".into(),
                source: "OpenAI-compatible".into(),
            }),
        )
        .await
        .unwrap();

        assert_eq!(model.status, "ready");
        assert_eq!(model.provider_kind, "external");
        assert_eq!(model.capability_status, "planned");
        assert!(model.capability_summary.contains("metadata only"));
        assert!(model.capability_summary.contains("does not proxy chat"));
        assert!(!is_runnable_model(&model));

        let Json(models_body) = v1_models(State(state.clone())).await;
        assert_eq!(models_body["data"].as_array().unwrap().len(), 0);

        let error = activate_model(State(state.clone()), Path(model.id.clone()))
            .await
            .unwrap_err();
        assert_api_error(
            error,
            StatusCode::NOT_IMPLEMENTED,
            "external_proxy_not_implemented",
        );

        let Json(runtime_body) = runtime_state(State(state)).await;
        assert_eq!(runtime_body["loaded_now"], false);
        assert_eq!(runtime_body["active_model_id"], serde_json::Value::Null);
    }

    #[tokio::test]
    async fn v1_chat_rejects_streaming_until_supported() {
        let state = test_state(vec![test_model("local-gpt2")], None);
        let mut req = test_chat_request(Some("local-gpt2"));
        req.stream = Some(true);

        let (status, Json(body)) = v1_chat_completions(State(state), Json(req)).await;

        assert_eq!(status, StatusCode::NOT_IMPLEMENTED);
        assert_v1_error_envelope(&body, "not_implemented");
        assert_no_fake_chat_success(&body);
        assert!(body["error"]["message"]
            .as_str()
            .unwrap()
            .contains("Streaming chat completions"));
    }

    #[tokio::test]
    async fn v1_chat_sampling_validation_rejects_bad_temperature_before_generation() {
        let state = test_state(vec![test_model("local-gpt2")], None);
        let mut req = test_chat_request(Some("local-gpt2"));
        req.temperature = Some(-0.1);

        let (status, Json(body)) = v1_chat_completions(State(state), Json(req)).await;

        assert_eq!(status, StatusCode::BAD_REQUEST);
        assert_eq!(body["error"]["type"], "invalid_request");
        assert!(body["error"]["message"]
            .as_str()
            .unwrap()
            .contains("temperature"));
    }

    #[tokio::test]
    async fn missing_context_strategy_returns_json_error() {
        let state = test_state(vec![], None);

        let error = model_context_strategy(State(state), Path("missing-model".into()))
            .await
            .unwrap_err();

        assert_api_error(error, StatusCode::NOT_FOUND, "model_not_found");
    }

    #[tokio::test]
    async fn rename_missing_conversation_returns_json_error() {
        let state = test_state(vec![], None);

        let error = rename_conversation(
            State(state),
            Path("missing-conversation".into()),
            Json(PatchConversationRequest {
                title: Some("Renamed".into()),
            }),
        )
        .await
        .unwrap_err();

        assert_api_error(error, StatusCode::NOT_FOUND, "conversation_not_found");
    }

    #[tokio::test]
    async fn delete_missing_conversation_returns_json_error() {
        let state = test_state(vec![], None);

        let error = delete_conversation(State(state), Path("missing-conversation".into()))
            .await
            .unwrap_err();

        assert_api_error(error, StatusCode::NOT_FOUND, "conversation_not_found");
    }

    #[tokio::test]
    async fn delete_existing_conversation_preserves_no_content() {
        let state = test_state(vec![], None);
        insert_test_conversation(&state, "conversation-a", Some("local-gpt2".into())).await;

        let status = delete_conversation(State(state), Path("conversation-a".into()))
            .await
            .unwrap();

        assert_eq!(status, StatusCode::NO_CONTENT);
    }

    #[tokio::test]
    async fn chat_missing_conversation_returns_json_error() {
        let state = test_state(vec![test_model("local-gpt2")], None);

        let error = chat_conversation(
            State(state),
            Path("missing-conversation".into()),
            Json(ChatRequest {
                content: "Hello".into(),
                model_id: Some("local-gpt2".into()),
            }),
        )
        .await
        .unwrap_err();

        assert_api_error(error, StatusCode::NOT_FOUND, "conversation_not_found");
    }

    #[tokio::test]
    async fn chat_empty_message_returns_json_error() {
        let state = test_state(vec![test_model("local-gpt2")], None);
        insert_test_conversation(&state, "conversation-a", Some("local-gpt2".into())).await;

        let error = chat_conversation(
            State(state),
            Path("conversation-a".into()),
            Json(ChatRequest {
                content: "   ".into(),
                model_id: Some("local-gpt2".into()),
            }),
        )
        .await
        .unwrap_err();

        assert_api_error(error, StatusCode::BAD_REQUEST, "invalid_chat_message");
    }

    #[tokio::test]
    async fn chat_external_placeholder_refuses_without_fake_answer() {
        let state = test_state(vec![test_external_model("external-gpt")], None);
        insert_test_conversation(&state, "conversation-a", Some("external-gpt".into())).await;

        let error = chat_conversation(
            State(state),
            Path("conversation-a".into()),
            Json(ChatRequest {
                content: "Hello".into(),
                model_id: Some("external-gpt".into()),
            }),
        )
        .await
        .unwrap_err();

        assert_api_error(
            error,
            StatusCode::NOT_IMPLEMENTED,
            "external_proxy_not_implemented",
        );
    }

    #[tokio::test]
    async fn create_invalid_memory_returns_json_error() {
        let state = test_state(vec![], None);

        let error = create_memory(
            State(state),
            Json(MemoryRequest {
                title: Some("".into()),
                body: Some("body".into()),
                scope: None,
            }),
        )
        .await
        .unwrap_err();

        assert_api_error(error, StatusCode::BAD_REQUEST, "invalid_memory");
    }

    #[tokio::test]
    async fn update_invalid_memory_returns_json_error() {
        let state = test_state(vec![], None);
        insert_test_memory(&state, "memory-a").await;

        let error = update_memory(
            State(state),
            Path("memory-a".into()),
            Json(MemoryRequest {
                title: Some("   ".into()),
                body: None,
                scope: None,
            }),
        )
        .await
        .unwrap_err();

        assert_api_error(error, StatusCode::BAD_REQUEST, "invalid_memory");
    }

    #[tokio::test]
    async fn update_missing_memory_returns_json_error() {
        let state = test_state(vec![], None);

        let error = update_memory(
            State(state),
            Path("missing-memory".into()),
            Json(MemoryRequest {
                title: Some("Title".into()),
                body: Some("Body".into()),
                scope: None,
            }),
        )
        .await
        .unwrap_err();

        assert_api_error(error, StatusCode::NOT_FOUND, "memory_not_found");
    }

    #[tokio::test]
    async fn delete_missing_memory_returns_json_error() {
        let state = test_state(vec![], None);

        let error = delete_memory(State(state), Path("missing-memory".into()))
            .await
            .unwrap_err();

        assert_api_error(error, StatusCode::NOT_FOUND, "memory_not_found");
    }

    #[tokio::test]
    async fn delete_existing_memory_preserves_no_content() {
        let state = test_state(vec![], None);
        insert_test_memory(&state, "memory-a").await;

        let status = delete_memory(State(state), Path("memory-a".into()))
            .await
            .unwrap();

        assert_eq!(status, StatusCode::NO_CONTENT);
    }

    fn assert_api_error(
        (status, Json(body)): ApiError,
        expected_status: StatusCode,
        expected_code: &str,
    ) {
        assert_eq!(status, expected_status);
        assert_v1_error_envelope(&body, expected_code);
    }

    fn assert_v1_error_envelope(body: &serde_json::Value, expected_code: &str) {
        assert!(
            body.get("error").is_some(),
            "missing top-level error: {body}"
        );
        assert_eq!(body["error"]["code"], expected_code);
        assert_eq!(body["error"]["type"], expected_code);
        assert!(body["error"]["message"]
            .as_str()
            .is_some_and(|message| !message.trim().is_empty()));
        assert!(body["error"].get("param").is_some());
        assert!(body["error"]["param"].is_null());
    }

    fn assert_no_fake_chat_success(body: &serde_json::Value) {
        assert!(
            body.get("choices").is_none(),
            "refusal must not include choices: {body}"
        );
        assert!(
            body.get("data").is_none(),
            "chat refusal must not include data: {body}"
        );
        assert_ne!(body["object"], "chat.completion");
    }

    fn assert_no_fake_embedding_success(body: &serde_json::Value) {
        assert!(
            body.get("data").is_none(),
            "refusal must not include embedding data: {body}"
        );
        assert!(
            body.get("choices").is_none(),
            "embedding refusal must not include choices: {body}"
        );
        assert_ne!(body["object"], "list");
    }

    async fn insert_test_conversation(state: &AppState, id: &str, model_id: Option<String>) {
        let now = now();
        state
            .inner
            .lock()
            .await
            .conversations
            .push(ConversationRecord {
                id: id.into(),
                title: "Test conversation".into(),
                model_id,
                created_at: now.clone(),
                updated_at: now,
                messages: Vec::new(),
                memory_context: None,
            });
    }

    async fn insert_test_memory(state: &AppState, id: &str) {
        let now = now();
        state.inner.lock().await.memories.push(MemoryRecord {
            id: id.into(),
            title: "Test memory".into(),
            body: "A useful memory".into(),
            scope: "General".into(),
            created_at: now.clone(),
            updated_at: now,
        });
    }

    fn test_state(models: Vec<ModelRecord>, active_model_id: Option<String>) -> AppState {
        test_state_at(
            std::env::temp_dir().join(format!("fathom-test-state-{}.json", Uuid::new_v4())),
            models,
            active_model_id,
        )
    }

    fn test_state_at(
        model_state_path: PathBuf,
        models: Vec<ModelRecord>,
        active_model_id: Option<String>,
    ) -> AppState {
        AppState {
            inner: Arc::new(Mutex::new(Store {
                models,
                active_model_id,
                ..Store::default()
            })),
            model_state_path,
        }
    }

    fn test_chunk(id: &str, document_id: &str, text: &str) -> VectorIndexChunk {
        VectorIndexChunk {
            id: id.into(),
            document_id: document_id.into(),
            text: text.into(),
            byte_start: 0,
            byte_end: text.len(),
        }
    }

    fn write_test_bert_safetensors_header(path: &std::path::Path) {
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
        let mut tensors = serde_json::Map::new();
        let mut data = Vec::new();
        let mut offset = 0usize;
        for (name, shape) in specs {
            let element_count = shape.iter().product::<usize>();
            let byte_len = element_count * 4;
            tensors.insert(
                name.to_string(),
                serde_json::json!({"dtype":"F32","shape":shape,"data_offsets":[offset, offset + byte_len]}),
            );
            for index in 0..element_count {
                let value = match name {
                    "embeddings.word_embeddings.weight" => match index / 4 {
                        0 => [0.5_f32, -0.5, 1.0, -1.0][index % 4],
                        1 => (index % 4 + 1) as f32,
                        2 => [2.0_f32, 1.0, 0.0, -1.0][index % 4],
                        _ => 0.0,
                    },
                    "embeddings.LayerNorm.weight"
                    | "encoder.layer.0.attention.output.LayerNorm.weight"
                    | "encoder.layer.0.output.LayerNorm.weight" => 1.0,
                    _ => 0.0,
                };
                data.extend_from_slice(&value.to_le_bytes());
            }
            offset += byte_len;
        }
        let header = serde_json::Value::Object(tensors).to_string();
        let mut bytes = (header.len() as u64).to_le_bytes().to_vec();
        bytes.extend_from_slice(header.as_bytes());
        bytes.extend_from_slice(&data);
        std::fs::write(path, bytes).unwrap();
    }

    fn test_chat_request(model: Option<&str>) -> V1ChatCompletionRequest {
        V1ChatCompletionRequest {
            model: model.map(str::to_string),
            messages: vec![ChatMessage {
                role: "user".into(),
                content: "Hello".into(),
            }],
            max_tokens: Some(1),
            temperature: None,
            top_p: None,
            top_k: None,
            stream: None,
            retrieval: None,
        }
    }

    fn test_external_model(id: &str) -> ModelRecord {
        let mut model = test_model(id);
        model.name = "External GPT".into();
        model.provider_kind = "external".into();
        model.model_path = None;
        model.runtime_model_name = Some("gpt-4o-mini".into());
        model.format = Some("OpenAI-compatible API".into());
        model.source = Some("OpenAI-compatible".into());
        model.engine = Some("External API".into());
        model.api_base = Some("https://api.example.test/v1".into());
        model.api_key_configured = Some(true);
        model.capability_status = "planned".into();
        model.capability_summary =
            "External OpenAI-compatible API details are connected as metadata only; Fathom does not proxy chat to external endpoints yet."
                .into();
        model.backend_lanes = vec!["external-openai".into()];
        model.task = None;
        model
    }

    fn test_model(id: &str) -> ModelRecord {
        ModelRecord {
            id: id.into(),
            name: "DistilGPT-2 fixture".into(),
            status: "ready".into(),
            provider_kind: "local".into(),
            model_path: Some("/tmp/distilgpt2".into()),
            runtime_model_name: Some(id.into()),
            format: Some("SafeTensors".into()),
            source: Some("Local artifact".into()),
            engine: Some("Fathom".into()),
            quant: None,
            hf_repo: None,
            hf_filename: None,
            bytes_downloaded: None,
            total_bytes: None,
            progress: None,
            install_error: None,
            api_base: None,
            api_key_configured: None,
            capability_status: "runnable".into(),
            capability_summary: "test model".into(),
            backend_lanes: vec!["safetensors-hf".into()],
            task: Some("text_generation".into()),
            download_manifest: None,
        }
    }
}
