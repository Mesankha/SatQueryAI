/**
 * SatQuery AI - Query Backend API Client
 * Real API calls to /query, /upload, /report endpoints
 */

const API_BASE = '/api';

async function request(endpoint, options = {}) {
  const res = await fetch(`${API_BASE}${endpoint}`, {
    headers: {
      'Content-Type': 'application/json',
      ...options.headers,
    },
    ...options,
  });

  if (!res.ok) {
    const err = await res.json().catch(() => ({ detail: `HTTP ${res.status}` }));
    throw new Error(err.detail || `HTTP ${res.status}`);
  }

  return res.json();
}

/**
 * Natural language query
 * @param {string} queryText - User query
 * @param {string} [imageId] - Optional scene ID
 * @param {object} [aoi_override] - Optional AOI override
 * @returns {Promise<QueryResponse>}
 */
export async function queryBackend(queryText, imageId = null, aoi_override = null) {
  return request('/query', {
    method: 'POST',
    body: JSON.stringify({ query_text: queryText, image_id: imageId, aoi_override }),
  });
}

/**
 * Direct image upload for caption/VQA/fusion/change
 * @param {File} file1 - Primary image
 * @param {File} [file2] - Optional second image
 * @param {string} [question] - VQA question
 * @param {string} [intent] - Explicit intent: 'caption', 'vqa', 'fusion', 'change'
 * @returns {Promise<UploadResponse>}
 */
export async function uploadBackend(file1, file2 = null, question = null, intent = null) {
  const formData = new FormData();
  formData.append('file', file1);
  if (file2) formData.append('file2', file2);
  if (question) formData.append('question', question);
  if (intent) formData.append('intent', intent);

  const res = await fetch(`${API_BASE}/upload`, {
    method: 'POST',
    body: formData,
  });

  if (!res.ok) {
    const err = await res.json().catch(() => ({ detail: `HTTP ${res.status}` }));
    throw new Error(err.detail || `HTTP ${res.status}`);
  }

  return res.json();
}

/**
 * Downloadable report
 * @param {string} queryId - Query ID from response
 * @param {string} [format='json'] - 'json' or 'pdf'
 * @param {string} [queryText] - Original query for context
 * @returns {Promise<ReportResponse>}
 */
export async function reportBackend(queryId, format = 'json', queryText = null) {
  const params = new URLSearchParams({ query_id: queryId, format });
  if (queryText) params.append('query_text', queryText);

  const res = await fetch(`${API_BASE}/report?${params}`);
  if (!res.ok) {
    const err = await res.json().catch(() => ({ detail: `HTTP ${res.status}` }));
    throw new Error(err.detail || `HTTP ${res.status}`);
  }
  return res.json();
}

/**
 * Health check
 */
export async function healthBackend() {
  const res = await fetch(`${API_BASE}/health`);
  return res.json();
}

/**
 * Config status (mock/real flags)
 */
export async function configStatusBackend() {
  const res = await fetch(`${API_BASE}/config-status`);
  return res.json();
}

/**
 * Catalog endpoints
 */
export async function catalogListBackend(params = {}) {
  const searchParams = new URLSearchParams();
  Object.entries(params).forEach(([k, v]) => { if (v !== undefined && v !== null) searchParams.append(k, v); });
  const res = await fetch(`${API_BASE}/catalog?${searchParams}`);
  return res.json();
}

export async function catalogSearchBackend(params) {
  const searchParams = new URLSearchParams();
  Object.entries(params).forEach(([k, v]) => { if (v !== undefined && v !== null) searchParams.append(k, v); });
  const res = await fetch(`${API_BASE}/catalog/search?${searchParams}`);
  return res.json();
}

export async function catalogGetSceneBackend(sceneId) {
  const res = await fetch(`${API_BASE}/catalog/${sceneId}`);
  return res.json();
}

export async function catalogGetPairBackend(sceneId) {
  const res = await fetch(`${API_BASE}/catalog/${sceneId}/pair`);
  return res.json();
}

/**
 * TypeScript-like type definitions for reference
 */
/*
export interface QueryResponse {
  query_id: string;
  results: ResultItem[];
  trace: ExecutionTrace;
}

export interface UploadResponse {
  query_id: string;
  results: ResultItem[];
  trace: ExecutionTrace;
  uploaded_files: string[];
}

export interface ResultItem {
  image_id: string;
  rank: number;
  score: number;
  score_breakdown: ScoreBreakdown;
  metadata: SceneMetadata;
  model_outputs: ModelOutputs;
  explanation_text: string;
  confidence_band: 'high' | 'medium' | 'low' | 'unknown';
  confidence_reason: string;
}

export interface ModelOutputs {
  caption?: string;
  vqa_answer?: string;
  grounding_bbox?: Record<string, any>;
  change_description?: string;
  fusion_statement?: string;
  sar_features?: SarFeatures;
  sar_caption?: string;
  sar_vqa_answer?: string;
  optical_caption?: string;
  sar_cross_check?: SarCrossCheck;
}

export interface SarFeatures {
  water_fraction: number;
  builtup_fraction: number;
  log_ratio_mean: number;
  notes?: string;
}

export interface SarCrossCheck {
  disagreement: boolean;
  severity: 'minor' | 'major';
  details: string;
  water_fraction: number;
  builtup_fraction: number;
  log_ratio_mean: number;
}

export interface ExecutionTrace {
  task_selected: string;
  tools_called: ToolCall[];
  candidates_considered: number;
  candidates_after_filter: number;
  fallback_used: boolean;
}

export interface ToolCall {
  tool_name: string;
  model_name: string;
  params: Record<string, any>;
  latency_ms: number;
  success: boolean;
}

export interface SceneMetadata {
  scene_id: string;
  sensor: string;
  modality: string;
  acquisition_time: string;
  geometry_wkt: string;
  file_path?: string;
  cloud_cover?: number;
  object?: string;
  paired_scene_id?: string;
}
*/