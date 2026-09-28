/**
 * SatQuery AI - Catalog API Client
 * Real API calls for scene catalog operations
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
 * Search catalog with query
 * @param {Object} params
 * @param {string} [params.query_text] - Natural language query
 * @param {string} [params.sensor] - 'Sentinel-1', 'Sentinel-2', 'both'
 * @param {string} [params.object] - Land cover object
 * @param {number} [params.top_k=10] - Max results
 * @returns {Promise<SceneMetadata[]>}
 */
export async function searchCatalog(params = {}) {
  const query = new URLSearchParams();
  Object.entries(params).forEach(([k, v]) => v !== undefined && query.append(k, v));
  return request(`/catalog/search?${query}`);
}

/**
 * Get all scenes (with optional filters)
 * @param {Object} params
 * @param {string} [params.sensor] - Filter by sensor
 * @param {string} [params.modality] - Filter by modality
 * @param {number} [params.limit=100] - Max results
 * @returns {Promise<SceneMetadata[]>}
 */
export async function getCatalog(params = {}) {
  const query = new URLSearchParams();
  Object.entries(params).forEach(([k, v]) => v !== undefined && query.append(k, v));
  return request(`/catalog?${query}`);
}

/**
 * Get scene by ID
 * @param {string} sceneId
 * @returns {Promise<SceneMetadata>}
 */
export async function getScene(sceneId) {
  return request(`/catalog/${sceneId}`);
}

/**
 * Get paired scene (for fusion/change)
 * @param {string} sceneId
 * @returns {Promise<SceneMetadata>}
 */
export async function getPairedScene(sceneId) {
  return request(`/catalog/${sceneId}/pair`);
}

/**
 * TypeScript-like type definitions for reference
 */
/*
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
  origin?: string;
  lat?: number;
  lon?: number;
}

export interface CatalogSearchParams {
  query_text?: string;
  sensor?: 'Sentinel-1' | 'Sentinel-2' | 'both';
  object?: string;
  start_date?: string;
  end_date?: string;
  top_k?: number;
}
*/