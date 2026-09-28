import seededFloat from "./SeededFloat";
import { SCENES } from "../components/Data/mockData";

export function hashSeed(str) {
  let h = 0;
  for (let i = 0; i < str.length; i++) h = (Math.imul(31, h) + str.charCodeAt(i)) | 0;
  return Math.abs(h);
}

export function classifyQuery(text) {
  const t = text.toLowerCase();
  let taskType = "search";
  if (/(changed|before and after|increase|decrease|difference|compare)/.test(t)) taskType = "change";
  else if (/(what|how many|how much|where is|is there)/.test(t)) taskType = "vqa";
  else if (/(describe|caption|what does .* look like)/.test(t)) taskType = "caption";
  else if (/(highlight|locate|find the|point out)/.test(t)) taskType = "grounding";
  else if (/(both|optical and sar|combine|fusion|multimodal)/.test(t)) taskType = "fusion";

  const seed = hashSeed(text);
  let scene = null;
  for (const s of SCENES) {
    if (s.tags.some((tag) => t.includes(tag))) { scene = s; break; }
  }
   if (!scene) {
     scene = SCENES[seed % SCENES.length];
   }
  const semantic = seededFloat(seed, 0.78, 0.97);
  const geo = seededFloat(seed + 1, 0.82, 0.99);
  const temporal = seededFloat(seed + 2, 0.7, 0.96);
  const modality = seededFloat(seed + 3, 0.75, 1.0);
  const change = seededFloat(seed + 4, 0.55, 0.94);
  const composite = semantic * 0.35 + geo * 0.25 + temporal * 0.2 + modality * 0.1 + change * 0.1;

  const explanations = {
    search: `Retrieved ${scene.name} as the closest catalog match. Ranking combined semantic similarity with geographic and temporal proximity to the query.`,
    change: `Comparing the ${scene.date} observation against its paired acquisition, index deltas indicate measurable surface change consistent with the query criteria.`,
    vqa: `Answer derived from a single-image visual query over ${scene.name}, cross-checked against the scene's spectral bands.`,
    caption: `Generated a structured description of ${scene.name} based on visual and spectral characteristics of the scene.`,
    grounding: `Localized the referring expression within ${scene.name} and returned a bounding region with confidence.`,
    fusion: `Synthesized optical and SAR observations over ${scene.name} into a single combined assessment.`,
  };

  return { taskType, scene, scores: { semantic, geo, temporal, modality, change, composite }, explanation: explanations[taskType] };
}