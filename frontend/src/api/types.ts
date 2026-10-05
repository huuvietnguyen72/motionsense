export type Split = 'train' | 'test';
export interface Health {
  app: 'motionsense';
  status: 'ok';
  data_ready: boolean;
  models_ready: boolean;
}
export interface ErrorDetail { row: number | null; column: string | null; message: string }
export interface ApiError { error: { code: string; message: string; details: ErrorDetail[] } }
export interface Page<T> { items: T[]; offset: number; limit: number; total: number }
export interface Probability { label_id: number; value: number }
export interface Prediction {
  sample_id: string | null;
  model_id: string;
  predicted_label: number;
  probabilities: Probability[];
  actual_label: number | null;
}
export interface Signal { time_seconds: number[]; x: number[]; y: number[]; z: number[]; unit: 'g' }
export interface Sample {
  sample_id: string;
  dataset_id: string;
  split: Split;
  subject_id: number;
  features: Record<string, number>;
  actual_label: number;
  signal: Signal;
}
export interface DatasetInfo {
  dataset_id: string | null;
  ready: boolean;
  source_url: string;
  license: string;
  feature_count: number;
  split_counts: { train: number; test: number };
  subjects: { subject_id: number; split: Split; sample_count: number }[];
  activities: { label_id: number; name_vi: string; count: number }[];
}
export interface FeatureInfo { feature_id: string; name: string }
export interface ModelInfo {
  model_id: string;
  dataset_id: string;
  profile: 'full' | 'reduced';
  trees: number;
  seed: number;
  feature_ids: string[];
  created_at: string;
  status: 'ready' | 'unavailable';
}
export interface ModelReport {
  model_id: string;
  labels: number[];
  accuracy: number;
  macro_f1: number;
  train_accuracy: number;
  oob_score: number | null;
  oob_warning: string | null;
  confusion_matrix: number[][];
  per_class: { label_id: number; precision: number; recall: number; f1: number; support: number }[];
  feature_importance: { feature_id: string; name: string; value: number }[];
  test_count: number;
  split_strategy: 'uci_subject_split';
}
export interface Job {
  job_id: string;
  status: 'queued' | 'running' | 'succeeded' | 'failed' | 'interrupted';
  stage: string;
  model_id: string | null;
  error: ApiError | null;
}
export interface Session {
  session_id: string;
  dataset_id: string;
  split: Split;
  subject_id: number;
  model_id: string;
  status: 'ready' | 'running' | 'paused' | 'finished';
  cursor: number;
  total: number;
  speed: 0.5 | 1 | 2;
  created_at: string;
  updated_at: string;
  finish_reason: 'complete' | 'user' | null;
  last_prediction: Prediction | null;
}
export interface SessionRow { ordinal: number; sample_id: string; prediction: Prediction; processed_at: string }
export interface SessionSummary {
  processed: number;
  labeled: number;
  correct: number;
  counts: { label_id: number; count: number }[];
  session_accuracy: number | null;
}
