export interface ChatRequest {
  message: string;
  session_id?: string;
}

export interface ChatResponse {
  final_response: string;
  category: string;
  confidence: 'high' | 'medium' | 'low' | 'none';
  escalated: boolean;
  lead_capture_requested: boolean;
  internal_note: string;
}

export interface ChatMessage {
  role: 'user' | 'assistant';
  text: string;
  escalated?: boolean;
  leadCaptureRequested?: boolean;
}

